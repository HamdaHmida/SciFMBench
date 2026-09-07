"""Adapter for the MPP (Axial ViT for PDE) model.

This is the ONLY file in `models/MPP/` that we author. The other files
(avit.py, spatial_modules.py, time_modules.py, mixed_modules.py,
shared_modules.py) are vendored from upstream and left untouched.

Why the adapter exists
----------------------
The vendored model's native surface is:

    model = build_avit(params)              # params is an attribute-namespace
    y = model(x, state_labels, bcs)         # x: (T, B, C, H, W); y: (B, C, H, W)
                                            # state_labels: list[int] length C
                                            # bcs: (B, 2) boundary flags

Our framework's `BaseModel.forward(x)` is single-arg. The adapter bridges
this by holding `state_labels` / `bcs` on the instance (set once per eval
or finetune run, since they're constant across the dataset) and re-attaching
them inside `forward()`. This matches how real eval pipelines use the model.

Normalization note
------------------
MPP normalizes its input internally (per-sample mean/std over T, H, W) and
denormalizes the output before returning. So the native output is already
in input-units — `to_canonical` just wraps it without further arithmetic.
The benchmark layer is responsible for any global normalization needed for
fair comparison.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from core.base_model import (
    BaseModel,
    CanonicalSample,
    Mode,
    ModelMetadata,
    TrainingResult,
    WeightSource,
    register,
)

# Vendored, unmodified. We import only what we need; the upstream files
# themselves import each other via try/except relative-imports that work
# whether or not we're inside a package.
from .avit import build_avit, AViT


# --- config -> params translation -----------------------------------------

# Default upstream parameter set; real adapters will pull these from YAML.
_DEFAULTS = dict(
    patch_size=(16, 16),
    embed_dim=768,
    processor_blocks=8,
    n_states=6,
    space_type="axial_attention",
    time_type="attention",
    bias_type="rel",
    block_type="axial",
    num_heads=12,
    gradient_checkpointing=False,
)


def _cfg_to_params(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Translate framework config dict to the attribute-namespace `build_avit` expects."""
    merged = {**_DEFAULTS, **cfg}
    # patch_size must be a tuple — YAML may give a list.
    if isinstance(merged.get("patch_size"), list):
        merged["patch_size"] = tuple(merged["patch_size"])
    return SimpleNamespace(**merged)


# --- the adapter ----------------------------------------------------------


@register
class MPPAdapter(BaseModel):
    """Thin wrapper around the vendored MPP (Axial ViT for PDE) model."""

    metadata = ModelMetadata(
        name="MPP",
        family="axial_vit",
        supported_modes=[Mode.FROM_SCRATCH, Mode.FINETUNE, Mode.ZERO_SHOT],
        default_mode=Mode.FINETUNE,  # the model ships `expand_projections` — finetune is its strength
        # Native layouts (documented so the evaluator knows what to canonicalize).
        native_input_layout="(T, B, C, Nx, Ny) float32 — time window of length T",
        native_output_layout="(B, C, Nx, Ny) float32 — only the last predicted step",
        # Framework canonical layout (the contract every adapter maps to):
        canonical_layout="(B, Nx, Ny, T, C) float32",
        # MPP normalizes internally (per-sample over T, Nx, Ny) and denormalizes
        # before returning — so native output is already in input units.
        upstream_version="vendored-0.1",
        paper=None,
    )

    # --- Construction -----------------------------------------------------

    def __init__(self, native: AViT, cfg: Dict[str, Any]):
        self._native = native
        self._cfg = cfg
        # `state_labels` and `bcs` are constant for a given (model, dataset)
        # pairing. Set them once via `set_inference_context()` before eval,
        # or rely on the upstream defaults if you call `forward(x)` directly.
        self._state_labels: Optional[List[int]] = None
        self._bcs: Optional[Tensor] = None

    @classmethod
    def build(cls, cfg: Dict[str, Any]) -> "MPPAdapter":
        params = _cfg_to_params(cfg)
        native = build_avit(params)
        return cls(native=native, cfg=cfg)

    def set_inference_context(self, state_labels: Sequence[int], bcs: Tensor) -> None:
        """Pin the auxiliary forward-time arguments.

        `state_labels`: list of int, length = C, indexing the channel vocab.
        `bcs`: (B, 2) tensor of boundary-condition flags (0=endpoint, 1=periodic).
        """
        self._state_labels = list(state_labels)
        self._bcs = bcs

    # --- Inference (NATIVE: 3-arg upstream) -------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Single-arg wrapper around the upstream 3-arg forward.

        If `state_labels` / `bcs` haven't been set, default to the upstream's
        `__main__` example values (T=10, bs=4, labels=[0,1]) so the adapter
        is callable in isolation. Real evaluation must call
        `set_inference_context()` first.

        The vendored `SubsampledLinear.forward` does `labels = labels[0]`
        then `len(labels)`, so the upstream expects `labels` to be a
        *list of label-lists* — one per batch element. We accept the
        flatter "channel-list" form (e.g. `[0, 1]`) and replicate it
        across the batch here, so callers can set the labels once and
        not worry about batch size.
        """
        # Determine batch size from the input. Native x is (T, B, C, Nx, Ny).
        B = x.shape[1] if x.ndim >= 2 else 1
        flat_labels = self._state_labels if self._state_labels is not None else [0, 1]
        labels = [list(flat_labels) for _ in range(B)]
        bcs = self._bcs if self._bcs is not None else torch.zeros(B, 2, dtype=torch.long, device=x.device)
        return self._native(x, labels, bcs)

    # --- Canonicalization (used by the evaluator) -------------------------
    #
    # Layout contract (set by the framework, not by MPP):
    #     canonical = (B, Nx, Ny, T, C) float32
    #     MPP native = (T, B, C, Nx, Ny) float32 (input window)
    #                  (B, C, Nx, Ny) float32 (single-step output)
    #
    # `to_canonical` is called on MPP's last-step prediction only.
    # `from_canonical` is called by the evaluator when feeding a canonical
    # window into MPP — T is preserved (MPP is auto-regressive over a window).

    def to_canonical(self, y_native: Tensor) -> CanonicalSample:
        """Convert MPP's native single-step output into a CanonicalSample.

        Permutes (B, C, Nx, Ny) -> (B, Nx, Ny, C), then adds a singleton
        time axis at position 3 so the result has the canonical
        (B, Nx, Ny, T=1, C) layout.
        """
        if y_native.ndim != 4:
            raise ValueError(
                f"MPPAdapter.to_canonical expects (B, C, Nx, Ny); got {tuple(y_native.shape)}"
            )
        # (B, C, Nx, Ny) -> (B, Nx, Ny, C) -> (B, Nx, Ny, T=1, C)
        y_canon = y_native.permute(0, 2, 3, 1).unsqueeze(3)
        return CanonicalSample(
            fields=y_canon.contiguous(),
            metadata={
                "source": "MPP",
                "layout": "single_step",
                "already_in_input_units": True,  # MPP denormalizes internally
            },
        )

    def from_canonical(self, sample: CanonicalSample) -> Tensor:
        """Inverse: convert a canonical (B, Nx, Ny, T, C) window to MPP's
        native (T, B, C, Nx, Ny) input layout.
        """
        x = sample.fields
        if x.ndim != 5:
            raise ValueError(
                f"MPPAdapter.from_canonical expects (B, Nx, Ny, T, C); "
                f"got {tuple(x.shape)}"
            )
        # (B, Nx, Ny, T, C) -> (T, B, C, Nx, Ny)
        return x.permute(3, 0, 4, 1, 2).contiguous()

    # --- Weights ----------------------------------------------------------

    def load_weights(self, source: WeightSource, *, strict: bool = True) -> None:
        """Load weights into the vendored AViT.

        Supports local paths. URL / HF Hub resolution can be added here
        without touching the vendored code.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"No MPP weights at {path}. Drop a state_dict .pt here or extend "
                f"`load_weights` to fetch from URL / HF Hub."
            )
        state = torch.load(path, map_location="cpu")
        # Upstream uses standard PyTorch state_dict; load directly.
        missing, unexpected = self._native.load_state_dict(state, strict=strict)
        if not strict and (missing or unexpected):
            # Surface info; the trainer / eval layer can decide what to do.
            print(f"[MPP] load_weights(strict=False): missing={len(missing)} unexpected={len(unexpected)}")

    def save_weights(self, destination: WeightSource) -> None:
        torch.save(self._native.state_dict(), Path(destination))

    # --- Finetuning helpers (exposed; native logic stays in vendored code) -

    def expand_projections(self, n_new_states: int) -> None:
        """Delegate to upstream — adds new state-variable slots for finetuning."""
        self._native.expand_projections(n_new_states)

    def freeze_middle(self) -> None:
        self._native.freeze_middle()

    def freeze_processor(self) -> None:
        self._native.freeze_processor()

    def unfreeze(self) -> None:
        self._native.unfreeze()

    # --- Training (NATIVE loop delegation) -------------------------------

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        *,
        cfg: Dict[str, Any],
        callbacks: Optional[list[Callable]] = None,
    ) -> TrainingResult:
        """Delegate to a PyTorch training loop using the vendored model.

        Per Option C, training is NATIVE: we don't impose the framework's
        optimizer / scheduler / loss on the model. The upstream code ships
        no training script, so we wire a minimal loop here — but the
        architecture, normalization, and forward pass are entirely upstream's.
        The framework's job is to feed data and call callbacks.
        """
        device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        lr = float(cfg.get("lr", 1e-4))
        epochs = int(cfg.get("epochs", 1))
        T = int(cfg.get("T", 10))  # time-window length

        # Optional finetuning hooks — pass through to vendored helpers.
        if cfg.get("freeze_processor"):
            self._native.freeze_processor()
        if cfg.get("unfreeze"):
            self._native.unfreeze()

        self._native.to(device).train()
        optim = torch.optim.AdamW(
            [p for p in self._native.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=cfg.get("weight_decay", 1e-5),
        )
        loss_fn = torch.nn.MSELoss()

        history: Dict[str, list[float]] = {"train_loss": []}
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_dataset:
                # Each batch must yield (window, target_window, state_labels, bcs).
                # The adapter's set_inference_context() should have been called
                # by the framework before train() to bind state_labels / bcs.
                window, target = batch[0], batch[1]
                window = window.to(device)
                target = target.to(device)

                # Roll the prediction: MPP predicts the next step from a window.
                # Use `is None` (not `or`) for the bcs tensor — `tensor or ...`
                # is illegal in PyTorch because it calls __bool__ on each element.
                b = window.shape[1] if window.ndim >= 2 else 1
                flat_labels = self._state_labels if self._state_labels is not None else [0, 1]
                labels = [list(flat_labels) for _ in range(b)]
                bcs = self._bcs if self._bcs is not None else torch.zeros(b, 2, dtype=torch.long, device=device)
                pred = self._native(window, labels, bcs)
                loss = loss_fn(pred, target[-1] if target.ndim == 5 else target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg)
            for cb in callbacks or []:
                cb(epoch=epoch, loss=avg)

        return TrainingResult(history=history)

    # --- Introspection ----------------------------------------------------

    @property
    def native_model(self) -> AViT:
        return self._native
