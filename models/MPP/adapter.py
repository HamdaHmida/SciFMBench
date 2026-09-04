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
        native_input_layout="(T, B, C, H, W) float32 — time window of length T",
        native_output_layout="(B, C, H, W) float32 — only the last predicted step",
        # MPP normalizes internally (per-sample over T, H, W) and denormalizes
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
        """
        labels = self._state_labels if self._state_labels is not None else [0, 1]
        bcs = self._bcs if self._bcs is not None else torch.zeros(1, 2, dtype=torch.long, device=x.device)
        return self._native(x, labels, bcs)

    # --- Canonicalization (used by the evaluator) -------------------------

    def to_canonical(self, y_native: Tensor) -> CanonicalSample:
        """Wrap native (B, C, H, W) output as a CanonicalSample.

        The final canonical layout (channels=vx/vy/p/vorticity convention,
        dtype, normalization) is finalized in task #1. For now: pass-through
        with provenance metadata so the evaluator can audit it.
        """
        return CanonicalSample(
            fields=y_native,
            metadata={
                "source": "MPP",
                "layout": "single_step",
                # MPP already denormalizes internally; record that fact so the
                # evaluator doesn't double-apply normalization.
                "already_in_input_units": True,
            },
        )

    def from_canonical(self, sample: CanonicalSample) -> Tensor:
        """Pull the canonical tensor back out as a native input.

        The upstream model expects (T, B, C, H, W). If the canonical sample
        is single-step (B, C, H, W), we add a T axis of length 1 here.
        Final canonical-vs-native axis mapping is locked in by task #1.
        """
        x = sample.fields
        if x.ndim == 4:  # (B, C, H, W) -> (T=1, B, C, H, W)
            x = x.unsqueeze(0)
        return x

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
                pred = self._native(window, self._state_labels or [0, 1], self._bcs or torch.zeros(1, 2, dtype=torch.long, device=device))
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
