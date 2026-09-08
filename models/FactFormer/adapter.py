"""Adapter for the FactFormer FABlock2D model.

This is the ONLY file in `models/FactFormer/` that we author. The other files
(`attention.py`, `basics.py`, `factorization_module.py`,
`positional_encoding_module.py`) are vendored from upstream and left untouched.

Native interface summary
------------------------
`FABlock2D` (factorization_module.py):
    forward(u, pos_lst)  # u: (B, Nx, Ny, C)  — channel-last 2D field
                         # pos_lst: tuple(pos_x, pos_y)
                         #   pos_x: (B, Nx, 2) — 2D coords along x-axis
                         #   pos_y: (B, Ny, 2) — 2D coords along y-axis
    returns: (B, Nx, Ny, dim_out)

The framework's smoke test builds canonical data `(B, Nx, Ny, T, C)` and
permutes it to native `(T, B, C, Nx, Ny)` before calling `forward()`.
This adapter bridges that layout:

    forward(x_native) where x_native: (T, B, C, Nx, Ny)
        -> take last timestep: (B, C, Nx, Ny)
        -> permute to channel-last: (B, Nx, Ny, C)
        -> generate positional encodings for Nx, Ny
        -> call FABlock2D -> (B, Nx, Ny, dim_out)
        -> permute to (B, dim_out, Nx, Ny)  [channel-first for framework]

For the canonical round-trip, `to_canonical` wraps the single-step
output as `(B, Nx, Ny, T=1, C_out)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

# Vendored, unmodified.
from .factorization_module import FABlock2D
from .positional_encoding_module import GaussianFourierFeatureTransform


# --- Defaults --------------------------------------------------------------

_FACTFORMER_DEFAULTS: Dict[str, Any] = dict(
    dim=64,              # input channel dim
    dim_head=32,         # attention head dim
    latent_dim=32,       # pooled latent dim
    heads=4,             # number of attention heads
    dim_out=64,          # output channel dim
    use_rope=True,       # rotary positional encoding
    kernel_multiplier=3,
    scaling_factor=1.0,
    # Adapter-side derived values (filled by smoke test at build time):
    # _T, _C, _Nx, _Ny
)


def _generate_grid_positions(Nx: int, Ny: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Generate 1D grid coordinates per axis for FactFormer positional encoding.

    FABlock2D uses LowRankKernel with pos_dim=1, which expects 1D coordinates
    along each axis (x-coords for x-axis attention, y-coords for y-axis).

    Returns:
        pos_x: (B=1, Nx, 1) — x-coordinates normalized to [-1, 1]
        pos_y: (B=1, Ny, 1) — y-coordinates normalized to [-1, 1]
    """
    xs = torch.linspace(-1, 1, Nx, device=device)
    ys = torch.linspace(-1, 1, Ny, device=device)

    # pos_x: (Nx,) -> (B=1, Nx, 1)
    pos_x = xs.unsqueeze(0).unsqueeze(-1)
    # pos_y: (Ny,) -> (B=1, Ny, 1)
    pos_y = ys.unsqueeze(0).unsqueeze(-1)

    return pos_x, pos_y


# --- adapter ---------------------------------------------------------------


@register
class FactFormerAdapter(BaseModel):
    """Adapter for the FactFormer 2D Factorized Attention Block (FABlock2D).

    The vendored model is a spatial attention block with factorized
    attention along each axis. It takes a channel-last 2D field
    (B, Nx, Ny, C) and returns (B, Nx, Ny, C_out).

    For PDE surrogate use, this adapter treats a (T, B, C, Nx, Ny) window
    as a time series and applies the block to the last timestep's field.
    The output is permuted to channel-first (B, C_out, Nx, Ny) to match
    the framework's expected native output layout.
    """

    metadata = ModelMetadata(
        name="FactFormer",
        family="factorized_attention",
        supported_modes=[Mode.FROM_SCRATCH, Mode.FINETUNE, Mode.ZERO_SHOT],
        default_mode=Mode.FROM_SCRATCH,
        native_input_layout=(
            "(T, B, C, Nx, Ny) float32 — adapter takes last timestep, "
            "permutes to (B, Nx, Ny, C), applies FABlock2D, "
            "returns (B, C_out, Nx, Ny)"
        ),
        native_output_layout="(B, C_out, Nx, Ny) float32 — single-step prediction",
        canonical_layout="(B, Nx, Ny, T, C) float32",
        upstream_version="factformer-vendored-0.1",
        paper="Li et al., 'FactFormer: Factorized Attention for Operator Learning' (2024)",
    )

    # --- Construction ------------------------------------------------------

    def __init__(self, native: FABlock2D, cfg: Dict[str, Any]):
        self._native = native
        self._cfg = cfg
        # Cached derived values (set at build time from per-model canonical shape).
        self._T: int = int(cfg.get("_T", 1))
        self._C: int = int(cfg.get("_C", cfg.get("dim", 64)))
        self._Nx: int = int(cfg.get("_Nx", 64))
        self._Ny: int = int(cfg.get("_Ny", 64))

    @classmethod
    def build(cls, cfg: Dict[str, Any]) -> "FactFormerAdapter":
        merged = {**_FACTFORMER_DEFAULTS, **cfg}

        native = FABlock2D(
            dim=int(merged["dim"]),
            dim_head=int(merged["dim_head"]),
            latent_dim=int(merged["latent_dim"]),
            heads=int(merged["heads"]),
            dim_out=int(merged["dim_out"]),
            use_rope=bool(merged["use_rope"]),
            kernel_multiplier=int(merged["kernel_multiplier"]),
            scaling_factor=float(merged["scaling_factor"]),
        )
        return cls(native=native, cfg=merged)

    # --- Inference (NATIVE: (T, B, C, Nx, Ny) -> (B, C_out, Nx, Ny)) ------

    def forward(self, x: Tensor) -> Tensor:
        """Run the vendored `FABlock2D` on a (T, B, C, Nx, Ny) window.

        Steps:
            1. Take the last timestep: (B, C, Nx, Ny).
            2. Permute to channel-last: (B, Nx, Ny, C).
            3. Generate positional encodings for the spatial grid.
            4. Call the vendored FABlock2D.
            5. Permute output to channel-first: (B, C_out, Nx, Ny).
        """
        if x.ndim != 5:
            raise ValueError(
                f"FactFormerAdapter.forward expects (T, B, C, Nx, Ny); got {tuple(x.shape)}"
            )
        T, B, C, Nx, Ny = x.shape

        # 1) Take last timestep: (B, C, Nx, Ny)
        x_last = x[-1]  # (B, C, Nx, Ny)

        # 2) Permute to channel-last: (B, Nx, Ny, C)
        x_cl = x_last.permute(0, 2, 3, 1).contiguous()  # (B, Nx, Ny, C)

        # Sanity check: C should match the model's dim
        if C != self._native.dim:
            raise RuntimeError(
                f"FABlock2D was built with dim={self._native.dim} but "
                f"forward received C={C}. Did you change channels without rebuilding?"
            )

        # 3) Generate positional encodings on the input's device.
        pos_x, pos_y = _generate_grid_positions(Nx, Ny, x.device)
        # Expand to batch size B
        pos_x = pos_x.expand(B, -1, -1)  # (B, Nx, 2)
        pos_y = pos_y.expand(B, -1, -1)  # (B, Ny, 2)
        pos_lst = (pos_x, pos_y)

        # 4) Native FABlock2D forward.
        # Note: FABlock2D output is (B, Nx*Ny, C_out) — it merges spatial dims.
        y_flat = self._native(x_cl, pos_lst)  # (B, Nx*Ny, C_out)

        # 5) Reshape to (B, Nx, Ny, C_out) then permute to channel-first.
        C_out = y_flat.shape[-1]
        y_cl = y_flat.view(B, Nx, Ny, C_out).contiguous()
        y = y_cl.permute(0, 3, 1, 2).contiguous()  # (B, C_out, Nx, Ny)
        return y

    # --- Canonicalization --------------------------------------------------

    def to_canonical(self, y_native: Tensor) -> CanonicalSample:
        """Convert the adapter's native (B, C_out, Nx, Ny) output to
        canonical (B, Nx, Ny, T=1, C_out) by adding a singleton time axis."""
        if y_native.ndim != 4:
            raise ValueError(
                f"FactFormerAdapter.to_canonical expects (B, C, Nx, Ny); "
                f"got {tuple(y_native.shape)}"
            )
        # (B, C_out, Nx, Ny) -> (B, Nx, Ny, C_out) -> (B, Nx, Ny, T=1, C_out)
        y_canon = y_native.permute(0, 2, 3, 1).unsqueeze(3)
        return CanonicalSample(
            fields=y_canon.contiguous(),
            metadata={
                "source": "FactFormer",
                "layout": "single_step",
                "already_in_input_units": True,
            },
        )

    def from_canonical(self, sample: CanonicalSample) -> Tensor:
        """Inverse: turn a canonical (B, Nx, Ny, T, C) sample into the
        adapter's native (T, B, C, Nx, Ny) input layout."""
        x = sample.fields
        if x.ndim != 5:
            raise ValueError(
                f"FactFormerAdapter.from_canonical expects (B, Nx, Ny, T, C); "
                f"got {tuple(x.shape)}"
            )
        # (B, Nx, Ny, T, C) -> (T, B, C, Nx, Ny)
        return x.permute(3, 0, 4, 1, 2).contiguous()

    # --- Weights -----------------------------------------------------------

    def load_weights(self, source: WeightSource, *, strict: bool = True) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"No FactFormer weights at {path}. The vendored code ships no "
                f"pretrained state_dict, so from-scratch is the default mode."
            )
        state = torch.load(path, map_location="cpu")
        missing, unexpected = self._native.load_state_dict(state, strict=strict)
        if not strict and (missing or unexpected):
            print(
                f"[FactFormer] load_weights(strict=False): "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    def save_weights(self, destination: WeightSource) -> None:
        torch.save(self._native.state_dict(), Path(destination))

    # --- Training (NATIVE loop delegation) ---------------------------------

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        *,
        cfg: Dict[str, Any],
        callbacks: Optional[list[Callable]] = None,
    ) -> TrainingResult:
        """Train the vendored FABlock2D with a native PyTorch loop.

        Per Option C, the framework does NOT impose its own optimizer
        / loss on the model. The vendored FABlock2D ships no training script,
        so we wire a minimal loop here.
        """
        device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        lr = float(cfg.get("lr", 1e-3))
        epochs = int(cfg.get("epochs", 1))
        weight_decay = float(cfg.get("weight_decay", 1e-5))

        self._native.to(device).train()
        optim = torch.optim.AdamW(
            [p for p in self._native.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        history: Dict[str, list[float]] = {"train_loss": []}
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_dataset:
                # Each batch yields (window, target).
                #   window: (T, b, C, Nx, Ny)
                #   target: (b, C_out, Nx, Ny)
                window, target = batch[0], batch[1]
                window = window.to(device)
                target = target.to(device)
                pred = self.forward(window)  # adapter.forward() handles the bridge
                loss = loss_fn(pred, target)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._native.parameters(), max_norm=cfg.get("grad_clip", 1.0)
                )
                optim.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg)
            for cb in callbacks or []:
                cb(epoch=epoch, loss=avg)

        return TrainingResult(history=history)

    # --- Introspection -----------------------------------------------------

    @property
    def native_model(self) -> FABlock2D:
        return self._native