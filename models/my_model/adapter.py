"""Adapter skeleton for `my_model`.

This is the template every model adapter follows. The pattern is:

  1. Subclass `BaseModel`.
  2. Declare `metadata` (registry id, supported modes, native layout).
  3. Decorate with `@register` so the framework can find it.
  4. Implement:
       - build(cfg)         -> construct the vendored upstream model
       - forward(x)         -> native forward pass
       - to_canonical / from_canonical  -> for the evaluator
       - load_weights / save_weights
       - train(...)         -> delegate to upstream training script
       - native_model       -> property exposing the vendored model

Everything else (data loaders, training loops, optimizers, schedulers)
stays in the upstream code — the adapter is glue, not a rewrite.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

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

from .model import PlaceholderModel


@register
class MyModelAdapter(BaseModel):
    """Adapter for `my_model`. The only file we author; `model.py` is vendored."""

    metadata = ModelMetadata(
        name="my_model",
        family="placeholder",
        supported_modes=[Mode.FROM_SCRATCH, Mode.FINETUNE, Mode.ZERO_SHOT],
        default_mode=Mode.FROM_SCRATCH,
        native_input_layout="(B, C, H, W) float32",
        native_output_layout="(B, C, H, W) float32",
        upstream_version="placeholder-0.1",
        paper=None,
    )

    # --- Construction --------------------------------------------------

    def __init__(self, native: PlaceholderModel, cfg: Dict[str, Any]):
        self._native = native
        self._cfg = cfg

    @classmethod
    def build(cls, cfg: Dict[str, Any]) -> "MyModelAdapter":
        """Construct the vendored model from a framework config dict."""
        native = PlaceholderModel(
            in_channels=cfg.get("in_channels", 3),
            out_channels=cfg.get("out_channels", 3),
            hidden=cfg.get("hidden", 32),
        )
        return cls(native=native, cfg=cfg)

    # --- Inference (NATIVE) --------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        return self._native(x)

    # --- Canonicalization ---------------------------------------------

    def to_canonical(self, y_native: Tensor) -> CanonicalSample:
        """Native layout is (B, C, H, W); canonical is a CanonicalSample."""
        # Final canonical layout is decided in task #1. For now, wrap as-is.
        return CanonicalSample(fields=y_native, metadata={"source": "native"})

    def from_canonical(self, sample: CanonicalSample) -> Tensor:
        """Inverse: pull `fields` back out as a native tensor."""
        return sample.fields

    # --- Weights -------------------------------------------------------

    def load_weights(self, source: WeightSource, *, strict: bool = True) -> None:
        """Load weights into the vendored model.

        Real implementations would dispatch on source type:
          - local Path  -> torch.load
          - URL         -> download then load
          - HF Hub id   -> hf_hub_download then load
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No weights at {path}")
        state = torch.load(path, map_location="cpu")
        self._native.load_state_dict(state, strict=strict)

    def save_weights(self, destination: WeightSource) -> None:
        torch.save(self._native.state_dict(), Path(destination))

    # --- Training (NATIVE loop) ----------------------------------------

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        *,
        cfg: Dict[str, Any],
        callbacks: Optional[list[Callable]] = None,
    ) -> TrainingResult:
        """Delegate to the upstream model's native training entry point.

        This stub demonstrates the shape; real adapters would call into the
        upstream training script (or its `trainer.fit()` if it ships one)
        and forward the framework's data + callbacks.
        """
        # PLACEHOLDER: a from-scratch PyTorch loop. Replace with the upstream
        # model's native train() in the real adapter.
        device = torch.device(cfg.get("device", "cpu"))
        optim = torch.optim.Adam(self._native.parameters(), lr=cfg.get("lr", 1e-3))
        loss_fn = torch.nn.MSELoss()

        self._native.to(device).train()
        history: Dict[str, list[float]] = {"train_loss": []}

        for epoch in range(int(cfg.get("epochs", 1))):
            epoch_loss = 0.0
            for batch in train_dataset:
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = self._native(x)
                loss = loss_fn(pred, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            history["train_loss"].append(epoch_loss)
            for cb in callbacks or []:
                cb(epoch=epoch, loss=epoch_loss)

        return TrainingResult(history=history)

    # --- Introspection -------------------------------------------------

    @property
    def native_model(self) -> PlaceholderModel:
        return self._native
