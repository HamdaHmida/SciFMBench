"""Adapters for the vendored Kolmogorov-Arnold Network code in `pykan.py`.

This is the ONLY file in `models/KAN/` that we author. `pykan.py` is
vendored from upstream and left untouched.

Why two adapters
----------------
`pykan.py` ships two distinct models, with different input/output layouts,
different hyperparameters, and different scientific uses:

  * `KAN` (lines 242-287 of pykan.py)
        A multilayer Kolmogorov-Arnold Network — a drop-in MLP replacement
        with learnable B-spline activations on the edges. It takes a flat
        feature vector and returns a flat feature vector. No spatial or
        temporal awareness.

  * `KAN_Convolutional_Layer` (lines 290-372 of pykan.py)
        A 2D convolutional KAN: kernel_size, stride, padding — Conv2d
        semantics, but the kernel is a KANLinear over the kernel pixels.
        Input is (B, C_in, H, W), output is (B, C_out, H_out, W_out).

The framework's `BaseModel` declares *one* native layout per adapter, and
the trainer/smoke-test builds inputs in the canonical-to-native layout
(`(T, B, C, Nx, Ny)`). Forcing both into a single adapter would mean
swallowing that layout into a feature vector for one and keeping it as
a 2D map for the other — that is exactly the bridge a per-model adapter
is supposed to own. One file, two `@register`ed classes, one per
vendored model. Each user picks the id that matches the use case
("I want a KAN MLP" vs "I want a KAN conv layer").

Native interface summary
------------------------
`KAN`:
    forward(x)          # x: (B, in_features); returns (B, out_features)
    regularization_loss(...)  # sum over layers; used by the training loop

`KAN_Convolutional_Layer`:
    forward(x)          # x: (B, in_channels, H, W); returns (B, out_channels, H', W')

How we bridge to the framework's `(T, B, C, Nx, Ny)` ↔ `(B, C, Nx, Ny)`
----------------------------------------------------------------------------
`KANAdapter`:
    Treat the whole (T, B, C, Nx, Ny) window as a flat feature vector of
    size T*C*Nx*Ny per sample, run it through the KAN, then take the
    *last* "T-step's worth" of features and reshape back to (B, C, Nx, Ny).
    That gives the same (B, C, Nx, Ny) output the framework expects.

`KANConvAdapter`:
    Apply the convolutional KAN per-timestep (loop over T) and keep the
    last step's output. Output is (B, out_channels, Nx', Ny'). For the
    smoke test we use out_channels=C and padding that preserves spatial
    dims, so the output shape is (B, C, Nx, Ny) — matching the test.
"""
from __future__ import annotations

from pathlib import Path
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

# Vendored, unmodified. We import only the two classes we wrap.
from .pykan import KAN, KAN_Convolutional_Layer


# --- Defaults (small variants; real configs live in YAML) ------------------

# These match the upstream `pykan.py` defaults where possible, scaled down
# for the smoke test. Adapters translate framework config keys to the
# upstream constructor's kwargs.

_KAN_DEFAULTS: Dict[str, Any] = dict(
    # `[in, hidden, hidden, out]` style layer sizes — `KAN` requires the
    # first and last entries to be equal to in_features and out_features.
    # The adapter fills in `in_features` / `out_features` automatically.
    layers_hidden=None,  # type: ignore[assignment]  # built by the adapter
    grid_size=5,
    spline_order=3,
    scale_noise=0.1,
    scale_base=1.0,
    scale_spline=1.0,
    base_activation="SiLU",
    grid_eps=0.02,
    grid_range=[-1.0, 1.0],
    # Adapter-side defaults (not part of upstream KAN's signature):
    hidden_width=32,  # width of the bottleneck hidden layers
    n_hidden=1,       # number of hidden layers between input and output
)

_KANCONV_DEFAULTS: Dict[str, Any] = dict(
    in_channels=1,
    out_channels=1,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),         # default = "same" for kernel_size=3
    dilation=(1, 1),
    grid_size=5,
    spline_order=3,
    scale_noise=0.1,
    scale_base=1.0,
    scale_spline=1.0,
    base_activation="SiLU",
    grid_eps=0.02,
    grid_range=[-1.0, 1.0],
)


# --- helpers --------------------------------------------------------------


def _activation_from_name(name: str):
    """Map a string name to a torch.nn activation *class* (upstream expects a class)."""
    table = {
        "SiLU": torch.nn.SiLU,
        "ReLU": torch.nn.ReLU,
        "Tanh": torch.nn.Tanh,
        "GELU": torch.nn.GELU,
        "Identity": torch.nn.Identity,
    }
    if name not in table:
        raise ValueError(
            f"Unknown base_activation '{name}'. Known: {sorted(table)}"
        )
    return table[name]


def _to_tuple(x: Any, length: int = 2) -> tuple:
    """Coerce a scalar/list/tuple to a tuple of the requested length.

    `KAN_Convolutional_Layer`'s constructor expects 2-tuples for
    kernel_size / stride / padding / dilation. YAML configs often give
    scalars, lists, or tuples.
    """
    if isinstance(x, (list, tuple)):
        if len(x) != length:
            raise ValueError(f"Expected tuple of length {length}; got {tuple(x)}")
        return tuple(x)
    if isinstance(x, int):
        return tuple([x] * length)
    raise TypeError(f"Cannot coerce {type(x).__name__} to a tuple of length {length}")


# --- adapter 1: the MLP-style `KAN` ---------------------------------------


@register
class KANAdapter(BaseModel):
    """Adapter for the classic `KAN` (Multilayer Kolmogorov-Arnold Network).

    The vendored model is a fully-connected network with learnable B-spline
    activations on the edges. It maps a flat feature vector to another flat
    feature vector and has no built-in notion of space or time.

    For PDE surrogate use, this adapter treats a (T, B, C, Nx, Ny) window
    as a flat (T*C*Nx*Ny)-dim feature vector and reduces the KAN's output
    back to a (B, C, Nx, Ny) single-step prediction (the last "T-slot"
    of the flat output). That output is then turned into the framework's
    canonical (B, Nx, Ny, T=1, C) form by `to_canonical`.
    """

    metadata = ModelMetadata(
        name="KAN",
        family="kan_mlp",
        supported_modes=[Mode.FROM_SCRATCH, Mode.FINETUNE, Mode.ZERO_SHOT],
        default_mode=Mode.FROM_SCRATCH,  # the upstream ships no pretrained weights
        # Native in/out (after the adapter's bridge):
        native_input_layout=(
            "(T, B, C, Nx, Ny) float32 — adapter flattens to "
            "(T*B, T*C*Nx*Ny) then takes the last T slot's output"
        ),
        native_output_layout="(B, C, Nx, Ny) float32 — single-step prediction",
        canonical_layout="(B, Nx, Ny, T, C) float32",
        upstream_version="pykan-vendored-0.1",
        paper="Liu et al., 'KAN: Kolmogorov-Arnold Networks' (2024)",
    )

    # --- Construction --------------------------------------------------

    def __init__(self, native: KAN, cfg: Dict[str, Any]):
        self._native = native
        self._cfg = cfg
        # Cached derived values (computed at build time so forward() is fast).
        self._T: int = int(cfg.get("_T", 1))  # populated by train(); not used at inference
        self._C: int = int(cfg["_C"])
        self._Nx: int = int(cfg["_Nx"])
        self._Ny: int = int(cfg["_Ny"])

    @classmethod
    def build(cls, cfg: Dict[str, Any]) -> "KANAdapter":
        merged = {**_KAN_DEFAULTS, **cfg}
        # Determine the input/output feature size: the adapter treats a
        # single (T, C, Nx, Ny) sample as a T*C*Nx*Ny-dim vector and the
        # output is the C*Nx*Ny last-step prediction. We tie the KAN's
        # in_features to T*C*Nx*Ny and out_features to C*Nx*Ny.
        T = int(merged.get("_T", 1))
        C = int(merged.get("_C", merged.get("in_channels", 1)))
        Nx = int(merged.get("_Nx", 64))
        Ny = int(merged.get("_Ny", 64))
        in_features = T * C * Nx * Ny
        out_features = T * C * Nx * Ny  # full T-slot output; we'll take the last step

        # Build the layer sizes list: [in, hidden, ..., hidden, out].
        n_hidden = int(merged["n_hidden"])
        hidden_width = int(merged["hidden_width"])
        if n_hidden == 0:
            layers_hidden = [in_features, out_features]
        else:
            layers_hidden = [in_features] + [hidden_width] * n_hidden + [out_features]

        # Translate scalar/lists to tuples where the constructor cares.
        grid_range = _to_tuple(merged["grid_range"], length=2)
        # Cast grid_range to a Python list (KAN expects an indexable of len 2).
        if isinstance(grid_range, tuple):
            grid_range = list(grid_range)

        base_activation = _activation_from_name(merged["base_activation"])

        native = KAN(
            layers_hidden=layers_hidden,
            grid_size=int(merged["grid_size"]),
            spline_order=int(merged["spline_order"]),
            scale_noise=float(merged["scale_noise"]),
            scale_base=float(merged["scale_base"]),
            scale_spline=float(merged["scale_spline"]),
            base_activation=base_activation,
            grid_eps=float(merged["grid_eps"]),
            grid_range=grid_range,
        )
        # Stash the derived sizes in the cfg so the adapter instance can use them.
        merged["_T"] = T
        merged["_C"] = C
        merged["_Nx"] = Nx
        merged["_Ny"] = Ny
        return cls(native=native, cfg=merged)

    # --- Inference (NATIVE: (T, B, C, Nx, Ny) -> (B, C, Nx, Ny)) ---------

    def forward(self, x: Tensor) -> Tensor:
        """Run the vendored `KAN` on a (T, B, C, Nx, Ny) window.

        Steps:
            1. Flatten everything except the batch dim:
                 (T, B, C, Nx, Ny) -> (B, T*C*Nx*Ny)
            2. Call the vendored `KAN`, which expects (B, in_features).
            3. Reshape the output to (B, T, C, Nx, Ny).
            4. Return the *last* timestep's slice as (B, C, Nx, Ny).

        Why the last timestep: the vendored MLP has no time axis, so we
        arrange it as in_features = T*C*Nx*Ny → out_features = T*C*Nx*Ny,
        then read off the last T-slot. The framework only consumes a
        single-step prediction per `forward()` call (matching MPP's
        contract), so this is the cleanest fit.
        """
        if x.ndim != 5:
            raise ValueError(
                f"KANAdapter.forward expects (T, B, C, Nx, Ny); got {tuple(x.shape)}"
            )
        T, B, C, Nx, Ny = x.shape
        in_features = T * C * Nx * Ny
        out_features = T * C * Nx *Ny  # the KAN predicts a full T-slot output

        # Sanity: in_features should match what the KAN was built with.
        first_layer = self._native.layers[0]
        if first_layer.in_features != in_features:
            raise RuntimeError(
                f"KAN was built with in_features={first_layer.in_features} but "
                f"forward received a window that flattens to {in_features}. "
                f"Did you change T, C, Nx, or Ny without rebuilding?"
            )

        # 1) Flatten: (T, B, C, Nx, Ny) -> (B, T*C*Nx*Ny)
        x_flat = x.permute(1, 0, 2, 3, 4).reshape(B, in_features)

        # 2) Native KAN forward.
        y_flat = self._native(x_flat)  # (B, T*C*Nx*Ny)

        # 3) Reshape to (B, T, C, Nx, Ny).
        y = y_flat.view(B, T, C, Nx, Ny)

        # 4) Take the last timestep's slice: (B, C, Nx, Ny).
        return y[:, -1].contiguous()

    # --- Canonicalization ---------------------------------------------

    def to_canonical(self, y_native: Tensor) -> CanonicalSample:
        """Convert the adapter's native (B, C, Nx, Ny) output to canonical
        (B, Nx, Ny, T=1, C) by adding a singleton time axis."""
        if y_native.ndim != 4:
            raise ValueError(
                f"KANAdapter.to_canonical expects (B, C, Nx, Ny); got {tuple(y_native.shape)}"
            )
        # (B, C, Nx, Ny) -> (B, Nx, Ny, C) -> (B, Nx, Ny, T=1, C)
        y_canon = y_native.permute(0, 2, 3, 1).unsqueeze(3)
        return CanonicalSample(
            fields=y_canon.contiguous(),
            metadata={
                "source": "KAN",
                "layout": "single_step",
                "already_in_input_units": True,  # KAN has no internal normalization
            },
        )

    def from_canonical(self, sample: CanonicalSample) -> Tensor:
        """Inverse: turn a canonical (B, Nx, Ny, T, C) sample into the
        adapter's native (T, B, C, Nx, Ny) input layout."""
        x = sample.fields
        if x.ndim != 5:
            raise ValueError(
                f"KANAdapter.from_canonical expects (B, Nx, Ny, T, C); "
                f"got {tuple(x.shape)}"
            )
        # (B, Nx, Ny, T, C) -> (T, B, C, Nx, Ny)
        return x.permute(3, 0, 4, 1, 2).contiguous()

    # --- Weights -------------------------------------------------------

    def load_weights(self, source: WeightSource, *, strict: bool = True) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"No KAN weights at {path}. The vendored pykan.py ships no "
                f"pretrained state_dict, so from-scratch is the default mode."
            )
        state = torch.load(path, map_location="cpu")
        missing, unexpected = self._native.load_state_dict(state, strict=strict)
        if not strict and (missing or unexpected):
            print(
                f"[KAN] load_weights(strict=False): "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    def save_weights(self, destination: WeightSource) -> None:
        torch.save(self._native.state_dict(), Path(destination))

    # --- Training (NATIVE loop delegation) -----------------------------

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        *,
        cfg: Dict[str, Any],
        callbacks: Optional[list[Callable]] = None,
    ) -> TrainingResult:
        """Train the vendored KAN with a native PyTorch loop.

        Per Option C, the framework does NOT impose its own optimizer
        / loss on the model. The vendored KAN ships no training script,
        so we wire a minimal loop here — but the model itself, the
        forward pass, and the architecture are entirely upstream's.
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
                #   target: (b, C, Nx, Ny)
                window, target = batch[0], batch[1]
                window = window.to(device)
                target = target.to(device)
                pred = self.forward(window)  # adapter.forward() handles the bridge
                loss = loss_fn(pred, target)
                optim.zero_grad()
                loss.backward()
                # The vendored KAN is known to be sensitive to exploding
                # spline weights early in training; clip as a safety net.
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

    # --- Introspection -------------------------------------------------

    @property
    def native_model(self) -> KAN:
        return self._native


# --- adapter 2: the convolutional KAN layer --------------------------------


@register
class KANConvAdapter(BaseModel):
    """Adapter for `KAN_Convolutional_Layer` (a 2D KAN conv).

    The vendored layer maps (B, C_in, H, W) to (B, C_out, H', W') with
    kernel_size/stride/padding/dilation — Conv2d semantics, but the kernel
    is a small KANLinear. The adapter applies it per-timestep and returns
    the last step's output as (B, C_out, Nx', Ny').

    For the smoke test we set out_channels = C and use kernel_size=3 with
    padding=(1, 1) so the spatial dims are preserved (the framework's
    forward_shape check expects the same Nx, Ny on input and output).
    """

    metadata = ModelMetadata(
        name="KANConv",
        family="kan_conv",
        supported_modes=[Mode.FROM_SCRATCH, Mode.FINETUNE, Mode.ZERO_SHOT],
        default_mode=Mode.FROM_SCRATCH,
        native_input_layout=(
            "(T, B, C, Nx, Ny) float32 — adapter applies the conv per "
            "timestep and returns the last step's output"
        ),
        native_output_layout="(B, out_channels, Nx', Ny') float32 — single-step prediction",
        canonical_layout="(B, Nx, Ny, T, C) float32",
        upstream_version="pykan-vendored-0.1",
        paper="Liu et al., 'KAN: Kolmogorov-Arnold Networks' (2024) — conv variant",
    )

    def __init__(self, native: KAN_Convolutional_Layer, cfg: Dict[str, Any]):
        self._native = native
        self._cfg = cfg
        # Cached for forward(): avoids re-parsing the config every step.
        self._out_channels: int = int(cfg.get("out_channels", cfg.get("_C", 1)))

    @classmethod
    def build(cls, cfg: Dict[str, Any]) -> "KANConvAdapter":
        merged = {**_KANCONV_DEFAULTS, **cfg}
        # Translate scalar/lists to the tuples the constructor expects.
        kernel_size = _to_tuple(merged["kernel_size"])
        stride = _to_tuple(merged["stride"])
        padding = _to_tuple(merged["padding"])
        dilation = _to_tuple(merged["dilation"])
        grid_range = _to_tuple(merged["grid_range"], length=2)
        if isinstance(grid_range, tuple):
            grid_range = list(grid_range)
        base_activation = _activation_from_name(merged["base_activation"])

        native = KAN_Convolutional_Layer(
            in_channels=int(merged["in_channels"]),
            out_channels=int(merged["out_channels"]),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            grid_size=int(merged["grid_size"]),
            spline_order=int(merged["spline_order"]),
            scale_noise=float(merged["scale_noise"]),
            scale_base=float(merged["scale_base"]),
            scale_spline=float(merged["scale_spline"]),
            base_activation=base_activation,
            grid_eps=float(merged["grid_eps"]),
            grid_range=grid_range,
            device="cpu",  # device is allocated from the input tensor in vendored code
        )
        return cls(native=native, cfg=merged)

    # --- Inference (NATIVE: (T, B, C, Nx, Ny) -> (B, C_out, Nx', Ny')) -

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolutional KAN per-timestep; return the last step.

        The vendored `KAN_Convolutional_Layer` operates on a single 2D
        feature map at a time, so we loop over the T axis. The output's
        spatial dims depend on kernel/stride/padding/dilation; for the
        default config (kernel=3, padding=1, stride=1) the output shape
        matches the input shape.
        """
        if x.ndim != 5:
            raise ValueError(
                f"KANConvAdapter.forward expects (T, B, C, Nx, Ny); "
                f"got {tuple(x.shape)}"
            )
        T, B, C, Nx, Ny = x.shape
        # Apply the layer to each timestep. The vendored conv's `device`
        # attribute is set inside its forward() from `x.device`, so we
        # don't need to move it ourselves.
        outputs: List[Tensor] = []
        for t in range(T):
            xt = x[t]  # (B, C, Nx, Ny)
            yt = self._native(xt)  # (B, C_out, Nx', Ny')
            outputs.append(yt)
        # Return the last step: (B, C_out, Nx', Ny').
        return outputs[-1].contiguous()

    # --- Canonicalization ---------------------------------------------

    def to_canonical(self, y_native: Tensor) -> CanonicalSample:
        """Native (B, C_out, Nx', Ny') -> canonical (B, Nx', Ny', T=1, C_out).

        Note: the channel count may differ from the original input (e.g.
        out_channels=2 when in_channels=1). The metadata records this so
        the benchmark layer knows the output's channel vocab.
        """
        if y_native.ndim != 4:
            raise ValueError(
                f"KANConvAdapter.to_canonical expects (B, C, Nx, Ny); "
                f"got {tuple(y_native.shape)}"
            )
        # (B, C, Nx, Ny) -> (B, Nx, Ny, C) -> (B, Nx, Ny, T=1, C)
        y_canon = y_native.permute(0, 2, 3, 1).unsqueeze(3)
        return CanonicalSample(
            fields=y_canon.contiguous(),
            metadata={
                "source": "KANConv",
                "layout": "single_step",
                "out_channels": int(y_native.shape[1]),
                "already_in_input_units": True,
            },
        )

    def from_canonical(self, sample: CanonicalSample) -> Tensor:
        """Canonical (B, Nx, Ny, T, C) -> native (T, B, C, Nx, Ny)."""
        x = sample.fields
        if x.ndim != 5:
            raise ValueError(
                f"KANConvAdapter.from_canonical expects (B, Nx, Ny, T, C); "
                f"got {tuple(x.shape)}"
            )
        return x.permute(3, 0, 4, 1, 2).contiguous()

    # --- Weights -------------------------------------------------------

    def load_weights(self, source: WeightSource, *, strict: bool = True) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"No KANConv weights at {path}. The vendored pykan.py ships no "
                f"pretrained state_dict for the conv variant."
            )
        state = torch.load(path, map_location="cpu")
        missing, unexpected = self._native.load_state_dict(state, strict=strict)
        if not strict and (missing or unexpected):
            print(
                f"[KANConv] load_weights(strict=False): "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    def save_weights(self, destination: WeightSource) -> None:
        torch.save(self._native.state_dict(), Path(destination))

    # --- Training (NATIVE loop delegation) -----------------------------

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        *,
        cfg: Dict[str, Any],
        callbacks: Optional[list[Callable]] = None,
    ) -> TrainingResult:
        """Train the vendored KAN_Convolutional_Layer with a native PyTorch loop.

        Same pattern as `KANAdapter.train`: framework hands us batches
        in (T, B, C, Nx, Ny) layout, the adapter applies the conv
        per-timestep and returns the last step's prediction.
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
                window, target = batch[0], batch[1]
                window = window.to(device)
                target = target.to(device)
                pred = self.forward(window)  # adapter.forward() bridges (T,...) -> (B,...)
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

    # --- Introspection -------------------------------------------------

    @property
    def native_model(self) -> KAN_Convolutional_Layer:
        return self._native
