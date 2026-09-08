"""Synthetic fluid-dynamics data for smoke-testing the training script.

Data layouts
------------
* Raw HDF5 format (what Parametric_2D_Dataset reads):
      (B, T, Nx, Ny, C) float32
  This matches the format expected by the real PDEBench dataset loader.

* Canonical format (what the framework's evaluator consumes):
      (B, Nx, Ny, T, C) float32
  Obtained by permuting the raw format.

* Native format (what MPP consumes during training):
      (T, B, C, Nx, Ny) float32
  Obtained by permuting the canonical format.

The values are random — this is purely a wiring test, not a benchmark.
A real dataset module (task #5) will replace this.

Torch is imported lazily so this module is safe to import in environments
where torch is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple, Optional

import numpy as np
import torch  # noqa: F401  — referenced in annotations only; actual use is deferred


def make_synthetic_raw(
    *,
    n_samples: int = 8,
    T: int = 10,
    C: int = 3,
    Nx: int = 64,
    Ny: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Generate synthetic data in the raw HDF5 format: (B, T, Nx, Ny, C).

    This is the format that `Parametric_2D_Dataset` expects to read from .h5 files.

    Returns:
        tensor of shape (n_samples, T, Nx, Ny, C) float32.
    """
    g = torch.Generator().manual_seed(seed)
    # Add a small temporal drift so consecutive timesteps are different
    base = torch.randn(n_samples, 1, Nx, Ny, C, generator=g)
    drift = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1, 1) * 0.01
    return (base + drift).expand(n_samples, T, Nx, Ny, C).contiguous()


def save_synthetic_raw(
    data: torch.Tensor,
    filepath: Path | str,
    dataset_name: str = "tensor",
) -> None:
    """Save a (B, T, Nx, Ny, C) tensor to an HDF5 file.

    Matches the format expected by `Parametric_2D_Dataset`:
    - dataset name "tensor" (hardcoded in the loader)
    - dtype float32
    """
    import h5py
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filepath, "w") as f:
        f.create_dataset(dataset_name, data=data.numpy().astype(np.float32))


def load_synthetic_raw(
    filepath: Path | str,
    dataset_name: str = "tensor",
) -> torch.Tensor:
    """Load a (B, T, Nx, Ny, C) tensor from an HDF5 file."""
    import h5py
    with h5py.File(filepath, "r") as f:
        return torch.from_numpy(f[dataset_name][:]).float()


def make_synthetic_loader(
    *,
    n_samples: int = 8,
    T: int = 10,
    C: int = 3,
    Nx: int = 64,
    Ny: int = 64,
    batch_size: int = 2,
    seed: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield (window, target) batches in NATIVE MPP layout forever.

    `n_samples` is the dataset size and the iterator cycles through it.

    Native layout (MPP): window=(T, b, C, Nx, Ny), target=(b, C, Nx, Ny)
    """
    g = torch.Generator().manual_seed(seed)
    # Generate raw format (B, T, Nx, Ny, C) then convert to native
    raw = make_synthetic_raw(
        n_samples=n_samples, T=T, C=C, Nx=Nx, Ny=Ny, seed=seed
    )  # (B, T, Nx, Ny, C)

    while True:
        # Shuffle the dataset deterministically per epoch.
        perm = torch.randperm(n_samples, generator=g)
        shuffled = raw[perm]  # (B, T, Nx, Ny, C)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_raw = shuffled[start:end]  # (b, T, Nx, Ny, C)
            b = batch_raw.shape[0]

            # Convert to canonical: (b, Nx, Ny, T, C)
            batch_canon = batch_raw.permute(0, 2, 3, 1, 4).contiguous()

            # Convert to native MPP: (T, b, C, Nx, Ny)
            window = batch_canon.permute(3, 0, 4, 1, 2).contiguous()

            # Target is the last timestep in native: (b, C, Nx, Ny)
            target = window[-1].contiguous()

            yield window, target


def make_canonical_sample(
    *,
    B: int = 1,
    Nx: int = 64,
    Ny: int = 64,
    T: int = 10,
    C: int = 3,
    seed: int = 0,
) -> torch.Tensor:
    """Produce a single random sample in the framework's canonical layout.

    Returns:
        tensor of shape (B, Nx, Ny, T, C) float32.
    """
    g = torch.Generator().manual_seed(seed)
    # Generate raw (B, T, Nx, Ny, C) then permute to canonical
    raw = torch.randn(B, T, Nx, Ny, C, generator=g)
    # Add temporal drift
    drift = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1, 1) * 0.01
    raw = raw + drift
    # Permute to canonical: (B, Nx, Ny, T, C)
    return raw.permute(0, 2, 3, 1, 4).contiguous()


def native_to_canonical(x_native: torch.Tensor) -> torch.Tensor:
    """Permute MPP's (T, B, C, Nx, Ny) → canonical (B, Nx, Ny, T, C).

    Used by the evaluator when it has a native batch and wants to score it
    against canonical ground truth without going through an adapter.
    """
    if x_native.ndim != 5:
        raise ValueError(
            f"native_to_canonical expects (T, B, C, Nx, Ny); got {tuple(x_native.shape)}"
        )
    return x_native.permute(1, 3, 4, 0, 2).contiguous()


def canonical_to_native(x_canon: torch.Tensor) -> torch.Tensor:
    """Inverse of `native_to_canonical`: (B, Nx, Ny, T, C) → (T, B, C, Nx, Ny)."""
    if x_canon.ndim != 5:
        raise ValueError(
            f"canonical_to_native expects (B, Nx, Ny, T, C); got {tuple(x_canon.shape)}"
        )
    return x_canon.permute(3, 0, 4, 1, 2).contiguous()


def raw_to_canonical(x_raw: torch.Tensor) -> torch.Tensor:
    """Convert raw HDF5 format (B, T, Nx, Ny, C) → canonical (B, Nx, Ny, T, C)."""
    if x_raw.ndim != 5:
        raise ValueError(
            f"raw_to_canonical expects (B, T, Nx, Ny, C); got {tuple(x_raw.shape)}"
        )
    return x_raw.permute(0, 2, 3, 1, 4).contiguous()


def canonical_to_raw(x_canon: torch.Tensor) -> torch.Tensor:
    """Inverse: canonical (B, Nx, Ny, T, C) → raw (B, T, Nx, Ny, C)."""
    if x_canon.ndim != 5:
        raise ValueError(
            f"canonical_to_raw expects (B, Nx, Ny, T, C); got {tuple(x_canon.shape)}"
        )
    return x_canon.permute(0, 3, 1, 2, 4).contiguous()