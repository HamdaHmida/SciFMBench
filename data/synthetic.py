"""Synthetic fluid-dynamics data for smoke-testing the training script.

Two layouts are produced, both with the same underlying content:

  * Native format (what MPP consumes during training):
        window: (T, B, C, Nx, Ny) float32
        target: (B, C, Nx, Ny) float32  — last step the model should hit

  * Canonical format (what the framework's evaluator consumes):
        window: (B, Nx, Ny, T, C) float32
        target: (B, Nx, Ny, T, C) float32  — full trajectory

The values are random — this is purely a wiring test, not a benchmark.
A real dataset module (task #5) will replace this.

Torch is imported lazily so this module is safe to import in environments
where torch is not installed.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import torch  # noqa: F401  — referenced in annotations only; actual use is deferred


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
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n_samples, C, Nx, Ny, generator=g)

    while True:
        # Shuffle the dataset deterministically per epoch.
        perm = torch.randperm(n_samples, generator=g)
        shuffled = base[perm]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = shuffled[start:end]                                  # (b, C, Nx, Ny)
            b = batch.shape[0]
            window = torch.stack([batch + 0.01 * i for i in range(T)], dim=0)  # (T, b, C, Nx, Ny)
            target = batch.clone()                                       # (b, C, Nx, Ny)
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
    return torch.randn(B, Nx, Ny, T, C, generator=g)


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
