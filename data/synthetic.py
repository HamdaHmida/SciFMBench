"""Synthetic fluid-dynamics data for smoke-testing the training script.

Produces an iterator over (window, target) batches shaped for MPP:

    window: (T, B, C, H, W) float32
    target: (B, C, H, W) float32

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
    H: int = 64,
    W: int = 64,
    batch_size: int = 2,
    seed: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield (window, target) batches forever — `n_samples` is the dataset size
    and the iterator cycles through it.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n_samples, C, H, W, generator=g)

    epoch = 0
    while True:
        # Shuffle the dataset deterministically per epoch.
        perm = torch.randperm(n_samples, generator=g)
        shuffled = base[perm]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = shuffled[start:end]                          # (b, C, H, W)
            B = batch.shape[0]
            window = torch.stack([batch + 0.01 * i for i in range(T)], dim=0)  # (T, b, C, H, W)
            target = batch.clone()                               # (b, C, H, W)
            yield window, target
        epoch += 1
