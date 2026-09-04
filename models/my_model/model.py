"""Placeholder for a vendored upstream model.

In a real adapter, this file is the upstream repo's model class — copied
in unmodified (or imported from an installed dep). The adapter in
`adapter.py` is the ONLY file you author.

This stub stands in until the first real model is added (task #3).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class PlaceholderModel(nn.Module):
    """Stub vendored model. Replace with real upstream code in task #3."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, hidden: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Native layout for this stub: (B, C, H, W) -> (B, C, H, W).
        # Real models would have their own layout documented in metadata.
        return self.net(x)
