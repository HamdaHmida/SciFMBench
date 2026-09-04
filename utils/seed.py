"""Global seeding for reproducibility.

NumPy and PyTorch are imported lazily so this module is safe to import in
environments where neither is installed (e.g. lightweight tooling, CI
smoke checks).
"""
from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) if available."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
