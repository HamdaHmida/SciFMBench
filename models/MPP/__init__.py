"""MPP — Axial ViT for PDE surrogates (vendored upstream + thin adapter).

Importing this package runs `MPPAdapter`'s `@register` decorator, which
makes the model discoverable via:

    from core.base_model import available_models, get_model
    get_model("MPP").build(cfg)

The vendored files (`avit.py`, `spatial_modules.py`, `time_modules.py`,
`mixed_modules.py`, `shared_modules.py`) are kept unmodified — only
`adapter.py` is authored here.
"""
from .adapter import MPPAdapter

__all__ = ["MPPAdapter"]
