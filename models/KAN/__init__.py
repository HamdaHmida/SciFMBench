"""KAN — Kolmogorov-Arnold Networks (vendored upstream + thin adapters).

Importing this package runs the `@register` decorators on both adapter
classes, which makes them discoverable via:

    from core.base_model import available_models, get_model
    get_model("KAN").build(cfg)      # the MLP-style KAN
    get_model("KANConv").build(cfg)  # the 2D convolutional KAN

The vendored file `pykan.py` is kept unmodified — only `adapter.py` is
authored here. See `adapter.py` for the rationale behind shipping two
adapters (different native layouts, different scientific uses).
"""
from .adapter import KANAdapter, KANConvAdapter

__all__ = ["KANAdapter", "KANConvAdapter"]
