"""FactFormer — Factorized Attention for Operator Learning (vendored upstream + thin adapter).

Importing this package runs the `@register` decorator on the adapter
class, which makes it discoverable via:

    from core.base_model import available_models, get_model
    get_model("FactFormer").build(cfg)

The vendored files (`attention.py`, `basics.py`, `factorization_module.py`,
`positional_encoding_module.py`) are kept unmodified — only `adapter.py` is
authored here.
"""
from .adapter import FactFormerAdapter

__all__ = ["FactFormerAdapter"]