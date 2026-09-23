"""OFormer model package.

Registers the OFormerAdapter when imported.
"""

# Import adapter to trigger @register decorator
from models.OFormer.adapter import OFormerAdapter  # noqa: F401

__all__ = ["OFormerAdapter"]