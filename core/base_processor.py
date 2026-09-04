"""BaseProcessor ABC — pre/post-processing for SciFMs.

Pre/post-processing in fluid-dynamics SciFMs is heavy and model-specific
(normalization conventions, grid sampling, coordinate transforms, FFT
shifts, etc.). Keeping it out of the model ABC means a model swap doesn't
have to rewrite a custom transform.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .base_model import BaseModel


class BaseProcessor(ABC):
    """Pre- or post-processor paired with a specific model adapter."""

    model_name: str  # registry id of the adapter this processor belongs to

    @abstractmethod
    def __call__(self, x: Any, model: BaseModel) -> Any:
        """Apply the transform. Receives the adapter so the processor can
        query metadata (e.g. channel order, normalization stats) if needed.
        """
