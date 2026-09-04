"""BaseMetric ABC — shared evaluation metrics.

Per Option C, metrics operate on CANONICAL outputs so all models are
scored with the same implementation. Model-specific quirks never leak
into the metric.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """A metric computed on canonical outputs (or canonical vs. ground truth)."""

    name: str  # e.g. "rmse", "spectrum_error", "cfl_stability"

    @abstractmethod
    def __call__(self, prediction: Any, target: Any, **kwargs) -> float:
        """Return a scalar score. Higher-is-better vs. lower-is-better is
        declared per metric; the evaluator handles aggregation.
        """
