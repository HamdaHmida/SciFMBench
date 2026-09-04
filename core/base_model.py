"""BaseModel ABC — the contract every model adapter implements.

Design notes
------------
* Upstream model code is **never** modified. The adapter is a thin layer that
  translates between the unified framework surface and the model's native API.
* Per Option C, the framework's training loop is NATIVE (each model trains the
  way its authors designed) and the evaluation path is CANONICAL (the framework
  enforces shared splits, normalization, and metrics).
* The adapter's `forward()` accepts and returns **native** tensors — whatever
  shape the upstream model expects. Canonicalization is opt-in via
  `to_canonical` / `from_canonical` and is invoked by the evaluator, not by
  the trainer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor, nn


class Mode(str, Enum):
    """Training/eval modes a model can support.

    A given model declares which modes it supports in its metadata. The
    framework dispatches based on the mode requested for a run.
    """

    FROM_SCRATCH = "from_scratch"  # train without public weights
    FINETUNE = "finetune"  # initialize from public weights, then train
    ZERO_SHOT = "zero_shot"  # public weights, no training


WeightSource = Union[str, Path]  # local path, URL, or HF Hub repo id


@dataclass
class ModelMetadata:
    """Static, declarative info about a model. Populated by each adapter."""

    name: str  # registry id, e.g. "fno2d"
    family: str  # e.g. "neural_operator", "gnn", "transformer"
    supported_modes: list[Mode] = field(default_factory=list)
    default_mode: Mode = Mode.ZERO_SHOT
    # Native tensor layout — what `forward()` expects and returns.
    # Documented here so the evaluator knows when to canonicalize.
    native_input_layout: str = "auto"
    native_output_layout: str = "auto"
    # Optional: upstream commit/tag this adapter was tested against.
    upstream_version: Optional[str] = None
    paper: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# --- Registry ---------------------------------------------------------------

_REGISTRY: Dict[str, type["BaseModel"]] = {}


def register(cls: type["BaseModel"]) -> type["BaseModel"]:
    """Class decorator that registers an adapter in the global registry.

    The adapter must set `metadata.name` as a class attribute (or pass one to
    `ModelMetadata`) — that's the id used by `get_model()` and CLI/configs.
    """
    if not cls.metadata or not cls.metadata.name:
        raise ValueError(
            f"{cls.__name__}.metadata.name must be set before registration."
        )
    if cls.metadata.name in _REGISTRY:
        raise ValueError(f"Model '{cls.metadata.name}' already registered.")
    _REGISTRY[cls.metadata.name] = cls
    return cls


def get_model(name: str) -> type["BaseModel"]:
    """Look up an adapter class by registry id."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Registered: {available}")
    return _REGISTRY[name]


def available_models() -> list[str]:
    return sorted(_REGISTRY)


# --- The ABC ---------------------------------------------------------------


class BaseModel(ABC):
    """Abstract base class for every SciFM adapter.

    Lifecycle:
        cfg = load_config(...)
        model = MyAdapter.build(cfg)         # construct native model
        model.load_weights(weights_src)     # optional, depending on mode
        out = model.forward(x)              # inference (native tensors)
        canon = model.to_canonical(out)     # evaluator uses this

    Training is delegated: the trainer calls `train()` (or step-wise hooks)
    which internally uses whatever native loop the upstream model ships with.
    """

    metadata: ModelMetadata  # subclasses MUST set this as a class attribute

    # --- Construction ------------------------------------------------------

    @classmethod
    @abstractmethod
    def build(cls, cfg: Dict[str, Any]) -> "BaseModel":
        """Construct the adapter (and underlying native model) from a config dict.

        `cfg` is the merged output of the framework's YAML config layer. The
        adapter is responsible for forwarding the relevant keys to the upstream
        model's constructor and storing any extra metadata it needs.
        """

    # --- Inference (NATIVE tensors in/out) ---------------------------------

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Single forward pass. Accepts and returns the model's NATIVE types.

        For most PyTorch models this is `Tensor -> Tensor`. For models that
        take a tuple (e.g. DeepONet's (branch, trunk)) or a dict, the adapter
        decides the shape and documents it in `metadata`.
        """

    # --- Canonicalization (used by the evaluator, not the trainer) ---------

    @abstractmethod
    def to_canonical(self, y_native: Any) -> "CanonicalSample":
        """Convert a NATIVE model output into the framework's canonical form.

        The benchmark layer calls this so it can compute shared metrics on
        canonical outputs regardless of which model produced them.
        """

    @abstractmethod
    def from_canonical(self, sample: "CanonicalSample") -> Any:
        """Inverse of `to_canonical`: build a NATIVE input from a canonical
        sample. Used by the evaluator when feeding canonical test data to
        models whose native input format differs.
        """

    # --- Weights -----------------------------------------------------------

    @abstractmethod
    def load_weights(self, source: WeightSource, *, strict: bool = True) -> None:
        """Load weights from a local path, URL, or HF Hub repo id.

        Implementations should resolve the source, fetch/cache if needed,
        then call the upstream model's native `load_state_dict`-equivalent.
        """

    @abstractmethod
    def save_weights(self, destination: WeightSource) -> None:
        """Persist current weights to a local path or HF Hub repo id."""

    # --- Training (NATIVE loop, framework delegates) -----------------------

    def supports(self, mode: Mode) -> bool:
        return mode in (self.metadata.supported_modes if self.metadata else [])

    @abstractmethod
    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        *,
        cfg: Dict[str, Any],
        callbacks: Optional[list[Callable]] = None,
    ) -> "TrainingResult":
        """Run the upstream model's NATIVE training loop.

        The framework does NOT impose its own optimizer / scheduler / loss
        here — those belong to the model. The adapter wires the upstream
        training script to the framework's data objects, runs it, and
        returns a `TrainingResult`. Callbacks receive framework-level
        events (epoch end, checkpoint, eval, etc.).
        """

    # --- Optional introspection -------------------------------------------

    def num_parameters(self) -> int:
        """Total trainable parameter count. Optional; default uses torch."""
        if isinstance(self.native_model, nn.Module):
            return sum(p.numel() for p in self.native_model.parameters() if p.requires_grad)
        return -1

    # Subclasses must expose the underlying upstream model object so the
    # framework can introspect / serialize it when needed.
    @property
    @abstractmethod
    def native_model(self) -> Any:
        """The vendored upstream model instance (or its closest handle)."""


# --- Canonical sample schema (placeholder; finalized in next phase) --------


@dataclass
class CanonicalSample:
    """Framework-canonical representation of a fluid-dynamics sample.

    The shape and field set are deliberately minimal here — task #1
    (canonical Sample schema) will refine this. For now, a single tensor
    plus named channels is enough for the ABC to compile.
    """

    fields: Tensor  # canonical-dtype tensor, layout to be finalized
    coords: Optional[Tensor] = None  # optional spatial coordinates
    time: Optional[Tensor] = None  # optional time axis
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """What `train()` returns. Finalized alongside the trainer."""

    best_checkpoint: Optional[str] = None
    history: Dict[str, list[float]] = field(default_factory=dict)
