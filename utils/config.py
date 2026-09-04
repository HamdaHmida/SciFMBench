"""Lightweight config loader.

Prefers YAML when PyYAML is installed; falls back to JSON so the framework
runs even before YAML support is added. Plain-dict-only — no fancy schema
validation, no env-var interpolation. Keep it simple.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a config file (YAML if available, JSON as fallback)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    suffix = p.suffix.lower()
    text = p.read_text()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load .yaml configs. Install with "
                "`pip install pyyaml`, or use a .json config instead."
            ) from e
        return yaml.safe_load(text) or {}

    if suffix == ".json":
        return json.loads(text) if text.strip() else {}

    raise ValueError(f"Unsupported config format '{suffix}'. Use .yaml or .json.")
