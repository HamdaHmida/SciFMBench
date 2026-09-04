"""Top-level CLI: train / finetune / test a registered SciFM.

Usage:
    # train from scratch using the 'basic_config' section of the MPP YAML
    python -m engine.run --model MPP --mode train   --config configs/models/mpp_avit_s_config.yaml

    # finetune using the 'finetune' section and a checkpoint
    python -m engine.run --model MPP --mode finetune --config configs/models/mpp_avit_s_config.yaml --weights /path/to/ckpt.tar

    # zero-shot eval
    python -m engine.run --model MPP --mode test    --config configs/models/mpp_avit_s_config.yaml --weights /path/to/ckpt.tar

    # auto-select section based on --mode
    python -m engine.run --model MPP --mode finetune --config mpp_avit_s_config.yaml --auto-section

    # list registered models
    python -m engine.run --list-models

What it does
------------
1. Loads the YAML/JSON config (model hparams + run hparams).
2. Selects a section by name (default: 'basic_config' for --mode train,
   'finetune' for --mode finetune/test). Pass --section to override.
3. Translates upstream MPP key names to the framework's flat keys:
     learning_rate   -> lr
     max_epochs      -> epochs
     n_steps         -> T
     n_states        -> channel vocab size
     state_names     -> state_labels (sorted unique names → indices)
     pretrained      -> bool flag
     pretrained_ckpt_path -> checkpoint path (if --weights not given)
4. Looks up the adapter in the registry (--model).
5. Builds the model with the translated config.
6. Loads weights from --weights or pretrained_ckpt_path (finetune/test).
7. Dispatches by --mode:
     - train    : from-scratch, native training loop
     - finetune : load weights, then native training loop
     - test     : zero-shot eval (forward + MSE vs. targets)

Per Option C, training is delegated to the adapter's native loop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Make `python -m engine.run` work whether called from repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.base_model import Mode, available_models, get_model  # noqa: E402
from data.synthetic import make_synthetic_loader              # noqa: E402
from utils.config import load_config                          # noqa: E402
from utils.seed import set_seed                                # noqa: E402


# --- section / key translation --------------------------------------------

# Upstream MPP YAML uses these section names. Order matters: --auto-section
# picks the first section whose key set matches the requested mode.
_KNOWN_SECTIONS = ["basic_config", "finetune", "frozen", "less_frozen"]


def _select_section(cfg: Dict[str, Any], mode: str, section: Optional[str], auto: bool) -> str:
    """Pick which top-level key of the YAML to use."""
    if section is not None:
        if section not in cfg:
            raise SystemExit(f"[run] --section '{section}' not in config. Found: {list(cfg)}")
        return section
    if not auto:
        # Default: 'basic_config' for train, 'finetune' otherwise.
        return "finetune" if mode in {"finetune", "test"} else "basic_config"
    # Auto: pick first known section that contains a sensible flag.
    for name in _KNOWN_SECTIONS:
        sub = cfg.get(name)
        if not isinstance(sub, dict):
            continue
        if mode == "train" and sub.get("pretrained", False) is False:
            return name
        if mode in {"finetune", "test"} and sub.get("pretrained", False) is True:
            return name
    # Fallback: first known section present in the file.
    for name in _KNOWN_SECTIONS:
        if name in cfg:
            return name
    raise SystemExit(f"[run] no usable section found. Config has: {list(cfg)}")


def _state_names_to_labels(state_names) -> list[int]:
    """Convert ['Pressure','Vx','Vy',...] to [0,1,2,...] preserving order."""
    seen: Dict[str, int] = {}
    for name in state_names:
        if name not in seen:
            seen[name] = len(seen)
    return [seen[n] for n in state_names]


def _translate_cfg(upstream: Dict[str, Any]) -> Dict[str, Any]:
    """Translate upstream MPP YAML keys → framework's flat keys.

    Returns a dict with both 'model' keys (passed to adapter.build())
    and 'run' keys (passed to adapter.train()).
    """
    out_model: Dict[str, Any] = {}
    out_run: Dict[str, Any] = {}

    # ---- model hparams ----
    for k in ("embed_dim", "num_heads", "processor_blocks", "n_states",
              "patch_size", "bias_type", "block_type", "time_type",
              "space_type", "drop_path", "tie_fields", "gradient_checkpointing"):
        if k in upstream:
            out_model[k] = upstream[k]

    if "state_names" in upstream:
        out_model["state_names"] = list(upstream["state_names"])
        out_run["state_labels"] = _state_names_to_labels(upstream["state_names"])

    # ---- run hparams ----
    if "learning_rate" in upstream:
        out_run["lr"] = float(upstream["learning_rate"])
    if "max_epochs" in upstream:
        out_run["epochs"] = int(upstream["max_epochs"])
    if "n_steps" in upstream:
        out_run["T"] = int(upstream["n_steps"])
    if "batch_size" in upstream:
        out_run["batch_size"] = int(upstream["batch_size"])
    if "accum_grad" in upstream:
        out_run["accum_grad"] = int(upstream["accum_grad"])
    if "epoch_size" in upstream:
        out_run["epoch_size"] = int(upstream["epoch_size"])
    if "weight_decay" in upstream:
        out_run["weight_decay"] = float(upstream["weight_decay"])
    if "warmup_steps" in upstream:
        out_run["warmup_steps"] = int(upstream["warmup_steps"])
    if "scheduler_epochs" in upstream:
        out_run["scheduler_epochs"] = int(upstream["scheduler_epochs"])
    if "optimizer" in upstream:
        out_run["optimizer"] = str(upstream["optimizer"])
    if "scheduler" in upstream:
        out_run["scheduler"] = str(upstream["scheduler"])
    if "freeze_middle" in upstream:
        out_run["freeze_middle"] = bool(upstream["freeze_middle"])
    if "freeze_processor" in upstream:
        out_run["freeze_processor"] = bool(upstream["freeze_processor"])
    if "embedding_offset" in upstream:
        out_run["embedding_offset"] = int(upstream["embedding_offset"])

    # Synthetic-shape knobs (used only when no real data is wired up yet).
    for k, key in (("C", "C"), ("H", "H"), ("W", "W"), ("n_samples", "n_samples")):
        if k in upstream:
            out_run[key] = int(upstream[k])

    # Seed (framework-level, for reproducibility of synthetic data).
    out_run.setdefault("seed", 0)

    return {"model": out_model, "run": out_run}


# --- helpers --------------------------------------------------------------

def _print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


# --- mode dispatch --------------------------------------------------------

def run_train(model, run_cfg: Dict[str, Any], weights: Optional[str], mode_label: str = "train") -> None:
    if weights is not None:
        print(f"[run] --mode {mode_label} ignoring --weights ({weights}); not used for from-scratch.")
    set_seed(int(run_cfg.get("seed", 0)))

    T = int(run_cfg.get("T", 10))
    state_labels = run_cfg.get("state_labels") or list(range(int(run_cfg.get("C", 3))))
    C = max(int(run_cfg.get("C", 3)), max(state_labels, default=-1) + 1)
    H = int(run_cfg.get("H", 64))
    W = int(run_cfg.get("W", 64))
    n_samples = int(run_cfg.get("n_samples", 8))
    batch_size = int(run_cfg.get("batch_size", 2))

    if hasattr(model, "set_inference_context"):
        model.set_inference_context(
            state_labels=state_labels,
            bcs=torch.zeros(1, 2, dtype=torch.long),
        )

    loader = make_synthetic_loader(
        n_samples=n_samples, T=T, C=C, H=H, W=W, batch_size=batch_size, seed=run_cfg.get("seed", 0)
    )

    train_cfg = {
        "epochs": run_cfg.get("epochs", 2),
        "lr": run_cfg.get("lr", 1e-4),
        "T": T,
        "device": run_cfg.get("device", "cpu"),
        "weight_decay": run_cfg.get("weight_decay", 1e-5),
        "freeze_processor": run_cfg.get("freeze_processor", False),
    }

    _print_section(f"{mode_label.upper()} ({'from-scratch' if mode_label == 'train' else 'finetune'})")
    print(f"model        : {model.metadata.name}")
    print(f"epochs       : {train_cfg['epochs']}")
    print(f"lr           : {train_cfg['lr']}")
    print(f"window shape : (T={T}, B={batch_size}, C={C}, H={H}, W={W})")
    print(f"state_labels : {state_labels}")

    result = model.train(train_dataset=loader, val_dataset=None, cfg=train_cfg, callbacks=None)
    print(f"final train loss: {result.history['train_loss'][-1]:.6f}")


def run_finetune(model, run_cfg: Dict[str, Any], weights: Optional[str], ckpt_from_cfg: Optional[str]) -> None:
    """Finetuning: load public weights, then native training."""
    src = weights or ckpt_from_cfg
    if src is None:
        raise SystemExit("[run] --mode finetune requires --weights <path> or 'pretrained_ckpt_path' in config.")
    model.load_weights(src, strict=False)
    print(f"[run] loaded weights from {src}")
    run_train(model, run_cfg, weights=None, mode_label="finetune")


def run_test(model, run_cfg: Dict[str, Any], weights: Optional[str], ckpt_from_cfg: Optional[str]) -> None:
    """Zero-shot evaluation: load weights, run forward on a few batches, report MSE."""
    src = weights or ckpt_from_cfg
    if src is None:
        print("[run] --mode test without --weights: running untrained model (sanity check only).")
    else:
        model.load_weights(src, strict=False)
        print(f"[run] loaded weights from {src}")

    T = int(run_cfg.get("T", 10))
    state_labels = run_cfg.get("state_labels") or list(range(int(run_cfg.get("C", 3))))
    C = max(int(run_cfg.get("C", 3)), max(state_labels, default=-1) + 1)
    H = int(run_cfg.get("H", 64))
    W = int(run_cfg.get("W", 64))
    n_batches = int(run_cfg.get("test_batches", 3))

    if hasattr(model, "set_inference_context"):
        model.set_inference_context(
            state_labels=state_labels,
            bcs=torch.zeros(1, 2, dtype=torch.long),
        )

    loader = make_synthetic_loader(
        n_samples=n_batches * 2, T=T, C=C, H=H, W=W, batch_size=1, seed=run_cfg.get("seed", 0)
    )

    _print_section("TEST (zero-shot)")
    print(f"model        : {model.metadata.name}")
    print(f"batches      : {n_batches}")
    print(f"window shape : (T={T}, B=1, C={C}, H={H}, W={W})")
    print(f"state_labels : {state_labels}")

    device = torch.device(run_cfg.get("device", "cpu"))
    model.native_model.to(device).eval()

    losses = []
    with torch.no_grad():
        for i, (window, target) in enumerate(loader):
            if i >= n_batches:
                break
            window = window.to(device)
            target = target.to(device)
            pred = model.forward(window)
            mse = torch.nn.functional.mse_loss(pred, target).item()
            losses.append(mse)
            print(f"  batch {i:>2d}  mse = {mse:.6f}  pred.shape = {tuple(pred.shape)}")
    if losses:
        avg = sum(losses) / len(losses)
        print(f"avg mse: {avg:.6f}  ({len(losses)} batches)")


# --- CLI ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="engine.run",
        description="Train / finetune / test a registered SciFM.",
    )
    p.add_argument("--model", help="Registry id of the adapter (e.g. 'MPP').")
    p.add_argument("--mode", choices=["train", "finetune", "test"],
                   help="What to do.")
    p.add_argument("--config", help="Path to YAML or JSON config.")
    p.add_argument("--weights", default=None,
                   help="Path to a state_dict file. Required for finetune; optional for test.")
    p.add_argument("--section", default=None,
                   help="Top-level YAML key to use (e.g. 'basic_config', 'finetune'). "
                        "Default: 'basic_config' for --mode train, 'finetune' otherwise.")
    p.add_argument("--auto-section", action="store_true",
                   help="Pick section by inspecting the 'pretrained' flag.")
    p.add_argument("--list-models", action="store_true",
                   help="Print the registry and exit.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_models:
        print("Registered models:", available_models() or "<none>")
        return

    if not args.model or not args.mode or not args.config:
        raise SystemExit("[run] --model, --mode, and --config are required (or pass --list-models).")

    # Trigger adapter registration by importing the package.
    try:
        import models.MPP  # noqa: F401
    except ImportError:
        pass

    cfg = load_config(args.config)
    section = _select_section(cfg, args.mode, args.section, args.auto_section)
    upstream = cfg[section]
    print(f"[run] using config section: {section}")
    flat = _translate_cfg(upstream)
    model_cfg, run_cfg = flat["model"], flat["run"]

    if args.model not in available_models():
        raise SystemExit(
            f"[run] model '{args.model}' is not registered. Available: {available_models()}"
        )
    Adapter = get_model(args.model)

    model = Adapter.build(model_cfg)
    print(f"[run] built adapter '{args.model}' (family={model.metadata.family})")
    print(f"[run] supported modes: {[m.value for m in model.metadata.supported_modes]}")
    if not model.supports(Mode(args.mode)):
        raise SystemExit(f"[run] model '{args.model}' does not support mode '{args.mode}'.")

    # `pretrained_ckpt_path` from the config is a fallback if --weights is absent.
    ckpt_from_cfg = upstream.get("pretrained_ckpt_path") if isinstance(upstream, dict) else None

    if args.mode == "train":
        run_train(model, run_cfg, args.weights, mode_label="train")
    elif args.mode == "finetune":
        run_finetune(model, run_cfg, args.weights, ckpt_from_cfg)
    elif args.mode == "test":
        run_test(model, run_cfg, args.weights, ckpt_from_cfg)


if __name__ == "__main__":
    main()
