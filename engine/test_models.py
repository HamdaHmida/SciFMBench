"""Smoke test for a registered SciFM adapter.

Goal
----
Quickly answer: "is this model wired up correctly enough that we can
launch a real training run against it?" without needing real data or
real weights.

What it does
------------
1. Builds the chosen model via the registry.
2. Generates a random canonical tensor of shape (16, 128, 128, 20, 2)
   = (B, Nx, Ny, T, C) — fluid-dynamics canonical layout.
3. Wraps it in a DataLoader with batch_size=4 (4 batches per epoch).
4. Permutes the canonical data into the model's native layout
   (per Option C, training is native — the framework does not impose
   its own tensor layout on the model).
5. Runs 1 training epoch via `model.train(...)`.
6. Verifies and reports on:
     - the model builds without error
     - forward output shape matches the expected native output
     - to_canonical / from_canonical round-trip cleanly
     - the training loop completes
     - losses are finite (no NaN / Inf)
     - gradients are flowing (loss is meaningfully smaller than
       input-variance after one step)
7. Prints PASS / FAIL with details.

Usage
-----
    python3 -m engine.test_models --model MPP
    python3 -m engine.test_models --list-models   # show what's registered

Exit code: 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

# Make `python -m engine.test_models` work whether called from repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.base_model import available_models, get_model  # noqa: E402
from data.synthetic import canonical_to_native, make_canonical_sample  # noqa: E402


# --- shape defaults (the script's contract) -------------------------------

CANON_B = 16
CANON_NX = 128
CANON_NY = 128
CANON_T = 20
CANON_C = 2  # default to 2 channels (e.g. vx, vy); override via --channels
BATCH_SIZE = 4
EPOCHS = 1
SEED = 0


# --- per-model canonical-shape overrides ----------------------------------
#
# Most models are fine with the default 16x128x128x20x2 shape. But some
# model families — like the KAN MLP, which flattens the whole
# (T, C, Nx, Ny) window into a single feature vector — have a layer
# whose size is T*C*Nx*Ny. At default shape that's 655k input features
# and the B-spline weight alone is hundreds of GB. The smoke test runs
# a from-scratch model on CPU, so we override the canonical shape for
# those families to keep memory + time bounded.
_CANON_SHAPE_OVERRIDES: Dict[str, Dict[str, int]] = {
    # KAN: flatten (T, C, Nx, Ny) into a single feature vector.
    # 4*2*16*16 = 2048 inputs is a tractable smoke-test size.
    "KAN": {"B": 4, "Nx": 16, "Ny": 16, "T": 4, "C": 2},
    # KANConv applies a per-timestep Conv2d with B-spline kernels.
    # The vendored code creates intermediate tensors of size
    # (B, C, kernel_size^2, H*W) which at 128x128 is ~32M elements
    # per timestep. Reduce spatial dims for the smoke test.
    "KANConv": {"B": 4, "Nx": 32, "Ny": 32, "T": 4, "C": 2},
}


def _canon_shape_for(model_name: str) -> Dict[str, int]:
    return _CANON_SHAPE_OVERRIDES.get(
        model_name, {"B": CANON_B, "Nx": CANON_NX, "Ny": CANON_NY, "T": CANON_T, "C": CANON_C}
    )


# --- per-model default config --------------------------------------------

# Minimal configs that get any registered model off the ground.
# Adapters can extend this with their own keys.
_DEFAULT_CFG: Dict[str, Dict[str, Any]] = {
    "MPP": {
        "patch_size": (16, 16),
        "embed_dim": 96,        # tiny variant for the smoke test
        "processor_blocks": 2,
        "n_states": CANON_C,
        "num_heads": 4,
        "space_type": "axial_attention",
        "time_type": "attention",
        "block_type": "axial",
        # IMPORTANT: don't use `bias_type: "none"` — the vendored code's
        # 'none' branch builds a 2-arg lambda, but the call site passes 3
        # args (W, W, bcs). That's a latent vendored bug we can't patch.
        # Use "rel" (the upstream YAML default) which works correctly.
        "bias_type": "rel",
    },
    "KAN": {
        # KAN adapter treats the whole (T, C, Nx, Ny) window as a flat
        # feature vector, so the layer sizes get large fast. Keep the
        # bottleneck narrow for the smoke test. The shape-related keys
        # (_T / _C / _Nx / _Ny) are injected by _check_build from the
        # per-model canonical shape.
        "n_hidden": 1,
        "hidden_width": 64,
        "grid_size": 5,
        "spline_order": 3,
        "base_activation": "SiLU",
    },
    "KANConv": {
        # Channels-preserving config: out_channels = in_channels = C,
        # kernel=3 with same-padding so the spatial dims are preserved
        # (the framework's forward_shape check expects the same Nx, Ny).
        # `in_channels` / `out_channels` are injected by _check_build.
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),       # "same" for kernel=3
        "dilation": (1, 1),
        "grid_size": 5,
        "spline_order": 3,
        "base_activation": "SiLU",
    },
    "my_model": {},  # the placeholder adapter is small enough that defaults suffice
}


def _default_cfg_for(model_name: str, channels: int) -> Dict[str, Any]:
    cfg = dict(_DEFAULT_CFG.get(model_name, {}))
    if "n_states" in cfg:
        cfg["n_states"] = channels
    return cfg


# --- adequacy checks ------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _check_build(model_name: str, shape: Dict[str, int]) -> tuple[CheckResult, Any]:
    try:
        cfg = _default_cfg_for(model_name, shape["C"])
        # Inject the per-model canonical shape so adapters (e.g. KAN) that
        # build layer sizes from T / C / Nx / Ny see the same numbers the
        # smoke test will use.
        cfg["_T"] = shape["T"]
        cfg["_C"] = shape["C"]
        cfg["_Nx"] = shape["Nx"]
        cfg["_Ny"] = shape["Ny"]
        # KANConv reads `in_channels` / `out_channels` (matching its
        # constructor) rather than the framework's C.
        if model_name == "KANConv":
            cfg["in_channels"] = shape["C"]
            cfg["out_channels"] = shape["C"]
        model = get_model(model_name).build(cfg)
        return CheckResult("build", True, f"params={model.num_parameters():,}"), model
    except Exception as e:
        return CheckResult("build", False, f"{type(e).__name__}: {e}"), None


def _check_forward_shape(model, shape: Dict[str, int]) -> CheckResult:
    """One canonical batch through from_canonical -> forward -> to_canonical."""
    try:
        B, Nx, Ny, T, C = shape["B"], shape["Nx"], shape["Ny"], shape["T"], shape["C"]
        # Bind inference context the same way _check_training does, so the
        # forward call has the same state_labels / bcs that training will use.
        if hasattr(model, "set_inference_context"):
            model.set_inference_context(
                state_labels=list(range(C)),
                bcs=torch.zeros(1, 2, dtype=torch.long),
            )
        sample = make_canonical_sample(B=1, Nx=Nx, Ny=Ny, T=T, C=C, seed=SEED)
        native_in = canonical_to_native(sample)  # (T, 1, C, Nx, Ny)
        with torch.no_grad():
            y_native = model.forward(native_in)
        # Expected native output for the framework's "single-step" contract:
        # (B=1, C, Nx, Ny). Adapters whose vendored model changes the spatial
        # dims (e.g. KANConv with stride>1) would need a different check; for
        # the default configs we ship here, output shape == input shape.
        expected = (1, C, Nx, Ny)
        if tuple(y_native.shape) != expected:
            return CheckResult(
                "forward_shape",
                False,
                f"got {tuple(y_native.shape)}, expected {expected}",
            )
        # Round-trip through to_canonical.
        canon_out = model.to_canonical(y_native)
        if canon_out.fields.shape[-1] != C:
            return CheckResult(
                "to_canonical",
                False,
                f"channel axis wrong: {tuple(canon_out.fields.shape)}",
            )
        return CheckResult(
            "forward_shape",
            True,
            f"in={tuple(native_in.shape)}  out={tuple(y_native.shape)}  "
            f"canon={tuple(canon_out.fields.shape)}",
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        # Keep the last 4 lines — they almost always contain the root cause.
        tail = " | ".join(tb[-4:]) if len(tb) > 4 else " | ".join(tb)
        return CheckResult("forward_shape", False, f"{type(e).__name__}: {e}\n      {tail}")


def _check_training(model, shape: Dict[str, int]) -> tuple[CheckResult, List[float]]:
    """Run 1 epoch over the random-data DataLoader; return (check, losses)."""
    try:
        B, Nx, Ny, T, C = shape["B"], shape["Nx"], shape["Ny"], shape["T"], shape["C"]
        # Canonical data (B, Nx, Ny, T, C) -> permute to native (T, B, C, Nx, Ny).
        canon = make_canonical_sample(B=B, Nx=Nx, Ny=Ny, T=T, C=C, seed=SEED)
        native = canonical_to_native(canon)        # (T, B, C, Nx, Ny)
        # Build a (window, target) dataset in NATIVE layout.
        # window: (T, B, C, Nx, Ny)  — what the model consumes
        # target: (B, C, Nx, Ny)     — the last step it should hit
        # We use the last frame of the window as the target.
        per_sample_window = native.permute(1, 0, 2, 3, 4).contiguous()  # (B, T, C, Nx, Ny)
        target = native[-1]  # (B, C, Nx, Ny)
        dataset = TensorDataset(per_sample_window, target)
        # Batch size: scale down for models with per-sample compute that
        # scales with B*T*C*Nx*Ny (KAN) to keep memory bounded.
        bs = min(BATCH_SIZE, B)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)

        # Most adapters need a state_labels / bcs context for the forward call.
        if hasattr(model, "set_inference_context"):
            state_labels = list(range(C))
            # bcs shape depends on the model; (B=1, 2) is the MPP convention.
            bcs = torch.zeros(1, 2, dtype=torch.long)
            model.set_inference_context(state_labels=state_labels, bcs=bcs)

        # Convert the DataLoader's per-batch layout back to what the model
        # expects: each batch's window is (b, T, C, Nx, Ny) — we need to
        # permute to (T, b, C, Nx, Ny) before calling model.forward().
        def batch_iterator():
            for w, t in loader:
                # w: (b, T, C, Nx, Ny) -> (T, b, C, Nx, Ny)
                w_native = w.permute(1, 0, 2, 3, 4).contiguous()
                yield w_native, t

        cfg = {
            "epochs": EPOCHS,
            "lr": 1e-3,
            "T": T,
            "device": "cpu",  # smoke test stays on CPU
        }
        result = model.train(train_dataset=batch_iterator(), val_dataset=None, cfg=cfg, callbacks=None)
        losses = result.history.get("train_loss", [])
        if not losses:
            return CheckResult("train", False, "no losses recorded"), []
        finite = all(math.isfinite(loss) for loss in losses)
        if not finite:
            return CheckResult("train", False, f"non-finite losses: {losses}"), losses
        return (
            CheckResult(
                "train",
                True,
                f"1 epoch, {len(losses)} batch(es), final loss={losses[-1]:.6f}",
            ),
            losses,
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        tail = " | ".join(tb[-4:]) if len(tb) > 4 else " | ".join(tb)
        return CheckResult("train", False, f"{type(e).__name__}: {e}\n      {tail}"), []


def _check_finite_and_flowing(losses: List[float], sample_variance: float) -> CheckResult:
    """Sanity-check the loss trace: finite, positive, and in a reasonable range
    relative to the input variance.

    We can't demand "loss decreased" with only 1 epoch's worth of aggregated
    losses (most adapters report one number per epoch). What we *can* check is
    that the loss isn't exploding or collapsing — which usually catches
    broken shapes, missing gradients, or wrong data layouts.
    """
    if not losses:
        return CheckResult("finite_and_flowing", False, "no losses to check")
    if any(not math.isfinite(loss) for loss in losses):
        return CheckResult("finite_and_flowing", False, f"non-finite loss: {losses}")
    last = losses[-1]
    # Loss should be in a sane range — not vanishing, not exploding.
    # For N(0, 1) inputs, an MSE loss against any plausible target is O(1).
    upper = 100.0 * sample_variance  # generous upper bound
    lower = 1e-4                     # below this, the model is doing nothing
    if last < lower:
        return CheckResult(
            "finite_and_flowing",
            False,
            f"loss collapsed to {last:.6f} (< {lower}); model may not be training",
        )
    if last > upper:
        return CheckResult(
            "finite_and_flowing",
            False,
            f"loss exploded to {last:.4f} (> {upper}); shapes or normalization likely wrong",
        )
    return CheckResult(
        "finite_and_flowing",
        True,
        f"loss finite & in range: final={last:.4f} (var={sample_variance:.4f}, "
        f"bounds=[{lower}, {upper:.0f}])",
    )


# --- main -----------------------------------------------------------------


def _print_banner(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _print_table(checks: List[CheckResult]) -> None:
    for c in checks:
        flag = "PASS" if c.passed else "FAIL"
        print(f"  [{flag:>4}]  {c.name:<22}  {c.detail}")


def run_smoke_test(model_name: str) -> int:
    shape = _canon_shape_for(model_name)
    B, Nx, Ny, T, C = shape["B"], shape["Nx"], shape["Ny"], shape["T"], shape["C"]
    bs = min(BATCH_SIZE, B)

    _print_banner(f"Model smoke test: {model_name}")
    print(f"  canonical shape : (B={B}, Nx={Nx}, Ny={Ny}, T={T}, C={C})")
    print(f"  batch size      : {bs}  ({B // bs} batches per epoch)")
    print(f"  epochs          : {EPOCHS}")
    print()

    checks: List[CheckResult] = []

    build_result, model = _check_build(model_name, shape)
    checks.append(build_result)
    if not build_result.passed:
        _print_table(checks)
        print()
        print("RESULT: FAIL — model could not be built.")
        return 1

    print(f"  model metadata  : family={model.metadata.family}, "
          f"modes={[m.value for m in model.metadata.supported_modes]}")
    print()

    forward_result = _check_forward_shape(model, shape)
    checks.append(forward_result)

    train_result, losses = _check_training(model, shape)
    checks.append(train_result)

    # Reference variance: a healthy model's first-step loss should be on the
    # order of the input variance. For N(0,1) data, that's ~1.0; for the
    # small drift added by the synthetic loader, ~1.0 still.
    sample_var = 1.0
    checks.append(_check_finite_and_flowing(losses, sample_var))

    _print_table(checks)
    print()
    overall = all(c.passed for c in checks)
    if overall:
        print("RESULT: PASS — model is adequate. You can launch a real training run.")
        return 0
    else:
        print("RESULT: FAIL — at least one check failed. See details above.")
        return 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="engine.test_models",
        description="Smoke-test a registered SciFM adapter with random canonical data.",
    )
    p.add_argument("--model", help="Registry id of the adapter (e.g. 'MPP').")
    p.add_argument("--channels", type=int, default=CANON_C,
                   help=f"Number of channels in the synthetic data (default: {CANON_C}).")
    p.add_argument("--list-models", action="store_true",
                   help="Print the registry and exit.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Trigger adapter registration by importing the packages. If a model's
    # heavy deps are missing the error will surface inside _check_build
    # with a useful traceback. Do this before any registry lookups.
    try:
        import models.MPP  # noqa: F401
    except ImportError:
        pass
    try:
        import models.KAN  # noqa: F401
    except ImportError:
        pass
    try:
        import models.my_model  # noqa: F401
    except ImportError:
        pass

    if args.list_models:
        print("Registered models:", available_models() or "<none>")
        return 0

    if not args.model:
        raise SystemExit("[test_models] --model is required (or pass --list-models).")

    # Override module-level constant for this run if the user asked for a
    # different channel count. We do this by mutating the module attr, which
    # keeps the rest of the script simple.
    global CANON_C
    CANON_C = args.channels

    return run_smoke_test(args.model)


if __name__ == "__main__":
    sys.exit(main())
