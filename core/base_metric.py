"""BaseMetric ABC — shared evaluation metrics.

Per Option C, metrics operate on CANONICAL outputs so all models are
scored with the same implementation. Model-specific quirks never leak
into the metric.

Canonical layout: (B, Nx, Ny, T, C) float32
  B  = batch
  Nx = spatial x dimension
  Ny = spatial y dimension
  T  = time dimension
  C  = channels (typically u, v velocity components at indices 0, 1)
"""
from __future__ import annotations

import numpy as np

# Reference numerical solver time for score_time (seconds per sample)
# Override via environment or config for your specific numerical solver.
T_NUMERICAL_SEC: float = 1.0


def _validate_canonical(arr: np.ndarray, name: str = "array") -> tuple[int, int, int, int, int]:
    """Validate and extract canonical shape (B, Nx, Ny, T, C)."""
    if arr.ndim != 5:
        raise ValueError(f"{name} must be 5D canonical (B, Nx, Ny, T, C); got {arr.shape}")
    return arr.shape


def rel_l2_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Relative L2 error per sample over spatial+temporal dims.

    Args:
        pred: (B, Nx, Ny, T, C) canonical prediction
        target: (B, Nx, Ny, T, C) canonical ground truth
    Returns:
        (B,) relative L2 per sample
    """
    B, Nx, Ny, T, C = _validate_canonical(pred, "pred")
    _, _, _, _, C_t = _validate_canonical(target, "target")
    if C != C_t:
        raise ValueError(f"Channel mismatch: pred C={C} vs target C={C_t}")

    # Flatten spatial+temporal+channel dims per sample
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


def rel_l2_per_timestep(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Relative L2 error per sample per timestep over spatial dims.

    Args:
        pred: (B, Nx, Ny, T, C) canonical prediction
        target: (B, Nx, Ny, T, C) canonical ground truth
    Returns:
        (B, T) relative L2 per sample per timestep
    """
    B, Nx, Ny, T, C = _validate_canonical(pred, "pred")
    _, _, _, _, C_t = _validate_canonical(target, "target")
    if C != C_t:
        raise ValueError(f"Channel mismatch: pred C={C} vs target C={C_t}")

    # Flatten spatial+channel dims per sample per timestep
    p = pred.transpose(0, 3, 1, 2, 4).reshape(B, T, -1)  # (B, T, Nx*Ny*C)
    t = target.transpose(0, 3, 1, 2, 4).reshape(B, T, -1)
    denom = np.linalg.norm(t, axis=2).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=2) / denom


def kinetic_energy(x: np.ndarray) -> np.ndarray:
    """Turbulent kinetic energy per spatial point, averaged over time.

    For canonical (B, Nx, Ny, T, C) with u=x[...,0], v=x[...,1]:
    Computes 0.5 * <(u - <u>_t)^2 + (v - <v>_t)^2>_t at each (Nx, Ny) per batch.

    Args:
        x: (B, Nx, Ny, T, C) canonical field, C >= 2
    Returns:
        (B, Nx, Ny) time-averaged TKE per spatial point
    """
    B, Nx, Ny, T, C = _validate_canonical(x, "x")
    if C < 2:
        raise ValueError(f"kinetic_energy requires C >= 2; got C={C}")

    u = x[..., 0]  # (B, Nx, Ny, T)
    v = x[..., 1]  # (B, Nx, Ny, T)

    # Velocity fluctuations: subtract temporal mean at each spatial point
    u_mean = np.mean(u, axis=3, keepdims=True)  # (B, Nx, Ny, 1)
    v_mean = np.mean(v, axis=3, keepdims=True)  # (B, Nx, Ny, 1)
    u_prime = u - u_mean
    v_prime = v - v_mean

    # TKE = 0.5 * <u'^2 + v'^2>_t
    tke = 0.5 * (np.mean(u_prime**2, axis=3) + np.mean(v_prime**2, axis=3))  # (B, Nx, Ny)
    return tke


def tke_rel_l2_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Relative L2 error of time-averaged TKE per sample.

    Args:
        pred: (B, Nx, Ny, T, C) canonical prediction, C >= 2
        target: (B, Nx, Ny, T, C) canonical ground truth, C >= 2
    Returns:
        (B,) relative L2 of TKE field per sample
    """
    B, Nx, Ny, T, C = _validate_canonical(pred, "pred")
    _, _, _, _, C_t = _validate_canonical(target, "target")
    if C < 2 or C_t < 2:
        return np.zeros((B,), dtype=np.float32)
    if C != C_t:
        raise ValueError(f"Channel mismatch: pred C={C} vs target C={C_t}")

    pred_tke = kinetic_energy(pred)   # (B, Nx, Ny)
    target_tke = kinetic_energy(target)  # (B, Nx, Ny)

    p = pred_tke.reshape(B, -1)
    t = target_tke.reshape(B, -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


def mvpe_rel_l2_per_sample(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    probe_x_frac: float = 0.5,
    probe_y_fracs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> np.ndarray:
    """Mean Velocity Profile Error (MVPE) relative L2 per sample.

    Computes the mean velocity at specified fractional x-locations
    across a range of y-locations, comparing pred vs target.

    Args:
        pred: (B, Nx, Ny, T, C) canonical prediction, C >= 2
        target: (B, Nx, Ny, T, C) canonical ground truth, C >= 2
        probe_x_frac: fractional x-location (0 to 1) for probe line
        probe_y_fracs: fractional y-locations (0 to 1) to sample along the line
    Returns:
        (B,) relative L2 of mean velocity profile per sample
    """
    B, Nx, Ny, T, C = _validate_canonical(pred, "pred")
    _, _, _, _, C_t = _validate_canonical(target, "target")
    if C < 2 or C_t < 2:
        return np.zeros((B,), dtype=np.float32)
    if C != C_t:
        raise ValueError(f"Channel mismatch: pred C={C} vs target C={C_t}")

    # Convert fractional coordinates to indices
    probe_x = int(probe_x_frac * (Nx - 1))
    probe_y = [int(frac * (Ny - 1)) for frac in probe_y_fracs]
    probe_y = [y for y in probe_y if 0 <= y < Ny]
    if not probe_y:
        return np.zeros((B,), dtype=np.float32)

    # Mean velocity over time at probe locations: (B, Ny_probe, 2)
    pred_mean = pred[:, probe_x, probe_y, :, :2].mean(axis=2)  # (B, n_probe, 2)
    target_mean = target[:, probe_x, probe_y, :, :2].mean(axis=2)

    # Flatten (B, n_probe*2) and compute rel L2
    p = pred_mean.reshape(B, -1)
    t = target_mean.reshape(B, -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


def mvpe_rel_l2(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    probe_x_frac: float = 0.5,
    probe_y_fracs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> float:
    """Aggregate MVPE relative L2 over all samples."""
    return float(np.mean(mvpe_rel_l2_per_sample(pred, target, probe_x_frac=probe_x_frac, probe_y_fracs=probe_y_fracs)))


def score_error(err: float, scale: float = 0.5) -> float:
    """Map relative L2 error to [0, 100] score. Higher is better.

    Score = 100 / (1 + scale * max(err, 0))
    """
    if not np.isfinite(err):
        return 0.0
    return float(100.0 / (1.0 + abs(scale) * max(float(err), 0.0)))


def score_time(t_neural: float, t_numerical: float = T_NUMERICAL_SEC, r_min: float = 1.0) -> float:
    """Time score: 100 * ST where ST = 1 / (1 + sqrt(r)), r = t_neural / t_numerical.

    Args:
        t_neural: Neural solver time per sample (seconds)
        t_numerical: Reference numerical solver time per sample (seconds)
        r_min: Minimum ratio threshold
    Returns:
        Score in [0, 100]. Higher is better (faster).
    """
    if not np.isfinite(t_neural) or t_neural <= 0.0:
        return 0.0
    if not np.isfinite(t_numerical) or t_numerical <= 0.0:
        return 0.0
    r = float(t_neural) / float(t_numerical)
    st = 1.0 / (1.0 + (r / r_min) ** 0.5)
    return float(100.0 * st)


def safe_prediction_score(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 3.0,
) -> float:
    """Safe Prediction Score (SPS): fraction of samples where max rel error < threshold.

    SPS = 100 * (num_safe / total) where safe = max_timestep_rel_l2 < threshold.

    Args:
        pred: (B, Nx, Ny, T, C) canonical prediction
        target: (B, Nx, Ny, T, C) canonical ground truth
        threshold: Maximum allowed relative L2 per timestep (default 3.0 = 300%)
    Returns:
        Score in [0, 100]. Higher is better (more samples within threshold).
    """
    B, Nx, Ny, T, C = _validate_canonical(pred, "pred")
    _, _, _, _, C_t = _validate_canonical(target, "target")
    if C != C_t:
        raise ValueError(f"Channel mismatch: pred C={C} vs target C={C_t}")

    rel_l2_t = rel_l2_per_timestep(pred, target)  # (B, T)
    max_rel_l2 = np.max(rel_l2_t, axis=1)  # (B,)
    safe = np.sum(max_rel_l2 < threshold)
    return float(100.0 * safe / B)


def composite_score(
    pred: np.ndarray,
    target: np.ndarray,
    t_neural: float,
    t_numerical: float = T_NUMERICAL_SEC,
    *,
    weights: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
    scales: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> dict:
    """Composite score combining error, TKE, MVPE, and time.

    Args:
        pred: (B, Nx, Ny, T, C) canonical prediction
        target: (B, Nx, Ny, T, C) canonical ground truth
        t_neural: Neural solver time per sample (seconds)
        t_numerical: Reference numerical solver time per sample (seconds)
        weights: (w_error, w_tke, w_mvpe, w_time) summing to 1.0
        scales: (scale_error, scale_tke, scale_mvpe) for score_error mapping
    Returns:
        Dict with individual scores and composite
    """
    w_err, w_tke, w_mvpe, w_time = weights
    s_err, s_tke, s_mvpe = scales

    err = float(np.mean(rel_l2_per_sample(pred, target)))
    tke = float(np.mean(tke_rel_l2_per_sample(pred, target)))
    mvpe = mvpe_rel_l2(pred, target)
    time_s = score_time(t_neural, t_numerical)

    s_error = score_error(err, s_err)
    s_tke = score_error(tke, s_tke)
    s_mvpe = score_error(mvpe, s_mvpe)

    composite = w_err * s_error + w_tke * s_tke + w_mvpe * s_mvpe + w_time * time_s

    return {
        "rel_l2": err,
        "tke_rel_l2": tke,
        "mvpe_rel_l2": mvpe,
        "time_score": time_s,
        "score_error": s_error,
        "score_tke": s_tke,
        "score_mvpe": s_mvpe,
        "composite": composite,
    }
