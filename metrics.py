def rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, c: int) -> np.ndarray:
    p = pred[..., :c].reshape(pred.shape[0], -1)
    t = target[..., :c].reshape(target.shape[0], -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


def kinetic_energy(x: np.ndarray) -> np.ndarray:
    u = x[..., 0]
    v = x[..., 1]
    u_prime = np.mean((u - np.mean(u, axis=1, keepdims=True)) ** 2, axis=1)
    v_prime = np.mean((v - np.mean(v, axis=1, keepdims=True)) ** 2, axis=1)
    return 0.5 * (u_prime + v_prime)


def tke_rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, c: int) -> np.ndarray:
    if c < 2:
        return np.zeros((pred.shape[0],), dtype=np.float32)
    pred_ke = kinetic_energy(pred[..., :c])
    target_ke = kinetic_energy(target[..., :c])
    p = pred_ke.reshape(pred.shape[0], -1)
    t = target_ke.reshape(target.shape[0], -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


def mvpe_rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, sub_s_real: int = 2) -> np.ndarray:
    d = 16
    center_x = 10
    center_y = 32
    n_probe = 9
    N, _, h, w, _ = pred.shape
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(h / (n_probe + 1)))
    probe_y = [
        probe_center_y + interval_y * j
        for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)
    ]
    probe_y = [y for y in probe_y if 0 <= y < h]
    if not probe_y:
        return np.zeros((N,), dtype=np.float32)

    errors = []
    for i in range(4):
        if int((2 * d + center_x) / sub_s_real) < w:
            probe_x = int(((i + 1) * d + center_x) / sub_s_real)
        else:
            probe_x = int((0.5 * (i + 2) * d + center_x) / sub_s_real)
        if not 0 <= probe_x < w:
            continue
        pp = pred[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(N, -1)
        tt = target[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(N, -1)
        denom = np.linalg.norm(tt, axis=1).clip(min=1e-8)
        errors.append(np.linalg.norm(pp - tt, axis=1) / denom)
    if not errors:
        return np.zeros((N,), dtype=np.float32)
    return np.mean(np.stack(errors, axis=0), axis=0)


def mvpe_rel_l2(pred: np.ndarray, target: np.ndarray, sub_s_real: int = 2) -> float:
    return float(np.mean(mvpe_rel_l2_per_sample(pred, target, sub_s_real)))


def score_error(err: float, scale: float = 0.5) -> float:
    if not np.isfinite(err):
        return 0.0
    return float(100.0 / (1.0 + abs(scale) * max(float(err), 0.0)))


def score_time(t_neural: float, r_min: float = 1.0) -> float:
    # ST = 1 / (1 + sqrt(r)), r = t_neural / t_numerical; score = 100 * ST.
    # A missing, zero, negative or non-finite time scores 0.
    if not np.isfinite(t_neural) or t_neural <= 0.0:
        return 0.0
    r = float(t_neural) / T_NUMERICAL_SEC
    st = 1.0 / (1.0 + (r / r_min) ** 0.5)
    return float(100.0 * st)
