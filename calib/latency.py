"""Cross-correlate logged commands to logged responses for latency estimation.

Two pairings (per the plan, Phase 3):
    steering (+right)  →  -gyro_z  (+right in sim's yaw frame)
    throttle_avg       →  accel_y  (+forward)

Returns peak lag in ms (and the correlation value at the peak). Use the
combined four driving logs to reduce variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .log_io import Log, LOG_DT_S


@dataclass
class LatencyResult:
    steer_to_gyro_ms: float
    throttle_to_accel_ms: float
    steer_peak_corr: float
    throttle_peak_corr: float


def _xcorr_peak_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag k (in samples) that maximizes corr(x[t], y[t+k]) over k in [0, max_lag].
    Positive k = y lags x = response is slower than command.
    """
    x = x - np.mean(x)
    y = y - np.mean(y)
    x_std = np.std(x) or 1.0
    y_std = np.std(y) or 1.0
    best_k = 0
    best_r = -np.inf
    for k in range(0, max_lag + 1):
        if k >= x.size:
            break
        xs = x[: x.size - k]
        ys = y[k : x.size]
        r = float(np.mean(xs * ys) / (x_std * y_std))
        if r > best_r:
            best_r = r
            best_k = k
    return best_k, best_r


def estimate(logs: list[Log], max_lag_ms: float = 500.0) -> LatencyResult:
    """Aggregate across logs. Each log contributes its signals stacked; we
    subtract per-log means to avoid DC bias-contamination of the xcorr."""
    steers, mgyros, throttles, accels = [], [], [], []
    for L in logs:
        steers.append(L.steering_deg - L.steering_deg.mean())
        mgyros.append(-L.gyro_z_lsb + L.gyro_z_lsb.mean())  # flip sign
        throttles.append(L.throttle_avg - L.throttle_avg.mean())
        accels.append(L.accel_y_lsb - L.accel_y_lsb.mean())
    S = np.concatenate(steers)
    G = np.concatenate(mgyros)
    T = np.concatenate(throttles)
    A = np.concatenate(accels)
    max_lag = int(round(max_lag_ms / (LOG_DT_S * 1000.0)))
    k_sg, r_sg = _xcorr_peak_lag(S, G, max_lag)
    k_ta, r_ta = _xcorr_peak_lag(T, A, max_lag)
    return LatencyResult(
        steer_to_gyro_ms=k_sg * LOG_DT_S * 1000.0,
        throttle_to_accel_ms=k_ta * LOG_DT_S * 1000.0,
        steer_peak_corr=r_sg,
        throttle_peak_corr=r_ta,
    )
