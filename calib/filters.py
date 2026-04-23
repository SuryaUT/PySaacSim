"""Offline filter emulation — applies the same IMU chain (44 Hz LPF + N-sample
block avg) that the sim uses, for analysis that needs to compare "unfiltered"
sim output against filtered real data, or vice versa. The sim path itself is
in `sim/imu.py`.

Also provides an 8-sample moving-average wrapper for TFLuna replay-mode
output (matches firmware's RAW_MM_AVG = 8).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np


def butter1_lpf(x: np.ndarray, fc_hz: float, fs_hz: float) -> np.ndarray:
    """1st-order IIR LPF, causal. Matches the discrete form used in sim/imu.py:
        alpha = 1 - exp(-dt/tau), tau = 1/(2π fc).
    """
    if fc_hz <= 0:
        return x.copy()
    dt = 1.0 / fs_hz
    tau = 1.0 / (2.0 * math.pi * fc_hz)
    alpha = 1.0 - math.exp(-dt / tau)
    out = np.empty_like(x, dtype=float)
    acc: Optional[float] = None
    for i, v in enumerate(x):
        acc = float(v) if acc is None else acc + (float(v) - acc) * alpha
        out[i] = acc
    return out


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Causal N-tap rolling mean. Output[0..n-1] averages the partial window."""
    if n <= 1:
        return x.astype(float).copy()
    out = np.empty_like(x, dtype=float)
    buf: deque[float] = deque(maxlen=n)
    for i, v in enumerate(x):
        buf.append(float(v))
        out[i] = sum(buf) / len(buf)
    return out


def apply_imu_chain(x: np.ndarray, fc_hz: float, avg_n: int, fs_hz: float) -> np.ndarray:
    return rolling_mean(butter1_lpf(x, fc_hz, fs_hz), avg_n)


def tfluna_replay_avg(x: np.ndarray, n: int = 8) -> np.ndarray:
    """8-sample moving average for replay-mode lidar comparison (Phase 0e)."""
    return rolling_mean(x, n)
