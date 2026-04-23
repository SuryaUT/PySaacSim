"""Estimate static IMU bias from a motors-off stationary capture.

The output is raw int16 LSB offsets per axis, matching what the firmware
would need to subtract to zero the sensor at rest. `IMU_steady_state.csv`
on this robot was captured with **only the sensor board powered** (motor
board off), so the IMU readings are pure sensor noise around the chip's
static offsets — no motor vibration mixed in. The means are clean biases.
(The throttle column shows 9999 because the firmware register held a stale
value; the motors had no power, so it isn't an actual command.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .log_io import Log
from .windows import Window, find_quiescent


@dataclass
class IMUBiasEstimate:
    # x, y, z for each; we only log z for gyro (the others aren't in the log
    # format) — so gyro_x / gyro_y are always 0 from this estimate.
    gyro_bias: tuple[int, int, int]     # (x, y, z)
    accel_bias: tuple[int, int, int]    # (x, y, z)
    gyro_std_lsb: float                 # per-sample std of gyro_z
    accel_x_std_lsb: float
    accel_y_std_lsb: float
    n_samples: int
    notes: str


def estimate_bias(log: Log, window: Window | None = None) -> IMUBiasEstimate:
    if window is None:
        wins = find_quiescent(log)
        if not wins:
            raise ValueError(
                "no quiescent window found in log; pass an explicit Window"
            )
        # Take the longest one.
        window = max(wins, key=len)
    i, j = window.start, window.stop
    gz = log.gyro_z_lsb[i:j]
    ax = log.accel_x_lsb[i:j]
    ay = log.accel_y_lsb[i:j]
    gyro_bias_z = int(round(float(np.mean(gz))))
    accel_bias_x = int(round(float(np.mean(ax))))
    accel_bias_y = int(round(float(np.mean(ay))))
    notes = ""
    return IMUBiasEstimate(
        gyro_bias=(0, 0, gyro_bias_z),
        accel_bias=(accel_bias_x, accel_bias_y, 0),
        gyro_std_lsb=float(np.std(gz, ddof=1)) if gz.size > 1 else 0.0,
        accel_x_std_lsb=float(np.std(ax, ddof=1)) if ax.size > 1 else 0.0,
        accel_y_std_lsb=float(np.std(ay, ddof=1)) if ay.size > 1 else 0.0,
        n_samples=int(j - i),
        notes=notes,
    )
