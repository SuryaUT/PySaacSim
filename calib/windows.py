"""Segment a log into quiescent and active windows.

Quiescent = the car is near-stationary (low |gyro|, near-zero throttle change,
no big accel spikes). We use these windows for noise-std and IMU bias
estimates. Active = everything else; used for dynamics fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .log_io import Log


@dataclass
class Window:
    start: int          # index (inclusive)
    stop: int           # index (exclusive)

    def __len__(self) -> int:
        return self.stop - self.start


def find_quiescent(
    log: Log,
    gyro_lsb_thresh: float = 800.0,       # ~6 deg/s — loose; motor vibration alone
    min_len: int = 20,
) -> list[Window]:
    """Find contiguous index ranges where |gyro_z| stays below threshold.

    This tolerates motor-on stationary capture (`IMU_steady_state.csv`): the
    car's throttle is nonzero but it's not rotating, so gyro stays low.
    """
    gz = np.abs(log.gyro_z_lsb)
    mask = gz < gyro_lsb_thresh
    out: list[Window] = []
    i = 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) >= min_len:
            out.append(Window(start=i, stop=j))
        i = j
    return out


def full(log: Log) -> Window:
    return Window(start=0, stop=log.n)
