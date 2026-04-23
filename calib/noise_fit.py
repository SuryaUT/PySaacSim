"""Fit per-channel sensor noise std on quiescent windows.

Produces:
  - IR per-side distance noise std (from ir_r, ir_l columns; mm)
  - IR per-side ADC noise std (inverse-mapped through the firmware formula)
  - TFLuna per-lidar noise std (tf_r, tf_l, tf_front)
  - IMU noise std per channel (already computed by imu_bias.estimate_bias,
    but we keep a channel-level summary here)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.calibration import IRCalibration, IRSideCalibration
from .log_io import Log
from .windows import Window, find_quiescent


@dataclass
class SensorNoiseFit:
    ir_right_mm_std: float
    ir_left_mm_std: float
    ir_right_adc_std: float   # propagated back through firmware formula
    ir_left_adc_std: float
    tf_right_mm_std: float
    tf_left_mm_std: float
    tf_front_mm_std: float
    n_samples: int


def _adc_std_from_mm_std(mm_std: float, mm_center: float, side: IRSideCalibration) -> float:
    """Propagate mm noise std back to ADC noise std via the firmware formula.

    d(adc) = a/(adc+b) + c  →  dd/dadc = -a/(adc+b)^2.
    If mm_center is the mean reported distance and we know the side fit,
    we can solve for adc_center and compute |dd/dadc|.
    """
    if mm_std <= 0 or not np.isfinite(mm_center):
        return 0.0
    # Don't propagate if at saturation sentinel (305 mm); slope is 0 there.
    if mm_center >= 304.0:
        return 0.0
    adc_center = side.a / (mm_center - side.c) - side.b
    denom = (adc_center + side.b) ** 2
    if denom <= 0:
        return 0.0
    slope_mm_per_adc = side.a / denom
    if slope_mm_per_adc <= 0:
        return 0.0
    return float(mm_std / slope_mm_per_adc)


def fit_noise(
    log: Log, ir_cal: IRCalibration, window: Window | None = None
) -> SensorNoiseFit:
    if window is None:
        wins = find_quiescent(log)
        if not wins:
            raise ValueError("no quiescent window found in log")
        window = max(wins, key=len)
    i, j = window.start, window.stop
    ir_r = log.ir_r_mm[i:j]
    ir_l = log.ir_l_mm[i:j]
    tf_r = log.tf_r_mm[i:j]
    tf_l = log.tf_l_mm[i:j]
    tf_f = log.tf_front_mm[i:j]

    # Exclude saturated IR samples (305 mm) from std calc — std of a constant
    # is 0. Require a reasonably large non-saturated sample count so noise on
    # a few outliers doesn't dominate.
    def _ir_mm_std(x: np.ndarray, min_n: int = 100) -> tuple[float, float]:
        mask = x < 304.0
        xx = x[mask]
        if xx.size < min_n:
            return 0.0, 0.0
        return float(np.std(xx, ddof=1)), float(np.mean(xx))

    # For TFLuna, reject signals that are clearly non-stationary (wide range /
    # fully-saturated). We look for stable sub-segments via rolling-IQR.
    def _tf_mm_std(x: np.ndarray) -> float:
        if x.size < 20:
            return 0.0
        # Drop samples at the 1000 mm cap (sensor max).
        xx = x[x < 999.0]
        if xx.size < 20:
            return 0.0
        # If the distribution is very wide, we're looking at a moving target,
        # not noise. Reject unless span stays within ~30 mm of the mode.
        q1, q3 = np.quantile(xx, [0.25, 0.75])
        iqr = q3 - q1
        if iqr > 15.0:
            return 0.0
        # Keep samples within ±2·IQR of the median and compute std there.
        med = float(np.median(xx))
        inliers = xx[np.abs(xx - med) <= max(2.0 * iqr, 5.0)]
        if inliers.size < 20:
            return 0.0
        return float(np.std(inliers, ddof=1))

    r_std, r_mean = _ir_mm_std(ir_r)
    l_std, l_mean = _ir_mm_std(ir_l)

    r_adc_std = _adc_std_from_mm_std(r_std, r_mean, ir_cal.right)
    l_adc_std = _adc_std_from_mm_std(l_std, l_mean, ir_cal.left)

    return SensorNoiseFit(
        ir_right_mm_std=r_std,
        ir_left_mm_std=l_std,
        ir_right_adc_std=r_adc_std,
        ir_left_adc_std=l_adc_std,
        tf_right_mm_std=_tf_mm_std(tf_r),
        tf_left_mm_std=_tf_mm_std(tf_l),
        tf_front_mm_std=_tf_mm_std(tf_f),
        n_samples=int(j - i),
    )
