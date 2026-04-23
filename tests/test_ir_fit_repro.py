"""Synthetic-recovery test for the IR xlsx fit.

Generate a table of (adc, d_mm) samples from a known (a, b, c), run the fit,
recover the same params within a small tolerance.
"""
from __future__ import annotations

import numpy as np

from PySaacSim.calib.ir_xlsx import _fit_side


def _synth(adc: np.ndarray, a: float, b: float, c: float, noise_mm: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mm = a / (adc + b) + c
    return mm + rng.normal(0.0, noise_mm, size=mm.shape)


def test_recover_left_constants() -> None:
    # Realistic operating range: adc 780..2900 (matching the IR_Calib.xlsx data).
    adc = np.array([2895, 2023, 1580, 1155, 1105, 1035, 963, 915, 840, 815, 790], dtype=float)
    a_true, b_true, c_true = 80_000.0, -500.0, 20.0
    mm = _synth(adc, a_true, b_true, c_true, noise_mm=1.0)
    fit = _fit_side(adc, mm, side="left")
    # Recover a, b, c within ~3 % (small data, noisy fit).
    assert abs(fit.a - a_true) / a_true < 0.03, f"a drift {fit.a} vs {a_true}"
    assert abs(fit.b - b_true) / abs(b_true) < 0.05, f"b drift {fit.b} vs {b_true}"
    assert abs(fit.c - c_true) < 2.0, f"c drift {fit.c} vs {c_true}"
    # RMSE should be near the injected noise level.
    assert fit.rmse_mm < 3.0


def test_adc_threshold_near_data_lower_bound() -> None:
    adc = np.array([2895, 2023, 1580, 1155, 1105, 1035, 963, 915, 840, 815, 790], dtype=float)
    a_true, b_true, c_true = 80_000.0, -500.0, 20.0
    mm = _synth(adc, a_true, b_true, c_true, noise_mm=0.0)
    fit = _fit_side(adc, mm, side="right")
    # Threshold should not exceed the smallest observed adc + some margin.
    assert fit.adc_threshold <= int(adc.min()) + 2
