"""Sanity-check the log loader against one of the real CSVs."""
from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from PySaacSim.calib.log_io import load_log, LOG_DT_S


REPO = Path(__file__).resolve().parents[2]
STEADY = REPO / "MSPM0LabProjects" / "IMU_steady_state.csv"
DRIVE  = REPO / "MSPM0LabProjects" / "robot_clockwise.csv"


@pytest.mark.skipif(not STEADY.exists(), reason="steady-state CSV not available")
def test_steady_log_shapes() -> None:
    L = load_log(STEADY)
    assert L.n > 1000
    # Nominal 80 ms cadence; the MSPM0 Robot() task actually runs at ~80-90 ms
    # depending on TFLuna semaphore timing. Accept ±15 ms.
    dts = np.diff(L.time_ms)
    assert abs(float(np.median(dts)) - LOG_DT_S * 1000) < 15.0


@pytest.mark.skipif(not DRIVE.exists(), reason="driving CSV not available")
def test_drive_log_sign_conventions() -> None:
    L = load_log(DRIVE)
    # Throttles live on 0..10000.
    assert L.throttle_l.min() >= 0 and L.throttle_l.max() <= 10_000
    # Steering stays within firmware CAP_STEERING=35.
    assert abs(L.steering_deg).max() <= 35
