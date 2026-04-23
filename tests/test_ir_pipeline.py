"""End-to-end IR sensor pipeline test. For a sequence of true distances, the
sim's `ir_reading` should produce output mm that matches
    d_mm = ir_firmware_convert( round( v(d)/Vref * adc_max ), side )
within a small tolerance driven by the noise knobs (which we set to 0 here)."""
from __future__ import annotations

import math

import pytest

from PySaacSim.sim.calibration import (
    IRCalibration, IRSideCalibration, SensorCalibration, IMUCalibration,
    LidarCalibration, SensorPlacement, ir_firmware_convert, ir_distance_to_volts,
)
from PySaacSim.sim.geometry import Segment, Vec2
from PySaacSim.sim.physics import Pose
from PySaacSim.sim.sensors import ir_reading


def _cal_noiseless() -> IRCalibration:
    return IRCalibration(
        curve_k=10.7489, curve_c=-2.7760, floor_v=0.7879,
        peak_cm=4.0, peak_floor_v=0.4, v_clamp=3.1,
        min_cm=4.0, max_cm=30.5,
        voltage_noise_std=0.0, adc_noise_std=0.0,
        adc_max_count=4095, adc_vref=3.3,
        left=IRSideCalibration(a=80_000, b=-500, c=20, adc_threshold=790),
        right=IRSideCalibration(a=76_000, b=-495, c=20, adc_threshold=765),
    )


@pytest.mark.parametrize("d_cm", [8.0, 12.0, 16.0, 20.0, 24.0, 28.0])
def test_round_trip_matches_firmware_formula(d_cm: float) -> None:
    cal = _cal_noiseless()
    # Put the sensor at origin pointing +x; wall at true distance d_cm.
    placement = SensorPlacement(id="ir_left", x=0.0, y=0.0, theta=0.0)
    pose = Pose(0.0, 0.0, 0.0)
    walls = [
        Segment(Vec2(d_cm, -50.0), Vec2(d_cm, 50.0)),
    ]
    out = ir_reading(walls, pose, placement, cal, cal.left)
    # Reference: apply the same pipeline by hand.
    v = ir_distance_to_volts(d_cm, cal)
    adc = int(round(v / cal.adc_vref * cal.adc_max_count))
    adc = max(0, min(adc, cal.adc_max_count))
    expected_mm = ir_firmware_convert(adc, cal.left, sat_mm=cal.sat_mm)
    # Allow 1 mm tolerance for ray-cast sub-cm snapping to the wall geometry.
    assert abs(out["distance_mm"] - expected_mm) < 1.0
    assert out["adc"] == adc


def test_out_of_range_saturates() -> None:
    # Threshold high enough that any ray-cast-produced ADC falls below it →
    # firmware OOR sentinel (305 mm) returned.
    cal = _cal_noiseless()
    cal.left = IRSideCalibration(a=80_000, b=-500, c=20, adc_threshold=4095)
    placement = SensorPlacement(id="ir_left", x=0.0, y=0.0, theta=0.0)
    pose = Pose(0.0, 0.0, 0.0)
    walls = [Segment(Vec2(25.0, -50.0), Vec2(25.0, 50.0))]
    out = ir_reading(walls, pose, placement, cal, cal.left)
    assert out["distance_mm"] == cal.sat_mm
    assert out["valid"] is False
