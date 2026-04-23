"""Sensor simulation — 3 TF-Luna lidars + 2 side IR sensors. All parameters
come from SensorCalibration; noise/scale/bias can be edited in
config/calibration.yaml without touching code."""
from __future__ import annotations

import math
import random
from typing import Optional

from .calibration import (
    IRCalibration, IRSideCalibration, LidarCalibration, SensorCalibration,
    SensorPlacement, ir_distance_to_volts, ir_firmware_convert,
)
from .geometry import Segment, Vec2, body_to_world, cast_ray
from .physics import Pose


def _gauss(std: float) -> float:
    return random.gauss(0.0, std) if std > 0 else 0.0


def _cast_from(
    walls: list[Segment],
    pose: Pose,
    bx: float,
    by: float,
    btheta: float,
    max_cm: float,
):
    pose_t = (pose.x, pose.y, pose.theta)
    origin = body_to_world(pose_t, bx, by)
    world_theta = pose_t[2] + btheta
    dir = Vec2(math.cos(world_theta), math.sin(world_theta))
    hit = cast_ray(walls, origin, dir, max_cm)
    true_dist = hit[0] if hit is not None else None
    hit_pt = hit[1] if hit is not None else None
    return origin, dir, true_dist, hit_pt


def lidar_reading(
    walls: list[Segment], pose: Pose, placement: SensorPlacement, cal: LidarCalibration
) -> dict:
    origin, direction, true_dist, hit_pt = _cast_from(
        walls, pose, placement.x, placement.y, placement.theta, cal.max_cm
    )
    valid = true_dist is not None
    raw = true_dist if true_dist is not None else cal.max_cm
    d = raw * cal.distance_scale + cal.distance_bias_cm + _gauss(cal.noise_std_cm)
    if d > cal.max_cm:
        d = cal.max_cm
        valid = False
    if d < 0:
        d = 0.0
    return {
        "id": placement.id,
        "distance_cm": d,
        "valid": valid,
        "origin": origin,
        "dir": direction,
        "hit": hit_pt,
    }


def ir_reading(
    walls: list[Segment],
    pose: Pose,
    placement: SensorPlacement,
    cal: IRCalibration,
    side: IRSideCalibration,
) -> dict:
    """Synthesize an IR reading identical in pipeline to the real firmware:
        true_d → analog volts → ADC quantize (+ noise) → firmware d=a/(adc+b)+c.
    Output `distance_cm` matches what the CSV log records (mm/10)."""
    # Cast a bit past IR max so we can still simulate out-of-range behavior.
    origin, direction, true_dist, hit_pt = _cast_from(
        walls, pose, placement.x, placement.y, placement.theta, cal.max_cm + 5
    )
    true_d_cm = true_dist if true_dist is not None else (cal.max_cm + 5)

    # 1-3: true dist → analog voltage (+ voltage noise).
    volts = max(0.0, ir_distance_to_volts(true_d_cm, cal) + _gauss(cal.voltage_noise_std))
    # 4: volts → ADC count (+ ADC-side noise before firmware formula).
    adc_f = (volts / cal.adc_vref) * cal.adc_max_count + _gauss(cal.adc_noise_std)
    adc = int(round(adc_f))
    if adc < 0:
        adc = 0
    if adc > cal.adc_max_count:
        adc = cal.adc_max_count
    # 5-6: firmware-side convert with saturation.
    reported_mm = ir_firmware_convert(adc, side, sat_mm=cal.sat_mm)
    reported_cm = reported_mm / 10.0
    # Valid = not at the out-of-range sentinel (matches firmware semantics).
    valid = adc >= side.adc_threshold and reported_mm < cal.sat_mm
    return {
        "id": placement.id,
        "distance_cm": reported_cm,
        "distance_mm": reported_mm,
        "valid": valid,
        "adc": adc,
        "volts": volts,
        "origin": origin,
        "dir": direction,
        "hit": hit_pt,
    }


def sample_sensors(walls: list[Segment], pose: Pose, cal: SensorCalibration) -> dict:
    lc, ll, lr = cal.lidar_placements
    il, ir_p = cal.ir_placements
    return {
        "lidar": {
            "center": lidar_reading(walls, pose, lc, cal.lidar),
            "left":   lidar_reading(walls, pose, ll, cal.lidar),
            "right":  lidar_reading(walls, pose, lr, cal.lidar),
        },
        "ir": {
            "left":  ir_reading(walls, pose, il,  cal.ir, cal.ir_side(il.id)),
            "right": ir_reading(walls, pose, ir_p, cal.ir, cal.ir_side(ir_p.id)),
        },
    }
