"""Sensor calibration surface — all tunable knobs that describe real-hardware
behavior, loaded from YAML so no code changes are needed when the physical
robot's calibration is updated.

One knob per physical quantity:
  IR:
    curve_k, curve_c, floor_v   analog front-end response v(d_cm)
    peak_cm                     non-monotonic regime boundary
    min_cm, max_cm              reported-valid range
    *_noise_std                 gaussian jitter (voltage + ADC counts)
    adc_*                       ADC converter params
    left / right                firmware formula constants (d_mm=A/(adc+B)+C)
  Lidar:
    max_cm, noise, scale/bias, rate, fov
  IMU:
    dlpf_fc_hz, avg_samples     firmware filter chain
    *_noise_lsb                 per-sample Gaussian std, raw LSB
    gyro_bias, accel_bias       static bias (used by calib/replay; sim stays
                                bias-free — see sim/imu.py docstring)
  Placements:
    per-sensor (x, y, theta) in body frame
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union

import yaml


_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "config" / "calibration.yaml"


@dataclass
class IRSideCalibration:
    """Firmware IR formula per side (RTOS_Labs_common/IRDistance.c lines 78-102).

    d_mm = a / (adc + b) + c, saturated at 305 mm when adc < adc_threshold.
    """
    a: float
    b: float
    c: float
    adc_threshold: int


@dataclass
class IRCalibration:
    # Analog front-end (shared: both sides use the same GP2Y0A sensor).
    # v(d_cm) = curve_k / (d_cm + curve_c) + floor_v  for d_cm >= peak_cm.
    curve_k: float = 12.79
    curve_c: float = 0.585
    floor_v: float = 0.0
    peak_cm: float = 4.0
    peak_floor_v: float = 0.4
    v_clamp: float = 3.1
    min_cm: float = 4.0
    max_cm: float = 30.5
    voltage_noise_std: float = 0.02
    adc_noise_std: float = 0.0          # counts, pre-firmware-formula
    adc_max_count: int = 4095
    adc_vref: float = 3.3
    # Firmware per-side constants (bootstrap values = current firmware).
    left: IRSideCalibration = field(default_factory=lambda: IRSideCalibration(
        a=137932.0, b=-859.0, c=32.0, adc_threshold=1376))
    right: IRSideCalibration = field(default_factory=lambda: IRSideCalibration(
        a=52850.0, b=-1239.0, c=69.0, adc_threshold=1476))
    sat_mm: float = 305.0               # firmware out-of-range sentinel


@dataclass
class LidarCalibration:
    max_cm: float = 800.0
    noise_std_cm: float = 2.0
    fov_rad: float = math.radians(2)
    rate_hz: float = 100.0
    # reported_cm = raw_cm * distance_scale + distance_bias_cm
    distance_scale: float = 1.0
    distance_bias_cm: float = 0.0


@dataclass
class IMUCalibration:
    """Firmware filter chain + static bias.

    The sim applies the LPF + block-avg chain to its synthesized signals (see
    sim/imu.py). `gyro_bias` and `accel_bias` are raw int16 LSB offsets
    measured on the real car; the sim emits bias-free outputs, and
    `calib/replay.py` subtracts these from the real log before comparing.
    """
    dlpf_fc_hz: float = 44.0
    avg_samples: int = 4
    gyro_noise_lsb: float = 0.0
    accel_noise_lsb: float = 0.0
    # (x, y, z). MPU-6050 LSB. Defaults to zero bias.
    gyro_bias: list[int] = field(default_factory=lambda: [0, 0, 0])
    accel_bias: list[int] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class SensorPlacement:
    id: str
    x: float
    y: float
    theta: float


@dataclass
class SensorCalibration:
    ir: IRCalibration
    lidar: LidarCalibration
    imu: IMUCalibration
    lidar_placements: list[SensorPlacement]
    ir_placements: list[SensorPlacement]

    @classmethod
    def default(cls) -> "SensorCalibration":
        return cls.from_yaml(_DEFAULT_YAML)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SensorCalibration":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "SensorCalibration":
        ir_data = dict(data["ir"])
        # Legacy no-op field tolerated for backward-compat with older YAMLs.
        ir_data.pop("distance_noise_std_cm", None)
        left = IRSideCalibration(**ir_data.pop("left")) if "left" in ir_data else IRSideCalibration(
            a=137932.0, b=-859.0, c=32.0, adc_threshold=1376)
        right = IRSideCalibration(**ir_data.pop("right")) if "right" in ir_data else IRSideCalibration(
            a=52850.0, b=-1239.0, c=69.0, adc_threshold=1476)
        ir = IRCalibration(left=left, right=right, **ir_data)
        lidar = LidarCalibration(**data["lidar"])
        imu_raw = data.get("imu", {}) or {}
        imu = IMUCalibration(**imu_raw)
        lidars = [SensorPlacement(**p) for p in data["lidar_placements"]]
        irs = [SensorPlacement(**p) for p in data["ir_placements"]]
        if len(lidars) != 3:
            raise ValueError(f"expected 3 lidar placements, got {len(lidars)}")
        if len(irs) != 2:
            raise ValueError(f"expected 2 IR placements, got {len(irs)}")
        return cls(ir=ir, lidar=lidar, imu=imu,
                   lidar_placements=lidars, ir_placements=irs)

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    def copy(self) -> "SensorCalibration":
        return copy.deepcopy(self)

    def ir_side(self, placement_id: str) -> IRSideCalibration:
        """Resolve a placement id ('ir_left' / 'ir_right') → side constants."""
        if placement_id.endswith("_left") or placement_id == "left":
            return self.ir.left
        if placement_id.endswith("_right") or placement_id == "right":
            return self.ir.right
        raise ValueError(f"cannot resolve IR side from placement id {placement_id!r}")


def ir_distance_to_volts(d_cm: float, cal: IRCalibration) -> float:
    """Voltage produced by the IR analog front-end for a true distance d_cm.

    Below peak_cm the sensor is non-monotonic (very-close object reads like a
    ~25 cm object); model that regime as a linear rise from peak_floor_v at
    d=0 up to the rational-curve value at peak_cm.
    """
    peak_v = cal.curve_k / (cal.peak_cm + cal.curve_c) + cal.floor_v
    clamped_peak = min(cal.v_clamp, peak_v)
    if d_cm < cal.peak_cm:
        t = d_cm / cal.peak_cm
        return cal.peak_floor_v + (clamped_peak - cal.peak_floor_v) * t
    v = cal.curve_k / (d_cm + cal.curve_c) + cal.floor_v
    return max(0.0, min(cal.v_clamp, v))


def ir_firmware_convert(adc: int, side: IRSideCalibration, sat_mm: float = 305.0) -> float:
    """Apply the on-device firmware formula d_mm = a/(adc+b) + c.

    Mirrors IRDistance_Left / IRDistance_Right byte-for-byte:
        if adc < adc_threshold: return sat_mm   (out-of-range sentinel)
        else:                   return a/(adc + b) + c
    """
    if adc < side.adc_threshold:
        return float(sat_mm)
    denom = adc + side.b
    if denom == 0:
        return float(sat_mm)
    return side.a / denom + side.c
