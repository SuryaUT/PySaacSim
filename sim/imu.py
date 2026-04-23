"""Synthesize raw int16 LSB IMU readings (GY-521 / MPU-6050) from physics.

The on-device firmware feeds raw chip-LSB values into the model — see
Section 3 of Model_Interface.md. To stay sim-to-real consistent, this module
emits the same int16 LSB representation and mirrors the chip+firmware filter
chain:

  raw (physics dv/dt, omega·v, omega)
    → + optional Gaussian noise (stand-in for chip noise)
    → 1st-order LPF at fc = IMU_DLPF_CFG (44 Hz default)   [chip]
    → N-sample rolling mean (N = IMU_AVG_SAMPLES, 4 default)  [firmware]
    → int16 saturate

Conventions (per IMU_behavior.md):
  - GyroZ raw > 0  ⇔ left turn (CCW from above; right-hand rule).
    The env negates this so model input "+ = right turn" matches steering.
  - AccelX raw > 0 ⇔ chassis accel toward the right (no flip needed).
  - AccelY raw > 0 ⇔ chassis accel forward (no flip needed).

Scale factors at the chip's default ranges (±2 g, ±250 dps):
  - 16384 LSB / g
  - 131   LSB / (deg/s)

Bias: the sim deliberately emits *bias-free* signals. The on-device firmware
does not subtract static IMU bias before feeding the model, so comparison
against a real log must subtract the per-channel bias from the log (see
IMUCalibration.gyro_bias / accel_bias, applied by `calib/replay.py`). This
keeps the fit about *dynamics*, not DC offsets.
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Optional

from .physics import RobotState

ACCEL_LSB_PER_G = 16384
GYRO_LSB_PER_DPS = 131
G_MPS2 = 9.80665
INT16_MIN = -32768
INT16_MAX = 32767


def _sat16(x: float) -> int:
    return max(INT16_MIN, min(INT16_MAX, int(round(x))))


def _gauss(std: float) -> float:
    return random.gauss(0.0, std) if std > 0 else 0.0


class _Channel:
    """Per-channel LPF + rolling-mean state.

    The LPF is a causal 1st-order IIR with time constant tau = 1/(2π·fc),
    discretized with alpha = 1 − exp(−dt / tau) so the corner frequency is
    honored regardless of the caller's cadence (training vs. replay vs.
    once-per-physics-tick). `avg_samples` then does the block average that
    firmware does on each IMU_Read call — at the 50 Hz read cadence it also
    behaves as a short moving average.
    """

    def __init__(self, fc_hz: float, avg_samples: int) -> None:
        self.fc_hz = max(1e-6, fc_hz)
        self.avg_samples = max(1, int(avg_samples))
        self._lpf: Optional[float] = None
        self._buf: deque[float] = deque(maxlen=self.avg_samples)

    def reset(self) -> None:
        self._lpf = None
        self._buf.clear()

    def push(self, x: float, dt: float) -> float:
        if self._lpf is None:
            self._lpf = x  # no bootstrap kick on first sample
        else:
            tau = 1.0 / (2.0 * math.pi * self.fc_hz)
            alpha = 1.0 - math.exp(-max(dt, 0.0) / max(tau, 1e-9))
            self._lpf += (x - self._lpf) * alpha
        self._buf.append(self._lpf)
        return sum(self._buf) / len(self._buf)


class IMUSimulator:
    """Tracks previous body velocity / time so dv/dt can give longitudinal accel.

    Parameters mirror the firmware chain: `dlpf_fc_hz` = MPU-6050 DLPF corner
    (IMU.h IMU_DLPF_CFG — 44 Hz at cfg=3), `avg_samples` = IMU_AVG_SAMPLES.
    Per-channel Gaussian noise std (`gyro_noise_lsb`, `accel_noise_lsb`) is
    applied *before* the filter chain, so it's attenuated the same way as a
    real sensor's HF noise."""

    def __init__(
        self,
        dlpf_fc_hz: float = 44.0,
        avg_samples: int = 4,
        gyro_noise_lsb: float = 0.0,
        accel_noise_lsb: float = 0.0,
    ) -> None:
        self._prev_v_cms: float = 0.0
        self._prev_t_s: float = 0.0
        self._gyro_noise_lsb = float(gyro_noise_lsb)
        self._accel_noise_lsb = float(accel_noise_lsb)
        self._ch_gyro_z = _Channel(dlpf_fc_hz, avg_samples)
        self._ch_accel_x = _Channel(dlpf_fc_hz, avg_samples)
        self._ch_accel_y = _Channel(dlpf_fc_hz, avg_samples)

    @classmethod
    def from_calibration(cls, imu_cal) -> "IMUSimulator":
        return cls(
            dlpf_fc_hz=imu_cal.dlpf_fc_hz,
            avg_samples=imu_cal.avg_samples,
            gyro_noise_lsb=imu_cal.gyro_noise_lsb,
            accel_noise_lsb=imu_cal.accel_noise_lsb,
        )

    def reset(self, v_cms: float = 0.0, t_s: float = 0.0) -> None:
        self._prev_v_cms = v_cms
        self._prev_t_s = t_s
        self._ch_gyro_z.reset()
        self._ch_accel_x.reset()
        self._ch_accel_y.reset()

    def read(self, state: RobotState, t_s: float) -> tuple[int, int, int]:
        """Sample IMU at simulated time `t_s`. Returns int16 (gyro_z, accel_x, accel_y)."""
        omega = state.omega                 # rad/s; CCW positive matches chip GyroZ
        v_mps = state.v / 100.0             # forward body velocity, m/s

        # Gyro Z directly: chip and sim agree on CCW = positive.
        raw_gyro_z = math.degrees(omega) * GYRO_LSB_PER_DPS + _gauss(self._gyro_noise_lsb)

        # Lateral accel is centripetal, magnitude omega*v, pointing toward
        # rotation center. Left turn (omega>0) ⇒ center on left ⇒ chip sees
        # -X (since +X = right). So accel_x_mps2 = -omega * v.
        accel_x_mps2 = -omega * v_mps
        raw_accel_x = accel_x_mps2 / G_MPS2 * ACCEL_LSB_PER_G + _gauss(self._accel_noise_lsb)

        # Longitudinal: numerical dv/dt from the previous read.
        dt = max(1e-6, t_s - self._prev_t_s)
        accel_y_mps2 = (v_mps - self._prev_v_cms / 100.0) / dt
        raw_accel_y = accel_y_mps2 / G_MPS2 * ACCEL_LSB_PER_G + _gauss(self._accel_noise_lsb)

        gz = self._ch_gyro_z.push(raw_gyro_z, dt)
        ax = self._ch_accel_x.push(raw_accel_x, dt)
        ay = self._ch_accel_y.push(raw_accel_y, dt)

        self._prev_v_cms = state.v
        self._prev_t_s = t_s
        return _sat16(gz), _sat16(ax), _sat16(ay)
