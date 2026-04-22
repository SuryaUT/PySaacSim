"""Synthesize raw int16 LSB IMU readings (GY-521 / MPU-6050) from physics.

The on-device firmware feeds raw chip-LSB values into the model — see
Section 3 of Model_Interface.md. To stay sim-to-real consistent, this module
emits the same int16 LSB representation. The downstream env applies the
same +right/+forward sign conventions and the GyroZ negation that
`Robot()` does at the input-packing site.

Conventions (per IMU_behavior.md):
  - GyroZ raw > 0  ⇔ left turn (CCW from above; right-hand rule).
    The env negates this so model input "+ = right turn" matches steering.
  - AccelX raw > 0 ⇔ chassis accel toward the right (no flip needed).
  - AccelY raw > 0 ⇔ chassis accel forward (no flip needed).

Scale factors at the chip's default ranges (±2 g, ±250 dps):
  - 16384 LSB / g
  - 131   LSB / (deg/s)
"""
from __future__ import annotations

import math

from .physics import RobotState

ACCEL_LSB_PER_G = 16384
GYRO_LSB_PER_DPS = 131
G_MPS2 = 9.80665
INT16_MIN = -32768
INT16_MAX = 32767


def _sat16(x: float) -> int:
    return max(INT16_MIN, min(INT16_MAX, int(round(x))))


class IMUSimulator:
    """Tracks previous body velocity / time so dv/dt can give longitudinal accel."""

    def __init__(self) -> None:
        self._prev_v_cms: float = 0.0
        self._prev_t_s: float = 0.0

    def reset(self, v_cms: float = 0.0, t_s: float = 0.0) -> None:
        self._prev_v_cms = v_cms
        self._prev_t_s = t_s

    def read(self, state: RobotState, t_s: float) -> tuple[int, int, int]:
        """Sample IMU at simulated time `t_s`. Returns int16 (gyro_z, accel_x, accel_y)."""
        omega = state.omega                 # rad/s; CCW positive matches chip GyroZ
        v_mps = state.v / 100.0             # forward body velocity, m/s

        # Gyro Z directly: chip and sim agree on CCW = positive.
        raw_gyro_z = _sat16(math.degrees(omega) * GYRO_LSB_PER_DPS)

        # Lateral accel is centripetal, magnitude omega*v, pointing toward
        # rotation center. Left turn (omega>0) ⇒ center on left ⇒ chip sees
        # -X (since +X = right). So accel_x_mps2 = -omega * v.
        accel_x_mps2 = -omega * v_mps
        raw_accel_x = _sat16(accel_x_mps2 / G_MPS2 * ACCEL_LSB_PER_G)

        # Longitudinal: numerical dv/dt from the previous read.
        dt = max(1e-6, t_s - self._prev_t_s)
        accel_y_mps2 = (v_mps - self._prev_v_cms / 100.0) / dt
        raw_accel_y = _sat16(accel_y_mps2 / G_MPS2 * ACCEL_LSB_PER_G)

        self._prev_v_cms = state.v
        self._prev_t_s = t_s
        return raw_gyro_z, raw_accel_x, raw_accel_y
