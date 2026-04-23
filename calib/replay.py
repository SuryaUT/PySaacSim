"""Inject logged throttle/steering commands into the sim at the firmware's
Robot() cadence (12.5 Hz, 80 ms per log row) and capture the sim's IMU trace
at the same indices. Used by `dynamics_fit` and the ablation/overlay plots.

Key points:
  * One log row == one Robot() tick == one command. Between consecutive rows
    we advance the 1 ms physics sim by 80 ms, applying the (constant)
    commanded throttle/steering across the whole interval.
  * Servo count formula: `count = 3120 + steering_deg * 1200 / 53` (matches
    the firmware's `bump.c`); sim convention flips sign because its yaw
    frame is CCW-positive.
  * Sim IMU is read at the same 12.5 Hz cadence, after 80 ms of physics. The
    sim IMU's internal LPF+block-avg is driven per physics tick (so the
    filter sees the real sample rate). We sample once per log row and pair
    it to the log's (gyro_z, accel_x, accel_y) for the MSE loss.
  * Optional: pose tracking disabled (no walls, no collisions) — we only
    care about the IMU outputs and body velocity, not map-relative state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..sim.calibration import IMUCalibration
from ..sim.constants import (
    PHYSICS_DT_S, SERVO_CENTER_COUNT, SERVO_COUNTS_PER_53DEG,
)
from ..sim.imu import IMUSimulator
from ..sim.physics import RobotState, apply_command, _integrate_dynamics
from .log_io import Log, LOG_DT_S


@dataclass
class ReplayTrace:
    t_s: np.ndarray
    gyro_z_lsb: np.ndarray
    accel_x_lsb: np.ndarray
    accel_y_lsb: np.ndarray
    v_cms: np.ndarray            # sim forward body velocity, cm/s
    omega_rad_s: np.ndarray

    @property
    def n(self) -> int:
        return int(self.t_s.size)


def steering_deg_to_servo_count(deg: float) -> int:
    """Firmware's formula: count = 3120 + deg * 1200 / 53."""
    return int(round(SERVO_CENTER_COUNT + deg * SERVO_COUNTS_PER_53DEG / 53.0))


def replay(
    log: Log,
    imu_cal: Optional[IMUCalibration] = None,
    dt_phys: float = PHYSICS_DT_S,
    dt_row: float = LOG_DT_S,
    # Optional parameter override hooks — used by `dynamics_fit` to search
    # over the 6-param vector without mutating the sim's module-level
    # constants. If None, the module-level defaults apply.
    overrides: Optional[dict] = None,
) -> ReplayTrace:
    """Replay `log` through the sim. Returns sim IMU + body-velocity traces
    at each logged row."""
    state = RobotState()
    imu = IMUSimulator.from_calibration(imu_cal) if imu_cal else IMUSimulator()
    n = log.n
    t_out = np.empty(n, dtype=float)
    gz = np.empty(n, dtype=float)
    ax = np.empty(n, dtype=float)
    ay = np.empty(n, dtype=float)
    v = np.empty(n, dtype=float)
    om = np.empty(n, dtype=float)

    t_now = 0.0
    steps_per_row = int(round(dt_row / dt_phys))

    # Allow per-call override of physics constants so the CMA-ES loop can
    # search dynamics params without reimporting modules. Done by temporarily
    # shadowing attributes on `sim.physics`.
    from ..sim import physics as _phys_mod
    saved: dict = {}
    if overrides:
        for k, v_val in overrides.items():
            if hasattr(_phys_mod, k):
                saved[k] = getattr(_phys_mod, k)
                setattr(_phys_mod, k, v_val)

    try:
        for i in range(n):
            servo = steering_deg_to_servo_count(float(log.steering_deg[i]))
            apply_command(
                state,
                int(log.throttle_l[i]),
                int(log.throttle_r[i]),
                servo,
                dir_l=1, dir_r=1,
            )
            # Advance physics for 80 ms with the command held constant. No
            # walls → no collision; we care about the IMU+dynamics response.
            # Capture the IMU output from the *last* in-loop tick so the
            # dt seen by the IMU's dv/dt computation is the real physics dt
            # (1 ms), not a near-zero "snapshot" that would zero out accel_y.
            g_last, ax_last, ay_last = 0, 0, 0
            for _ in range(steps_per_row):
                _integrate_dynamics(state, dt_phys)
                t_now += dt_phys
                # Drive the IMU filter at physics cadence so fc=44 Hz is honored.
                g_last, ax_last, ay_last = imu.read(state, t_now)
            t_out[i] = t_now
            gz[i] = g_last; ax[i] = ax_last; ay[i] = ay_last
            v[i] = state.v
            om[i] = state.omega
    finally:
        # Restore any module-level overrides we applied.
        for k, val in saved.items():
            setattr(_phys_mod, k, val)

    return ReplayTrace(
        t_s=t_out, gyro_z_lsb=gz, accel_x_lsb=ax, accel_y_lsb=ay,
        v_cms=v, omega_rad_s=om,
    )
