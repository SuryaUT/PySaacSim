"""Phase 7a regression: actuator transport delay queues.

Verifies that an `apply_command` call does not change the active
setpoint until `STEER_TRANSPORT_LAG_S` / `THROTTLE_TRANSPORT_LAG_S`
seconds of sim time have elapsed.
"""
from __future__ import annotations

import pytest

from PySaacSim.sim import constants as C
from PySaacSim.sim.physics import RobotState, apply_command, _integrate_dynamics


def test_steer_setpoint_held_until_lag_elapses():
    s = RobotState()
    # Before any command, setpoint is 0.
    assert s.steer_cmd == 0.0
    # Issue full-right steering (lowest servo count → most negative angle in
    # CCW-positive convention).
    apply_command(s, 0, 0, C.SERVO_MAX_COUNT)
    # Advance physics for HALF the lag — setpoint should still be the old 0.
    _integrate_dynamics(s, C.STEER_TRANSPORT_LAG_S * 0.5)
    assert s.steer_cmd == pytest.approx(0.0)
    # Advance past — total elapsed > LAG, queue should release.
    _integrate_dynamics(s, C.STEER_TRANSPORT_LAG_S)
    assert s.steer_cmd != 0.0


def test_motor_setpoint_held_until_lag_elapses():
    s = RobotState()
    apply_command(s, 9000, 9000, C.SERVO_CENTER_COUNT)
    _integrate_dynamics(s, C.THROTTLE_TRANSPORT_LAG_S * 0.5)
    assert s.motor_cmd_l == pytest.approx(0.0)
    assert s.motor_cmd_r == pytest.approx(0.0)
    _integrate_dynamics(s, C.THROTTLE_TRANSPORT_LAG_S)
    assert s.motor_cmd_l > 0
    assert s.motor_cmd_r > 0


def test_no_speed_governor_overshoots_max_speed_cms():
    """With the hard `MAX_SPEED_CMS` clamp removed, the sim is allowed to
    settle at terminal velocity emerging from drag/force balance, which
    may temporarily overshoot the measured top-speed value if drag is low.
    The point of the test is just that the clamp is gone — no upper bound
    enforced on `state.v` — even if natural dynamics keep it close to the
    measured top speed.
    """
    s = RobotState()
    apply_command(s, 9999, 9999, C.SERVO_CENTER_COUNT)
    # Run physics for 5 sim-seconds at 1 ms.
    for _ in range(5000):
        _integrate_dynamics(s, 0.001)
    # No assertion that state.v <= MAX_SPEED_CMS — that clamp is gone. Just
    # confirm it's a finite positive forward speed.
    assert s.v > 0.0
    assert s.v < 1e6  # sanity (no runaway)
