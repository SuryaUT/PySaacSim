"""Bicycle-model dynamics integrated at 1 ms with first-order motor lag, a
servo slew limiter, and per-actuator transport delay (queues that hold
recently-commanded setpoints until LAG seconds have elapsed). Rear-wheel
differential drive gives asymmetric thrust; the front steer angle +
wheelbase set the geometric turn radius (slip allowed through lateral
friction).

Top speed is **not** clamped by `MAX_SPEED_CMS` in this module — terminal
velocity emerges naturally from the force/drag balance. The constant is
kept in `constants.py` as a measured reference value and a fit-bound
sanity check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constants import (
    G, INERTIA_KG_M2, LINEAR_DRAG, MASS_KG, MOTOR_LAG_TAU_S,
    MOTOR_MAX_FORCE_N, MU_KINETIC, MU_STATIC, PHYSICS_DT_S, REAR_TRACK_CM,
    ROLLING_RESIST, SERVO_RAD_PER_SEC, STEER_LIMIT_RAD,
    STEER_TRANSPORT_LAG_S, THROTTLE_TRANSPORT_LAG_S, WHEELBASE_CM,
    duty_count_to_pwm01, servo_count_to_steer_angle,
)
from .geometry import Segment, chassis_segments, seg_intersect


_WB_M = WHEELBASE_CM / 100.0
_TRACK_M = REAR_TRACK_CM / 100.0


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def copy(self) -> "Pose":
        return Pose(self.x, self.y, self.theta)


@dataclass
class RobotState:
    pose: Pose = field(default_factory=Pose)
    v: float = 0.0            # forward body speed, cm/s
    omega: float = 0.0        # yaw rate, rad/s
    steer_angle: float = 0.0  # actual wheel angle (rad), servo-limited
    steer_cmd: float = 0.0    # active wheel-angle setpoint (rad), post-lag
    motor_force_l: float = 0.0  # filtered rear-left force, N
    motor_force_r: float = 0.0
    motor_cmd_l: float = 0.0  # active rear-left force setpoint (N), post-lag
    motor_cmd_r: float = 0.0
    collided: bool = False
    # Sim time accumulator used by the transport-delay queues. Reset to 0
    # whenever apply_command is first called on a fresh state.
    sim_t_s: float = 0.0
    # Pending command queues: (release_time_s, value) tuples written by
    # apply_command, drained by _integrate_dynamics once sim_t_s >=
    # release_time_s. Cap kept short by drop-on-consume.
    steer_cmd_queue: list = field(default_factory=list)
    motor_cmd_queue: list = field(default_factory=list)


def _lag(current: float, target: float, tau: float, dt: float) -> float:
    alpha = 1 - math.exp(-dt / max(tau, 1e-6))
    return current + (target - current) * alpha


def apply_command(
    state: RobotState,
    duty_l: int,
    duty_r: int,
    servo: int,
    dir_l: int = 1,
    dir_r: int = 1,
) -> None:
    """Queue a firmware-style motor command. The command becomes active on
    the state setpoint after `STEER_TRANSPORT_LAG_S` / `THROTTLE_TRANSPORT_LAG_S`
    of sim time elapses, modelling the CAN + PWM-cycle + servo-deadband chain
    between the firmware Robot() loop emitting a command and the actuator
    physically responding."""
    steer_target = servo_count_to_steer_angle(servo)
    motor_l_target = dir_l * duty_count_to_pwm01(duty_l) * MOTOR_MAX_FORCE_N
    motor_r_target = dir_r * duty_count_to_pwm01(duty_r) * MOTOR_MAX_FORCE_N
    t_now = state.sim_t_s
    state.steer_cmd_queue.append((t_now + STEER_TRANSPORT_LAG_S, steer_target))
    state.motor_cmd_queue.append(
        (t_now + THROTTLE_TRANSPORT_LAG_S, motor_l_target, motor_r_target)
    )


def _drain_queue_setpoint(queue: list, t_now: float):
    """Pop entries whose release time has passed, return the latest released
    value (or None if nothing has been released yet). Drops earlier consumed
    entries except the most-recent-released one (so LAG can be rebound on the
    fly without losing the current setpoint)."""
    latest_released = None
    while queue and queue[0][0] <= t_now:
        latest_released = queue.pop(0)
    return latest_released


def _integrate_dynamics(state: RobotState, dt: float) -> None:
    # Advance sim clock so transport-lag queues know when commands release.
    state.sim_t_s += dt

    # Drain transport-delay queues: any commands whose release time has
    # passed become the active setpoint.
    released = _drain_queue_setpoint(state.steer_cmd_queue, state.sim_t_s)
    if released is not None:
        state.steer_cmd = released[1]
    released = _drain_queue_setpoint(state.motor_cmd_queue, state.sim_t_s)
    if released is not None:
        state.motor_cmd_l = released[1]
        state.motor_cmd_r = released[2]

    # Servo slew.
    steer_err = state.steer_cmd - state.steer_angle
    max_step = SERVO_RAD_PER_SEC * dt
    if abs(steer_err) <= max_step:
        state.steer_angle = state.steer_cmd
    else:
        state.steer_angle += math.copysign(max_step, steer_err)
    state.steer_angle = max(-STEER_LIMIT_RAD, min(STEER_LIMIT_RAD, state.steer_angle))

    # Motor first-order lag.
    state.motor_force_l = _lag(state.motor_force_l, state.motor_cmd_l, MOTOR_LAG_TAU_S, dt)
    state.motor_force_r = _lag(state.motor_force_r, state.motor_cmd_r, MOTOR_LAG_TAU_S, dt)

    v_ms = state.v / 100.0

    # Longitudinal dynamics: drive - rolling resist - drag, bounded by tire mu.
    # Terminal velocity emerges from f_drive == f_roll + f_drag — no hard
    # speed governor (the old `MAX_SPEED_CMS` clamp was removed because it
    # made `accel_y` flat at zero in sim once the cap was hit, breaking the
    # IMU-based dynamics fit).
    f_drive = state.motor_force_l + state.motor_force_r
    f_roll = math.copysign(ROLLING_RESIST * MASS_KG * G, v_ms) if v_ms != 0 else 0.0
    f_drag = LINEAR_DRAG * v_ms
    f_max = MU_STATIC * MASS_KG * G
    f_net = max(-f_max, min(f_max, f_drive)) - f_roll - f_drag
    v_new = v_ms + (f_net / MASS_KG) * dt

    # Dead-zone friction (stick when at rest and drive < breakout).
    if abs(v_new) < 0.005 and abs(f_drive) < ROLLING_RESIST * MASS_KG * G * 1.5:
        v_new = 0.0

    # Yaw: differential torque from rear wheels + bicycle geometric turn rate.
    yaw_torque = (state.motor_force_r - state.motor_force_l) * (_TRACK_M / 2.0)
    delta = state.steer_angle
    omega_geom = 0.0
    if abs(v_new) > 1e-4 and abs(delta) > 1e-4:
        omega_geom = v_new * math.tan(delta) / _WB_M

    # 15 ms smoothing keeps numerical stability without eating reaction latency
    # (longer smoothing here was making the wall-follower start correcting late).
    kin_alpha = 1 - math.exp(-dt / 0.015)
    omega_from_kin = state.omega + (omega_geom - state.omega) * kin_alpha
    omega_from_torque = omega_from_kin + (yaw_torque / INERTIA_KG_M2) * dt

    # Lateral slip: centripetal accel bounded by kinetic friction.
    max_lat_a = MU_KINETIC * G
    centripetal = v_new * omega_from_torque
    if abs(centripetal) > max_lat_a:
        state.omega = omega_from_torque * (max_lat_a / abs(centripetal))
    else:
        state.omega = omega_from_torque

    state.v = v_new * 100.0
    state.pose.theta += state.omega * dt
    state.pose.x += v_new * math.cos(state.pose.theta) * 100.0 * dt
    state.pose.y += v_new * math.sin(state.pose.theta) * 100.0 * dt


def _handle_collision(
    state: RobotState, prev: Pose, walls: list[Segment], length_cm: float, width_cm: float
) -> bool:
    pose_t = (state.pose.x, state.pose.y, state.pose.theta)
    segs = chassis_segments(pose_t, length_cm, width_cm)
    for wall in walls:
        for rs in segs:
            if seg_intersect(rs.a, rs.b, wall.a, wall.b) is not None:
                state.pose.x = prev.x
                state.pose.y = prev.y
                state.pose.theta = prev.theta
                state.v = 0.0
                state.omega = 0.0
                return True
    return False


def step_physics(
    state: RobotState,
    walls: list[Segment],
    length_cm: float,
    width_cm: float,
    dt: float = PHYSICS_DT_S,
) -> None:
    """Advance `state` by one physics tick. On collision, rollback pose and
    zero velocities, setting `state.collided = True`."""
    prev = state.pose.copy()
    _integrate_dynamics(state, dt)
    state.collided = _handle_collision(state, prev, walls, length_cm, width_cm)


def initial_robot(x: float, y: float, theta: float = 0.0) -> RobotState:
    return RobotState(pose=Pose(x, y, theta))
