"""Gymnasium environment wrapping the PySaacSim physics + sensors + calibration.

Observation (8-dim float32, all normalized):
  [lidar_center, lidar_left, lidar_right, ir_left, ir_right, v, omega, steer]

Action (2-dim float32):
  [servo_norm ∈ [-1, +1], throttle ∈ [0, 1]]

Can be driven either by:
  - RL agents (via step(action))
  - A C/Python controller (pass `controller=` to __init__; step() ignores
    its action argument and polls the controller each control period).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np

from ..control.base import AbstractController, MotorCommand
from ..sim.calibration import SensorCalibration
from ..sim.constants import (
    CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, MAX_SPEED_CMS, MOTOR_PWM_MAX_COUNT,
    PHYSICS_DT_S, SERVO_CENTER_COUNT, SERVO_MAX_COUNT, SERVO_MIN_COUNT,
    STEER_LIMIT_RAD,
)
from ..sim.physics import RobotState, apply_command, initial_robot, step_physics
from ..sim.sensors import sample_sensors
from ..sim.world import DEFAULT_SPAWN, build_default_world


CalibrationArg = Union[SensorCalibration, str, Path, None]

# Floor on the throttle action. Keeps the robot always moving so PPO gets
# gradient from the reward shape rather than stalling at 0 throttle.
THROTTLE_MIN = 0.3


class RobotEnv(gym.Env):
    """Single-robot gymnasium env. Use gymnasium.make_vec_env with
    SubprocVecEnv to run N of these in parallel for RL training."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        dt_ms: float = 1.0,
        ctrl_period_ms: float = 80.0,
        calibration: CalibrationArg = None,
        controller: Optional[AbstractController] = None,
        max_episode_steps: int = 2000,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._dt = dt_ms / 1000.0
        self._ctrl_steps = max(1, int(round(ctrl_period_ms / dt_ms)))
        self._max_episode_steps = max_episode_steps

        self.set_calibration(calibration)
        self._controller = controller

        self._world = build_default_world()
        self._walls = self._world["walls"]
        self._state: Optional[RobotState] = None
        self._t_ms: float = 0.0
        self._step_count: int = 0
        # Lap-progress tracker: cumulative unwrapped angle around the track
        # centroid. Reward is proportional to Δangle per step, so one full
        # lap (2π rad of progress) contributes a big positive return.
        self._track_center: Optional[tuple[float, float]] = None
        self._prev_angle: float = 0.0
        self._lap_progress: float = 0.0
        # Episode-sticky driving direction (+1 = CCW, -1 = CW, 0 = not yet
        # decided). The car's own first decisive motion picks it, so the
        # policy can settle into either lap direction per episode instead
        # of being forced CCW by the reward. Prevents oscillation hacks
        # because reversing flips lap reward negative.
        self._ep_direction: int = 0

        # Observation: 5 sensor distances (normalized 0..1) + v, omega, steer.
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, +1, +1, +1], dtype=np.float32),
            dtype=np.float32,
        )
        # Action: [servo_norm, throttle_L_norm, throttle_R_norm], all ∈ [-1, +1].
        # Two throttles because the real robot is differential-drive (firmware's
        # CAN_SetMotors takes throttle_l, throttle_r, steering independently —
        # it slows the inside wheel on sharp turns). Each throttle is remapped
        # to [THROTTLE_MIN, 1.0] inside step() so the Gaussian action dist
        # (which centers at 0) starts at ~0.5 duty and always keeps the robot
        # moving (avoids the 0-gradient failure mode).
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([+1.0, +1.0, +1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ---- Calibration API --------------------------------------------------

    def set_calibration(self, cal: CalibrationArg) -> None:
        """Hot-swap calibration. Pass a SensorCalibration, a YAML path, or
        None to reload the default YAML."""
        if cal is None:
            self.calibration = SensorCalibration.default()
        elif isinstance(cal, SensorCalibration):
            self.calibration = cal.copy()
        elif isinstance(cal, (str, Path)):
            self.calibration = SensorCalibration.from_yaml(cal)
        else:
            raise TypeError(f"unsupported calibration arg type: {type(cal)}")

    # ---- Gymnasium API ----------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        spawn = dict(DEFAULT_SPAWN)
        if options and "spawn" in options:
            spawn.update(options["spawn"])
        else:
            # Jitter spawn so the policy doesn't memorize one pose, and
            # coin-flip the heading so half of episodes face CW and half
            # CCW. In real life a car drives the way it's placed; we want
            # training to cover both placements so the policy generalizes.
            rng = self.np_random
            spawn["x"] += float(rng.uniform(-30.0, 30.0))
            spawn["y"] += float(rng.uniform(-6.0, 6.0))
            base_theta = spawn.get("theta", 0.0)
            if rng.random() < 0.5:
                base_theta += float(np.pi)        # flip 180° → opposite lap dir
            spawn["theta"] = base_theta + float(rng.uniform(-0.5, 0.5))
        self._state = initial_robot(spawn["x"], spawn["y"], spawn.get("theta", 0.0))
        self._t_ms = 0.0
        self._step_count = 0
        self._lap_progress = 0.0
        self._ep_direction = 0
        self._track_center = self._compute_track_center()
        self._prev_angle = self._angle_from_center(self._state.pose.x,
                                                   self._state.pose.y)
        if self._controller is not None:
            self._controller.reset()
        sensors = self._sample()
        return self._obs(sensors), {}

    def _compute_track_center(self) -> tuple[float, float]:
        """Centroid of the world bounds. Works for any closed track."""
        b = self._world.get("bounds", {})
        if not b:
            return (0.0, 0.0)
        return (0.5 * (b["min_x"] + b["max_x"]),
                0.5 * (b["min_y"] + b["max_y"]))

    def _angle_from_center(self, x: float, y: float) -> float:
        cx, cy = self._track_center or (0.0, 0.0)
        return float(np.arctan2(y - cy, x - cx))

    def step(self, action):
        assert self._state is not None, "call reset() before step()"
        if self._controller is not None:
            sensors = self._sample()
            cmd = self._controller.tick(sensors, self._t_ms)
            apply_command(self._state, cmd.duty_l, cmd.duty_r, cmd.servo,
                          cmd.dir_l, cmd.dir_r)
        else:
            servo_norm = float(np.clip(action[0], -1.0, 1.0))
            tL_norm   = float(np.clip(action[1], -1.0, 1.0))
            tR_norm   = float(np.clip(action[2], -1.0, 1.0))
            # Remap [-1, +1] → [THROTTLE_MIN, 1.0] per wheel.
            throttle_l = THROTTLE_MIN + (tL_norm + 1.0) * 0.5 * (1.0 - THROTTLE_MIN)
            throttle_r = THROTTLE_MIN + (tR_norm + 1.0) * 0.5 * (1.0 - THROTTLE_MIN)
            if servo_norm >= 0:
                servo = int(SERVO_CENTER_COUNT + servo_norm * (SERVO_MAX_COUNT - SERVO_CENTER_COUNT))
            else:
                servo = int(SERVO_CENTER_COUNT + servo_norm * (SERVO_CENTER_COUNT - SERVO_MIN_COUNT))
            duty_l = int(throttle_l * MOTOR_PWM_MAX_COUNT)
            duty_r = int(throttle_r * MOTOR_PWM_MAX_COUNT)
            apply_command(self._state, duty_l, duty_r, servo)

        for _ in range(self._ctrl_steps):
            step_physics(self._state, self._walls,
                         CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, self._dt)
            self._t_ms += self._dt * 1000.0
            if self._state.collided:
                break

        sensors = self._sample()
        obs = self._obs(sensors)
        reward = self._reward(sensors)
        terminated = self._state.collided
        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps
        info = {
            "t_ms": self._t_ms,
            "pose": (self._state.pose.x, self._state.pose.y, self._state.pose.theta),
            "v": self._state.v,
            "collided": self._state.collided,
            "sensors": sensors,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        # Lightweight inspection hook; a pygame/matplotlib renderer can consume
        # the state externally. Left as a no-op so headless training is cheap.
        return None

    def close(self):
        if self._controller is not None:
            self._controller.close()

    # ---- Internals --------------------------------------------------------

    def _sample(self) -> dict:
        return sample_sensors(self._walls, self._state.pose, self.calibration)

    def _obs(self, sensors: dict) -> np.ndarray:
        lidar_max = self.calibration.lidar.max_cm
        ir_max = self.calibration.ir.max_cm
        lc = sensors["lidar"]["center"]["distance_cm"] / lidar_max
        ll = sensors["lidar"]["left"]["distance_cm"] / lidar_max
        lr = sensors["lidar"]["right"]["distance_cm"] / lidar_max
        il = sensors["ir"]["left"]["distance_cm"] / ir_max
        ir = sensors["ir"]["right"]["distance_cm"] / ir_max
        v = self._state.v / MAX_SPEED_CMS
        omega = np.clip(self._state.omega / 5.0, -1.0, 1.0)
        steer = self._state.steer_angle / STEER_LIMIT_RAD
        return np.array([lc, ll, lr, il, ir, v, omega, steer], dtype=np.float32)

    def _reward(self, sensors: dict) -> float:
        """Controller-agnostic racing reward — direction-agnostic. Does NOT
        target any specific wall-distance setpoint so it stays swappable
        with any controller (firmware PD, learned residual, etc.).

        Importance ordering (crash >> laps > speed > proximity > smoothness):
          crash       : -100  (one-shot terminal)
          laps        : +20 per full lap (linear on Δangle, signed vs chosen dir)
          speed       : up to +0.2/tick, only while progressing in chosen dir
          proximity   : up to -0.05/tick inside 10 cm, floor at 3 cm
          smoothness  : up to -0.01/tick for hard steering
        Over a 1500-tick (~120 s) episode this caps near-best total at
        ~+400 (several laps at full speed) and worst at ~-100 (crash)."""
        if self._state.collided:
            return -100.0

        # ---- Lap progress (signed Δangle around track center) ------------
        cur_angle = self._angle_from_center(self._state.pose.x,
                                            self._state.pose.y)
        d_angle = cur_angle - self._prev_angle
        if d_angle >  np.pi: d_angle -= 2 * np.pi
        if d_angle < -np.pi: d_angle += 2 * np.pi
        self._prev_angle = cur_angle
        self._lap_progress += d_angle

        # Episode direction: the car's first decisive motion picks CCW (+1)
        # or CW (-1). Once set, lap reward is signed against that choice,
        # so reversing *loses* accumulated reward. Works in either direction
        # without biasing the policy — cars in ghost mode (or just separate
        # envs) will split into ~half CCW / half CW naturally.
        if self._ep_direction == 0 and abs(self._lap_progress) > 0.05:
            self._ep_direction = 1 if self._lap_progress > 0 else -1
        direction = self._ep_direction or 1  # default CCW until decided

        # +20 per full lap (2π rad) in the chosen direction.
        lap_r = float(d_angle) * direction * (20.0 / (2.0 * np.pi))

        # ---- Speed bonus (gated on progressing in chosen direction) ------
        # No "sprint backwards" loophole: oscillation won't earn speed_r
        # because d_angle * direction flips sign each reversal.
        speed_frac = max(0.0, self._state.v) / MAX_SPEED_CMS
        progressing = 1.0 if d_angle * direction > 0 else 0.0
        speed_r = 0.2 * speed_frac * progressing

        # ---- Proximity safety (bounded well below crash) -----------------
        min_dist = min(
            sensors["lidar"]["center"]["distance_cm"],
            sensors["lidar"]["left"]["distance_cm"],
            sensors["lidar"]["right"]["distance_cm"],
            sensors["ir"]["left"]["distance_cm"]
                if sensors["ir"]["left"]["valid"] else 99.0,
            sensors["ir"]["right"]["distance_cm"]
                if sensors["ir"]["right"]["valid"] else 99.0,
        )
        if min_dist < 10.0:
            proximity_pen = (10.0 - max(3.0, min_dist)) * 0.05  # max ~0.35/tick
        else:
            proximity_pen = 0.0

        # ---- Steering smoothness — gentle, just to discourage chatter ----
        steer_pen = 0.01 * (self._state.steer_angle / STEER_LIMIT_RAD) ** 2

        return lap_r + speed_r - proximity_pen - steer_pen
