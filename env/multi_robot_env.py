"""Same-scene multi-agent training env.

N robots share one track. Each robot's sensors see other robots' chassis
edges as obstacles and physics treats them as collidable walls. All N
agents feed experience into a single PPO model (Isaac-Sim-style).

Exposes the stable_baselines3 VecEnv interface so PPO can drive it directly:
    reset()  -> obs shape (N, obs_dim)
    step(actions)  -> obs (N, obs_dim), rewards (N,), dones (N,), infos [N]

Auto-reset policy: when any agent terminates (collision) or truncates
(max steps), it is reset to its spawn pose at the start of the next step.
The terminal transition's final obs is stashed in info["terminal_observation"]
per SB3 convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from ..sim.calibration import SensorCalibration
from ..sim.constants import (
    CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, MAX_SPEED_CMS, MOTOR_PWM_MAX_COUNT,
    PHYSICS_DT_S, SERVO_CENTER_COUNT, SERVO_MAX_COUNT, SERVO_MIN_COUNT,
    STEER_LIMIT_RAD,
)
from ..sim.geometry import Segment, chassis_segments
from ..sim.physics import RobotState, apply_command, initial_robot, step_physics
from ..sim.sensors import sample_sensors
from ..sim.world import DEFAULT_SPAWN, build_default_world
from .robot_env import THROTTLE_MIN


CalibrationArg = Union[SensorCalibration, str, Path, None]


class MultiRobotVecEnv(VecEnv):
    """N cars sharing one track; each car is an SB3 vector slot."""

    def __init__(
        self,
        n_agents: int = 4,
        walls: Optional[list[Segment]] = None,
        calibration: CalibrationArg = None,
        dt_ms: float = 1.0,
        ctrl_period_ms: float = 80.0,
        max_episode_steps: int = 1500,
        spawns: Optional[Sequence[dict]] = None,
    ) -> None:
        self._n = max(1, int(n_agents))
        self._dt = dt_ms / 1000.0
        self._ctrl_steps = max(1, int(round(ctrl_period_ms / dt_ms)))
        self._max_episode_steps = max_episode_steps

        self._world = build_default_world()
        if walls is not None:
            self._world["walls"] = list(walls)
        self._walls = self._world["walls"]

        if isinstance(calibration, SensorCalibration):
            self._cal = calibration.copy()
        elif isinstance(calibration, (str, Path)):
            self._cal = SensorCalibration.from_yaml(calibration)
        else:
            self._cal = SensorCalibration.default()

        # Evenly stagger spawns around the oval so cars don't start on top
        # of each other. User can override via `spawns`.
        if spawns is None:
            spawns = self._default_spawns(self._n)
        self._spawns = list(spawns)

        self._rng = np.random.default_rng()
        self._states: list[RobotState] = [
            initial_robot(s["x"], s["y"], s.get("theta", 0.0)) for s in self._spawns
        ]
        self._step_counts = [0] * self._n
        self._track_center = self._compute_track_center()
        self._prev_angles = [self._angle_from_center(s.pose.x, s.pose.y)
                             for s in self._states]
        # Per-agent lap progress + sticky episode direction (see RobotEnv).
        self._lap_progress = [0.0] * self._n
        self._ep_direction = [0] * self._n

        obs_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, +1, +1, +1], dtype=np.float32),
            dtype=np.float32,
        )
        act_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([+1.0, +1.0, +1.0], dtype=np.float32),
            dtype=np.float32,
        )
        super().__init__(self._n, obs_space, act_space)

        self._pending_actions: Optional[np.ndarray] = None

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _default_spawns(n: int) -> list[dict]:
        # Strung out along the bottom straight near the default spawn,
        # spaced 40 cm apart so they fit on a 300+ cm straight.
        base = dict(DEFAULT_SPAWN)
        return [{"x": base["x"] + i * 40.0, "y": base["y"], "theta": 0.0}
                for i in range(n)]

    def _compute_track_center(self) -> tuple[float, float]:
        b = self._world.get("bounds", {})
        if not b:
            return (0.0, 0.0)
        return (0.5 * (b["min_x"] + b["max_x"]),
                0.5 * (b["min_y"] + b["max_y"]))

    def _angle_from_center(self, x: float, y: float) -> float:
        cx, cy = self._track_center
        return float(np.arctan2(y - cy, x - cx))

    def _other_segments(self, me: int) -> list[Segment]:
        segs: list[Segment] = []
        for j in range(self._n):
            if j == me:
                continue
            s = self._states[j]
            segs.extend(chassis_segments(
                (s.pose.x, s.pose.y, s.pose.theta),
                CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM))
        return segs

    def _obs_for(self, i: int, sensors: dict) -> np.ndarray:
        s = self._states[i]
        cal = self._cal
        return np.array([
            sensors["lidar"]["center"]["distance_cm"] / cal.lidar.max_cm,
            sensors["lidar"]["left"]["distance_cm"]   / cal.lidar.max_cm,
            sensors["lidar"]["right"]["distance_cm"]  / cal.lidar.max_cm,
            sensors["ir"]["left"]["distance_cm"]  / cal.ir.max_cm,
            sensors["ir"]["right"]["distance_cm"] / cal.ir.max_cm,
            s.v / MAX_SPEED_CMS,
            np.clip(s.omega / 5.0, -1.0, 1.0),
            s.steer_angle / STEER_LIMIT_RAD,
        ], dtype=np.float32)

    def _reward_for(self, i: int, sensors: dict) -> float:
        """Controller-agnostic, direction-agnostic racing reward
        (mirrors RobotEnv._reward; see that file for the rationale)."""
        s = self._states[i]
        if s.collided:
            return -100.0
        cur = self._angle_from_center(s.pose.x, s.pose.y)
        d = cur - self._prev_angles[i]
        if d >  np.pi: d -= 2 * np.pi
        if d < -np.pi: d += 2 * np.pi
        self._prev_angles[i] = cur
        self._lap_progress[i] += d

        if self._ep_direction[i] == 0 and abs(self._lap_progress[i]) > 0.05:
            self._ep_direction[i] = 1 if self._lap_progress[i] > 0 else -1
        direction = self._ep_direction[i] or 1

        lap_r = float(d) * direction * (20.0 / (2.0 * np.pi))

        speed_frac = max(0.0, s.v) / MAX_SPEED_CMS
        progressing = 1.0 if d * direction > 0 else 0.0
        speed_r = 0.2 * speed_frac * progressing

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
            proximity_pen = (10.0 - max(3.0, min_dist)) * 0.05
        else:
            proximity_pen = 0.0

        steer_pen = 0.01 * (s.steer_angle / STEER_LIMIT_RAD) ** 2
        return lap_r + speed_r - proximity_pen - steer_pen

    def _apply_action(self, i: int, action: np.ndarray) -> None:
        servo_norm = float(np.clip(action[0], -1.0, 1.0))
        tL_norm   = float(np.clip(action[1], -1.0, 1.0))
        tR_norm   = float(np.clip(action[2], -1.0, 1.0))
        throttle_l = THROTTLE_MIN + (tL_norm + 1.0) * 0.5 * (1.0 - THROTTLE_MIN)
        throttle_r = THROTTLE_MIN + (tR_norm + 1.0) * 0.5 * (1.0 - THROTTLE_MIN)
        if servo_norm >= 0:
            servo = int(SERVO_CENTER_COUNT + servo_norm * (SERVO_MAX_COUNT - SERVO_CENTER_COUNT))
        else:
            servo = int(SERVO_CENTER_COUNT + servo_norm * (SERVO_CENTER_COUNT - SERVO_MIN_COUNT))
        duty_l = int(throttle_l * MOTOR_PWM_MAX_COUNT)
        duty_r = int(throttle_r * MOTOR_PWM_MAX_COUNT)
        apply_command(self._states[i], duty_l, duty_r, servo)

    def _reset_one(self, i: int) -> np.ndarray:
        spawn = self._spawns[i]
        # Jitter around the staggered spawn + 50/50 heading flip so the agent
        # experiences both CW and CCW starts. See RobotEnv.reset() for rationale.
        jx = spawn["x"] + float(self._rng.uniform(-15.0, 15.0))
        jy = spawn["y"] + float(self._rng.uniform(-6.0, 6.0))
        base_t = spawn.get("theta", 0.0)
        if self._rng.random() < 0.5:
            base_t += float(np.pi)
        jt = base_t + float(self._rng.uniform(-0.5, 0.5))
        self._states[i] = initial_robot(jx, jy, jt)
        self._step_counts[i] = 0
        self._prev_angles[i] = self._angle_from_center(jx, jy)
        self._lap_progress[i] = 0.0
        self._ep_direction[i] = 0
        sensors = sample_sensors(self._walls + self._other_segments(i),
                                 self._states[i].pose, self._cal)
        return self._obs_for(i, sensors)

    # ---- VecEnv interface -------------------------------------------------

    def reset(self) -> np.ndarray:  # type: ignore[override]
        return np.stack([self._reset_one(i) for i in range(self._n)])

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        assert self._pending_actions is not None
        actions = self._pending_actions
        self._pending_actions = None

        # Apply all actions first so cars move "simultaneously" in a tick.
        for i in range(self._n):
            self._apply_action(i, actions[i])

        # Step physics for all cars. Walls for car i = static + other cars'
        # chassis at their CURRENT poses. We use pre-step positions for the
        # obstacle set; the usual 80 ms control period means cars don't
        # move more than a few cm per tick, so this is fine.
        for i in range(self._n):
            other_segs = self._other_segments(i)
            eff_walls = self._walls + other_segs
            for _ in range(self._ctrl_steps):
                step_physics(self._states[i], eff_walls,
                             CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, self._dt)
                if self._states[i].collided:
                    break
            self._step_counts[i] += 1

        # Build obs / reward / done per agent.
        obs = np.zeros((self._n, 8), dtype=np.float32)
        rewards = np.zeros(self._n, dtype=np.float32)
        dones = np.zeros(self._n, dtype=bool)
        infos: list[dict] = [{} for _ in range(self._n)]

        for i in range(self._n):
            sensors = sample_sensors(self._walls + self._other_segments(i),
                                     self._states[i].pose, self._cal)
            obs[i] = self._obs_for(i, sensors)
            rewards[i] = self._reward_for(i, sensors)
            terminated = self._states[i].collided
            truncated = self._step_counts[i] >= self._max_episode_steps
            dones[i] = terminated or truncated
            infos[i]["TimeLimit.truncated"] = truncated and not terminated
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]
                obs[i] = self._reset_one(i)

        return obs, rewards, dones, infos

    def close(self) -> None:
        pass

    # SB3 VecEnv plumbing. Return empty/sensible defaults; none of these are
    # used by PPO's default training loop.
    def get_attr(self, attr_name, indices=None):
        indices = self._get_indices(indices)
        return [getattr(self, attr_name, None) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        for _ in self._get_indices(indices):
            setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed] * self._n

    def _get_indices(self, indices):
        if indices is None:
            return range(self._n)
        if isinstance(indices, int):
            return [indices]
        return list(indices)
