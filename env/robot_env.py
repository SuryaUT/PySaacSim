"""Gymnasium env for the residual-on-PD architecture (see Model_Interface.md).

Observation (13-dim float32, all in [0, 1] — Q16/65536 semantics):
  matches input_t in Model.h:
    [ir_right, ir_left, tf_left, tf_middle, tf_right,
     throttle_left_prev, throttle_right_prev, steering_prev,
     angle_left, angle_right,
     yaw_rate, accel_lat, accel_long]

Action (3-dim float32 ∈ [-1, +1]):
  matches output_t in Model.h:
    [throttle_left_delta, throttle_right_delta, steering_delta]
  action 0 ⇒ pure PD baseline; ±1 ⇒ ±CAP_DELTA_*.

Step pipeline (matches spec section 7):
  1. Sample sensors → mm
  2. IR correction (firmware lines 448-449)
  3. Compute geometry → run PD baseline
  4. Build model input vector (uses prev-applied actions)
  5. RL action gives the delta — env.apply_residual + clamp
  6. Convert applied (throttle_l, throttle_r, steering_deg) → physics command
  7. Integrate physics for one control period
  8. Read IMU; build next observation
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np

from ..control.pd_baseline import (
    PDBaseline, clamp_angle_for_model, compute_geometry, ir_correction,
)
from ..sim.calibration import SensorCalibration
from ..sim.constants import (
    CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, MAX_SPEED_CMS, MOTOR_PWM_MAX_COUNT,
    PHYSICS_DT_S, SERVO_CENTER_COUNT, SERVO_MAX_COUNT, SERVO_MIN_COUNT,
    STEER_LIMIT_RAD,
)
from ..sim.imu import IMUSimulator
from ..sim.model import (
    CAP_ACCEL, CAP_IR, CAP_STEERING, CAP_THROTTLE, CAP_TFLUNA, CAP_YAW,
    Inp, NUM_INPUTS, NUM_OUTPUTS, Q16_HALF, action_to_delta, apply_residual,
    encode_angle, normalize, normalize_signed,
)
from ..sim.physics import RobotState, apply_command, initial_robot, step_physics
from ..sim.sensors import sample_sensors
from ..sim.world import DEFAULT_SPAWN, build_default_world


CalibrationArg = Union[SensorCalibration, str, Path, None]

# L2 penalty on the residual delta — discourages the model from fighting PD
# when PD is already good. Tuned conservatively per spec section 9.
ACTION_L2_PENALTY = 0.05

# Proximity-shaping knobs used by _reward (and MultiRobotVecEnv).
# LiDARs are always penalized for closeness — they detect forward / ±45°
# crash paths. Side IRs get a sweet-spot bonus on the *inside* of the chosen
# lap direction, so the policy learns to hug the inside wall for shorter laps.
LIDAR_PROX_THRESHOLD_CM = 10.0
LIDAR_PROX_FLOOR_CM     = 3.0
LIDAR_PROX_SCALE        = 0.05
IR_SAFETY_FLOOR_CM      = 5.0   # "couple of inches" — below this both IRs hurt
IR_SAFETY_FLOOR_MIN_CM  = 3.0   # clamp so penalty doesn't grow unbounded
IR_SAFETY_SCALE         = 0.05
IR_INSIDE_BAND_MAX_CM   = 12.0  # inside IR gets bonus when reading ≤ this
IR_INSIDE_BONUS         = 0.05  # per step while inside IR sits in sweet spot


def lidar_proximity_penalty(sensors: dict) -> float:
    """Always-on crash-imminent penalty across the 3 LiDARs. Returns ≥ 0."""
    min_d = min(
        sensors["lidar"]["center"]["distance_cm"],
        sensors["lidar"]["left"]["distance_cm"],
        sensors["lidar"]["right"]["distance_cm"],
    )
    if min_d >= LIDAR_PROX_THRESHOLD_CM:
        return 0.0
    return (LIDAR_PROX_THRESHOLD_CM - max(LIDAR_PROX_FLOOR_CM, min_d)) * LIDAR_PROX_SCALE


def ir_reward_component(sensors: dict, direction: int) -> float:
    """Signed reward contribution from the two side IRs.

    Returns positive when the inside IR is in the racing-line sweet spot,
    negative when either IR drops below the safety floor, zero otherwise.
    `direction`: +1 CCW (left inside), -1 CW (right inside), 0 unknown
    (no bonus yet, only safety floor is active).
    """
    ir_l = sensors["ir"]["left"]
    ir_r = sensors["ir"]["right"]
    total = 0.0

    # Safety floor — active regardless of direction. Don't let the policy
    # shave the wall off even on the "rewarded" side.
    if ir_l["valid"] and ir_l["distance_cm"] < IR_SAFETY_FLOOR_CM:
        d = max(IR_SAFETY_FLOOR_MIN_CM, ir_l["distance_cm"])
        total -= (IR_SAFETY_FLOOR_CM - d) * IR_SAFETY_SCALE
    if ir_r["valid"] and ir_r["distance_cm"] < IR_SAFETY_FLOOR_CM:
        d = max(IR_SAFETY_FLOOR_MIN_CM, ir_r["distance_cm"])
        total -= (IR_SAFETY_FLOOR_CM - d) * IR_SAFETY_SCALE

    # Inside-wall bonus — only once direction is locked.
    if direction == 0:
        return total
    inside = ir_l if direction == +1 else ir_r
    if (inside["valid"]
            and IR_SAFETY_FLOOR_CM <= inside["distance_cm"] <= IR_INSIDE_BAND_MAX_CM):
        total += IR_INSIDE_BONUS
    return total


def steer_deg_to_servo_count(steer_deg: float) -> int:
    """Inverse of `servo_count_to_steer_angle`. +deg = right turn ⇒ count > center."""
    t = max(-1.0, min(1.0, steer_deg / CAP_STEERING))
    if t >= 0:
        return int(round(SERVO_CENTER_COUNT + t * (SERVO_MAX_COUNT - SERVO_CENTER_COUNT)))
    return int(round(SERVO_CENTER_COUNT + t * (SERVO_CENTER_COUNT - SERVO_MIN_COUNT)))


def sensors_to_mm(sensors: dict) -> tuple[int, int, int, int, int]:
    """Pull out the 5 distance readings in mm (firmware units)."""
    d_ir   = int(round(sensors["ir"]["right"]["distance_cm"]    * 10))
    ld_ir  = int(round(sensors["ir"]["left"]["distance_cm"]     * 10))
    d2     = int(round(sensors["lidar"]["right"]["distance_cm"] * 10))
    ld2    = int(round(sensors["lidar"]["left"]["distance_cm"]  * 10))
    front  = int(round(sensors["lidar"]["center"]["distance_cm"] * 10))
    return d_ir, ld_ir, d2, ld2, front


def build_observation(
    sensors: dict,
    raw_gyro_z: int, raw_accel_x: int, raw_accel_y: int,
    prev_thr_l: float, prev_thr_r: float, prev_steer_deg: float,
) -> np.ndarray:
    """Build the 13-element float32 observation in [0, 1].

    Mirrors the firmware's `Model_Inputs[...]` packing in `Robot()` lines
    491-512. The IR correction has already been applied to (d_ir, ld_ir).
    """
    d_ir, ld_ir, d2, ld2, front = sensors_to_mm(sensors)
    d_ir, ld_ir = ir_correction(d_ir, ld_ir, d2, ld2)
    geom = compute_geometry(d_ir, ld_ir, d2, ld2, front)

    obs = np.zeros(NUM_INPUTS, dtype=np.float32)
    obs[Inp.ir_right]            = normalize(d_ir,  CAP_IR)
    obs[Inp.ir_left]             = normalize(ld_ir, CAP_IR)
    obs[Inp.tf_left]             = normalize(ld2,   CAP_TFLUNA)
    obs[Inp.tf_middle]           = normalize(front, CAP_TFLUNA)
    obs[Inp.tf_right]            = normalize(d2,    CAP_TFLUNA)
    obs[Inp.throttle_left_prev]  = normalize(prev_thr_l, CAP_THROTTLE)
    obs[Inp.throttle_right_prev] = normalize(prev_thr_r, CAP_THROTTLE)
    obs[Inp.steering_prev]       = normalize_signed(prev_steer_deg, CAP_STEERING)
    obs[Inp.angle_left]          = encode_angle(clamp_angle_for_model(geom.angle_left))
    obs[Inp.angle_right]         = encode_angle(clamp_angle_for_model(geom.angle_right))
    # Spec: yaw_rate input uses -GyroZ so + = right turn (matches steering).
    obs[Inp.yaw_rate]            = normalize_signed(-raw_gyro_z, CAP_YAW)
    obs[Inp.accel_lat]           = normalize_signed(raw_accel_x, CAP_ACCEL)
    obs[Inp.accel_long]          = normalize_signed(raw_accel_y, CAP_ACCEL)
    return obs


class RobotEnv(gym.Env):
    """Single-robot env. Observation is the 13-input model vector; action is
    the 3-output residual delta. Use SubprocVecEnv to run N in parallel."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        dt_ms: float = 1.0,
        ctrl_period_ms: float = 80.0,   # ~12.5 Hz, matches firmware
        calibration: CalibrationArg = None,
        max_episode_steps: int = 1500,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._dt = dt_ms / 1000.0
        self._ctrl_steps = max(1, int(round(ctrl_period_ms / dt_ms)))
        self._max_episode_steps = max_episode_steps

        self.set_calibration(calibration)

        self._world = build_default_world()
        self._walls = self._world["walls"]
        self._state: Optional[RobotState] = None
        self._t_ms: float = 0.0
        self._step_count: int = 0

        self._pd = PDBaseline()
        self._imu = IMUSimulator()

        # Per-spec initial conditions (sec 7): throttles 0, steering centered.
        self._prev_thr_l: float = 0.0
        self._prev_thr_r: float = 0.0
        self._prev_steer_deg: float = 0.0

        # Lap-progress reward state — direction-agnostic, see _reward.
        self._track_center: Optional[tuple[float, float]] = None
        self._prev_angle: float = 0.0
        self._lap_progress: float = 0.0
        self._ep_direction: int = 0

        # Observation: 13 features in [0, 1]. Action: 3 deltas in [-1, +1].
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(NUM_INPUTS,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_OUTPUTS,), dtype=np.float32,
        )

    # ---- Calibration API --------------------------------------------------

    def set_calibration(self, cal: CalibrationArg) -> None:
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
            rng = self.np_random
            spawn["x"] += float(rng.uniform(-30.0, 30.0))
            spawn["y"] += float(rng.uniform(-6.0, 6.0))
            base_theta = spawn.get("theta", 0.0)
            if rng.random() < 0.5:
                base_theta += float(np.pi)
            spawn["theta"] = base_theta + float(rng.uniform(-0.5, 0.5))

        self._state = initial_robot(spawn["x"], spawn["y"], spawn.get("theta", 0.0))
        self._t_ms = 0.0
        self._step_count = 0

        self._pd.reset()
        self._imu.reset(v_cms=0.0, t_s=0.0)

        self._prev_thr_l = 0.0
        self._prev_thr_r = 0.0
        self._prev_steer_deg = 0.0

        self._lap_progress = 0.0
        self._ep_direction = 0
        self._track_center = self._compute_track_center()
        self._prev_angle = self._angle_from_center(self._state.pose.x, self._state.pose.y)

        sensors = self._sample()
        # IMU at rest at t=0 reads zeros; matches firmware boot.
        obs = build_observation(
            sensors, 0, 0, 0,
            self._prev_thr_l, self._prev_thr_r, self._prev_steer_deg,
        )
        return obs, {}

    def step(self, action):
        assert self._state is not None, "call reset() before step()"

        sensors = self._sample()
        d_ir, ld_ir, d2, ld2, front = sensors_to_mm(sensors)
        d_ir, ld_ir = ir_correction(d_ir, ld_ir, d2, ld2)
        geom = compute_geometry(d_ir, ld_ir, d2, ld2, front)
        pd = self._pd.tick(geom)

        delta_thr_l, delta_thr_r, delta_steer = action_to_delta(np.asarray(action))
        thr_l, thr_r, steer_deg = apply_residual(
            pd.throttle_l, pd.throttle_r, pd.steering,
            delta_thr_l, delta_thr_r, delta_steer,
        )

        # Convert to physics command. Throttle is already in PWM units that
        # map directly to the firmware's CAN_SetMotors duty (CAP_THROTTLE=9000
        # < MOTOR_PWM_MAX_COUNT=10000).
        servo = steer_deg_to_servo_count(steer_deg)
        apply_command(self._state, int(thr_l), int(thr_r), servo)

        for _ in range(self._ctrl_steps):
            step_physics(self._state, self._walls,
                         CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, self._dt)
            self._t_ms += self._dt * 1000.0
            if self._state.collided:
                break

        self._prev_thr_l = thr_l
        self._prev_thr_r = thr_r
        self._prev_steer_deg = steer_deg

        raw_gyro_z, raw_accel_x, raw_accel_y = self._imu.read(self._state, self._t_ms / 1000.0)
        next_sensors = self._sample()
        obs = build_observation(
            next_sensors, raw_gyro_z, raw_accel_x, raw_accel_y,
            self._prev_thr_l, self._prev_thr_r, self._prev_steer_deg,
        )

        reward = self._reward(next_sensors, action)
        terminated = self._state.collided
        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps
        info = {
            "t_ms": self._t_ms,
            "pose": (self._state.pose.x, self._state.pose.y, self._state.pose.theta),
            "v": self._state.v,
            "collided": self._state.collided,
            "sensors": next_sensors,
            "pd": {"throttle_l": pd.throttle_l, "throttle_r": pd.throttle_r,
                   "steering": pd.steering},
            "applied": {"throttle_l": thr_l, "throttle_r": thr_r,
                        "steering": steer_deg},
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass

    # ---- Internals --------------------------------------------------------

    def _sample(self) -> dict:
        return sample_sensors(self._walls, self._state.pose, self.calibration)

    def _compute_track_center(self) -> tuple[float, float]:
        b = self._world.get("bounds", {})
        if not b:
            return (0.0, 0.0)
        return (0.5 * (b["min_x"] + b["max_x"]),
                0.5 * (b["min_y"] + b["max_y"]))

    def _angle_from_center(self, x: float, y: float) -> float:
        cx, cy = self._track_center or (0.0, 0.0)
        return float(np.arctan2(y - cy, x - cx))

    def _reward(self, sensors: dict, action) -> float:
        """Direction-aware lap reward + inside-wall shaping + L2 residual penalty.

        Lap progress: signed Δangle around the track centroid; episode picks
        its own direction on first decisive motion. Speed bonus only while
        progressing in the chosen direction. Crash terminates with -100.
        LiDARs always incur a proximity penalty (crash paths). Inside IR
        (left for CCW, right for CW) gets a sweet-spot bonus once direction
        is locked — encourages tight racing lines. Safety floor on both IRs.
        Action L2 penalty discourages large residuals — matches spec section 9.
        """
        if self._state.collided:
            return -100.0

        cur_angle = self._angle_from_center(self._state.pose.x, self._state.pose.y)
        d_angle = cur_angle - self._prev_angle
        if d_angle >  np.pi: d_angle -= 2 * np.pi
        if d_angle < -np.pi: d_angle += 2 * np.pi
        self._prev_angle = cur_angle
        self._lap_progress += d_angle

        if self._ep_direction == 0 and abs(self._lap_progress) > 0.05:
            self._ep_direction = 1 if self._lap_progress > 0 else -1

        # Zero lap and speed reward until direction is locked — keeps the
        # first decisive-motion window fully symmetric between CCW and CW.
        if self._ep_direction == 0:
            lap_r = 0.0
            speed_r = 0.0
        else:
            direction = self._ep_direction
            lap_r = float(d_angle) * direction * (20.0 / (2.0 * np.pi))
            speed_frac = max(0.0, self._state.v) / MAX_SPEED_CMS
            progressing = 1.0 if d_angle * direction > 0 else 0.0
            speed_r = 0.2 * speed_frac * progressing

        lidar_pen = lidar_proximity_penalty(sensors)
        ir_r_signed = ir_reward_component(sensors, self._ep_direction)

        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action_pen = ACTION_L2_PENALTY * float(np.dot(a, a)) / NUM_OUTPUTS

        return lap_r + speed_r - lidar_pen + ir_r_signed - action_pen
