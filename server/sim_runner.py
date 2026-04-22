"""Headless simulation loop — powers the optional browser dashboard
(plan §7.11) and, later, manual-drive debugging from the iOS app.

Design notes:
  * Zero PyQt dependency. Reuses ``sim.physics`` and ``sim.sensors`` directly.
  * One singleton per server. The active track and any loaded policy are
    swappable at runtime.
  * Broadcasts pose + sensor snapshots at ~50 Hz only when at least one
    client is subscribed (plan §14.15)."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
from typing import Any, Optional

from ..control.base import MotorCommand
from ..sim.calibration import SensorCalibration
from ..sim.constants import CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, PHYSICS_DT_S
from ..sim.geometry import Segment, Vec2
from ..sim.physics import RobotState, apply_command, initial_robot, step_physics
from ..sim.sensors import sample_sensors
from ..sim.world import DEFAULT_SPAWN, build_default_world


logger = logging.getLogger(__name__)


def _segments_from_dict(walls: Any) -> list[Segment]:
    """Accept either a list of ``Segment`` (already loaded from sim.world) or
    JSON-decoded ``[[[ax,ay],[bx,by]], ...]``."""
    out: list[Segment] = []
    for w in walls:
        if isinstance(w, Segment):
            out.append(w)
            continue
        a, b = w[0], w[1]
        out.append(Segment(Vec2(float(a[0]), float(a[1])),
                           Vec2(float(b[0]), float(b[1]))))
    return out


class SimRunner:
    """Async sim loop. ``run()`` is the body; call ``set_command`` /
    ``load_track`` from request handlers to mutate state safely."""

    def __init__(self, ws_broadcaster, *, calibration: Optional[SensorCalibration] = None) -> None:
        self._broadcaster = ws_broadcaster
        self._cal = calibration or SensorCalibration.default()

        world = build_default_world()
        self._walls: list[Segment] = list(world["walls"])
        self._spawn: dict[str, float] = dict(DEFAULT_SPAWN)
        self._state: RobotState = initial_robot(self._spawn["x"], self._spawn["y"],
                                                self._spawn["theta"])
        self._cmd: MotorCommand = MotorCommand()
        self._policy = None          # SB3 PPO model, or None for manual drive
        self._c_controller = None    # CBridge custom controller
        self._t_ms = 0.0
        self._obs_buf = None         # last numpy obs for the policy
        self._running: bool = False
        self._task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()

    # ---- public API -------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def set_command(self, throttle: float, steer: float) -> None:
        """Manual drive override. Accepts normalized [-1, 1] inputs and maps
        them to firmware-style duty/servo counts (matching RobotEnv)."""
        throttle = max(-1.0, min(1.0, float(throttle)))
        steer = max(-1.0, min(1.0, float(steer)))
        from ..sim.constants import (
            MOTOR_PWM_MAX_COUNT, SERVO_CENTER_COUNT, SERVO_MAX_COUNT, SERVO_MIN_COUNT,
        )
        if steer >= 0:
            servo = int(SERVO_CENTER_COUNT + steer * (SERVO_MAX_COUNT - SERVO_CENTER_COUNT))
        else:
            servo = int(SERVO_CENTER_COUNT + steer * (SERVO_CENTER_COUNT - SERVO_MIN_COUNT))
        duty = int(abs(throttle) * MOTOR_PWM_MAX_COUNT)
        dir_sign = 1 if throttle >= 0 else -1
        async with self._lock:
            self._cmd = MotorCommand(duty_l=duty, duty_r=duty, servo=servo,
                                     dir_l=dir_sign, dir_r=dir_sign)

    async def load_track(self, track: dict[str, Any]) -> None:
        """Swap in a confirmed track (plan §5.1 shape)."""
        walls = _segments_from_dict(track["walls"])
        spawn = dict(track.get("spawn") or DEFAULT_SPAWN)
        async with self._lock:
            self._walls = walls
            self._spawn = spawn
            self._state = initial_robot(spawn["x"], spawn["y"],
                                        float(spawn.get("theta", 0.0)))
            self._cmd = MotorCommand()

    async def reset(self) -> None:
        async with self._lock:
            self._state = initial_robot(self._spawn["x"], self._spawn["y"],
                                        float(self._spawn.get("theta", 0.0)))
            self._cmd = MotorCommand()

    async def load_policy(self, policy_path: str) -> None:
        """Load an SB3 PPO checkpoint and switch to autonomous mode.

        Uses a minimal wrapper that calls ``model.predict`` with the current
        sensor observation each control tick. The predict call is synchronous
        and cheap (one forward pass through an MLP); running it on the event
        loop is fine for 50 Hz control."""
        from stable_baselines3 import PPO
        from pathlib import Path

        p = Path(policy_path)
        load_target = str(p.with_suffix("") if p.suffix == ".zip" else p)
        model = await asyncio.get_event_loop().run_in_executor(
            None, lambda: PPO.load(load_target, device="cpu")
        )
        async with self._lock:
            self._policy = model
            self._obs_buf = None   # reset obs so we rebuild it on next tick

    async def load_c_controller(self, name: str, code: str) -> str:
        """Compile and load a C controller from source string."""
        from ..control.c_bridge import CController
        import tempfile
        from pathlib import Path
        
        tmp = Path(tempfile.mkdtemp(prefix="pysim_cctrl_")) / f"{name}.c"
        tmp.write_text(code)
        
        async with self._lock:
            try:
                # Perform the compilation
                ctrl = CController(tmp)
                if self._c_controller:
                    try:
                        self._c_controller.close()
                    except:
                        pass
                self._c_controller = ctrl
                self._c_controller.init()
                self._t_ms = 0.0
                self._policy = None # Mutually exclusive with PPO policy
                return "ok"
            except Exception as e:
                return str(e)
                
    async def unload_c_controller(self) -> None:
        async with self._lock:
            if self._c_controller:
                try:
                    self._c_controller.close()
                except:
                    pass
                self._c_controller = None

    async def unload_policy(self) -> None:
        async with self._lock:
            self._policy = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "walls": [[list(w.a), list(w.b)] for w in self._walls],
            "spawn": self._spawn,
            "pose": {"x": self._state.pose.x, "y": self._state.pose.y,
                     "theta": self._state.pose.theta},
        }

    # ---- loop -------------------------------------------------------------

    async def _run(self) -> None:
        # 50 Hz control: 20 × 1 ms physics ticks per iteration, matching the
        # env-training control rate (plan §3).
        ctrl_dt_s = 0.02
        phys_steps = int(round(ctrl_dt_s / PHYSICS_DT_S))
        try:
            while self._running:
                if self._broadcaster.client_count() > 0:
                    async with self._lock:
                        # If a policy is loaded, compute the next command from
                        # the current sensor reading before stepping physics.
                        if self._policy is not None or self._c_controller is not None:
                            sensors_pre = sample_sensors(
                                self._walls, self._state.pose, self._cal)
                            
                            if self._policy is not None:
                                obs = self._build_obs(sensors_pre)
                                action, _ = self._policy.predict(obs, deterministic=True)
                                self._apply_policy_action(action)
                            elif self._c_controller is not None:
                                try:
                                    cmd = self._c_controller.tick(sensors_pre, self._t_ms)
                                    self._cmd = cmd
                                except Exception as e:
                                    logger.error(f"C controller tick failed: {e}")
                                    self._c_controller = None
                                self._t_ms += ctrl_dt_s * 1000.0
                        
                        apply_command(self._state, self._cmd.duty_l, self._cmd.duty_r, self._cmd.servo, self._cmd.dir_l, self._cmd.dir_r)
                        for _ in range(phys_steps):
                            step_physics(self._state, self._walls,
                                         CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM,
                                         PHYSICS_DT_S)
                            if self._state.collided:
                                break
                        sensors = sample_sensors(self._walls, self._state.pose, self._cal)
                        payload = self._snapshot_msg(sensors)
                    await self._broadcaster.broadcast(json.dumps(payload,
                                                                 separators=(",", ":")))
                await asyncio.sleep(ctrl_dt_s)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Sim runner crashed")

    def _build_obs(self, sensors: dict[str, Any]):
        """Build the 8-dim obs vector matching ``RobotEnv._obs``."""
        import numpy as np
        from ..sim.constants import MAX_SPEED_CMS, STEER_LIMIT_RAD
        lidar_max = self._cal.lidar.max_cm
        ir_max = self._cal.ir.max_cm
        obs = np.array([
            sensors["lidar"]["center"]["distance_cm"] / lidar_max,
            sensors["lidar"]["left"]["distance_cm"]   / lidar_max,
            sensors["lidar"]["right"]["distance_cm"]  / lidar_max,
            sensors["ir"]["left"]["distance_cm"]  / ir_max,
            sensors["ir"]["right"]["distance_cm"] / ir_max,
            self._state.v / MAX_SPEED_CMS,
            float(np.clip(self._state.omega / 5.0, -1.0, 1.0)),
            self._state.steer_angle / STEER_LIMIT_RAD,
        ], dtype=np.float32)
        return obs

    def _apply_policy_action(self, action) -> None:
        """Map policy action (3-dim [servo, tL, tR]) → MotorCommand."""
        from ..sim.constants import (
            MOTOR_PWM_MAX_COUNT, SERVO_CENTER_COUNT, SERVO_MAX_COUNT, SERVO_MIN_COUNT,
        )
        from ..env.robot_env import THROTTLE_MIN
        servo_norm = float(max(-1.0, min(1.0, action[0])))
        tL_norm = float(max(-1.0, min(1.0, action[1])))
        tR_norm = float(max(-1.0, min(1.0, action[2])))
        throttle_l = THROTTLE_MIN + (tL_norm + 1.0) * 0.5 * (1.0 - THROTTLE_MIN)
        throttle_r = THROTTLE_MIN + (tR_norm + 1.0) * 0.5 * (1.0 - THROTTLE_MIN)
        if servo_norm >= 0:
            servo = int(SERVO_CENTER_COUNT + servo_norm * (SERVO_MAX_COUNT - SERVO_CENTER_COUNT))
        else:
            servo = int(SERVO_CENTER_COUNT + servo_norm * (SERVO_CENTER_COUNT - SERVO_MIN_COUNT))
        duty_l = int(throttle_l * MOTOR_PWM_MAX_COUNT)
        duty_r = int(throttle_r * MOTOR_PWM_MAX_COUNT)
        self._cmd.duty_l = duty_l
        self._cmd.duty_r = duty_r
        self._cmd.servo = servo

    def _snapshot_msg(self, sensors: dict[str, Any]) -> dict[str, Any]:
        # Flatten sensor hit points for easy canvas rendering on the client.
        def _flat(reading: dict[str, Any]) -> dict[str, Any]:
            hit = reading.get("hit")
            return {
                "distance_cm": float(reading["distance_cm"]),
                "valid": bool(reading["valid"]),
                "origin": [reading["origin"].x, reading["origin"].y],
                "hit": [hit.x, hit.y] if hit is not None else None,
            }
        return {
            "kind": "sim",
            "pose": {"x": self._state.pose.x, "y": self._state.pose.y,
                     "theta": self._state.pose.theta},
            "v": self._state.v,
            "omega": self._state.omega,
            "steer": self._state.steer_angle,
            "collided": bool(self._state.collided),
            "lidar": {k: _flat(v) for k, v in sensors["lidar"].items()},
            "ir": {k: _flat(v) for k, v in sensors["ir"].items()},
        }
