"""Drive the sim with a compiled-from-C controller. Smoke-test the C bridge.

Bypasses RobotEnv (which is RL-only — residual on PD). The C controller is
the firmware's full pipeline (PD + Model_ApplyResidual baked in), so we drive
the physics loop directly here at the same 12.5 Hz control rate.

Requires gcc on PATH.

Run:
    python -m PySaacSim.examples.run_c_controller
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySaacSim.control.c_bridge import CController
from PySaacSim.sim.calibration import SensorCalibration
from PySaacSim.sim.constants import (
    CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, PHYSICS_DT_S,
)
from PySaacSim.sim.physics import apply_command, initial_robot, step_physics
from PySaacSim.sim.sensors import sample_sensors
from PySaacSim.sim.world import DEFAULT_SPAWN, build_default_world


CTRL_PERIOD_MS = 80.0


def main():
    c_src = Path(__file__).with_name("controller_template.c")
    controller = CController(c_src)
    controller.init()

    world = build_default_world()
    walls = world["walls"]
    cal = SensorCalibration.default()
    state = initial_robot(DEFAULT_SPAWN["x"], DEFAULT_SPAWN["y"], DEFAULT_SPAWN["theta"])

    t_ms = 0.0
    ctrl_steps = max(1, int(round(CTRL_PERIOD_MS / (PHYSICS_DT_S * 1000.0))))
    max_ctrl_iters = 500

    for _ in range(max_ctrl_iters):
        sensors = sample_sensors(walls, state.pose, cal)
        cmd = controller.tick(sensors, t_ms)
        apply_command(state, cmd.duty_l, cmd.duty_r, cmd.servo)
        for _ in range(ctrl_steps):
            step_physics(state, walls, CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, PHYSICS_DT_S)
            t_ms += PHYSICS_DT_S * 1000.0
            if state.collided:
                break
        if state.collided:
            break

    print(f"Done. t_ms={t_ms:.0f} pose=({state.pose.x:.1f}, {state.pose.y:.1f}) "
          f"collided={state.collided}")
    controller.close()


if __name__ == "__main__":
    main()
