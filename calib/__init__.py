"""Sim↔real calibration from on-device CSV logs.

The firmware logs one row per Robot() tick (~12.5 Hz, 80 ms cadence) with
columns `time_ms, ir_r, ir_l, tf_r, tf_l, tf_front, throttle_l, throttle_r,
steering, gyro_z, accel_x, accel_y`. This package fits the sim's sensor and
dynamics knobs against those logs so a policy trained in sim transfers to
hardware with minimal delta.

Pipeline (see SIMULATION_REALISM_PLAN.md Phases 2–5):
    IR_Calib.xlsx         -> calib.ir_xlsx       -> per-side IR (a, b, c)
    IMU_steady_state.csv  -> calib.imu_bias +    -> bias + noise std
                             calib.noise_fit
    4x driving CSVs       -> calib.replay +      -> dynamics params
                             calib.dynamics_fit
"""
from __future__ import annotations
