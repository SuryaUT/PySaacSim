"""Load a firmware CSV log into a typed `Log` dataclass.

Columns (confirmed from RTOS_SensorBoard.c and the plan):
    time_ms, ir_r, ir_l, tf_r, tf_l, tf_front,
    throttle_l, throttle_r, steering,
    gyro_z, accel_x, accel_y

Sign conventions (from SIMULATION_REALISM_PLAN.md):
    steering  :  positive = right turn (degrees)
    gyro_z    :  positive = left turn  (int16 LSB, 131 LSB / (deg/s))
    accel_x   :  positive = chassis accel to the right (int16, 16384 LSB/g)
    accel_y   :  positive = chassis accel forward       (int16, 16384 LSB/g)

The firmware logs IR as already-converted distance in mm (saturates at 305 mm)
and TFLuna as raw mm (averaged 8 samples in firmware).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np


LOG_COLUMNS = (
    "time_ms", "ir_r", "ir_l", "tf_r", "tf_l", "tf_front",
    "throttle_l", "throttle_r", "steering",
    "gyro_z", "accel_x", "accel_y",
)

LOG_DT_S = 0.080            # 80 ms CSV cadence
LOG_RATE_HZ = 1.0 / LOG_DT_S  # 12.5 Hz


@dataclass
class Log:
    path: Path
    time_ms: np.ndarray
    ir_r_mm: np.ndarray
    ir_l_mm: np.ndarray
    tf_r_mm: np.ndarray
    tf_l_mm: np.ndarray
    tf_front_mm: np.ndarray
    throttle_l: np.ndarray   # PWM duty 0..9999
    throttle_r: np.ndarray
    steering_deg: np.ndarray # +right
    gyro_z_lsb: np.ndarray   # +left (int16)
    accel_x_lsb: np.ndarray
    accel_y_lsb: np.ndarray

    @property
    def n(self) -> int:
        return int(self.time_ms.size)

    @property
    def t_s(self) -> np.ndarray:
        return self.time_ms.astype(float) / 1000.0

    @property
    def throttle_avg(self) -> np.ndarray:
        return 0.5 * (self.throttle_l + self.throttle_r).astype(float)


def load_log(path: Union[str, Path]) -> Log:
    p = Path(path)
    cols: dict[str, list[float]] = {name: [] for name in LOG_COLUMNS}
    with open(p, "r") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        # Header row in these logs has leading spaces on some files; DictReader
        # keeps them unless we strip manually.
        fieldnames = [f.strip() for f in reader.fieldnames or []]
        if tuple(fieldnames) != LOG_COLUMNS:
            raise ValueError(
                f"{p.name}: expected columns {LOG_COLUMNS}, got {tuple(fieldnames)}"
            )
        for row in reader:
            # Guard against truncated trailing rows (some logs end mid-line).
            if any(row.get(f) is None or row.get(f) == "" for f in (reader.fieldnames or [])):
                continue
            for orig, name in zip(reader.fieldnames or [], fieldnames):
                cols[name].append(float(row[orig].strip()))
    arr = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    return Log(
        path=p,
        time_ms=arr["time_ms"].astype(np.int64),
        ir_r_mm=arr["ir_r"],
        ir_l_mm=arr["ir_l"],
        tf_r_mm=arr["tf_r"],
        tf_l_mm=arr["tf_l"],
        tf_front_mm=arr["tf_front"],
        throttle_l=arr["throttle_l"],
        throttle_r=arr["throttle_r"],
        steering_deg=arr["steering"],
        gyro_z_lsb=arr["gyro_z"],
        accel_x_lsb=arr["accel_x"],
        accel_y_lsb=arr["accel_y"],
    )


def load_logs(paths: list[Union[str, Path]]) -> list[Log]:
    return [load_log(p) for p in paths]
