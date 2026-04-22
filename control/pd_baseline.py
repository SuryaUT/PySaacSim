"""PD baseline controller — Python port of RTOS_SensorBoard.c::Robot().

Stateful (carries `prev_error`, `prev_e_a` between ticks). Distances in mm,
throttle in PWM units, steering in degrees with the project-wide convention
**+steering = right turn**. The residual model in `sim/model.py` adds small
deltas on top of this baseline; with zero residual, the simulated robot
behaves identically to the firmware running model bias = 32768.

Where the on-device firmware and the Model_Interface spec disagree, this
implementation follows the spec (which is the authoritative training target):
  - urgency  = (600-front) >> 4   (firmware uses >> 3)
  - final clamp ±CAP_STEERING=30  (firmware clamps to ±53)
  - diff steering threshold ±15   (firmware uses ±30)
The firmware will be updated to match; sim-side weights trained against the
spec will then transfer.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from ..sim.model import CAP_ANGLE, CAP_STEERING


# PD gains (mirror RTOS_SensorBoard.c lines 10-17)
KP_D = 1
KD_D = 2
KP_A = 5
KD_A = 2
ANGLE_REF = 5
DIST_REF = 0

BASE_THROTTLE = 9000      # spec section 6
CORNER_THROTTLE_DROP = 2000
CORNER_TRIGGER_MM = 600   # front TFLuna distance
DIFF_STEER_THRESH = 15    # |steer| ≥ this triggers inside-wheel slow-down
DIFF_STEER_DROP = 2000


class Geometry(NamedTuple):
    """Pre-computed wall geometry used by both PD and the model input vector."""
    d_ir: int           # right IR, mm
    ld_ir: int          # left  IR, mm
    d2: int             # right TFLuna, mm
    ld2: int            # left  TFLuna, mm
    front: int          # front TFLuna, mm
    angle_right: float  # degrees, post angle_ref subtraction (clamp ±64 in caller)
    angle_left: float   # degrees
    real_dist: float    # perpendicular right-wall distance, mm
    l_real_dist: float  # perpendicular left-wall distance, mm


class PDOutput(NamedTuple):
    throttle_l: int      # 0..CAP_THROTTLE
    throttle_r: int
    steering: int        # degrees, signed, clamped ±CAP_STEERING


def ir_correction(d_ir: int, ld_ir: int, d2: int, ld2: int) -> tuple[int, int]:
    """Firmware lines 448-449: clamp IR to 800mm when TFLuna sees open space.

    The IR is past its calibrated range when the parallel TFLuna shows
    >600mm AND the IR reading lags by >150mm. This mirrors the on-device
    behavior so the model sees the same "wall lost" signal in sim.
    """
    if d2 > 600 and d2 > d_ir + 150:
        d_ir = 800
    if ld2 > 600 and ld2 > ld_ir + 150:
        ld_ir = 800
    return d_ir, ld_ir


def compute_geometry(d_ir: int, ld_ir: int, d2: int, ld2: int, front: int) -> Geometry:
    """Wall-angle + perpendicular-distance math from Robot() lines 451-455.

    Side geometry: IR points perpendicular, TF-Luna at 45°, 224 mm offset
    between them. For a wall at angle θ to the chassis, tan(θ) ≈
        (d_ir * √2 - d2) / (224 + d2).
    The firmware codes the √2 as 1414 and `d2*1000` so the integer
    `(d_ir*1414 - d2*1000) / (224 + d2)` already encodes tan(θ)·1000;
    its `arctan` LUT consumes that scaled tangent and returns degrees. In
    Python we just divide out the 1000 and take a float atan.
    """
    denom_r = max(1, 224 + d2)
    denom_l = max(1, 224 + ld2)
    tan_r = (d_ir  * 1414 - d2  * 1000) / (denom_r * 1000)
    tan_l = (ld_ir * 1414 - ld2 * 1000) / (denom_l * 1000)
    angle_right = math.degrees(math.atan(tan_r)) - ANGLE_REF
    angle_left  = math.degrees(math.atan(tan_l))

    real_dist   = d_ir  * math.cos(math.radians(angle_right))
    l_real_dist = ld_ir * math.cos(math.radians(angle_left))
    return Geometry(
        d_ir=d_ir, ld_ir=ld_ir, d2=d2, ld2=ld2, front=front,
        angle_right=angle_right, angle_left=angle_left,
        real_dist=real_dist, l_real_dist=l_real_dist,
    )


def clamp_angle_for_model(angle_deg: float) -> int:
    """Clamp wall angle to ±CAP_ANGLE so the Q16 angle encoder doesn't overflow."""
    return max(-CAP_ANGLE, min(CAP_ANGLE, int(angle_deg)))


class PDBaseline:
    """Stateful PD controller. Reset between episodes."""

    def __init__(self) -> None:
        self.prev_error: float = 0.0
        self.prev_e_a: float = 0.0

    def reset(self) -> None:
        self.prev_error = 0.0
        self.prev_e_a = 0.0

    def tick(self, geom: Geometry) -> PDOutput:
        """Run one PD iteration on pre-computed geometry."""
        # Distance PD
        e_d = geom.real_dist - geom.l_real_dist - DIST_REF
        intend_angle = (KP_D * e_d) / 10.0 + (KD_D * (e_d - self.prev_error)) / 10.0
        self.prev_error = e_d

        # Angle PD (steering target follows wall-distance error)
        e_a = intend_angle - geom.angle_right
        steering = (e_a * KP_A) / 10.0 + ((e_a - self.prev_e_a) * KD_A) / 10.0
        self.prev_e_a = e_a

        # Throttle baseline + corner override
        throttle = BASE_THROTTLE
        if geom.front < CORNER_TRIGGER_MM:
            throttle -= CORNER_THROTTLE_DROP
            urgency = (CORNER_TRIGGER_MM - geom.front) >> 4   # spec: >> 4
            steering = -urgency if (geom.ld2 > geom.d2) else urgency

        # Final steering clamp at the spec limit
        steering = max(-CAP_STEERING, min(CAP_STEERING, steering))

        # Differential steering: slow the inside wheel on hard turns
        thr_l = throttle
        thr_r = throttle
        if steering <= -DIFF_STEER_THRESH:
            thr_r -= DIFF_STEER_DROP
        elif steering >= DIFF_STEER_THRESH:
            thr_l -= DIFF_STEER_DROP

        return PDOutput(int(thr_l), int(thr_r), int(round(steering)))
