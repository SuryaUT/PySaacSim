"""Python port of the MSPM0 sensor board's Model.h / Model.c.

Mirrors the on-device residual-policy model exactly:
  - Caps and normalization formulas match Model.c byte-for-byte semantics.
  - Inputs are Q16 (0..65536). Signed inputs use offset Q16, center 32768.
  - Outputs are 3 offset-Q16 values that the firmware turns into a delta on
    top of a PD baseline (see Model_ApplyResidual).

The simulator carries observations as float in [0, 1] for PPO (cheaper math,
identical semantics — Q16 / 65536). At export time the linear policy is
re-quantized into the firmware's int Q16 layout via `export_q16` and emitted
as a Model.c initializer block via `export_to_c_source`.

See: ../docs/Model_Interface.md (RTOS_SensorBoard project Model_Interface.md)
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


# -- Caps (mirror Model.h) -------------------------------------------------
CAP_IR = 800              # mm
CAP_TFLUNA = 1000         # mm
CAP_THROTTLE = 9000       # PWM units
CAP_STEERING = 35         # degrees (matches firmware Model.h)
CAP_ANGLE = 64            # degrees
CAP_ANGLE_POW = 6         # log2(CAP_ANGLE)
CAP_DELTA_STEERING = 10   # degrees
CAP_DELTA_THROTTLE = 2000 # PWM units
CAP_YAW = 16384           # raw LSB (~125 deg/s at 131 LSB/(deg/s))
CAP_ACCEL = 8192          # raw LSB (~0.5 g at 16384 LSB/g)

Q16_ONE = 65536
Q16_HALF = 32768

NUM_INPUTS = 13
NUM_OUTPUTS = 3


class Inp(IntEnum):
    """Indices into the 13-element input vector (matches input_t in Model.h)."""
    ir_right = 0
    ir_left = 1
    tf_left = 2
    tf_middle = 3
    tf_right = 4
    throttle_left_prev = 5
    throttle_right_prev = 6
    steering_prev = 7
    angle_left = 8
    angle_right = 9
    yaw_rate = 10
    accel_lat = 11
    accel_long = 12


class Out(IntEnum):
    """Indices into the 3-element output vector (matches output_t in Model.h)."""
    throttle_left = 0
    throttle_right = 1
    steering = 2


INPUT_NAMES = [Inp(i).name for i in range(NUM_INPUTS)]
OUTPUT_NAMES = [Out(i).name for i in range(NUM_OUTPUTS)]


# -- Q16 fixed-point normalization (mirrors Model.c) -----------------------

def normalize_q16(x_mm: int, cap: int) -> int:
    """Unsigned Q16 normalize. Saturates at 65536. Matches Model_Normalize."""
    out = (int(x_mm) << 16) // cap
    if out > Q16_ONE:
        return Q16_ONE
    if out < 0:
        return 0
    return out


def normalize_signed_q16(x: int, cap: int) -> int:
    """Signed Q16 normalize centered on 32768. Matches Model_NormalizeSigned."""
    x = max(-cap, min(cap, int(x)))
    return ((x + cap) << 16) // (2 * cap)


def encode_angle_q16(angle_deg: int) -> int:
    """Wall-angle encoding: clamp to ±64°, then (a+64) << 9. Matches firmware."""
    angle = max(-CAP_ANGLE, min(CAP_ANGLE, int(angle_deg)))
    return (angle + CAP_ANGLE) << (16 - CAP_ANGLE_POW - 1)


# -- Float helpers (used by env / training, in [0, 1]) ---------------------
# These give the same answer as the Q16 forms divided by 65536, but stay in
# float for PPO. The Q16 equivalents above are reserved for export checks.

def normalize(x_mm: float, cap: float) -> float:
    out = x_mm / cap
    return max(0.0, min(1.0, out))


def normalize_signed(x: float, cap: float) -> float:
    x = max(-cap, min(cap, x))
    return (x + cap) / (2.0 * cap)


def encode_angle(angle_deg: float) -> float:
    a = max(-CAP_ANGLE, min(CAP_ANGLE, angle_deg))
    return (a + CAP_ANGLE) / (2.0 * CAP_ANGLE)


# -- Residual application (mirrors Model_ApplyResidual) --------------------

def apply_residual(
    pd_throttle_l: float, pd_throttle_r: float, pd_steering: float,
    delta_thr_l: float, delta_thr_r: float, delta_steer: float,
) -> tuple[float, float, float]:
    """Add deltas to PD outputs, clamp to physical limits. Returns (thr_l, thr_r, steer)."""
    thr_l = max(0.0, min(CAP_THROTTLE, pd_throttle_l + delta_thr_l))
    thr_r = max(0.0, min(CAP_THROTTLE, pd_throttle_r + delta_thr_r))
    steer = max(-CAP_STEERING, min(CAP_STEERING, pd_steering + delta_steer))
    return thr_l, thr_r, steer


def action_to_delta(action: np.ndarray) -> tuple[float, float, float]:
    """Map RL action ∈ [-1, +1]^3 to physical deltas matching the firmware path.

    PPO action 0 ⇒ delta 0 ⇒ pure PD. ±1 ⇒ ±CAP_DELTA_*. The firmware uses
    `delta = ((out_q16 - 32768) * CAP) >> 15`, with `out_q16 = 32768*(1+a)`
    that simplifies to `delta = a * CAP` (modulo Q16 rounding).
    """
    a = np.clip(action, -1.0, 1.0)
    return (
        float(a[Out.throttle_left])  * CAP_DELTA_THROTTLE,
        float(a[Out.throttle_right]) * CAP_DELTA_THROTTLE,
        float(a[Out.steering])       * CAP_DELTA_STEERING,
    )


# -- Q16 weight export -----------------------------------------------------

def export_q16(W_float: np.ndarray, b_float: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a SB3-trained linear policy to firmware Q16.

    Derivation. The Python policy gives action = W·obs + b with obs ∈ [0,1]
    and target action ∈ [-1, +1]. The firmware computes
        out_q16 = sum_j (Wq[i,j] * obs_q16[j]) >> 16 + bq[i]
    with obs_q16 = obs * 65536. We want out_q16 = 32768 + 32768·action so
    that out_q16 = 32768 ⇒ delta = 0 ⇒ pure PD. Matching coefficients:
        Wq[i,j] = round(W_float[i,j] * 32768)
        bq[i]   = round(32768 + b_float[i] * 32768)
    """
    Wq = np.round(W_float * Q16_HALF).astype(np.int32)
    bq = np.round(Q16_HALF + b_float * Q16_HALF).astype(np.int32)
    return Wq, bq


def export_to_c_source(Wq: np.ndarray, bq: np.ndarray) -> str:
    """Render Q16 weights/biases as a drop-in replacement for Model.c initializers.

    Output column order matches Inp enum; row order matches Out enum, both
    of which match the firmware's input_t / output_t enums in Model.h.
    """
    if Wq.shape != (NUM_OUTPUTS, NUM_INPUTS):
        raise ValueError(f"Wq shape {Wq.shape} != ({NUM_OUTPUTS}, {NUM_INPUTS})")
    if bq.shape != (NUM_OUTPUTS,):
        raise ValueError(f"bq shape {bq.shape} != ({NUM_OUTPUTS},)")
    rows = []
    for i in range(NUM_OUTPUTS):
        nums = ", ".join(f"{int(v):>8d}" for v in Wq[i])
        rows.append(f"  {{ {nums} }},  // {Out(i).name}")
    bias_nums = ", ".join(f"{int(v)}" for v in bq)
    return (
        "const fixed_t Model_Weights[NUM_OUTPUTS][NUM_INPUTS] = {\n"
        + "\n".join(rows) + "\n};\n\n"
        f"const fixed_t Model_Bias[NUM_OUTPUTS] = {{ {bias_nums} }};\n"
    )
