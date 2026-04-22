"""Linear policy helpers for SB3 PPO with `net_arch=[]` — residual policy.

With `net_arch=[]`, the SB3 MLP collapses to a 13→3 affine map:
    action = W @ obs + b
where action is the residual delta in normalized [-1, +1] space (action 0
⇒ pure PD baseline). Names below match the spec's input_t / output_t
enums in Model.h.
"""
from __future__ import annotations

import numpy as np

from ...sim.model import (
    INPUT_NAMES, OUTPUT_NAMES, NUM_INPUTS, NUM_OUTPUTS, export_q16,
    export_to_c_source,
)


# Re-exported under the GUI-friendly names so existing widgets (heatmap labels)
# don't need to know about sim.model.
FEATURE_NAMES = INPUT_NAMES   # 13 inputs in input_t order
ACTION_NAMES = OUTPUT_NAMES   # 3 deltas in output_t order


def extract_weights(model) -> tuple[np.ndarray, np.ndarray]:
    """Pull the linear policy (W, b) out of a trained SB3 PPO model.

    Returns W shape (NUM_OUTPUTS, NUM_INPUTS) and b shape (NUM_OUTPUTS,)."""
    action_net = model.policy.action_net
    W = action_net.weight.detach().cpu().numpy().copy()
    b = action_net.bias.detach().cpu().numpy().copy()
    return W, b


def policy_forward(W: np.ndarray, b: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Deterministic forward pass. obs shape (13,) -> action shape (3,) in [-1, +1].

    Output components are the residual deltas in normalized space:
      [throttle_left_delta, throttle_right_delta, steering_delta]
    Each ±1 maps to ±CAP_DELTA_* in the env (see sim.model.action_to_delta)."""
    a = W @ obs + b
    return np.clip(a, -1.0, 1.0).astype(np.float32, copy=False)


def export_firmware(W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Quantize a trained linear policy and emit a Model.c initializer block.

    Returns (Wq, bq, c_source). Drop `c_source` into the firmware's Model.c
    in place of the existing `Model_Weights` / `Model_Bias` arrays.
    """
    if W.shape != (NUM_OUTPUTS, NUM_INPUTS):
        raise ValueError(f"W shape {W.shape} != ({NUM_OUTPUTS}, {NUM_INPUTS})")
    Wq, bq = export_q16(W, b)
    return Wq, bq, export_to_c_source(Wq, bq)
