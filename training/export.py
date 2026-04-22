"""Policy export (plan §5.6).

Two deliverables:
  * ``policy.npz`` — numeric dump of weights + obs-normalization stats.
  * ``policy.h`` — a C header with the same weights laid out as
    ``static const float[]`` plus an inline ``policy_forward`` function.

Both use the **action mean head** only (ignore log-std); we apply the
training-time ``tanh`` squashing at inference and clip to [-1, 1]. The
observation normalization is identity for now (we don't wrap the training
env in ``VecNormalize``), but the arrays are written regardless so the
firmware ingests the same schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


_POLICY_H_TEMPLATE = """\
/* policy.h — auto-generated. Do not edit. */
#pragma once

#include <math.h>

#define POLICY_OBS_DIM  {obs_dim}
#define POLICY_ACT_DIM  {act_dim}

static const float POLICY_OBS_MEAN[POLICY_OBS_DIM] = {{ {obs_mean} }};
static const float POLICY_OBS_STD[POLICY_OBS_DIM]  = {{ {obs_std} }};

{layer_consts}

static inline void policy_forward(const float *obs_raw, float *action) {{
    float obs[POLICY_OBS_DIM];
    for (int i = 0; i < POLICY_OBS_DIM; ++i) {{
        float d = (obs_raw[i] - POLICY_OBS_MEAN[i]) / POLICY_OBS_STD[i];
        obs[i] = d;
    }}
{forward_body}
    for (int i = 0; i < POLICY_ACT_DIM; ++i) {{
        float a = action[i];
        if (a > 1.0f) a = 1.0f;
        if (a < -1.0f) a = -1.0f;
        action[i] = a;
    }}
}}
"""


def _fmt_array(name: str, arr: np.ndarray) -> str:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    dims = " * ".join(str(d) for d in arr.shape)
    body = ", ".join(f"{v:.8g}f" for v in flat)
    return f"static const float {name}[{dims}] = {{ {body} }};"


def _extract_mlp_weights(model) -> dict[str, np.ndarray]:
    """Pull weights from an SB3 PPO model with MlpPolicy net_arch=[128, 128].

    SB3's MlpPolicy splits the network: a shared ``mlp_extractor`` (policy
    tower + value tower) and an ``action_net`` linear head that maps to the
    action-mean. We want the policy tower + action head."""
    import torch

    policy = model.policy
    ext = policy.mlp_extractor
    # Collect Linear layers in the policy tower in order.
    layers = []
    for m in ext.policy_net:
        if isinstance(m, torch.nn.Linear):
            layers.append(m)
    layers.append(policy.action_net)
    weights: dict[str, np.ndarray] = {}
    for i, lin in enumerate(layers):
        W = lin.weight.detach().cpu().numpy()
        b = lin.bias.detach().cpu().numpy()
        if i < len(layers) - 1:
            weights[f"W{i}"] = W
            weights[f"b{i}"] = b
        else:
            weights["Wout"] = W
            weights["bout"] = b
    return weights


def export_policy(policy_path: str, out_dir: str) -> dict[str, str]:
    """Load an SB3 PPO checkpoint, emit ``policy.npz`` + ``policy.h`` to
    ``out_dir``. Returns the paths."""
    from stable_baselines3 import PPO

    p = Path(policy_path)
    load_target = str(p.with_suffix("") if p.suffix == ".zip" else p)
    model = PPO.load(load_target, device="cpu")

    weights = _extract_mlp_weights(model)
    obs_space = model.observation_space
    act_space = model.action_space
    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(np.prod(act_space.shape))

    # We don't currently wrap training in VecNormalize — export identity.
    obs_mean = np.zeros(obs_dim, dtype=np.float32)
    obs_std = np.ones(obs_dim, dtype=np.float32)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "activation": "tanh",
        "note": "action is the Gaussian mean head, tanh-squashed, then clipped to [-1, 1]",
    }
    npz_path = out / "policy.npz"
    np.savez(npz_path,
             obs_mean=obs_mean, obs_std=obs_std,
             meta_json=np.array(json.dumps(meta), dtype=object),
             **weights)

    # ---- C header --------------------------------------------------------
    layer_defs: list[str] = []
    hidden_sizes: list[int] = []
    i = 0
    while f"W{i}" in weights:
        W = weights[f"W{i}"]
        b = weights[f"b{i}"]
        layer_defs.append(_fmt_array(f"POLICY_W{i}", W))
        layer_defs.append(_fmt_array(f"POLICY_B{i}", b))
        hidden_sizes.append(W.shape[0])
        i += 1
    layer_defs.append(_fmt_array("POLICY_WOUT", weights["Wout"]))
    layer_defs.append(_fmt_array("POLICY_BOUT", weights["bout"]))

    # Build the C forward pass. We allocate stack buffers of max hidden size
    # so the code stays simple.
    max_hidden = max([obs_dim, act_dim] + hidden_sizes)
    body_lines = [
        f"    float buf_a[{max_hidden}];",
        f"    float buf_b[{max_hidden}];",
        "    const float *cur = obs;",
        f"    int cur_size = {obs_dim};",
    ]
    for idx in range(len(hidden_sizes)):
        W = weights[f"W{idx}"]
        out_size = W.shape[0]
        tgt = "buf_a" if idx % 2 == 0 else "buf_b"
        body_lines += [
            f"    for (int j = 0; j < {out_size}; ++j) {{",
            f"        float s = POLICY_B{idx}[j];",
            f"        for (int k = 0; k < cur_size; ++k) s += POLICY_W{idx}[j * cur_size + k] * cur[k];",
            f"        {tgt}[j] = tanhf(s);",
            "    }",
            f"    cur = {tgt};",
            f"    cur_size = {out_size};",
        ]
    body_lines += [
        f"    for (int j = 0; j < {act_dim}; ++j) {{",
        "        float s = POLICY_BOUT[j];",
        "        for (int k = 0; k < cur_size; ++k) s += POLICY_WOUT[j * cur_size + k] * cur[k];",
        "        action[j] = tanhf(s);",
        "    }",
    ]

    header = _POLICY_H_TEMPLATE.format(
        obs_dim=obs_dim, act_dim=act_dim,
        obs_mean=", ".join(f"{v:.8g}f" for v in obs_mean),
        obs_std=", ".join(f"{v:.8g}f" for v in obs_std),
        layer_consts="\n".join(layer_defs),
        forward_body="\n".join(body_lines),
    )
    h_path = out / "policy.h"
    with open(h_path, "w") as f:
        f.write(header)

    return {"policy_npz": str(npz_path), "policy_h": str(h_path)}
