"""Scale sanity check (plan §6.1a).

Given the detected rectangles (from ``segment``), verify the median long-axis
length matches the known plank length within tolerance. Surfaces a
``SCALE_MISMATCH`` error when the iOS rectification was wrong."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BlockRect:
    cx: float                # cm, rectified-image frame
    cy: float
    w: float                 # long axis
    h: float                 # short axis
    theta: float             # rad


@dataclass
class ScaleCheck:
    ok: bool
    observed_cm: Optional[float]
    expected_cm: float
    n_blocks: int
    lower: float
    upper: float


def check_scale(
    blocks: list[BlockRect], *,
    plank_length_cm: float = 80.0,
    lower: float = 0.72,
    upper: float = 1.10,
    min_blocks: int = 4,
) -> ScaleCheck:
    """Median long-axis length vs plank length. Returns ``ok=True`` with
    ``observed_cm=None`` when there aren't enough blocks to check — callers
    should surface that as a *warning*, not a hard failure."""
    if len(blocks) < min_blocks:
        return ScaleCheck(ok=True, observed_cm=None,
                          expected_cm=plank_length_cm,
                          n_blocks=len(blocks), lower=lower, upper=upper)
    long_axes = np.array([max(b.w, b.h) for b in blocks], dtype=np.float32)
    obs = float(np.median(long_axes))
    ratio = obs / plank_length_cm
    ok = lower <= ratio <= upper
    return ScaleCheck(ok=ok, observed_cm=obs,
                      expected_cm=plank_length_cm,
                      n_blocks=len(blocks), lower=lower, upper=upper)
