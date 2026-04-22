"""SAM3 segmentation + filtering (plan §6.2).

SAM3 is access-gated, so this module imports it lazily. If the wheel isn't
present, ``segment_blocks`` raises ``SAM3Unavailable`` which the server
converts to a ``SAM3_UNAVAILABLE`` error code.

Filter logic is standalone and unit-testable: it consumes raw mask arrays
and emits ``BlockRect``s. The unit tests feed hand-drawn masks directly,
bypassing SAM3 entirely."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from .validate import BlockRect


class SAM3Unavailable(RuntimeError):
    """Raised when the SAM3 wheel is not installed or its weights aren't on
    disk. The server catches this and returns ``SAM3_UNAVAILABLE``."""


@dataclass
class SegmentConfig:
    score_threshold: float = 0.5
    text_prompts: tuple[str, ...] = ("rectangular wooden plank on the floor",)
    weights_path: str = "models/sam3.pt"
    mode: str = "text_prompt"          # "text_prompt" | "automatic"


def _lazy_import_sam3():
    try:
        import sam3  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise SAM3Unavailable(f"SAM3 is not installed: {e}")
    return sam3


def run_sam3(image_rgb: np.ndarray, cfg: SegmentConfig) -> list[np.ndarray]:
    """Returns a list of boolean masks, shape ``(H, W)``. Raises
    ``SAM3Unavailable`` if the SAM3 backend can't be loaded.

    The exact SAM3 Python API may shift between releases; this wrapper is a
    thin shim. Swap in the real call sites once the access SDK is in-hand —
    the return type is pinned so downstream code doesn't move."""
    sam3 = _lazy_import_sam3()
    if cfg.mode == "text_prompt":
        masks = sam3.segment(                         # type: ignore[attr-defined]
            image=image_rgb,
            text_prompt=cfg.text_prompts,
            score_threshold=cfg.score_threshold,
        )
    else:
        masks = sam3.auto_segment(image=image_rgb)    # type: ignore[attr-defined]
    out: list[np.ndarray] = []
    for m in masks:
        arr = np.asarray(m)
        if arr.dtype != bool:
            arr = arr.astype(bool)
        out.append(arr)
    return out


# --------------------------------------------------------------------------
# Filtering — consumes raw masks, emits BlockRect candidates (§6.2).
# --------------------------------------------------------------------------

def filter_masks(
    masks: Iterable[np.ndarray],
    *, px_per_cm: float,
    min_area_cm2: float = 50.0,
    max_area_cm2: float = 2000.0,
    min_aspect: float = 1.0,
    max_aspect: float = 6.0,
    max_rect_residual: float = 0.15,
    border_px: int = 20,
) -> list[BlockRect]:
    import cv2
    keep: list[BlockRect] = []
    for m in masks:
        m = np.asarray(m, dtype=bool)
        H, W = m.shape[:2]
        area_px = int(m.sum())
        if area_px <= 0:
            continue
        area_cm2 = area_px / (px_per_cm ** 2)
        if area_cm2 < min_area_cm2 or area_cm2 > max_area_cm2:
            continue

        ys, xs = np.where(m)
        if (xs.min() < border_px or ys.min() < border_px
                or xs.max() > W - border_px or ys.max() > H - border_px):
            continue

        # Rotated rectangle fit.
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        rect = cv2.minAreaRect(pts)
        (cx_px, cy_px), (w_px, h_px), angle_deg = rect
        w_cm = w_px / px_per_cm
        h_cm = h_px / px_per_cm
        long_cm = max(w_cm, h_cm)
        short_cm = max(1e-6, min(w_cm, h_cm))
        aspect = long_cm / short_cm
        if aspect < min_aspect or aspect > max_aspect:
            continue

        rect_area_px = max(1.0, w_px * h_px)
        residual = abs(rect_area_px - area_px) / rect_area_px
        if residual > max_rect_residual:
            continue

        theta_rad = float(np.radians(angle_deg))
        # Flip cy: OpenCV pixel y runs top-down; world cm y runs bottom-up
        # on the rectified canvas. Apply the flip so downstream coords match
        # the track/world frame.
        cy_cm_world = (H - cy_px) / px_per_cm
        keep.append(BlockRect(
            cx=float(cx_px) / px_per_cm,
            cy=float(cy_cm_world),
            w=float(long_cm),
            h=float(short_cm),
            theta=theta_rad,
        ))
    return keep
