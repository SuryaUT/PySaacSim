"""Turn ``BlockRect``s into a track dict (plan §6.3).

Each block contributes its four world-frame edges as ``sim.geometry.Segment``.
The drivable interior is skeletonized to produce a centerline."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..sim.geometry import Segment, Vec2
from .validate import BlockRect


@dataclass
class BuildResult:
    track: dict[str, Any]
    errors: list[tuple[str, str]]
    warnings: list[tuple[str, str]]


def block_to_segments(b: BlockRect) -> list[Segment]:
    hw, hh = b.w / 2.0, b.h / 2.0
    c, s = math.cos(b.theta), math.sin(b.theta)
    corners = []
    for (x, y) in [(+hw, +hh), (+hw, -hh), (-hw, -hh), (-hw, +hh)]:
        wx = b.cx + x * c - y * s
        wy = b.cy + x * s + y * c
        corners.append(Vec2(wx, wy))
    return [
        Segment(corners[0], corners[1]),
        Segment(corners[1], corners[2]),
        Segment(corners[2], corners[3]),
        Segment(corners[3], corners[0]),
    ]


def _skeletonize_to_centerline(
    drivable: np.ndarray, *, px_per_cm: float,
) -> np.ndarray:
    """Return an ordered (K, 2) centerline in cm, walking the largest
    skeleton component. Returns ``np.empty((0, 2))`` on failure."""
    try:
        from skimage.morphology import skeletonize
        from scipy.signal import savgol_filter
    except Exception:  # noqa: BLE001
        # Dependencies missing; let the caller record a warning and move on.
        return np.empty((0, 2), dtype=np.float32)

    sk = skeletonize(drivable.astype(bool))
    ys, xs = np.where(sk)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Order by nearest-neighbor walk from the point closest to the image
    # centroid — crude but adequate for the unit tests; the iOS app's
    # confirmation screen lets the user reject bad centerlines regardless.
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    d0 = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    order = [int(np.argmin(d0))]
    remaining = set(range(len(pts))) - {order[0]}
    while remaining:
        last = pts[order[-1]]
        rem_idx = np.fromiter(remaining, dtype=np.int64)
        rem_pts = pts[rem_idx]
        d = np.hypot(rem_pts[:, 0] - last[0], rem_pts[:, 1] - last[1])
        nxt = int(rem_idx[int(np.argmin(d))])
        # Stop if the next hop is huge — suggests multiple components.
        if np.min(d) > 6.0:
            break
        order.append(nxt)
        remaining.discard(nxt)

    ordered = pts[order]
    if ordered.shape[0] > 21:
        w = min(21, ordered.shape[0] if ordered.shape[0] % 2 == 1 else ordered.shape[0] - 1)
        ordered = np.stack([
            savgol_filter(ordered[:, 0], w, 3, mode="interp"),
            savgol_filter(ordered[:, 1], w, 3, mode="interp"),
        ], axis=1)

    # Resample to 200 equispaced points along cumulative arc length.
    diffs = np.diff(ordered, axis=0)
    segs = np.hypot(diffs[:, 0], diffs[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(segs)])
    if arc[-1] < 1e-3:
        return np.empty((0, 2), dtype=np.float32)
    u = np.linspace(0.0, arc[-1], 200)
    x_s = np.interp(u, arc, ordered[:, 0])
    y_s = np.interp(u, arc, ordered[:, 1])
    # Pixels → cm; flip y to world frame (plan uses bottom-up cm).
    H = drivable.shape[0]
    cm = np.stack([x_s / px_per_cm, (H - y_s) / px_per_cm], axis=1).astype(np.float32)
    return cm


def build_track(
    blocks: list[BlockRect], *,
    image_hw_px: tuple[int, int],
    px_per_cm: float,
    image_bounds_cm: tuple[float, float],
    chassis_width_cm: float,
    lane_width_cm: float = 81.28,
) -> BuildResult:
    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []

    if len(blocks) < 3:
        errors.append(("TRACK_TOO_SPARSE",
                       f"Need ≥ 3 blocks; got {len(blocks)}"))
        return BuildResult(track={}, errors=errors, warnings=warnings)

    walls: list[Segment] = []
    for b in blocks:
        walls.extend(block_to_segments(b))

    # Build a drivable mask: full rectified extent minus blocks dilated by
    # chassis_width / 2. Numpy-only so we don't *require* OpenCV here, but
    # dilation via cv2 is cheaper — fall back if unavailable.
    H, W = image_hw_px
    drivable = np.ones((H, W), dtype=np.uint8)
    try:
        import cv2
        block_mask = np.zeros((H, W), dtype=np.uint8)
        for b in blocks:
            cx_px = b.cx * px_per_cm
            cy_px = H - b.cy * px_per_cm
            size = (b.w * px_per_cm, b.h * px_per_cm)
            rect = ((cx_px, cy_px), size, math.degrees(b.theta))
            pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(block_mask, [pts], 1)
        k = max(1, int(round(chassis_width_cm * 0.5 * px_per_cm)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        block_mask = cv2.dilate(block_mask, kernel)
        drivable[block_mask > 0] = 0
    except Exception:  # noqa: BLE001
        warnings.append(("CV_SLOW_PATH",
                         "OpenCV not available; skipping dilation."))

    centerline = _skeletonize_to_centerline(drivable, px_per_cm=px_per_cm)

    if centerline.shape[0] == 0:
        errors.append(("TRACK_DISCONNECTED",
                       "Could not extract a centerline skeleton"))
        spawn = {"x": 0.0, "y": 0.0, "theta": 0.0}
    else:
        first = centerline[0]
        nxt = centerline[min(1, centerline.shape[0] - 1)]
        theta = float(math.atan2(nxt[1] - first[1], nxt[0] - first[0]))
        spawn = {"x": float(first[0]), "y": float(first[1]), "theta": theta}

    w_cm, h_cm = image_bounds_cm
    bounds = {"min_x": 0.0, "min_y": 0.0, "max_x": float(w_cm), "max_y": float(h_cm)}

    track = {
        "walls": walls,
        "centerline": [[float(x), float(y)] for x, y in centerline.tolist()],
        "spawn": spawn,
        "lane_width_cm": float(lane_width_cm),
        "bounds": bounds,
    }
    return BuildResult(track=track, errors=errors, warnings=warnings)
