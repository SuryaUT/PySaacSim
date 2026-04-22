"""Top-level orchestration: rectified-photo bytes in → track dict + errors out.

Called by the server's ``/tracks`` handler. Long, mostly I/O — the heavy
lifting lives in the peer modules (``segment``, ``validate``, ``build_track``).
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..sim.constants import CHASSIS_WIDTH_CM
from .build_track import build_track
from .segment import SAM3Unavailable, SegmentConfig, filter_masks, run_sam3
from .validate import check_scale


logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    track: dict[str, Any]
    blocks: list[dict[str, float]]
    preview_png_bytes: bytes
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    sam3_mode: str


def _decode_image(photo_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes → RGB ``(H, W, 3)`` uint8. Prefers OpenCV;
    falls back to PIL so tests that don't need CV can still exercise the
    pipeline wiring."""
    try:
        import cv2
        arr = np.frombuffer(photo_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image
        img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        return np.asarray(img)


def _preview_png(image_rgb: np.ndarray, blocks: list[Any], centerline: list) -> bytes:
    try:
        import cv2
        canvas = image_rgb.copy()
        # Blocks as red rectangles.
        for b in blocks:
            cx_px = b.cx * _LAST_PX_PER_CM[0]
            H = canvas.shape[0]
            cy_px = H - b.cy * _LAST_PX_PER_CM[0]
            size = (b.w * _LAST_PX_PER_CM[0], b.h * _LAST_PX_PER_CM[0])
            rect = ((cx_px, cy_px), size, float(np.degrees(b.theta)))
            pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(canvas, [pts], True, (255, 0, 0), 2)
        # Centerline as a green polyline.
        if centerline:
            H = canvas.shape[0]
            pts = np.array([[p[0] * _LAST_PX_PER_CM[0], H - p[1] * _LAST_PX_PER_CM[0]]
                            for p in centerline], dtype=np.int32)
            if pts.shape[0] > 1:
                cv2.polylines(canvas, [pts], False, (0, 200, 0), 2)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if ok:
            return buf.tobytes()
    except Exception:  # noqa: BLE001
        pass
    # PIL fallback (preview without overlays — sufficient for dev).
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return b""


# Little mutable closure-state so ``_preview_png`` can read the active scale
# without threading another argument through. Kept explicit + local to this
# module; the caller is single-threaded per request.
_LAST_PX_PER_CM: list[float] = [10.0]


def run_cv_pipeline(
    photo_bytes: bytes, *,
    px_per_cm: float,
    image_bounds_cm: tuple[float, float],
    arkit_confidence: float,
    camera_height_m: float,
    sam3_cfg: Optional[SegmentConfig] = None,
    plank_length_cm: float = 80.0,
    plank_lower: float = 0.72,
    plank_upper: float = 1.10,
    min_arkit_confidence: float = 0.5,
    segmenter=run_sam3,
) -> CVResult:
    """Main entrypoint. ``segmenter`` is injectable so tests can feed
    hand-crafted masks in without touching SAM3."""
    _LAST_PX_PER_CM[0] = px_per_cm
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    image_rgb = _decode_image(photo_bytes)
    H, W = image_rgb.shape[:2]
    if H * W < 1_000_000:
        warnings.append({"code": "IMAGE_TOO_SMALL",
                         "message": f"rectified image is {W}×{H} (< 1 MP)"})
    if arkit_confidence < min_arkit_confidence:
        errors.append({"code": "ARKIT_LOW_CONFIDENCE",
                       "message": f"ARKit confidence {arkit_confidence:.2f} "
                                  f"below threshold {min_arkit_confidence:.2f}"})

    cfg = sam3_cfg or SegmentConfig()
    sam3_mode = cfg.mode
    masks: list[np.ndarray] = []
    try:
        masks = segmenter(image_rgb, cfg)
    except SAM3Unavailable as e:
        errors.append({"code": "SAM3_UNAVAILABLE", "message": str(e)})
        return CVResult(track={}, blocks=[], preview_png_bytes=b"",
                        errors=errors, warnings=warnings, sam3_mode=sam3_mode)

    blocks = filter_masks(masks, px_per_cm=px_per_cm)
    scale = check_scale(blocks, plank_length_cm=plank_length_cm,
                        lower=plank_lower, upper=plank_upper)
    if not scale.ok:
        errors.append({
            "code": "SCALE_MISMATCH",
            "message": (f"median plank {scale.observed_cm:.1f} cm vs expected "
                        f"{scale.expected_cm:.1f} cm (ratio "
                        f"{(scale.observed_cm or 0) / scale.expected_cm:.2f})"),
            "expected_cm": scale.expected_cm,
            "observed_cm": scale.observed_cm,
        })
    elif scale.observed_cm is None:
        warnings.append({
            "code": "SCALE_UNCHECKED",
            "message": f"only {scale.n_blocks} block(s) detected; skipping plank-length check",
        })

    build = build_track(
        blocks,
        image_hw_px=(H, W),
        px_per_cm=px_per_cm,
        image_bounds_cm=image_bounds_cm,
        chassis_width_cm=CHASSIS_WIDTH_CM,
    )
    errors.extend({"code": c, "message": m} for c, m in build.errors)
    warnings.extend({"code": c, "message": m} for c, m in build.warnings)

    block_dicts = [
        {"cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h, "theta": b.theta}
        for b in blocks
    ]
    preview = _preview_png(image_rgb, blocks,
                           build.track.get("centerline", []) if build.track else [])
    return CVResult(
        track=build.track, blocks=block_dicts,
        preview_png_bytes=preview,
        errors=errors, warnings=warnings, sam3_mode=sam3_mode,
    )
