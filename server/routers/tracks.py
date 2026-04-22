"""``/tracks`` endpoints — photo upload, CV pipeline trigger, get, patch,
confirm. Photo upload is multipart with a handful of metadata fields the
iOS app fills in from ARKit."""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Request, Response,
    UploadFile, status,
)
from fastapi.responses import FileResponse

from ...cv.pipeline import run_cv_pipeline
from ...cv.segment import SegmentConfig
from ...cv.validate import BlockRect as CVBlockRect
from ...cv.build_track import build_track
from ..auth import User, current_user
from ..schemas import (
    BlockRect, CVError, TrackPatchBody, TrackResponse,
)
from ..storage import new_id


logger = logging.getLogger(__name__)
router = APIRouter()


# 25 MB upload cap (plan §7.9b). FastAPI/Starlette streams the body; we
# enforce the cap by reading .read() into memory with a ceiling.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


async def _read_capped(upload: UploadFile, cap: int) -> bytes:
    data = await upload.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"photo exceeds {cap} bytes")
    return data


@router.post("/tracks", response_model=TrackResponse, status_code=202)
async def upload_track(
    request: Request,
    photo: UploadFile = File(...),
    px_per_cm: float = Form(...),
    arkit_confidence: float = Form(...),
    camera_height_m: float = Form(...),
    image_bounds_w_cm: float = Form(...),
    image_bounds_h_cm: float = Form(...),
    user: User = Depends(current_user),
) -> TrackResponse:
    cfg = request.app.state.config
    storage = request.app.state.storage

    photo_bytes = await _read_capped(photo, MAX_UPLOAD_BYTES)

    track_id = new_id()
    tdir = storage.track_dir(track_id)
    (tdir / "photo.jpg").write_bytes(photo_bytes)

    meta: dict[str, Any] = {
        "track_id": track_id,
        "user_id": user.sub,
        "px_per_cm": float(px_per_cm),
        "arkit_confidence": float(arkit_confidence),
        "camera_height_m": float(camera_height_m),
        "image_bounds_cm": [float(image_bounds_w_cm), float(image_bounds_h_cm)],
        "state": "pending",
    }
    storage.save_track_meta(track_id, meta)

    sam3_cfg = SegmentConfig(
        score_threshold=cfg.cv.sam3_score_threshold,
        text_prompts=tuple(cfg.cv.sam3_text_prompts),
        weights_path=cfg.cv.sam3_weights,
        mode=cfg.cv.sam3_mode,
    )

    # CV is GPU-heavy; run it in a thread so the event loop stays free.
    def _run() -> Any:
        return run_cv_pipeline(
            photo_bytes,
            px_per_cm=px_per_cm,
            image_bounds_cm=(image_bounds_w_cm, image_bounds_h_cm),
            arkit_confidence=arkit_confidence,
            camera_height_m=camera_height_m,
            sam3_cfg=sam3_cfg,
            plank_length_cm=cfg.cv.plank_length_cm,
            plank_lower=cfg.cv.plank_tolerance.lower,
            plank_upper=cfg.cv.plank_tolerance.upper,
            min_arkit_confidence=cfg.cv.min_arkit_confidence,
        )

    try:
        result = await asyncio.to_thread(_run)
    except Exception as e:  # noqa: BLE001
        logger.exception("CV pipeline crashed for track %s", track_id)
        meta["state"] = "failed"
        meta["errors"] = [{"code": "CV_CRASH", "message": str(e)}]
        storage.save_track_meta(track_id, meta)
        return TrackResponse(track_id=track_id, state="failed",
                             errors=[CVError(code="CV_CRASH", message=str(e))])

    # Persist CV outputs.
    if result.preview_png_bytes:
        (tdir / "preview.png").write_bytes(result.preview_png_bytes)
    meta["blocks"] = result.blocks
    meta["errors"] = result.errors
    meta["warnings"] = result.warnings
    meta["sam3_mode"] = result.sam3_mode

    has_fatal = any(e for e in result.errors)
    if result.track:
        storage.save_track_json(track_id, _track_to_disk(result.track))
    meta["state"] = "failed" if has_fatal else "ready"
    storage.save_track_meta(track_id, meta)

    return _response_from_meta(track_id, meta, result.track)


@router.get("/tracks/{track_id}", response_model=TrackResponse)
async def get_track(track_id: str, request: Request,
                    user: User = Depends(current_user)) -> TrackResponse:
    storage = request.app.state.storage
    meta = storage.track_meta(track_id)
    if meta is None:
        raise HTTPException(404, "track not found")
    if meta.get("user_id") != user.sub:
        raise HTTPException(403, "forbidden")
    track = storage.load_track_json(track_id) or {}
    return _response_from_meta(track_id, meta, track)


@router.patch("/tracks/{track_id}", response_model=TrackResponse)
async def patch_track(
    track_id: str, body: TrackPatchBody, request: Request,
    user: User = Depends(current_user),
) -> TrackResponse:
    storage = request.app.state.storage
    meta = storage.track_meta(track_id)
    if meta is None:
        raise HTTPException(404, "track not found")
    if meta.get("user_id") != user.sub:
        raise HTTPException(403, "forbidden")
    if meta.get("state") == "confirmed":
        raise HTTPException(409, "track already confirmed")

    # Rebuild from user-edited blocks.
    px_per_cm = float(meta["px_per_cm"])
    w_cm, h_cm = meta["image_bounds_cm"]
    H_px = max(1, int(round(float(h_cm) * px_per_cm)))
    W_px = max(1, int(round(float(w_cm) * px_per_cm)))
    blocks = [CVBlockRect(cx=b.cx, cy=b.cy, w=b.w, h=b.h, theta=b.theta)
              for b in body.blocks]
    from ...sim.constants import CHASSIS_WIDTH_CM
    build = build_track(
        blocks, image_hw_px=(H_px, W_px), px_per_cm=px_per_cm,
        image_bounds_cm=(float(w_cm), float(h_cm)),
        chassis_width_cm=CHASSIS_WIDTH_CM,
    )
    meta["blocks"] = [
        {"cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h, "theta": b.theta}
        for b in blocks
    ]
    meta["errors"] = [{"code": c, "message": m} for c, m in build.errors]
    meta["warnings"] = [{"code": c, "message": m} for c, m in build.warnings]
    if build.track:
        storage.save_track_json(track_id, _track_to_disk(build.track))
        meta["state"] = "ready"
    else:
        meta["state"] = "failed"
    storage.save_track_meta(track_id, meta)
    return _response_from_meta(track_id, meta, build.track)


@router.post("/tracks/{track_id}/confirm", response_model=TrackResponse)
async def confirm_track(track_id: str, request: Request,
                        user: User = Depends(current_user)) -> TrackResponse:
    storage = request.app.state.storage
    meta = storage.track_meta(track_id)
    if meta is None:
        raise HTTPException(404, "track not found")
    if meta.get("user_id") != user.sub:
        raise HTTPException(403, "forbidden")
    if meta.get("state") != "ready":
        raise HTTPException(409,
                            f"can only confirm a 'ready' track; state is "
                            f"{meta.get('state')!r}")
    meta["state"] = "confirmed"
    storage.save_track_meta(track_id, meta)
    track = storage.load_track_json(track_id) or {}
    return _response_from_meta(track_id, meta, track)


@router.get("/tracks/{track_id}/preview.png")
async def get_preview(track_id: str, request: Request,
                      user: User = Depends(current_user)) -> Response:
    storage = request.app.state.storage
    meta = storage.track_meta(track_id)
    if meta is None:
        raise HTTPException(404, "track not found")
    if meta.get("user_id") != user.sub:
        raise HTTPException(403, "forbidden")
    p = storage.track_dir(track_id) / "preview.png"
    if not p.exists():
        raise HTTPException(404, "no preview available")
    return FileResponse(p, media_type="image/png")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _track_to_disk(track: dict[str, Any]) -> dict[str, Any]:
    """Serialize the CV-pipeline track dict (Segments as ``Segment`` named
    tuples) for JSON persistence. ``walls`` become nested lists; everything
    else is already plain."""
    out = dict(track)
    walls = track.get("walls")
    if walls:
        out["walls"] = [[[w.a.x, w.a.y], [w.b.x, w.b.y]] for w in walls]
    return out


def _response_from_meta(
    track_id: str, meta: dict[str, Any], track: dict[str, Any],
) -> TrackResponse:
    def _cv_err(d: dict[str, Any]) -> CVError:
        return CVError(
            code=str(d.get("code", "")),
            message=str(d.get("message", "")),
            expected_cm=d.get("expected_cm"),
            observed_cm=d.get("observed_cm"),
        )
    blocks = [BlockRect(**b) for b in (meta.get("blocks") or [])]
    centerline = track.get("centerline") if track else None
    return TrackResponse(
        track_id=track_id,
        state=meta.get("state", "pending"),
        blocks=blocks or None,
        centerline=[tuple(p) for p in centerline] if centerline else None,
        spawn=track.get("spawn") if track else None,
        lane_width_cm=track.get("lane_width_cm") if track else None,
        bounds=track.get("bounds") if track else None,
        preview_url=f"/tracks/{track_id}/preview.png",
        errors=[_cv_err(e) for e in (meta.get("errors") or [])],
        warnings=[_cv_err(w) for w in (meta.get("warnings") or [])],
    )
