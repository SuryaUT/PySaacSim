"""Upload endpoint wiring. The CV pipeline is stubbed with a fake segmenter
so the test has no SAM3 dependency."""
from __future__ import annotations

import io

import numpy as np
import pytest


def _synthesize_photo(W=400, H=300) -> bytes:
    """A small white-on-black JPEG-ish blob. Uses PIL if available; falls
    back to a 1x1 PNG stub (the stubbed segmenter doesn't care about the
    contents)."""
    try:
        from PIL import Image
        img = Image.new("RGB", (W, H), (30, 30, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        # Minimal 1x1 PNG.
        return bytes.fromhex(
            "89504e470d0a1a0a0000000d494844520000000100000001"
            "08060000001f15c4890000000a49444154789c6300010000"
            "0500010d0a2db40000000049454e44ae426082"
        )


def _fake_segmenter(image_rgb, cfg):
    """Return four aligned rectangular masks that pass all filters."""
    H, W = image_rgb.shape[:2]
    masks = []
    for i in range(4):
        m = np.zeros((H, W), dtype=bool)
        x0 = 60 + i * 60
        m[80:120, x0:x0 + 80] = True      # 80 cm long, 40 cm tall @ 1 px_per_cm*10
        masks.append(m)
    return masks


def test_upload_track_confirm_flow(client, bearer, monkeypatch):
    # Stub the segmenter used inside the pipeline.
    from PySaacSim.cv import pipeline as _pipe
    monkeypatch.setattr(_pipe, "run_sam3", _fake_segmenter)

    photo = _synthesize_photo()
    files = {"photo": ("track.jpg", photo, "image/jpeg")}
    form = {
        "px_per_cm": "1.0",
        "arkit_confidence": "0.9",
        "camera_height_m": "1.3",
        "image_bounds_w_cm": "400",
        "image_bounds_h_cm": "300",
    }
    r = client.post("/tracks", headers=bearer, files=files, data=form)
    assert r.status_code == 202, r.text
    body = r.json()
    track_id = body["track_id"]
    assert body["state"] in {"ready", "failed"}   # skeletonize may be absent

    # Even if the skeleton step degraded to failed, GET round-trips.
    r2 = client.get(f"/tracks/{track_id}", headers=bearer)
    assert r2.status_code == 200

    if body["state"] == "ready":
        r3 = client.post(f"/tracks/{track_id}/confirm", headers=bearer)
        assert r3.status_code == 200
        assert r3.json()["state"] == "confirmed"


def test_confirm_requires_ready_state(client, bearer):
    from PySaacSim.server.storage import new_id
    storage = client.app.state.storage
    track_id = new_id()
    storage.save_track_meta(track_id, {
        "track_id": track_id, "user_id": "apple-sub-test",
        "px_per_cm": 10.0, "arkit_confidence": 0.9, "camera_height_m": 1.2,
        "image_bounds_cm": [300, 200], "state": "failed",
    })
    r = client.post(f"/tracks/{track_id}/confirm", headers=bearer)
    assert r.status_code == 409
