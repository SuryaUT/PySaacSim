"""Smoke test for the filesystem layout."""
from __future__ import annotations

from pathlib import Path


def test_storage_roundtrip(tmp_data_dir: Path):
    from PySaacSim.server.storage import Storage, new_id

    s = Storage(root=tmp_data_dir)
    sub = "apple-sub-xyz"
    s.upsert_user(sub, email="x@example.com")
    s.add_apns_token(sub, "token-1")
    s.add_apns_token(sub, "token-1")   # idempotent
    s.add_apns_token(sub, "token-2")

    rec = s.load_user(sub)
    assert rec is not None
    assert rec["email"] == "x@example.com"
    assert sorted(rec["apns_tokens"]) == ["token-1", "token-2"]

    tid = new_id()
    s.save_track_meta(tid, {"track_id": tid, "user_id": sub, "state": "ready"})
    assert s.track_meta(tid)["state"] == "ready"

    jid = new_id()
    s.save_job_state(jid, {"job_id": jid, "state": "queued"})
    s.append_progress(jid, {"kind": "progress", "step": 10})
    s.append_progress(jid, {"kind": "progress", "step": 20})
    rows = list(s.read_progress(jid))
    assert [r["step"] for r in rows] == [10, 20]
