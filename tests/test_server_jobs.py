"""Job submission + state transitions, with the training subprocess stubbed
out so the test never imports torch or SB3."""
from __future__ import annotations

import asyncio
import json
import time

import pytest


def _seed_confirmed_track(client, bearer):
    """Write a minimal confirmed track straight to disk — bypasses the CV
    pipeline so this test has zero CV dependencies."""
    from PySaacSim.server.storage import new_id

    storage = client.app.state.storage
    track_id = new_id()
    storage.save_track_meta(track_id, {
        "track_id": track_id, "user_id": "apple-sub-test",
        "px_per_cm": 10.0, "arkit_confidence": 0.9, "camera_height_m": 1.2,
        "image_bounds_cm": [300, 200], "state": "confirmed",
    })
    storage.save_track_json(track_id, {
        "walls": [[[0, 0], [100, 0]], [[100, 0], [100, 100]]],
        "centerline": [[10.0, 10.0], [20.0, 10.0]],
        "spawn": {"x": 10.0, "y": 10.0, "theta": 0.0},
        "lane_width_cm": 81.28,
        "bounds": {"min_x": 0, "min_y": 0, "max_x": 300, "max_y": 200},
    })
    return track_id


def test_submit_job_and_state_transitions(client, bearer, monkeypatch):
    track_id = _seed_confirmed_track(client, bearer)

    # Stub the queue's _run_one so the job transitions queued→running→done
    # without spawning a real subprocess.
    queue = client.app.state.jobs

    async def _fake_run(job):
        from PySaacSim.server.storage import Storage
        job.state = "running"
        queue._persist(job)
        await queue._ws.broadcast(job.job_id, {"kind": "state", "state": "running"})
        await queue._ws.broadcast(job.job_id, {
            "kind": "progress", "step": 100, "mean_reward": 1.5,
            "fps": 100.0, "ts": "2026-04-22T00:00:00Z",
        })
        queue._storage.append_progress(job.job_id, {
            "kind": "progress", "step": 100, "mean_reward": 1.5, "fps": 100.0,
            "ts": "2026-04-22T00:00:00Z",
        })
        job.last_progress = {"step": 100, "mean_reward": 1.5}
        job.eval = {"completion_rate": 0.9, "collision_rate": 0.0,
                    "mean_reward": 1.5, "mean_lap_time_s": 20.0}
        job.state = "done"
        job.ended_at = int(time.time())
        queue._persist(job)
        await queue._emit_terminal(job)
        queue._user_running.pop(job.user_id, None)
        # Write a fake artifact file so GET /jobs/{id}/artifact works.
        (queue._storage.job_dir(job.job_id) / "policy.npz").write_bytes(b"test")

    monkeypatch.setattr(queue, "_run_one", _fake_run)

    r = client.post("/jobs/train",
                    json={"track_id": track_id, "minutes": 2, "n_envs": 2},
                    headers=bearer)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    # Wait briefly for the async worker task to drain. Bounded so the test
    # never hangs.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        r2 = client.get(f"/jobs/{job_id}", headers=bearer)
        if r2.status_code == 200 and r2.json()["state"] == "done":
            break
        time.sleep(0.05)
    r2 = client.get(f"/jobs/{job_id}", headers=bearer)
    assert r2.status_code == 200
    body = r2.json()
    assert body["state"] == "done"
    assert body["eval"]["completion_rate"] == 0.9

    # Second concurrent submit for same user → 409 (job already done, so no
    # longer running). Actually plan §7.4 says only running jobs block new
    # submits, so a new one should succeed. Verify by submitting again.
    r3 = client.post("/jobs/train",
                     json={"track_id": track_id, "minutes": 1},
                     headers=bearer)
    assert r3.status_code == 202

    # Artifact download.
    r4 = client.get(f"/jobs/{job_id}/artifact", headers=bearer)
    assert r4.status_code == 200
    assert r4.content == b"test"


def test_unconfirmed_track_rejected(client, bearer, monkeypatch):
    from PySaacSim.server.storage import new_id
    storage = client.app.state.storage
    track_id = new_id()
    storage.save_track_meta(track_id, {
        "track_id": track_id, "user_id": "apple-sub-test",
        "px_per_cm": 10.0, "arkit_confidence": 0.9, "camera_height_m": 1.2,
        "image_bounds_cm": [300, 200], "state": "ready",
    })

    r = client.post("/jobs/train",
                    json={"track_id": track_id, "minutes": 1},
                    headers=bearer)
    assert r.status_code == 409
