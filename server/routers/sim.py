"""Optional headless-sim endpoints (plan §7.11).

Available even without the dashboard static files — a developer can drive
the sim with ``POST /sim/command`` and consume ``WS /sim/events`` directly."""
from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, WebSocket

from ..auth import User, current_user, current_user_ws


router = APIRouter()


@router.get("/sim/state")
async def sim_state(request: Request,
                    user: User = Depends(current_user)) -> dict[str, Any]:
    return request.app.state.sim_runner.snapshot()


@router.post("/sim/command", status_code=204)
async def sim_command(
    request: Request,
    body: dict[str, Any] = Body(...),
    user: User = Depends(current_user),
) -> None:
    throttle = float(body.get("throttle", 0.0))
    steer = float(body.get("steer", 0.0))
    await request.app.state.sim_runner.set_command(throttle, steer)


@router.post("/sim/reset", status_code=204)
async def sim_reset(
    request: Request,
    body: dict[str, Any] = Body(default_factory=dict),
    user: User = Depends(current_user),
) -> None:
    track_id: Optional[str] = body.get("track_id")
    if track_id:
        storage = request.app.state.storage
        meta = storage.track_meta(track_id)
        if meta is None or meta.get("user_id") != user.sub:
            raise HTTPException(404, "track not found")
        if meta.get("state") != "confirmed":
            raise HTTPException(409, "track must be confirmed")
        track = storage.load_track_json(track_id)
        if not track:
            raise HTTPException(404, "track geometry missing")
        await request.app.state.sim_runner.load_track(track)
    else:
        await request.app.state.sim_runner.reset()


@router.post("/sim/policy", status_code=204)
async def load_policy(
    request: Request,
    body: dict[str, Any] = Body(...),
    user: User = Depends(current_user),
) -> None:
    """Load a finished job's policy weights into the running sim."""
    job_id: Optional[str] = body.get("job_id")
    if not job_id:
        raise HTTPException(400, "job_id required")
    queue = request.app.state.jobs
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.user_id != user.sub:
        raise HTTPException(403, "forbidden")
    if job.state != "done":
        raise HTTPException(409, f"job is not done (state={job.state!r})")
    policy_path = str(request.app.state.storage.job_dir(job_id) / "policy.zip")
    await request.app.state.sim_runner.load_policy(policy_path)


@router.delete("/sim/policy", status_code=204)
async def unload_policy(
    request: Request,
    user: User = Depends(current_user),
) -> None:
    await request.app.state.sim_runner.unload_policy()


@router.websocket("/sim/events")
async def sim_events(websocket: WebSocket,
                     token: str = Query(...)) -> None:
    try:
        await current_user_ws(token, app=websocket.app)
    except HTTPException:
        await websocket.close(code=4401)
        return
    hub = websocket.app.state.sim_ws_hub
    await websocket.accept()
    await hub.add(websocket)
    try:
        # Keep the socket open; the sim runner broadcasts; we only need to
        # detect client disconnect.
        while True:
            try:
                await websocket.receive_text()
            except Exception:  # noqa: BLE001
                break
    finally:
        await hub.remove(websocket)
