"""``/jobs`` endpoints — submit, status, cancel, artifact download, WS."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, status
from fastapi.responses import FileResponse, JSONResponse

from ..auth import User, current_user, current_user_ws
from ..jobs import JobError
from ..schemas import JobSubmitBody, JobStatus


logger = logging.getLogger(__name__)
router = APIRouter()


# Plan §14.10: `minutes` → `total_timesteps` via a lookup table, not a
# linear formula — throughput varies with track complexity.
_MINUTES_TO_STEPS = {
    1: 500_000,
    2: 1_000_000,
    3: 1_500_000,
    4: 2_000_000,
    5: 2_500_000,
    6: 3_000_000,
    7: 3_500_000,
    8: 4_000_000,
    9: 4_500_000,
    10: 5_000_000,
}


@router.post("/jobs/train", response_model=JobStatus, status_code=202)
async def submit_job(
    body: JobSubmitBody, request: Request,
    user: User = Depends(current_user),
) -> JobStatus:
    cfg = request.app.state.config
    storage = request.app.state.storage
    queue = request.app.state.jobs

    meta = storage.track_meta(body.track_id)
    if meta is None:
        raise HTTPException(404, "track not found")
    if meta.get("user_id") != user.sub:
        raise HTTPException(403, "forbidden")
    if meta.get("state") != "confirmed":
        raise HTTPException(409, "track must be confirmed before training")

    if body.total_timesteps is not None:
        total = int(body.total_timesteps)
    elif body.minutes is not None:
        total = _MINUTES_TO_STEPS.get(int(body.minutes),
                                      cfg.training.default_timesteps)
    else:
        total = cfg.training.default_timesteps

    n_envs = int(body.n_envs) if body.n_envs is not None else cfg.training.n_envs
    lr = float(body.learning_rate) if body.learning_rate is not None else 3e-4

    try:
        job = queue.submit(
            user_id=user.sub,
            track_id=body.track_id,
            total_timesteps=total,
            n_envs=n_envs,
            learning_rate=lr,
        )
    except JobError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))

    return _to_status(job)


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, request: Request,
                  user: User = Depends(current_user)) -> JobStatus:
    queue = request.app.state.jobs
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.user_id != user.sub:
        raise HTTPException(403, "forbidden")
    return _to_status(job)


@router.delete("/jobs/{job_id}", status_code=200)
async def cancel_job(job_id: str, request: Request,
                     user: User = Depends(current_user)) -> dict[str, Any]:
    queue = request.app.state.jobs
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.user_id != user.sub:
        raise HTTPException(403, "forbidden")
    ok = await queue.cancel(job_id)
    return {"cancelled": ok, "state": queue.get(job_id).state}


@router.get("/jobs/{job_id}/artifact")
async def get_artifact(job_id: str, request: Request,
                       user: User = Depends(current_user)) -> FileResponse:
    storage = request.app.state.storage
    queue = request.app.state.jobs
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.user_id != user.sub:
        raise HTTPException(403, "forbidden")
    p = storage.job_dir(job_id) / "policy.npz"
    if not p.exists():
        raise HTTPException(404, "artifact not yet available")
    return FileResponse(p, media_type="application/octet-stream",
                        filename="policy.npz")


@router.get("/jobs/{job_id}/artifact.h")
async def get_artifact_h(job_id: str, request: Request,
                         user: User = Depends(current_user)) -> FileResponse:
    storage = request.app.state.storage
    queue = request.app.state.jobs
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.user_id != user.sub:
        raise HTTPException(403, "forbidden")
    p = storage.job_dir(job_id) / "policy.h"
    if not p.exists():
        raise HTTPException(404, "artifact not yet available")
    return FileResponse(p, media_type="text/x-c-header",
                        filename="policy.h")


@router.websocket("/jobs/{job_id}/events")
async def job_events_ws(
    websocket: WebSocket, job_id: str,
    token: str = Query(..., description="app JWT"),
) -> None:
    try:
        user = await current_user_ws(token, app=websocket.app)
    except HTTPException:
        await websocket.close(code=4401)
        return
    queue = websocket.app.state.jobs
    job = queue.get(job_id)
    if job is None:
        await websocket.close(code=4404)
        return
    if job.user_id != user.sub:
        await websocket.close(code=4403)
        return
    await websocket.app.state.ws_hub.serve_job(job_id, websocket)


def _to_status(job) -> JobStatus:
    return JobStatus(
        job_id=job.job_id, user_id=job.user_id, track_id=job.track_id,
        state=job.state,
        created_at=job.created_at, started_at=job.started_at, ended_at=job.ended_at,
        total_timesteps=job.total_timesteps,
        last_progress=job.last_progress, eval=job.eval, error=job.error,
    )
