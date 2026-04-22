"""FastAPI entrypoint.

Run:
    uvicorn PySaacSim.server.app:app --host 127.0.0.1 --port 8787

Do the multiprocessing start-method pin *before* any heavy imports (plan
§7.9a). Torch is fork-hostile; SB3's SubprocVecEnv nested inside our own
child process inherits from the parent, so ``spawn`` is the only safe mode."""
from __future__ import annotations

import multiprocessing as mp
# Must happen before torch / cuda / any heavy imports.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import logging
import os
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .apns import APNsClient
from .config import ServerConfig
from .jobs import JobQueue
from .storage import Storage
from .sim_runner import SimRunner
from .ws import SimWSHub, WSHub


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg: ServerConfig = ServerConfig.load()
    app.state.config = cfg

    storage = Storage(root=cfg.data_dir)
    app.state.storage = storage

    ws_hub = WSHub(storage=storage)
    app.state.ws_hub = ws_hub

    sim_ws_hub = SimWSHub()
    app.state.sim_ws_hub = sim_ws_hub

    apns = APNsClient(cfg.apns)
    app.state.apns = apns

    async def _on_job_done(job) -> None:
        """Plan §7.7: APNs on every terminal transition, best-effort."""
        if job.state not in {"done", "failed", "cancelled"}:
            return
        title = {
            "done": "Training complete",
            "failed": "Training failed",
            "cancelled": "Training cancelled",
        }[job.state]
        body = f"Job {job.job_id[:8]}: {job.state}"
        try:
            await apns.push_to_user(storage, job.user_id,
                                    title=title, body=body,
                                    extra={"job_id": job.job_id})
        except Exception:  # noqa: BLE001
            logger.exception("APNs push failed for job %s", job.job_id)

    queue = JobQueue(storage=storage, ws_hub=ws_hub, cfg=cfg,
                     on_job_finished=_on_job_done)
    app.state.jobs = queue
    queue.start()

    sim_runner = SimRunner(sim_ws_hub)
    app.state.sim_runner = sim_runner
    sim_runner.start()

    logger.info("PysaacRC server ready. data_dir=%s", cfg.data_dir)

    try:
        yield
    finally:
        await queue.stop()
        await sim_runner.stop()
        await apns.aclose()


app = FastAPI(title="PysaacRC", version="0.1.0", lifespan=lifespan)

# Permissive CORS for local dev and the future dashboard. In production the
# Cloudflare tunnel in front of the server handles origin gating too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers.
from .routers.auth import router as auth_router            # noqa: E402
from .routers.jobs import router as jobs_router            # noqa: E402
from .routers.sim import router as sim_router              # noqa: E402
from .routers.tracks import router as tracks_router        # noqa: E402
from .routers.gui_state import router as gui_router        # noqa: E402

app.include_router(auth_router)
app.include_router(tracks_router)
app.include_router(jobs_router)
app.include_router(sim_router)
app.include_router(gui_router)


# Optional: serve the dashboard if static files exist. Never fails boot.
_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_static_dir), html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui", status_code=302)


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    cfg: ServerConfig = app.state.config
    cuda: object
    try:
        import torch
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:  # noqa: BLE001
        cuda = {"available": False, "device": None, "note": "torch not importable"}
    return {
        "ok": True,
        "data_dir": str(cfg.data_dir),
        "cuda": cuda,
    }
