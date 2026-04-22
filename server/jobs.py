"""Training job queue.

Plan §7.4:
  - In-process, one worker at a time (one GPU).
  - Training runs in a dedicated ``multiprocessing.Process`` (``spawn`` start
    method) so the event loop stays responsive and we can terminate on cancel.
  - Parent ↔ child: ``mp.Queue`` for progress rows, ``mp.Event`` for stop.
  - States: queued → running → done | failed | cancelled.

The child entrypoint is ``training.finetune.finetune_child`` — it knows how
to re-initialize torch, import SB3, and stream progress rows. This module is
transport; it doesn't know the learning details."""
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import ServerConfig
from .storage import Storage, new_id
from .ws import WSHub


logger = logging.getLogger(__name__)


# The training entrypoint is imported lazily inside the child process so the
# parent doesn't need torch / SB3 at import time (lets the server boot even
# when those aren't installed, useful for tests).


@dataclass
class Job:
    job_id: str
    user_id: str
    track_id: str
    total_timesteps: int
    n_envs: int
    learning_rate: float
    state: str = "queued"                 # plan §7.4 state machine
    created_at: int = field(default_factory=lambda: int(time.time()))
    started_at: Optional[int] = None
    ended_at: Optional[int] = None
    last_progress: Optional[dict[str, Any]] = None
    eval: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "track_id": self.track_id,
            "total_timesteps": self.total_timesteps,
            "n_envs": self.n_envs,
            "learning_rate": self.learning_rate,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "last_progress": self.last_progress,
            "eval": self.eval,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Job":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


class JobError(Exception):
    pass


class JobQueue:
    """Async-driven single-worker queue. Owns the training subprocess."""

    def __init__(
        self,
        storage: Storage,
        ws_hub: WSHub,
        cfg: ServerConfig,
        *,
        on_job_finished=None,        # async callable(job: Job)
    ) -> None:
        self._storage = storage
        self._ws = ws_hub
        self._cfg = cfg
        self._on_finished = on_job_finished

        self._jobs: dict[str, Job] = {}
        self._user_running: dict[str, str] = {}   # user_id -> job_id
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None

        self._current_job: Optional[str] = None
        self._current_proc: Optional[mp.Process] = None
        self._current_stop_evt: Optional[Any] = None   # mp.Event

    # ---------- lifecycle --------------------------------------------------

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        # Kill any in-flight training.
        if self._current_stop_evt is not None:
            self._current_stop_evt.set()
        if self._current_proc is not None and self._current_proc.is_alive():
            self._current_proc.terminate()
            self._current_proc.join(timeout=5)

    # ---------- public API -------------------------------------------------

    def submit(
        self, user_id: str, track_id: str, *,
        total_timesteps: int,
        n_envs: int,
        learning_rate: float,
    ) -> Job:
        # Plan §7.4: at most one running job per user. Queued jobs fine.
        running = self._user_running.get(user_id)
        if running and self._jobs[running].state in {"queued", "running"}:
            raise JobError("User already has a job in flight")
        job = Job(
            job_id=new_id(),
            user_id=user_id,
            track_id=track_id,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            learning_rate=learning_rate,
        )
        self._jobs[job.job_id] = job
        self._user_running[user_id] = job.job_id
        self._storage.save_job_state(job.job_id, job.to_dict())
        self._queue.put_nowait(job.job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        if job_id in self._jobs:
            return self._jobs[job_id]
        disk = self._storage.load_job_state(job_id)
        if disk is None:
            return None
        job = Job.from_dict(disk)
        self._jobs[job_id] = job
        return job

    async def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        if job.state == "queued":
            job.state = "cancelled"
            job.ended_at = int(time.time())
            self._persist(job)
            await self._ws.broadcast(job_id, {"kind": "state", "state": "cancelled"})
            self._user_running.pop(job.user_id, None)
            return True
        if job.state == "running" and self._current_job == job_id:
            if self._current_stop_evt is not None:
                self._current_stop_evt.set()
            if self._current_proc is not None and self._current_proc.is_alive():
                self._current_proc.terminate()
            return True
        return False

    # ---------- worker loop ------------------------------------------------

    async def _run_forever(self) -> None:
        while True:
            try:
                job_id = await self._queue.get()
                job = self.get(job_id)
                if job is None or job.state != "queued":
                    continue
                await self._run_one(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("Job worker loop error")

    async def _run_one(self, job: Job) -> None:
        job.state = "running"
        job.started_at = int(time.time())
        self._persist(job)
        await self._ws.broadcast(job.job_id, {"kind": "state", "state": "running"})

        # Build IPC channels. We use spawn start method for torch safety
        # (plan §7.9a). Parent only *creates* context; the child re-imports
        # torch on its own.
        ctx = mp.get_context("spawn")
        progress_q: Any = ctx.Queue(maxsize=1024)
        stop_evt = ctx.Event()
        self._current_stop_evt = stop_evt

        track = self._storage.load_track_json(job.track_id)
        if track is None:
            await self._fail(job, "TRACK_NOT_FOUND", "Track has no confirmed geometry")
            return

        child_args = dict(
            job_id=job.job_id,
            out_dir=str(self._storage.job_dir(job.job_id)),
            track=track,
            base_ckpt_path=self._cfg.training.base_policy_path,
            total_timesteps=job.total_timesteps,
            n_envs=job.n_envs,
            learning_rate=job.learning_rate,
            progress_q=progress_q,
            stop_evt=stop_evt,
        )
        proc = ctx.Process(
            target=_child_entrypoint,
            kwargs=child_args,
            name=f"pysaac-train-{job.job_id[:8]}",
        )
        self._current_job = job.job_id
        self._current_proc = proc
        proc.start()

        # Drain the progress queue. We poll asynchronously — mp.Queue doesn't
        # have an async API, so run in the default executor.
        try:
            while proc.is_alive() or not progress_q.empty():
                row = await asyncio.get_event_loop().run_in_executor(
                    None, _q_get_nowait, progress_q,
                )
                if row is None:
                    await asyncio.sleep(0.1)
                    continue
                await self._handle_progress_row(job, row)
            proc.join()
        finally:
            self._current_proc = None
            self._current_stop_evt = None
            self._current_job = None

        # Decide terminal state from the child's exit code + last message.
        if stop_evt.is_set() and job.state == "running":
            job.state = "cancelled"
        elif job.state == "running":
            # No explicit error, no cancel, process exited cleanly: done.
            job.state = "done" if proc.exitcode == 0 else "failed"
            if job.state == "failed" and not job.error:
                job.error = f"trainer exited with code {proc.exitcode}"
        job.ended_at = int(time.time())
        self._persist(job)
        await self._emit_terminal(job)

        self._user_running.pop(job.user_id, None)
        if self._on_finished:
            try:
                await self._on_finished(job)
            except Exception:  # noqa: BLE001
                logger.exception("on_job_finished hook failed")

    async def _handle_progress_row(self, job: Job, row: dict[str, Any]) -> None:
        kind = row.get("kind")
        if kind == "progress":
            job.last_progress = row
            self._storage.append_progress(job.job_id, row)
        elif kind == "eval":
            job.eval = row.get("eval")
        elif kind == "error":
            job.state = "failed"
            job.error = row.get("message") or row.get("code") or "unknown"
        elif kind == "done":
            # Child signals done before exit so we capture eval + paths.
            job.eval = row.get("eval") or job.eval
        self._persist(job)
        # Fan out to WS subscribers.
        await self._ws.broadcast(job.job_id, row)

    async def _fail(self, job: Job, code: str, message: str) -> None:
        job.state = "failed"
        job.error = f"{code}: {message}"
        job.ended_at = int(time.time())
        self._persist(job)
        await self._ws.broadcast(job.job_id,
                                 {"kind": "error", "code": code, "message": message})
        await self._emit_terminal(job)
        self._user_running.pop(job.user_id, None)

    async def _emit_terminal(self, job: Job) -> None:
        if job.state == "done":
            await self._ws.broadcast(job.job_id, {
                "kind": "done",
                "artifact_url": f"/jobs/{job.job_id}/artifact",
                "eval": job.eval or {},
            })
        else:
            await self._ws.broadcast(job.job_id,
                                     {"kind": "state", "state": job.state})

    def _persist(self, job: Job) -> None:
        self._storage.save_job_state(job.job_id, job.to_dict())


def _q_get_nowait(q: Any) -> Optional[dict[str, Any]]:
    try:
        return q.get_nowait()
    except Exception:  # queue.Empty in child process context
        return None


# --------------------------------------------------------------------------
# Child process entrypoint
# --------------------------------------------------------------------------

def _child_entrypoint(
    *, job_id: str, out_dir: str, track: dict[str, Any],
    base_ckpt_path: str, total_timesteps: int, n_envs: int, learning_rate: float,
    progress_q: Any, stop_evt: Any,
) -> None:
    """Runs in the spawned subprocess. Re-imports torch after spawn, wires
    SB3 to the training module's ``finetune`` with a progress callback."""
    # plan §14.8: pin threading env vars in the child, before torch import.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    def progress_cb(row: dict[str, Any]) -> None:
        try:
            progress_q.put(row, timeout=1.0)
        except Exception:  # noqa: BLE001
            pass

    try:
        # Heavy imports happen here so the parent stays light.
        from ..training.finetune import finetune
        from ..training.eval_gate import eval_policy
        from ..training.export import export_policy

        result = finetune(
            track=track,
            base_ckpt_path=base_ckpt_path,
            out_dir=out_dir,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            learning_rate=learning_rate,
            progress_cb=progress_cb,
            stop_event=stop_evt,
        )
        if stop_evt.is_set():
            return

        eval_result = eval_policy(
            policy_path=result["policy_path"],
            track=track,
            n_episodes=20,
        )
        progress_cb({"kind": "eval", "eval": eval_result})

        export_policy(
            policy_path=result["policy_path"],
            out_dir=out_dir,
        )
        progress_cb({"kind": "done", "eval": eval_result,
                     "artifact_url": f"/jobs/{job_id}/artifact"})
    except Exception as e:  # noqa: BLE001
        progress_cb({
            "kind": "error",
            "code": type(e).__name__,
            "message": f"{e}\n{traceback.format_exc()}",
        })
        raise
