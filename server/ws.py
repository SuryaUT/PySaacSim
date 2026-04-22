"""WebSocket hub — fans out per-job progress events to any number of
connected clients. Also handles heartbeats and progress-replay on reconnect.

Plan §7.5: ``Dict[job_id, Set[WebSocket]]``, replay from ``progress.jsonl``
on connect, 20 s heartbeat. When a job reaches a terminal state, persist the
final state in ``state.json`` *before* closing WS — otherwise a reconnecting
client could miss the ``done`` event (plan §14.16)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

from .storage import Storage


logger = logging.getLogger(__name__)


class WSHub:
    """Broadcast channels keyed by job id. One channel = N subscribers."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._subs: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._subs.setdefault(job_id, set()).add(ws)

    async def unsubscribe(self, job_id: str, ws: WebSocket) -> None:
        async with self._lock:
            subs = self._subs.get(job_id)
            if not subs:
                return
            subs.discard(ws)
            if not subs:
                self._subs.pop(job_id, None)

    async def broadcast(self, job_id: str, message: dict[str, Any]) -> None:
        """Non-fatal broadcast: bad sockets are dropped silently."""
        text = json.dumps(message, separators=(",", ":"))
        async with self._lock:
            subs = list(self._subs.get(job_id, ()))
        if not subs:
            return
        dead: list[WebSocket] = []
        for ws in subs:
            try:
                await ws.send_text(text)
            except Exception as e:  # noqa: BLE001
                logger.debug("ws send failed, dropping: %s", e)
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    for s in self._subs.values():
                        s.discard(ws)

    def subscriber_count(self, job_id: str) -> int:
        return len(self._subs.get(job_id, ()))

    # ---- client lifecycle (called from the WS route) ---------------------

    async def serve_job(self, job_id: str, ws: WebSocket) -> None:
        """Handle one client: replay history, stream live, heartbeat until
        the socket closes or the job reaches a terminal state."""
        await ws.accept()
        await self.subscribe(job_id, ws)
        try:
            # 1) Replay.
            for row in self._storage.read_progress(job_id):
                await ws.send_text(json.dumps(row, separators=(",", ":")))
            state = self._storage.load_job_state(job_id) or {}
            if state.get("state") in {"done", "failed", "cancelled"}:
                # Late subscriber: replay was the whole story. Send terminal
                # state, then close. The client polls GET /jobs/{id} for the
                # artifact URL + eval (plan §14.16 reconnect race).
                await ws.send_text(json.dumps({
                    "kind": "state", "state": state["state"],
                }))
                if state.get("state") == "done":
                    await ws.send_text(json.dumps({
                        "kind": "done",
                        "artifact_url": f"/jobs/{job_id}/artifact",
                        "eval": state.get("eval") or {},
                    }))
                return
            # 2) Heartbeat loop; the job worker drives actual progress broadcasts.
            while True:
                try:
                    await asyncio.wait_for(ws.receive_text(), timeout=20.0)
                except asyncio.TimeoutError:
                    await ws.send_text(json.dumps({"kind": "ping"}))
                except Exception:  # noqa: BLE001
                    break
        finally:
            await self.unsubscribe(job_id, ws)


# ---------- Headless sim WS channel ---------------------------------------
# Separate from job WS because every connected client sees the same stream
# — it's not keyed by anything. A singleton set is enough.

class SimWSHub:
    """Single-channel broadcast for the headless sim runner (plan §7.11)."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    def client_count(self) -> int:
        return len(self._clients)

    async def broadcast(self, message: str) -> None:
        if not self._clients:
            return
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(message)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)
