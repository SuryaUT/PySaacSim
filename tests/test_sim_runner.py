"""Headless sim advances pose and broadcasts when a client is subscribed."""
from __future__ import annotations

import asyncio
import json

import pytest


class _RecordingHub:
    def __init__(self) -> None:
        self._count = 1
        self.messages: list[str] = []

    def client_count(self) -> int:
        return self._count

    async def broadcast(self, msg: str) -> None:
        self.messages.append(msg)


@pytest.mark.asyncio
async def test_sim_runner_broadcasts_pose_updates():
    from PySaacSim.server.sim_runner import SimRunner

    hub = _RecordingHub()
    runner = SimRunner(hub)
    # Drive the car so pose changes rather than stalling at rest.
    await runner.set_command(throttle=0.8, steer=0.0)
    runner.start()
    await asyncio.sleep(0.25)
    await runner.stop()

    assert len(hub.messages) > 0, "expected ≥ 1 broadcast in 250 ms"
    msg = json.loads(hub.messages[-1])
    assert msg["kind"] == "sim"
    assert {"pose", "lidar", "ir"}.issubset(msg.keys())
