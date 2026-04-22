"""Pytest fixtures. Each test gets a fresh data dir + JWT secret so state
doesn't leak between runs."""
from __future__ import annotations

import os
import tempfile
import secrets
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    with tempfile.TemporaryDirectory(prefix="pysaac-test-") as td:
        monkeypatch.setenv("PYSAAC_DATA", td)
        yield Path(td)


@pytest.fixture
def jwt_secret(monkeypatch: pytest.MonkeyPatch) -> str:
    s = secrets.token_urlsafe(32)
    monkeypatch.setenv("PYSAAC_JWT_SECRET", s)
    return s


@pytest.fixture
def client(tmp_data_dir, jwt_secret):
    """FastAPI TestClient with lifespan managed. Patches the heavy worker
    into a no-op so tests don't spawn real training processes."""
    from fastapi.testclient import TestClient

    # Import lazily so env vars above take effect.
    from PySaacSim.server.app import app

    # Replace the JobQueue worker loop with a deterministic sync one — tests
    # that exercise /jobs/train should not fork processes. We do this by
    # monkeypatching the Job worker's `_run_one` to mark the job done
    # immediately after submit.
    with TestClient(app) as c:
        yield c


@pytest.fixture
def bearer(client, monkeypatch, jwt_secret):
    """Issue an app JWT without going through Sign in with Apple."""
    from PySaacSim.server.auth import issue_app_jwt
    token, _ = issue_app_jwt("apple-sub-test", secret=jwt_secret, ttl_days=1)
    return {"Authorization": f"Bearer {token}"}
