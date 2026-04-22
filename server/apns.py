"""APNs HTTP/2 push client (plan §7.7).

Best-effort: we log and continue on failure. Never block a job completion
on push delivery.

Keyed by the ``.p8`` auth key + APNs JWT (ES256, 55-min cache)."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from jose import jwt

from .config import APNsConfig
from .storage import Storage


logger = logging.getLogger(__name__)


APNS_PROD = "https://api.push.apple.com"
APNS_DEV = "https://api.sandbox.push.apple.com"


class APNsClient:
    """One instance, shared across requests. Caches the signing JWT."""

    def __init__(self, cfg: APNsConfig) -> None:
        self._cfg = cfg
        self._key_id = os.environ.get(cfg.key_id_env, "")
        self._team_id = os.environ.get(cfg.team_id_env, "")
        self._auth_key_path = os.environ.get(cfg.auth_key_path_env, "")
        self._auth_key: Optional[str] = None
        self._jwt: Optional[str] = None
        self._jwt_issued: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def configured(self) -> bool:
        return bool(self._key_id and self._team_id and self._auth_key_path
                    and Path(self._auth_key_path).exists())

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(http2=True, timeout=10.0)
        return self._client

    def _load_key(self) -> str:
        if self._auth_key is None:
            with open(self._auth_key_path, "r") as f:
                self._auth_key = f.read()
        return self._auth_key

    def _provider_jwt(self) -> str:
        # Refresh once per ~55 minutes (APNs rejects > 60 min old).
        if self._jwt and (time.time() - self._jwt_issued) < 55 * 60:
            return self._jwt
        now = int(time.time())
        token = jwt.encode(
            {"iss": self._team_id, "iat": now},
            self._load_key(),
            algorithm="ES256",
            headers={"kid": self._key_id},
        )
        self._jwt = token
        self._jwt_issued = time.time()
        return token

    async def push(
        self, apns_token: str, *, title: str, body: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not self.configured:
            logger.info("APNs not configured; skipping push to %s", apns_token[:8])
            return False
        url = (APNS_PROD if self._cfg.use_production else APNS_DEV) \
            + f"/3/device/{apns_token}"
        headers = {
            "authorization": f"bearer {self._provider_jwt()}",
            "apns-topic": self._cfg.bundle_id,
            "apns-push-type": "alert",
        }
        payload: dict[str, Any] = {
            "aps": {"alert": {"title": title, "body": body}, "sound": "default"},
        }
        if extra:
            payload.update(extra)
        try:
            client = await self._ensure_client()
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                logger.warning("APNs %s: %s", r.status_code, r.text[:200])
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("APNs push failed: %s", e)
            return False

    async def push_to_user(
        self, storage: Storage, sub: str, *,
        title: str, body: str, extra: Optional[dict[str, Any]] = None,
    ) -> int:
        """Push to every APNs token registered for this user. Returns the
        number of successful sends."""
        rec = storage.load_user(sub) or {}
        tokens = rec.get("apns_tokens", [])
        ok = 0
        for tok in tokens:
            if await self.push(tok, title=title, body=body, extra=extra):
                ok += 1
        return ok

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
