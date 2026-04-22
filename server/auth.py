"""Sign in with Apple verification → app JWT.

Flow (plan §7.6):
  1. iOS app POSTs its Apple ``identityToken`` to ``/auth/apple``.
  2. We fetch Apple's JWKs (cached 24 h), verify the token's signature,
     ``iss``, ``aud``, and ``exp``.
  3. We mint an app JWT (HS256, 7-day) keyed by the Apple ``sub``.
  4. Every protected endpoint validates that app JWT via ``current_user``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from fastapi import Depends, Header, HTTPException, Request, status
from jose import JWTError, jwk, jwt

from .config import ServerConfig
from .storage import Storage, default_storage


APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISS = "https://appleid.apple.com"
_JWKS_TTL_SECONDS = 24 * 3600


@dataclass
class User:
    sub: str                   # Apple user id


# ---------- Apple JWKs cache ----------------------------------------------

class _JWKSCache:
    """Process-local cache of Apple's public JWKs. httpx does an async GET,
    but validation is sync JOSE — the cache hands back a dict ready for
    ``jose.jwt.decode``."""

    def __init__(self) -> None:
        self._keys: dict[str, dict[str, Any]] = {}
        self._fetched_at: float = 0.0

    async def get(self, kid: str, *, client: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
        if kid not in self._keys or (time.time() - self._fetched_at) > _JWKS_TTL_SECONDS:
            await self._refresh(client=client)
        if kid not in self._keys:
            # Refresh once more in case Apple rotated mid-session.
            await self._refresh(client=client)
        if kid not in self._keys:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                                f"Unknown Apple signing key: {kid}")
        return self._keys[kid]

    async def _refresh(self, *, client: Optional[httpx.AsyncClient]) -> None:
        owned = False
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            owned = True
        try:
            r = await client.get(APPLE_JWKS_URL)
            r.raise_for_status()
            data = r.json()
        finally:
            if owned:
                await client.aclose()
        self._keys = {k["kid"]: k for k in data.get("keys", [])}
        self._fetched_at = time.time()


_jwks_cache = _JWKSCache()


# ---------- Apple token verification --------------------------------------

async def verify_apple_identity_token(
    token: str, *, bundle_id: str,
    client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    """Decode + verify an Apple ``identityToken``. Returns the claim set.

    Raises HTTPException(401) on any failure."""
    try:
        headers = jwt.get_unverified_header(token)
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            f"Malformed identity token: {e}")
    kid = headers.get("kid")
    alg = headers.get("alg", "RS256")
    if not kid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "Identity token missing kid")
    key_dict = await _jwks_cache.get(kid, client=client)
    public_key = jwk.construct(key_dict).to_pem().decode("utf-8")
    try:
        claims = jwt.decode(
            token, public_key,
            algorithms=[alg],
            audience=bundle_id,
            issuer=APPLE_ISS,
            options={"verify_at_hash": False},
        )
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            f"Identity token verification failed: {e}")
    if not claims.get("sub"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "Identity token missing sub")
    return claims


# ---------- App JWT issue / verify ----------------------------------------

def issue_app_jwt(sub: str, *, secret: str, ttl_days: int) -> tuple[str, int]:
    now = int(time.time())
    exp = now + ttl_days * 86400
    payload = {"sub": sub, "iat": now, "exp": exp, "iss": "pysaacrc"}
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token, exp


def decode_app_jwt(token: str, *, secret: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, secret, algorithms=["HS256"], issuer="pysaacrc")
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}")


# ---------- FastAPI dependency --------------------------------------------

from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
) -> User:
    cfg: ServerConfig = request.app.state.config
    claims = decode_app_jwt(token, secret=cfg.require_jwt_secret())
    return User(sub=str(claims["sub"]))


async def current_user_ws(token: Optional[str], *, app) -> User:
    """WebSocket auth: JWT via ``?token=`` query param (browsers can't set
    Authorization on upgrade). Still HS256."""
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    cfg: ServerConfig = app.state.config
    claims = decode_app_jwt(token, secret=cfg.require_jwt_secret())
    return User(sub=str(claims["sub"]))
