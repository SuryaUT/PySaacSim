"""JWT round-trip + Apple-identity-token wiring.

The real Apple JWKs fetch is mocked; the test verifies the server accepts
an identity token that validates against a test RSA key, and that the
resulting app JWT unlocks a protected endpoint."""
from __future__ import annotations

import time
from unittest import mock

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

BUNDLE_ID = "com.example.pysaacrc"


def _make_rsa_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    # Build a JWK for the public key.
    pub_numbers = key.public_key().public_numbers()
    import base64
    def _b64(n: int) -> str:
        b = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    jwk = {
        "kty": "RSA", "alg": "RS256", "use": "sig", "kid": "test-kid",
        "n": _b64(pub_numbers.n),
        "e": _b64(pub_numbers.e),
    }
    return priv, jwk


def test_apple_login_then_protected_call(client, jwt_secret, monkeypatch):
    priv_pem, jwk_dict = _make_rsa_keypair()
    now = int(time.time())
    identity = jwt.encode(
        {"iss": "https://appleid.apple.com", "aud": BUNDLE_ID,
         "sub": "apple-sub-abc", "iat": now, "exp": now + 300},
        priv_pem, algorithm="RS256", headers={"kid": "test-kid"},
    )

    # Patch the JWKs fetch.
    async def _fake_get(self, kid, client=None):
        assert kid == "test-kid"
        return jwk_dict
    from PySaacSim.server.auth import _JWKSCache
    monkeypatch.setattr(_JWKSCache, "get", _fake_get)

    # Point the app's config bundle id at the test value.
    client.app.state.config.apple_bundle_id = BUNDLE_ID

    r = client.post("/auth/apple", json={"identityToken": identity})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["sub"] == "apple-sub-abc"
    assert data["token_type"] == "bearer"
    assert data["access_token"]

    # Use the app JWT to call a protected endpoint.
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    r = client.post("/devices", json={"apns_token": "aabb"}, headers=headers)
    assert r.status_code == 204
