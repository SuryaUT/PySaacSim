"""``/auth/apple`` and ``/devices`` endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import os

from ..auth import (
    User, current_user, issue_app_jwt, verify_apple_identity_token,
)
from ..schemas import AppleLoginBody, AppleLoginResponse, DeviceRegisterBody


router = APIRouter()

@router.post("/auth/token")
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    expected_username = os.environ.get("PYSAAC_USERNAME", "admin")
    expected_password = os.environ.get("PYSAAC_PASSWORD", "admin")
    
    print(f"DEBUG: expected='{expected_username}'/'{expected_password}', got='{form_data.username}'/'{form_data.password}'")
    
    if form_data.username != expected_username or form_data.password != expected_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    cfg = request.app.state.config
    sub = form_data.username
    storage = request.app.state.storage
    
    # Optional: track login time if needed
    import time
    storage.upsert_user(sub, last_login=int(time.time()))
    
    token, exp = issue_app_jwt(
        sub, 
        secret=cfg.require_jwt_secret(),
        ttl_days=cfg.jwt_ttl_days
    )
    return {"access_token": token, "token_type": "bearer"}


@router.post("/auth/apple", response_model=AppleLoginResponse)
async def auth_apple(body: AppleLoginBody, request: Request) -> AppleLoginResponse:
    cfg = request.app.state.config
    claims = await verify_apple_identity_token(
        body.identityToken, bundle_id=cfg.apple_bundle_id,
    )
    sub = str(claims["sub"])
    storage = request.app.state.storage
    storage.upsert_user(sub, last_login=claims.get("iat"))
    token, exp = issue_app_jwt(sub, secret=cfg.require_jwt_secret(),
                               ttl_days=cfg.jwt_ttl_days)
    return AppleLoginResponse(access_token=token, expires_at=exp, sub=sub)


@router.post("/devices", status_code=204)
async def register_device(
    body: DeviceRegisterBody,
    request: Request,
    user: User = Depends(current_user),
) -> None:
    storage = request.app.state.storage
    storage.add_apns_token(user.sub, body.apns_token)
