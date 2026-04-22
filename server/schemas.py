"""Pydantic schemas shared across routers."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------- Auth -----------------------------------------------------------

class AppleLoginBody(BaseModel):
    identityToken: str = Field(..., description="JWT from Sign in with Apple")


class AppleLoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: int            # unix seconds
    sub: str                   # Apple sub claim == our user id


class DeviceRegisterBody(BaseModel):
    apns_token: str
    # hex-encoded 32-byte token; ~64 chars. Not enforced strictly.


# ---------- Tracks ---------------------------------------------------------

class BlockRect(BaseModel):
    """Rotated rectangle, cm, world frame."""
    cx: float
    cy: float
    w: float
    h: float
    theta: float               # radians, CCW positive


class TrackMeta(BaseModel):
    px_per_cm: float
    arkit_confidence: float
    camera_height_m: float
    image_bounds_cm: tuple[float, float]   # (w_cm, h_cm)


class CVError(BaseModel):
    code: str
    message: str
    expected_cm: Optional[float] = None
    observed_cm: Optional[float] = None


class TrackResponse(BaseModel):
    track_id: str
    state: Literal["pending", "ready", "failed", "confirmed"]
    blocks: Optional[list[BlockRect]] = None
    centerline: Optional[list[tuple[float, float]]] = None
    spawn: Optional[dict[str, float]] = None
    lane_width_cm: Optional[float] = None
    bounds: Optional[dict[str, float]] = None
    preview_url: Optional[str] = None
    errors: list[CVError] = Field(default_factory=list)
    warnings: list[CVError] = Field(default_factory=list)


class TrackPatchBody(BaseModel):
    blocks: list[BlockRect]


# ---------- Jobs -----------------------------------------------------------

JobState = Literal["queued", "running", "done", "failed", "cancelled"]


class JobSubmitBody(BaseModel):
    track_id: str
    minutes: Optional[int] = Field(None, ge=1, le=10)
    n_envs: Optional[int] = Field(None, ge=1, le=16)
    learning_rate: Optional[float] = Field(None, gt=0.0, lt=1.0)
    total_timesteps: Optional[int] = Field(None, ge=10_000, le=20_000_000)


class JobStatus(BaseModel):
    job_id: str
    user_id: str
    track_id: str
    state: JobState
    created_at: int
    started_at: Optional[int] = None
    ended_at: Optional[int] = None
    total_timesteps: int
    last_progress: Optional[dict[str, Any]] = None
    eval: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# ---------- Progress / WS events ------------------------------------------

class WSProgress(BaseModel):
    kind: Literal["progress"] = "progress"
    step: int
    mean_reward: float
    fps: Optional[float] = None
    ts: str


class WSState(BaseModel):
    kind: Literal["state"] = "state"
    state: JobState


class WSDone(BaseModel):
    kind: Literal["done"] = "done"
    artifact_url: str
    eval: dict[str, Any]


class WSError(BaseModel):
    kind: Literal["error"] = "error"
    code: str
    message: str

# ---------- GUI State / Control ------------------------------------------

class ControllerCompileBody(BaseModel):
    name: str
    code: str

class ControllerCompileResponse(BaseModel):
    status: Literal["ok", "error"]
    error: Optional[str] = None
