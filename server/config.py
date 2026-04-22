"""Load ``config/server.yaml`` into a typed object.

One loader, one place where defaults live. Env vars override a few
security-sensitive fields (JWT secret, APNs key path) — everything else
stays in YAML so it round-trips through a PR."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "config" / "server.yaml"


@dataclass
class APNsConfig:
    key_id_env: str = "APNS_KEY_ID"
    team_id_env: str = "APPLE_TEAM_ID"
    auth_key_path_env: str = "APNS_AUTH_KEY_PATH"
    bundle_id: str = "com.example.pysaacrc"
    use_production: bool = False


@dataclass
class RateLimits:
    auth: str = "10/minute"
    tracks: str = "20/hour"
    jobs: str = "6/hour"


@dataclass
class TrainingConfig:
    base_policy_path: str = "models/base_policy.zip"
    default_timesteps: int = 2_000_000
    n_envs: int = 8
    max_minutes: int = 10


@dataclass
class PlankTolerance:
    lower: float = 0.72
    upper: float = 1.10


@dataclass
class CVConfig:
    sam3_weights: str = "models/sam3.pt"
    sam3_mode: str = "text_prompt"
    sam3_text_prompts: list[str] = field(default_factory=lambda: ["wooden plank"])
    sam3_score_threshold: float = 0.5
    plank_length_cm: float = 80.0
    plank_tolerance: PlankTolerance = field(default_factory=PlankTolerance)
    min_arkit_confidence: float = 0.5
    target_px_per_cm: float = 10.0


@dataclass
class ServerConfig:
    data_dir: Path = Path("~/.pysaac").expanduser()
    host: str = "127.0.0.1"
    port: int = 8787
    jwt_secret_env: str = "PYSAAC_JWT_SECRET"
    jwt_ttl_days: int = 7
    apple_bundle_id: str = "com.example.pysaacrc"
    apns: APNsConfig = field(default_factory=APNsConfig)
    rate_limits: RateLimits = field(default_factory=RateLimits)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ServerConfig":
        p = Path(path) if path else _DEFAULT_YAML
        if not p.exists():
            return cls()
        with open(p, "r") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> "ServerConfig":
        data_dir = Path(str(d.get("data_dir", "~/.pysaac"))).expanduser()
        apns = APNsConfig(**(d.get("apns") or {}))
        rl = RateLimits(**(d.get("rate_limits") or {}))
        tr = TrainingConfig(**(d.get("training") or {}))
        cv_raw = dict(d.get("cv") or {})
        tol_raw = cv_raw.pop("plank_tolerance", None) or {}
        cv = CVConfig(**cv_raw, plank_tolerance=PlankTolerance(**tol_raw))
        return cls(
            data_dir=data_dir,
            host=str(d.get("host", "127.0.0.1")),
            port=int(d.get("port", 8787)),
            jwt_secret_env=str(d.get("jwt_secret_env", "PYSAAC_JWT_SECRET")),
            jwt_ttl_days=int(d.get("jwt_ttl_days", 7)),
            apple_bundle_id=str(d.get("apple_bundle_id", "com.example.pysaacrc")),
            apns=apns,
            rate_limits=rl,
            training=tr,
            cv=cv,
        )

    @property
    def jwt_secret(self) -> str:
        """Read the JWT secret from env, lazily, so tests can override it.

        The server refuses to start in production if this is unset; the caller
        checks ``jwt_secret_raw`` to decide."""
        v = os.environ.get(self.jwt_secret_env, "")
        return v

    def require_jwt_secret(self) -> str:
        v = self.jwt_secret
        if not v:
            raise RuntimeError(
                f"{self.jwt_secret_env} is not set. Export a long random string "
                f"(e.g. `python -c 'import secrets; print(secrets.token_urlsafe(48))'`) "
                f"before starting the server."
            )
        return v
