"""Filesystem layout under ``$PYSAAC_DATA`` (default ``~/.pysaac``).

One module owns paths; nothing else touches the filesystem directly. IDs are
ULIDs (time-sortable), produced via ``new_id``. See plan §7.2 for the layout."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import ulid


def new_id() -> str:
    return str(ulid.new())


@dataclass
class Storage:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()
        for sub in ("tracks", "jobs", "users"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ---- users ------------------------------------------------------------

    def user_path(self, apple_sub: str) -> Path:
        return self.root / "users" / f"{apple_sub}.json"

    def load_user(self, apple_sub: str) -> Optional[dict[str, Any]]:
        p = self.user_path(apple_sub)
        if not p.exists():
            return None
        with open(p, "r") as f:
            return json.load(f)

    def upsert_user(self, apple_sub: str, **fields: Any) -> dict[str, Any]:
        p = self.user_path(apple_sub)
        if p.exists():
            with open(p, "r") as f:
                rec = json.load(f)
        else:
            rec = {"sub": apple_sub, "created_at": int(time.time()),
                   "apns_tokens": []}
        rec.update(fields)
        rec["updated_at"] = int(time.time())
        self._atomic_write(p, rec)
        return rec

    def add_apns_token(self, apple_sub: str, token: str) -> None:
        rec = self.load_user(apple_sub) or {
            "sub": apple_sub, "created_at": int(time.time()), "apns_tokens": []
        }
        tokens = set(rec.get("apns_tokens", []))
        tokens.add(token)
        rec["apns_tokens"] = sorted(tokens)
        rec["updated_at"] = int(time.time())
        self._atomic_write(self.user_path(apple_sub), rec)

    # ---- tracks -----------------------------------------------------------

    def track_dir(self, track_id: str) -> Path:
        d = self.root / "tracks" / track_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def track_meta(self, track_id: str) -> Optional[dict[str, Any]]:
        p = self.track_dir(track_id) / "meta.json"
        if not p.exists():
            return None
        with open(p, "r") as f:
            return json.load(f)

    def save_track_meta(self, track_id: str, meta: dict[str, Any]) -> None:
        self._atomic_write(self.track_dir(track_id) / "meta.json", meta)

    def save_track_json(self, track_id: str, track: dict[str, Any]) -> None:
        self._atomic_write(self.track_dir(track_id) / "track.json", track)

    def load_track_json(self, track_id: str) -> Optional[dict[str, Any]]:
        p = self.track_dir(track_id) / "track.json"
        if not p.exists():
            return None
        with open(p, "r") as f:
            return json.load(f)

    # ---- jobs -------------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        d = self.root / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_job_state(self, job_id: str, state: dict[str, Any]) -> None:
        self._atomic_write(self.job_dir(job_id) / "state.json", state)

    def load_job_state(self, job_id: str) -> Optional[dict[str, Any]]:
        p = self.job_dir(job_id) / "state.json"
        if not p.exists():
            return None
        with open(p, "r") as f:
            return json.load(f)

    def append_progress(self, job_id: str, row: dict[str, Any]) -> None:
        p = self.job_dir(job_id) / "progress.jsonl"
        line = json.dumps(row, separators=(",", ":"))
        # O_APPEND writes are atomic up to PIPE_BUF on POSIX — safe for the
        # single-worker setup; still wrap in a small helper for clarity.
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_progress(self, job_id: str) -> Iterator[dict[str, Any]]:
        p = self.job_dir(job_id) / "progress.jsonl"
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    # ---- helpers ----------------------------------------------------------

    def _atomic_write(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False, default=str)
        os.replace(tmp, path)


def default_storage() -> Storage:
    root = Path(os.environ.get("PYSAAC_DATA", "~/.pysaac")).expanduser()
    return Storage(root=root)
