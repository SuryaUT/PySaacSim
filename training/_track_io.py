"""Track JSON helpers. The plan §3 calls out that ``sim.geometry.Segment``
is a NamedTuple of NamedTuples; JSON only sees tuples or lists. Keep the
packing/unpacking in one place."""
from __future__ import annotations

from typing import Any, Iterable

from ..sim.geometry import Segment, Vec2


def walls_to_json(walls: Iterable[Segment]) -> list[list[list[float]]]:
    return [[[w.a.x, w.a.y], [w.b.x, w.b.y]] for w in walls]


def walls_from_json(raw: Any) -> list[Segment]:
    out: list[Segment] = []
    for w in raw or []:
        a, b = w[0], w[1]
        out.append(Segment(Vec2(float(a[0]), float(a[1])),
                           Vec2(float(b[0]), float(b[1]))))
    return out


def track_to_json(track: dict[str, Any]) -> dict[str, Any]:
    out = dict(track)
    walls = track.get("walls")
    if walls is not None:
        out["walls"] = walls_to_json(walls)
    cl = track.get("centerline")
    if cl is not None:
        # Accept either np.ndarray or nested list.
        out["centerline"] = [[float(p[0]), float(p[1])] for p in cl]
    return out


def track_from_json(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    walls = d.get("walls")
    if walls is not None:
        out["walls"] = walls_from_json(walls)
    return out
