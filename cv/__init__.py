"""Computer-vision pipeline (plan §6).

The server calls ``run_cv_pipeline`` on an uploaded, already-rectified
top-down photo and gets back a track dict (plan §5.1 shape), a preview PNG,
and any error/warning codes.

SAM3 is access-gated. The import in ``segment.py`` is guarded so this module
loads on machines without the wheel; run_cv_pipeline will return a
``SAM3_UNAVAILABLE`` error in that case and a developer can use the
``synthetic`` fallback for tests."""
from .pipeline import run_cv_pipeline  # re-export for convenience

__all__ = ["run_cv_pipeline"]
