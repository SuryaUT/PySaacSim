"""Training module — called from the server in a spawned subprocess.

Heavy imports (torch, stable_baselines3) are deferred to function bodies so
the parent (FastAPI) process can import this module at boot time without
dragging GPU state in."""
