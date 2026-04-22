"""Post-train evaluation (plan §5.5).

Runs the trained policy on the same track used for fine-tuning and reports
coarse performance numbers. The server attaches this dict to the job
artifact; it's a smoke test, not a pass/fail gate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._track_io import track_from_json


def eval_policy(
    policy_path: str,
    track: dict[str, Any],
    *,
    n_episodes: int = 20,
) -> dict[str, Any]:
    from stable_baselines3 import PPO
    import numpy as np

    from ..env.robot_env import RobotEnv

    track = track_from_json(dict(track))
    env = RobotEnv(max_episode_steps=1500)
    if track.get("walls"):
        env._world["walls"] = list(track["walls"])
        env._walls = env._world["walls"]
    if track.get("bounds"):
        env._world["bounds"] = dict(track["bounds"])

    p = Path(policy_path)
    load_target = str(p.with_suffix("") if p.suffix == ".zip" else p)
    model = PPO.load(load_target, device="auto")

    completions = 0
    collisions = 0
    total_reward = 0.0
    lap_times: list[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        laps = 0
        prev_lap = 0.0
        terminated = truncated = False
        t_ms_start = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_r += float(r)
            # RobotEnv tracks lap_progress in radians (2π = 1 lap).
            lp = env._lap_progress if hasattr(env, "_lap_progress") else 0.0
            d_lap = lp - prev_lap
            prev_lap = lp
            if abs(lp) >= 2 * 3.14159265 and laps == 0:
                lap_times.append(float(info.get("t_ms", 0.0)) / 1000.0)
                laps += 1
                completions += 1
        if terminated and not truncated:
            # Reward terminal was a crash (collided bool set in info).
            if info.get("collided"):
                collisions += 1
        total_reward += ep_r

    return {
        "n_episodes": n_episodes,
        "completion_rate": completions / max(1, n_episodes),
        "collision_rate": collisions / max(1, n_episodes),
        "mean_reward": total_reward / max(1, n_episodes),
        "mean_lap_time_s": float(np.mean(lap_times)) if lap_times else None,
    }
