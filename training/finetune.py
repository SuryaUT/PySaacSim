"""Per-track fine-tuner (plan §5.4).

Runs inside a spawned child process (see ``server.jobs``). The parent polls
``progress_cb`` rows off a multiprocessing.Queue; ``stop_event`` lets a
cancel interrupt ``model.learn``."""
from __future__ import annotations

import datetime as _dt
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ._track_io import track_from_json


# Factory used by SubprocVecEnv. It has to be top-level so ``spawn`` can
# pickle it. It reads the track from a JSON file on disk (plan §14.4).

def _build_env(track_json_path: str):
    import os
    import json
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor
    from ..env.robot_env import RobotEnv

    with open(track_json_path, "r") as f:
        track = json.load(f)
    track = track_from_json(track)

    def _factory():
        env = RobotEnv(max_episode_steps=1500)
        # RobotEnv holds a default world; swap the walls in. We leave the
        # reward's lap-progress centroid computation as-is (it uses world
        # bounds which defaults to the oval; good enough for fine-tuning
        # until `bounds` is plumbed through from the track dict).
        if track.get("walls"):
            env._world["walls"] = list(track["walls"])
            env._walls = env._world["walls"]
        if track.get("bounds"):
            env._world["bounds"] = dict(track["bounds"])
        return Monitor(env)

    return _factory


def finetune(
    track: dict[str, Any],
    base_ckpt_path: str,
    out_dir: str,
    *,
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    progress_cb: Optional[Callable[[dict[str, Any]], None]] = None,
    stop_event: Any = None,
) -> dict[str, Any]:
    """Fine-tune the base policy on a single confirmed track.

    Returns ``{"policy_path": str, "total_timesteps": int}``. ``progress_cb``
    is invoked with ``{"kind": "progress", "step": ..., "mean_reward": ...,
    "fps": ..., "ts": ...}`` on every rollout end."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stash the track on disk once so child envs can re-read it cheaply and
    # picklably (avoids shipping a large dict through mp pickling per env).
    import json
    from ._track_io import track_to_json
    track_json_path = out / "track.child.json"
    with open(track_json_path, "w") as f:
        json.dump(track_to_json(track), f)

    # Build vec env.
    factories = [_build_env(str(track_json_path)) for _ in range(max(1, n_envs))]
    if n_envs > 1:
        vec = SubprocVecEnv(factories, start_method="spawn")
    else:
        vec = DummyVecEnv(factories)

    # Load or construct the policy.
    base_path = Path(base_ckpt_path)
    if base_path.exists():
        model = PPO.load(str(base_path.with_suffix("")), env=vec, device="auto")
        model.learning_rate = learning_rate
        model._setup_lr_schedule()
    else:
        # No base checkpoint on disk — train from scratch (useful for dev /
        # CI until an overnight base-policy run has produced `base_policy.zip`).
        model = PPO(
            "MlpPolicy", vec,
            n_steps=2048, batch_size=512,
            learning_rate=learning_rate,
            gamma=0.995, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.005,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log=str(out / "tb"),
            verbose=0,
        )

    class ProgressCb(BaseCallback):
        def __init__(self):
            super().__init__()
            self._t0 = time.time()

        def _on_step(self) -> bool:
            if stop_event is not None and stop_event.is_set():
                return False
            return True

        def _on_rollout_end(self) -> None:
            if progress_cb is None:
                return
            import numpy as np
            ep = self.model.ep_info_buffer
            mean_r = float(np.mean([e["r"] for e in ep])) if ep else 0.0
            elapsed = max(1e-6, time.time() - self._t0)
            fps = self.num_timesteps / elapsed
            progress_cb({
                "kind": "progress",
                "step": int(self.num_timesteps),
                "mean_reward": mean_r,
                "fps": fps,
                "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            })

    # Checkpoint cadence (plan §5.3 / §14 risks table): save every 200k steps.
    class CheckpointCb(BaseCallback):
        def __init__(self, every: int = 200_000):
            super().__init__()
            self._every = every
            self._next = every

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next:
                try:
                    self.model.save(str(out / "ckpt"))
                except Exception:  # noqa: BLE001
                    pass
                self._next += self._every
            return True

    from stable_baselines3.common.callbacks import CallbackList

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([ProgressCb(), CheckpointCb()]),
            reset_num_timesteps=True,
            progress_bar=False,
        )
    finally:
        policy_path = out / "policy"
        model.save(str(policy_path))
        vec.close()

    return {
        "policy_path": str(policy_path) + ".zip",
        "total_timesteps": int(total_timesteps),
    }
