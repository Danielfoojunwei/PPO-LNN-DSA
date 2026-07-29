"""Synchronous vectorised wrapper around :class:`~dsa.envs.spectrum.SpectrumEnv`.

Vectorising the rollout is the single largest throughput win available on a
CPU-only box (measured ~12x from 1 to 16 lanes), so the learner always steps a
batch of lanes.

Two properties matter for correctness:

* **Every lane is a pure function of its own episode seed.**  A lane's stream
  does not depend on which lane index it occupies, on how many lanes there are,
  or on how many environments were constructed earlier.  ``reseed`` rebuilds the
  lanes from a new seed list, which is how the trainer gets a fresh set of
  episodes per rollout batch without ever mutating a counter.
* **No auto-reset.**  ``horizon == max_steps`` for every registered scenario, so
  all lanes truncate together at the end of a rollout.  Stepping past truncation
  raises rather than silently starting a new episode with a stream nobody
  recorded.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import ScenarioConfig
from .spectrum import SpectrumEnv

__all__ = ["SyncVectorEnv"]


class SyncVectorEnv:
    """A fixed batch of independent :class:`SpectrumEnv` lanes.

    Parameters
    ----------
    config
        Shared frozen scenario configuration.
    episode_seeds
        One seed per lane.  ``num_envs == len(episode_seeds)``.
    """

    def __init__(self, config: ScenarioConfig, episode_seeds: Sequence[int]) -> None:
        seeds = [int(s) for s in episode_seeds]
        if not seeds:
            raise ValueError("episode_seeds must be non-empty")
        self.config = config
        self.episode_seeds = seeds
        self.num_envs = len(seeds)
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.max_steps = int(config.max_steps)
        self.envs: list[SpectrumEnv] = []
        self._build()

    def _build(self) -> None:
        self.envs = [SpectrumEnv(self.config, s) for s in self.episode_seeds]

    # ------------------------------------------------------------------- api #

    def reseed(self, episode_seeds: Sequence[int]) -> None:
        """Replace the lane seeds and rebuild every lane for the next batch."""
        seeds = [int(s) for s in episode_seeds]
        if len(seeds) != self.num_envs:
            raise ValueError(
                f"reseed expects {self.num_envs} seeds, got {len(seeds)}"
            )
        self.episode_seeds = seeds
        self._build()

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Restart every lane.  Idempotent and free of construction-order effects."""
        self._build()
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        dt = np.zeros(self.num_envs, dtype=np.float32)
        for i, env in enumerate(self.envs):
            state, _info = env.reset()
            obs[i] = state["obs"]
            dt[i] = state["dt"]
        return obs, dt

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step every lane once.  Does **not** auto-reset."""
        acts = np.asarray(actions, dtype=np.int64).reshape(-1)
        if acts.shape[0] != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} actions, got {acts.shape[0]}"
            )
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        dt = np.zeros(self.num_envs, dtype=np.float32)
        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []
        for i, env in enumerate(self.envs):
            state, r, terminated, truncated, info = env.step(int(acts[i]))
            obs[i] = state["obs"]
            dt[i] = state["dt"]
            reward[i] = r
            done[i] = bool(terminated or truncated)
            infos.append(info)
        return obs, dt, reward, done, infos

    def episode_metrics(self) -> list[dict[str, float]]:
        """Per-lane episode metrics, in lane order."""
        return [env.episode_metrics() for env in self.envs]
