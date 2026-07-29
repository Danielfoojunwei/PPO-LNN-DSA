"""Sequence-major rollout storage for recurrent PPO.

Design notes (these are load-bearing; see REBUILD_SPEC §4.4):

* Storage is ``(N, T, ...)`` -- sequence-major, NOT flattened ``(N*T, ...)``.  The
  update replays whole sequences through :meth:`RecurrentActorCritic.unroll`, so the
  time axis must survive into the minibatch.
* **Per-step hidden states are deliberately NOT stored.**  Storing them and feeding
  them back as update inputs is the single most common fatal PPO-RNN bug: after the
  first gradient step the stored states are stale, the recomputed log-probs no longer
  match the behaviour policy, and the PPO ratio silently drifts away from 1.0 on the
  very first minibatch.  Only ``init_state`` (the state at t=0) is stored, and the
  whole hidden trajectory is recomputed from it inside every minibatch.
* ``dones`` stores **true termination only**, never time-limit truncation.  A
  time-limited episode must still bootstrap ``V(s_T)``; that is what ``last_value``
  is for.  Mixing the two biases every return estimate at the horizon boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["SequenceBatch", "SequenceRolloutBuffer"]


def _as_tensor(x, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


@dataclass
class SequenceBatch:
    """One rollout batch of ``N`` sequences of length ``T``."""

    obs: torch.Tensor  # (N, T, obs_dim) float32
    dt: torch.Tensor  # (N, T)          float32
    actions: torch.Tensor  # (N, T)          int64
    log_probs: torch.Tensor  # (N, T)          float32
    values: torch.Tensor  # (N, T)          float32
    rewards: torch.Tensor  # (N, T)          float32
    dones: torch.Tensor  # (N, T)          float32  -- TRUE TERMINATION ONLY
    init_state: tuple[torch.Tensor, ...]  # each (N, state_size)
    last_value: torch.Tensor  # (N,)            float32 -- V(s_T), the truncation bootstrap

    def num_sequences(self) -> int:
        return int(self.obs.shape[0])

    def horizon(self) -> int:
        return int(self.obs.shape[1])

    def select(self, index: torch.Tensor) -> "SequenceBatch":
        """Return the sub-batch of sequences named by ``index`` (a 1-D LongTensor)."""
        return SequenceBatch(
            obs=self.obs[index],
            dt=self.dt[index],
            actions=self.actions[index],
            log_probs=self.log_probs[index],
            values=self.values[index],
            rewards=self.rewards[index],
            dones=self.dones[index],
            init_state=tuple(s[index] for s in self.init_state),
            last_value=self.last_value[index],
        )


class SequenceRolloutBuffer:
    """Fixed-capacity ``(N, T)`` buffer filled one time-step at a time.

    Unlike the buffer this repository previously shipped, there is no flush
    threshold and no "when the buffer is long enough" condition: capacity is
    ``num_envs * horizon`` by construction and :meth:`finish` always yields a full
    batch of ``num_envs`` sequences.  Minibatching over the sequence axis is
    therefore reachable for any ``num_sequence_minibatches <= num_envs``.
    """

    def __init__(
        self,
        num_envs: int,
        horizon: int,
        obs_dim: int,
        state_sizes: tuple[int, ...],
        device: str | torch.device = "cpu",
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.num_envs = int(num_envs)
        self.horizon_len = int(horizon)
        self.obs_dim = int(obs_dim)
        self.state_sizes = tuple(int(s) for s in state_sizes)
        self.device = torch.device(device)

        n, t = self.num_envs, self.horizon_len
        self.obs = torch.zeros(n, t, self.obs_dim, dtype=torch.float32, device=self.device)
        self.dt = torch.zeros(n, t, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(n, t, dtype=torch.int64, device=self.device)
        self.log_probs = torch.zeros(n, t, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(n, t, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(n, t, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(n, t, dtype=torch.float32, device=self.device)
        self._init_state: tuple[torch.Tensor, ...] | None = None
        self._pos = 0

    # ------------------------------------------------------------------ #

    def start(self, init_state: tuple[torch.Tensor, ...]) -> None:
        """Begin a rollout from ``init_state`` (each element ``(N, state_size)``)."""
        init_state = tuple(init_state)
        if len(init_state) != len(self.state_sizes):
            raise ValueError(
                f"init_state has {len(init_state)} blocks, expected {len(self.state_sizes)}"
            )
        for blk, (s, size) in enumerate(zip(init_state, self.state_sizes)):
            if tuple(s.shape) != (self.num_envs, size):
                raise ValueError(
                    f"init_state[{blk}] has shape {tuple(s.shape)}, "
                    f"expected {(self.num_envs, size)}"
                )
        self._init_state = tuple(s.detach().to(self.device).clone() for s in init_state)
        self._pos = 0

    def add(self, obs, dt, action, log_prob, value, reward, done) -> None:
        """Append one synchronous time-step across all ``N`` lanes.

        ``done`` is the **terminal** flag (the MDP genuinely ended).  Do not pass
        time-limit truncation here -- truncation is handled by ``last_value``.
        """
        if self._init_state is None:
            raise RuntimeError("call start(init_state) before add()")
        if self._pos >= self.horizon_len:
            raise RuntimeError(
                f"buffer is full ({self.horizon_len} steps); call finish() before reusing"
            )
        t = self._pos
        self.obs[:, t] = _as_tensor(obs, torch.float32, self.device)
        self.dt[:, t] = _as_tensor(dt, torch.float32, self.device).reshape(self.num_envs)
        self.actions[:, t] = _as_tensor(action, torch.int64, self.device).reshape(self.num_envs)
        self.log_probs[:, t] = _as_tensor(log_prob, torch.float32, self.device).reshape(self.num_envs)
        self.values[:, t] = _as_tensor(value, torch.float32, self.device).reshape(self.num_envs)
        self.rewards[:, t] = _as_tensor(reward, torch.float32, self.device).reshape(self.num_envs)
        self.dones[:, t] = _as_tensor(done, torch.float32, self.device).reshape(self.num_envs)
        self._pos += 1

    def finish(self, last_value: torch.Tensor) -> SequenceBatch:
        """Close the rollout with ``V(s_T)`` and emit an immutable batch."""
        if self._init_state is None:
            raise RuntimeError("call start(init_state) before finish()")
        if self._pos != self.horizon_len:
            raise RuntimeError(
                f"buffer holds {self._pos} of {self.horizon_len} steps; rollout is incomplete"
            )
        lv = _as_tensor(last_value, torch.float32, self.device).reshape(self.num_envs)
        batch = SequenceBatch(
            obs=self.obs.clone(),
            dt=self.dt.clone(),
            actions=self.actions.clone(),
            log_probs=self.log_probs.clone(),
            values=self.values.clone(),
            rewards=self.rewards.clone(),
            dones=self.dones.clone(),
            init_state=tuple(s.clone() for s in self._init_state),
            last_value=lv.clone(),
        )
        self._pos = 0
        self._init_state = None
        return batch

    def __len__(self) -> int:
        """Number of time-steps currently stored (not sequences)."""
        return self._pos
