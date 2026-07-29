"""Recurrent PPO with carried hidden state.

Contract (REBUILD_SPEC §4.4).  Every numbered item below is enforced by a test in
``tests/test_ppo.py``:

1.  Rollout runs under ``model.eval()`` and ``torch.no_grad()``.
2.  Only ``init_state`` is stored; the hidden trajectory is recomputed from it in
    every minibatch of every epoch via :meth:`RecurrentActorCritic.unroll`.
3.  Minibatching is over the **sequence** axis and is reachable by construction.
4.  GAE bootstraps ``V(s_T)`` at the horizon (time-limit truncation) and masks the
    bootstrap only on genuine termination.
5.  ``first_epoch_max_ratio_deviation`` must be ~0: on epoch 0 / minibatch 0 the
    recomputed policy is the behaviour policy, so the PPO ratio is exactly 1.
6.  Action sampling uses an explicit ``torch.Generator``; the global RNG is never
    consulted, so evaluation cannot contaminate the training stream.
7.  ``total_env_steps`` is incremented by ``num_envs`` per synchronous vector step
    and is the only permitted source of the reported ``train_steps``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from .buffer import SequenceBatch, SequenceRolloutBuffer

__all__ = [
    "PPOHyperParams",
    "RecurrentPPO",
    "compute_gae",
    "train",
]


@dataclass(frozen=True)
class PPOHyperParams:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 6
    num_sequence_minibatches: int = 4
    num_envs: int = 16
    horizon: int = 64

    def to_dict(self) -> dict[str, float | int]:
        return dict(asdict(self))

    def minibatch_size(self) -> int:
        """Sequences per minibatch.  Ceil-division so the split is always reachable."""
        return max(1, math.ceil(self.num_envs / max(1, self.num_sequence_minibatches)))

    def minibatches_per_epoch(self, num_sequences: int | None = None) -> int:
        n = self.num_envs if num_sequences is None else int(num_sequences)
        return math.ceil(n / self.minibatch_size())

    def gradient_steps_per_update(self, num_sequences: int | None = None) -> int:
        return self.update_epochs * self.minibatches_per_epoch(num_sequences)


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over an ``(N, T)`` grid.

    ``dones[:, t]`` is 1.0 iff the state reached after step ``t`` is a genuine
    terminal state (value zero by definition).  Time-limit truncation is NOT a
    terminal: at ``t == T-1`` with ``dones == 0`` the estimator bootstraps
    ``last_value = V(s_T)``, which is the correct treatment of a truncated episode.

    Returns ``(advantages, returns)`` each ``(N, T)``.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values and dones must share shape (N, T)")
    n, t_len = rewards.shape
    if last_value.shape != (n,):
        raise ValueError(f"last_value must have shape ({n},), got {tuple(last_value.shape)}")

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(n, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(t_len)):
        next_non_terminal = 1.0 - dones[:, t]
        next_value = last_value if t == t_len - 1 else values[:, t + 1]
        delta = rewards[:, t] + gamma * next_value * next_non_terminal - values[:, t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[:, t] = gae
    returns = advantages + values
    return advantages, returns


def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true, unbiased=False)
    if float(var_y) == 0.0:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class RecurrentPPO:
    """PPO for a policy that carries hidden state across environment steps."""

    def __init__(
        self,
        model,
        hparams: PPOHyperParams,
        action_generator: torch.Generator,
        shuffle_generator: torch.Generator,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.hparams = hparams
        self.action_generator = action_generator
        self.shuffle_generator = shuffle_generator
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=hparams.learning_rate, eps=1e-5
        )
        self.total_env_steps: int = 0
        self.total_gradient_steps: int = 0
        self.total_rollouts: int = 0
        # Real, distributional policy entropy (nats) under the behaviour policy of
        # the most recent rollout.  NOT an action histogram -- see evaluate.py.
        self.last_rollout_policy_entropy: float = float("nan")
        self.last_rollout_mean_return: float = float("nan")

    # -------------------------------------------------------------- rollout #

    def collect(self, vec_env) -> SequenceBatch:
        """Run one synchronous rollout of ``horizon`` steps over ``num_envs`` lanes."""
        hp = self.hparams
        if vec_env.num_envs != hp.num_envs:
            raise ValueError(
                f"vec_env has {vec_env.num_envs} lanes but hparams.num_envs={hp.num_envs}"
            )
        was_training = self.model.training
        self.model.eval()  # rule 1: no dropout / no batchnorm drift during rollout
        try:
            buf = SequenceRolloutBuffer(
                hp.num_envs, hp.horizon, vec_env.obs_dim, self.model.state_sizes, self.device
            )
            state = self.model.initial_state(hp.num_envs)
            state = tuple(s.to(self.device) for s in state)
            buf.start(state)

            obs_np, dt_np = vec_env.reset()
            entropy_sum = 0.0
            with torch.no_grad():
                for _ in range(hp.horizon):
                    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
                    dt = torch.as_tensor(dt_np, dtype=torch.float32, device=self.device).reshape(-1)
                    logits, value, next_state = self.model.step(obs, dt, state)
                    dist = Categorical(logits=logits)
                    # rule 6: explicit generator, never Categorical.sample()
                    action = torch.multinomial(
                        dist.probs, 1, generator=self.action_generator
                    ).squeeze(-1)
                    log_prob = dist.log_prob(action)
                    entropy_sum += float(dist.entropy().mean())

                    nobs_np, ndt_np, reward_np, _done_np, infos = vec_env.step(
                        action.detach().cpu().numpy()
                    )
                    # Genuine termination only.  The spectrum environment never
                    # terminates -- it truncates at max_steps -- so this is all
                    # zeros in practice, and V(s_T) is bootstrapped at the horizon.
                    terminal = np.array(
                        [float(bool(i.get("terminated", False))) for i in infos],
                        dtype=np.float32,
                    )
                    buf.add(obs, dt, action, log_prob, value, reward_np, terminal)

                    state = next_state
                    obs_np, dt_np = nobs_np, ndt_np
                    self.total_env_steps += hp.num_envs  # rule 7

                last_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
                last_dt = torch.as_tensor(dt_np, dtype=torch.float32, device=self.device).reshape(-1)
                _, last_value, _ = self.model.step(last_obs, last_dt, state)

            batch = buf.finish(last_value)
        finally:
            self.model.train(was_training)

        self.total_rollouts += 1
        self.last_rollout_policy_entropy = entropy_sum / hp.horizon
        self.last_rollout_mean_return = float(batch.rewards.sum(dim=1).mean())
        return batch

    # --------------------------------------------------------------- update #

    def update(self, batch: SequenceBatch) -> dict[str, float]:
        hp = self.hparams
        self.model.train()  # rule 1 (mirror): training mode only inside update

        n_seq = batch.num_sequences()
        advantages, returns = compute_gae(
            batch.rewards, batch.values, batch.dones, batch.last_value, hp.gamma, hp.gae_lambda
        )
        # rule 5: normalise once over the whole (N*T) batch, never per-minibatch.
        adv_flat = advantages.reshape(-1)
        adv_norm = (advantages - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)

        mb_size = hp.minibatch_size()
        n_minibatches = hp.minibatches_per_epoch(n_seq)

        first_epoch_max_ratio_deviation = float("nan")
        later_max_ratio_deviation = 0.0
        acc = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        n_steps = 0

        for epoch in range(hp.update_epochs):
            perm = torch.randperm(n_seq, generator=self.shuffle_generator).to(self.device)
            for mb_i in range(n_minibatches):
                idx = perm[mb_i * mb_size : (mb_i + 1) * mb_size]
                if idx.numel() == 0:
                    continue
                obs_mb = batch.obs[idx]
                dt_mb = batch.dt[idx]
                act_mb = batch.actions[idx]
                old_logp_mb = batch.log_probs[idx]
                adv_mb = adv_norm[idx]
                ret_mb = returns[idx]
                init_mb = tuple(s[idx] for s in batch.init_state)

                # rule 2: replay the whole hidden trajectory from init_state.
                # No stored per-step states, no detach inside the unroll.
                logits, values_new, _ = self.model.unroll(obs_mb, dt_mb, init_mb)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                logratio = new_logp - old_logp_mb
                ratio = logratio.exp()
                max_dev = float((ratio - 1.0).abs().max().detach())
                if epoch == 0 and mb_i == 0:
                    first_epoch_max_ratio_deviation = max_dev
                else:
                    later_max_ratio_deviation = max(later_max_ratio_deviation, max_dev)

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - hp.clip_epsilon, 1.0 + hp.clip_epsilon) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (values_new - ret_mb).pow(2).mean()
                loss = policy_loss + hp.value_coef * value_loss - hp.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), hp.max_grad_norm
                )
                self.optimizer.step()
                self.total_gradient_steps += 1

                with torch.no_grad():
                    acc["policy_loss"] += float(policy_loss)
                    acc["value_loss"] += float(value_loss)
                    acc["entropy"] += float(entropy)
                    acc["approx_kl"] += float(((ratio - 1.0) - logratio).mean())
                    acc["clip_fraction"] += float(
                        ((ratio - 1.0).abs() > hp.clip_epsilon).float().mean()
                    )
                    acc["grad_norm"] += float(grad_norm)
                n_steps += 1

        out = {k: (v / n_steps if n_steps else float("nan")) for k, v in acc.items()}
        out["explained_variance"] = _explained_variance(
            batch.values.reshape(-1), returns.reshape(-1)
        )
        out["first_epoch_max_ratio_deviation"] = first_epoch_max_ratio_deviation
        out["later_max_ratio_deviation"] = later_max_ratio_deviation
        out["gradient_steps"] = float(n_steps)
        out["minibatches_per_epoch"] = float(n_minibatches)
        out["minibatch_size"] = float(mb_size)
        # The genuine distributional entropy of the behaviour policy that produced
        # this batch.  Distinct from the eval-time action histogram entropy.
        out["policy_entropy_mean"] = self.last_rollout_policy_entropy
        out["mean_rollout_return"] = self.last_rollout_mean_return
        return out

    # ------------------------------------------------------------ snapshot #

    def run_snapshot(self) -> dict[str, float | int | str]:
        """Everything the runner needs to reproduce this agent's trajectory.

        ``train_steps`` here is the *measured* count -- it is incremented by
        ``num_envs`` inside :meth:`collect` and is never a declared or derived
        figure.  The old repository recorded ``rounds * local_timesteps`` while
        executing six times that many steps (defect D8); reading this field is the
        only sanctioned way to populate a ``train_steps`` column.
        """
        snap: dict[str, float | int | str] = {
            "train_steps": int(self.total_env_steps),
            "total_env_steps": int(self.total_env_steps),
            "total_gradient_steps": int(self.total_gradient_steps),
            "total_rollouts": int(self.total_rollouts),
            "action_generator_seed": int(self.action_generator.initial_seed()),
            "shuffle_generator_seed": int(self.shuffle_generator.initial_seed()),
            "device": str(self.device),
        }
        snap.update({f"ppo_{k}": v for k, v in self.hparams.to_dict().items()})
        return snap

    # ----------------------------------------------------------- checkpoint #

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, sd: Mapping[str, torch.Tensor]) -> None:
        self.model.load_state_dict({k: v for k, v in sd.items()})


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def train(
    agent: RecurrentPPO,
    vec_env,
    total_env_steps: int,
    rollout_seed_fn: Callable[[int], Sequence[int]],
) -> list[dict[str, float]]:
    """Run ``ceil(total_env_steps / (num_envs*horizon))`` rollout+update cycles.

    Before each cycle ``r`` the vector environment is reseeded with
    ``rollout_seed_fn(r)``, so the environment stream is a pure function of the
    base seed and the rollout index -- never of construction order.

    ``torch.set_num_threads(1)`` is the caller's responsibility (worker entry point).
    """
    hp = agent.hparams
    batch_steps = hp.num_envs * hp.horizon
    num_rollouts = math.ceil(int(total_env_steps) / batch_steps)
    logs: list[dict[str, float]] = []
    for r in range(num_rollouts):
        seeds = list(rollout_seed_fn(r))
        vec_env.reseed(seeds)
        batch = agent.collect(vec_env)
        metrics = agent.update(batch)
        metrics["rollout_index"] = float(r)
        metrics["total_env_steps"] = float(agent.total_env_steps)
        metrics["total_gradient_steps"] = float(agent.total_gradient_steps)
        logs.append(metrics)
    return logs
