"""Deterministic policy evaluation.

Two entropy quantities are reported, under two unambiguous names, because the
previous version of this repository reported one under the other's name:

``mean_action_histogram_entropy``
    Entropy of the empirical histogram of the actions actually taken during a
    greedy (argmax) episode.  It measures how much the *realised behaviour* varied
    across channels.  It is not a property of the policy distribution.
    Also exposed as ``eval_action_histogram_entropy``.

``mean_policy_entropy_nats``
    Mean of ``Categorical(logits=pi(s)).entropy()`` over the states visited during
    evaluation.  This is the genuine distributional policy entropy.  It is ``0.0``
    for zero-parameter heuristic policies, which have no distribution.

Evaluation never touches the training RNG stream: greedy evaluation consumes no
randomness at all, and stochastic evaluation consumes only the explicitly supplied
``action_generator``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from ..envs.vector import SyncVectorEnv

__all__ = ["evaluate_policy", "EVAL_CHUNK_LANES"]

EVAL_CHUNK_LANES = 16


def _is_neural(policy) -> bool:
    """Duck-type on the recurrent-policy contract."""
    return hasattr(policy, "unroll") and hasattr(policy, "step")


def _chunks(seq: Sequence[int], size: int) -> list[list[int]]:
    seq = list(seq)
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def evaluate_policy(
    policy,
    config,
    episode_seeds: Sequence[int],
    deterministic: bool = True,
    device: str = "cpu",
    action_generator: torch.Generator | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate ``policy`` over ``len(episode_seeds)`` episodes.

    Episodes are run in synchronous chunks of at most ``EVAL_CHUNK_LANES`` lanes.
    Each episode's environment stream is a pure function of its seed, so the same
    ``episode_seeds`` produce byte-identical streams for every policy -- which is
    what makes the seed-paired statistics in the analysis layer valid.

    Returns ``(summary, per_episode)``.
    """
    episode_seeds = [int(s) for s in episode_seeds]
    if not episode_seeds:
        raise ValueError("episode_seeds must be non-empty")
    if not deterministic and _is_neural(policy) and action_generator is None:
        raise ValueError("stochastic evaluation requires an explicit action_generator")

    dev = torch.device(device)
    neural = _is_neural(policy)
    prior_mode = policy.training if neural else None
    if neural:
        policy.eval()  # guaranteed for the whole call, restored on exit

    per_episode: list[dict[str, float]] = []
    try:
        for chunk in _chunks(episode_seeds, EVAL_CHUNK_LANES):
            vec = SyncVectorEnv(config, chunk)
            obs_np, dt_np = vec.reset()
            n = vec.num_envs
            state = None
            if neural:
                state = tuple(s.to(dev) for s in policy.initial_state(n))
            else:
                policy.reset(n)

            entropy_sum = np.zeros(n, dtype=np.float64)
            entropy_steps = 0
            max_steps = int(getattr(config, "max_steps"))
            done_all = np.zeros(n, dtype=bool)

            with torch.no_grad():
                for _ in range(max_steps):
                    if neural:
                        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=dev)
                        dt = torch.as_tensor(dt_np, dtype=torch.float32, device=dev).reshape(-1)
                        logits, _value, state = policy.step(obs, dt, state)
                        dist = Categorical(logits=logits)
                        entropy_sum += dist.entropy().detach().cpu().numpy().astype(np.float64)
                        entropy_steps += 1
                        if deterministic:
                            action = torch.argmax(logits, dim=-1)
                        else:
                            action = torch.multinomial(
                                dist.probs, 1, generator=action_generator
                            ).squeeze(-1)
                        action_np = action.detach().cpu().numpy().astype(np.int64)
                    else:
                        action_np = np.asarray(
                            policy.act(obs_np, dt_np), dtype=np.int64
                        ).reshape(n)

                    obs_np, dt_np, _reward, done, _infos = vec.step(action_np)
                    done_all |= np.asarray(done, dtype=bool)
                    if bool(done_all.all()):
                        break

            metrics = vec.episode_metrics()
            ent = (entropy_sum / entropy_steps) if entropy_steps else np.zeros(n)
            for lane, m in enumerate(metrics):
                row = {str(k): float(v) for k, v in m.items()}
                row["policy_entropy_nats"] = float(ent[lane]) if neural else 0.0
                row["episode_seed"] = float(chunk[lane])
                per_episode.append(row)
    finally:
        if neural and prior_mode is not None:
            policy.train(prior_mode)

    metric_keys = [k for k in per_episode[0].keys() if k != "episode_seed"]
    summary: dict[str, float] = {}
    for k in metric_keys:
        vals = np.asarray([row[k] for row in per_episode], dtype=np.float64)
        summary[f"mean_{k}"] = float(vals.mean())
        summary[f"std_{k}"] = float(vals.std(ddof=0))

    if "mean_episode_return" not in summary:
        raise KeyError(
            "episode_metrics() must expose 'episode_return'; got "
            f"{sorted(metric_keys)}"
        )
    summary["mean_eval_return"] = summary["mean_episode_return"]
    summary["eval_action_histogram_entropy"] = summary.get(
        "mean_action_histogram_entropy", float("nan")
    )
    summary["num_eval_episodes"] = float(len(per_episode))
    return summary, per_episode
