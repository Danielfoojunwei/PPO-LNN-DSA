"""PPO training and evaluation utilities for the empirical DSA benchmark."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from empirical.models import build_model, count_parameters


@dataclass
class PPOHyperParams:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 128
    rollout_steps: int = 512

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


class RolloutBuffer:
    """Rollout storage for PPO."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.dt: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.infos: List[Dict[str, float]] = []

    def add(
        self,
        state: Dict[str, np.ndarray],
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        info: Dict[str, float],
    ) -> None:
        self.obs.append(np.asarray(state["obs"], dtype=np.float32))
        self.dt.append(np.asarray(state["dt"], dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.infos.append(info)

    def __len__(self) -> int:
        return len(self.actions)

    def clear(self) -> None:
        self.__init__()


class PPOAgent:
    """Model-agnostic PPO agent for the empirical benchmark suite."""

    def __init__(
        self,
        model_name: str,
        obs_dim: int,
        seq_len: int,
        action_dim: int,
        device: str = "cpu",
        hyperparams: PPOHyperParams | None = None,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.hparams = hyperparams or PPOHyperParams()

        self.model = build_model(model_name, obs_dim=obs_dim, seq_len=seq_len, action_dim=action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        self.buffer = RolloutBuffer()
        self.total_env_steps = 0

    @property
    def parameter_count(self) -> int:
        return count_parameters(self.model)

    def state_to_torch(self, state: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs = torch.tensor(state["obs"], dtype=torch.float32, device=self.device).unsqueeze(0)
        dt = torch.tensor(state["dt"], dtype=torch.float32, device=self.device).unsqueeze(0)
        return {"obs": obs, "dt": dt}

    def act(self, state: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[int, float, float, float]:
        state_t = self.state_to_torch(state)
        with torch.no_grad():
            action, log_prob, value, entropy = self.model.act(state_t, deterministic=deterministic)
        return int(action.item()), float(log_prob.item()), float(value.item()), float(entropy.mean().item())

    def predict_value(self, state: Dict[str, np.ndarray]) -> float:
        state_t = self.state_to_torch(state)
        with torch.no_grad():
            _, value = self.model.forward(state_t)
        return float(value.item())

    def store_transition(
        self,
        state: Dict[str, np.ndarray],
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        info: Dict[str, float],
    ) -> None:
        self.buffer.add(state, action, log_prob, value, reward, done, info)
        self.total_env_steps += 1

    def _buffer_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.tensor(np.stack(self.buffer.obs), dtype=torch.float32, device=self.device),
            "dt": torch.tensor(np.stack(self.buffer.dt), dtype=torch.float32, device=self.device),
            "actions": torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device),
            "log_probs": torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device),
            "values": torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device),
        }

    def update(self, next_state: Dict[str, np.ndarray]) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        tensors = self._buffer_tensors()
        with torch.no_grad():
            next_value = torch.tensor(self.predict_value(next_state), dtype=torch.float32, device=self.device)

        rewards = tensors["rewards"]
        dones = tensors["dones"]
        values = tensors["values"]

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            delta = rewards[t] + self.hparams.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.hparams.gamma * self.hparams.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        updates = 0

        batch_size = rewards.shape[0]
        minibatch_size = min(self.hparams.minibatch_size, batch_size)

        for _ in range(self.hparams.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                state_mb = {
                    "obs": tensors["obs"][mb_idx],
                    "dt": tensors["dt"][mb_idx],
                }
                actions_mb = tensors["actions"][mb_idx]
                old_log_probs_mb = tensors["log_probs"][mb_idx]
                advantages_mb = advantages[mb_idx]
                returns_mb = returns[mb_idx]

                new_log_probs, entropy, value = self.model.evaluate_actions(state_mb, actions_mb)
                ratio = (new_log_probs - old_log_probs_mb).exp()
                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1.0 - self.hparams.clip_epsilon, 1.0 + self.hparams.clip_epsilon) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, returns_mb)
                entropy_loss = entropy.mean()
                loss = policy_loss + self.hparams.value_coef * value_loss - self.hparams.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs_mb - new_log_probs).mean().abs().item()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_loss.item())
                total_kl += approx_kl
                updates += 1

        self.buffer.clear()
        return {
            "policy_loss": total_policy_loss / max(1, updates),
            "value_loss": total_value_loss / max(1, updates),
            "entropy": total_entropy / max(1, updates),
            "approx_kl": total_kl / max(1, updates),
        }

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def set_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_name": self.model_name,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_env_steps": self.total_env_steps,
                "hyperparams": self.hparams.to_dict(),
            },
            path,
        )


def run_training_episode(env, agent: PPOAgent, deterministic: bool = False) -> Dict[str, float]:
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    action_start = time.perf_counter()
    forward_times: List[float] = []

    while not done:
        t0 = time.perf_counter()
        action, log_prob, value, _ = agent.act(state, deterministic=deterministic)
        forward_times.append(time.perf_counter() - t0)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if not deterministic:
            agent.store_transition(state, action, log_prob, value, reward, done, info)
        total_reward += reward
        steps += 1
        state = next_state

    metrics = env.get_metrics()
    metrics.update(
        {
            "episode_reward": float(total_reward),
            "episode_length": float(steps),
            "mean_forward_time": float(np.mean(forward_times) if forward_times else 0.0),
            "wall_clock_episode_seconds": float(time.perf_counter() - action_start),
        }
    )
    return metrics


def evaluate_agent(agent: PPOAgent, env_factory, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
    episode_metrics: List[Dict[str, float]] = []
    for _ in range(num_episodes):
        env = env_factory()
        metrics = run_training_episode(env, agent, deterministic=deterministic)
        episode_metrics.append(metrics)

    keys = sorted(episode_metrics[0].keys()) if episode_metrics else []
    summary: Dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in episode_metrics], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std(ddof=0))
    return summary


def train_agent(
    agent: PPOAgent,
    env_factory,
    total_timesteps: int,
    eval_every: int,
    eval_episodes: int,
) -> Dict[str, List[Dict[str, float]]]:
    train_history: List[Dict[str, float]] = []
    eval_history: List[Dict[str, float]] = []
    update_history: List[Dict[str, float]] = []

    last_eval_step = 0
    while agent.total_env_steps < total_timesteps:
        env = env_factory()
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        wall_start = time.perf_counter()

        while not done and agent.total_env_steps < total_timesteps:
            action, log_prob, value, _ = agent.act(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, log_prob, value, reward, done, info)
            episode_reward += reward
            episode_steps += 1
            state = next_state

            if len(agent.buffer) >= agent.hparams.rollout_steps:
                update_history.append(agent.update(next_state))

            if agent.total_env_steps - last_eval_step >= eval_every:
                eval_history.append({"step": float(agent.total_env_steps), **evaluate_agent(agent, env_factory, num_episodes=eval_episodes)})
                last_eval_step = agent.total_env_steps

        if len(agent.buffer) > 0:
            update_history.append(agent.update(state))

        env_metrics = env.get_metrics()
        env_metrics.update(
            {
                "train_step": float(agent.total_env_steps),
                "episode_reward": float(episode_reward),
                "episode_length": float(episode_steps),
                "wall_clock_training_seconds": float(time.perf_counter() - wall_start),
            }
        )
        train_history.append(env_metrics)

    return {
        "train_history": train_history,
        "eval_history": eval_history,
        "update_history": update_history,
    }
