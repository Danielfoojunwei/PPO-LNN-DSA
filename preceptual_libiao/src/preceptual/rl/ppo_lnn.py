"""
PPO-LNN Hierarchical Policy for Airtime OS

Implements Proximal Policy Optimization with LNN backbone for:
- Slice budget allocation
- Congestion mode selection
- Scan scheduling
- Handover decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from .lnn_ltc import LNNBackbone, DTAwareLNNEncoder


class GlobalHead(nn.Module):
    """
    Global policy head for system-wide decisions.

    Outputs:
    - Slice budgets (6 continuous values)
    - Congestion mode (4-way categorical)
    - Scan quota (continuous)
    """

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Slice budgets (mean and log_std)
        self.slice_mean = nn.Linear(hidden_size, 6)
        self.slice_log_std = nn.Parameter(torch.zeros(6))

        # Congestion mode
        self.mode_logits = nn.Linear(hidden_size, 4)

        # Scan quota (mean and log_std)
        self.scan_mean = nn.Linear(hidden_size, 1)
        self.scan_log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Encoded state (batch, input_size)

        Returns:
            Dict with distribution parameters
        """
        h = self.shared(x)

        # Slice budgets (clamped to valid range)
        slice_mean = torch.sigmoid(self.slice_mean(h))  # [0, 1]
        slice_std = torch.exp(self.slice_log_std.clamp(-5, 2))

        # Congestion mode
        mode_logits = self.mode_logits(h)

        # Scan quota
        scan_mean = torch.sigmoid(self.scan_mean(h))  # [0, 1] fraction
        scan_std = torch.exp(self.scan_log_std.clamp(-5, 2))

        return {
            "slice_mean": slice_mean,
            "slice_std": slice_std,
            "mode_logits": mode_logits,
            "scan_mean": scan_mean,
            "scan_std": scan_std,
        }


class RobotHead(nn.Module):
    """
    Per-robot policy head for switching decisions.

    Outputs:
    - HOLD/ARM/EXECUTE decision (3-way categorical)
    - Target candidate index (if switching)
    """

    def __init__(self, input_size: int, hidden_size: int = 64, max_candidates: int = 10):
        super().__init__()
        self.max_candidates = max_candidates

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Action type: HOLD, ARM_SWITCH, EXECUTE_SWITCH
        self.action_logits = nn.Linear(hidden_size, 3)

        # Candidate selection (if switching)
        self.candidate_logits = nn.Linear(hidden_size, max_candidates)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a single robot.

        Args:
            x: Robot encoding (batch, input_size)

        Returns:
            Dict with distribution parameters
        """
        h = self.shared(x)

        action_logits = self.action_logits(h)
        candidate_logits = self.candidate_logits(h)

        return {
            "action_logits": action_logits,
            "candidate_logits": candidate_logits,
        }


class ValueHead(nn.Module):
    """Value function head for advantage estimation."""

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOLNNPolicy(nn.Module):
    """
    PPO Policy with LNN backbone for Airtime OS.

    Hierarchical structure:
    1. Shared LNN encoder processes fleet state with dt-awareness
    2. Global head outputs slice budgets, congestion mode, scan quota
    3. Robot heads output switching decisions for top-K risk robots
    """

    def __init__(
        self,
        global_obs_size: int = 42,
        robot_obs_size: int = 14,
        hidden_size: int = 128,
        num_lnn_layers: int = 2,
        max_robots: int = 50,
        top_k_robots: int = 10,
    ):
        super().__init__()
        self.global_obs_size = global_obs_size
        self.robot_obs_size = robot_obs_size
        self.hidden_size = hidden_size
        self.max_robots = max_robots
        self.top_k_robots = top_k_robots

        # Shared LNN encoder
        self.encoder = DTAwareLNNEncoder(
            global_input_size=global_obs_size,
            robot_input_size=robot_obs_size,
            hidden_size=hidden_size,
            num_layers=num_lnn_layers,
            max_robots=max_robots,
        )

        # Policy heads
        self.global_head = GlobalHead(hidden_size, hidden_size)
        self.robot_head = RobotHead(hidden_size // 2 + robot_obs_size, hidden_size // 2)

        # Value head
        self.value_head = ValueHead(hidden_size)

        # Robot feature projection
        self.robot_proj = nn.Linear(robot_obs_size, hidden_size // 2)

    def init_hidden(self, batch_size: int, device: torch.device) -> dict:
        """Initialize hidden states."""
        return self.encoder.init_hidden(batch_size, device)

    def forward(
        self,
        global_obs: torch.Tensor,
        robot_obs: torch.Tensor,
        robot_mask: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        hidden: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            global_obs: Global observation (batch, global_obs_size)
            robot_obs: Robot observations (batch, max_robots, robot_obs_size)
            robot_mask: Valid robot mask (batch, max_robots)
            dt: Time delta (batch,)
            hidden: Previous hidden states

        Returns:
            (action_dist_params, value, new_hidden)
        """
        batch_size = global_obs.size(0)
        device = global_obs.device

        # Encode state
        encoded, new_hidden = self.encoder(
            global_obs, robot_obs, robot_mask, dt, hidden
        )

        # Global policy
        global_params = self.global_head(encoded)

        # Robot policy for top-K robots
        # In practice, select based on risk scores from observation
        robot_params_list = []
        for k in range(self.top_k_robots):
            if k < self.max_robots:
                robot_feat = robot_obs[:, k, :]
                robot_enc = self.robot_proj(robot_feat)
                robot_input = torch.cat([encoded[:, :self.hidden_size // 2], robot_feat], dim=-1)
                robot_params = self.robot_head(robot_input)
                robot_params_list.append(robot_params)

        # Value function
        value = self.value_head(encoded)

        # Combine all parameters
        action_params = {
            "global": global_params,
            "robots": robot_params_list,
        }

        return action_params, value, new_hidden

    def get_action(
        self,
        global_obs: torch.Tensor,
        robot_obs: torch.Tensor,
        robot_mask: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        hidden: Optional[dict] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """
        Sample action from policy.

        Returns:
            (actions, log_probs, value, new_hidden)
        """
        action_params, value, new_hidden = self.forward(
            global_obs, robot_obs, robot_mask, dt, hidden
        )

        # Sample global actions
        global_p = action_params["global"]

        # Slice budgets
        if deterministic:
            slice_budgets = global_p["slice_mean"]
        else:
            slice_dist = Normal(global_p["slice_mean"], global_p["slice_std"])
            slice_budgets = slice_dist.sample()
            slice_budgets = torch.clamp(slice_budgets, 0, 1)

        slice_log_prob = Normal(global_p["slice_mean"], global_p["slice_std"]).log_prob(slice_budgets).sum(-1)

        # Congestion mode
        mode_dist = Categorical(logits=global_p["mode_logits"])
        if deterministic:
            cong_mode = global_p["mode_logits"].argmax(-1)
        else:
            cong_mode = mode_dist.sample()
        mode_log_prob = mode_dist.log_prob(cong_mode)

        # Scan quota
        if deterministic:
            scan_quota = global_p["scan_mean"]
        else:
            scan_dist = Normal(global_p["scan_mean"], global_p["scan_std"])
            scan_quota = scan_dist.sample()
            scan_quota = torch.clamp(scan_quota, 0, 1)

        scan_log_prob = Normal(global_p["scan_mean"], global_p["scan_std"]).log_prob(scan_quota).sum(-1)

        # Robot actions
        robot_actions = []
        robot_log_probs = []
        for robot_params in action_params["robots"]:
            action_dist = Categorical(logits=robot_params["action_logits"])
            if deterministic:
                action = robot_params["action_logits"].argmax(-1)
            else:
                action = action_dist.sample()

            candidate_dist = Categorical(logits=robot_params["candidate_logits"])
            if deterministic:
                candidate = robot_params["candidate_logits"].argmax(-1)
            else:
                candidate = candidate_dist.sample()

            robot_actions.append({
                "action": action,
                "candidate": candidate,
            })
            robot_log_probs.append(
                action_dist.log_prob(action) + candidate_dist.log_prob(candidate)
            )

        # Aggregate log probs
        total_log_prob = (
            slice_log_prob +
            mode_log_prob +
            scan_log_prob +
            sum(robot_log_probs) / max(1, len(robot_log_probs))
        )

        actions = {
            "slice_budgets": slice_budgets,
            "congestion_mode": cong_mode,
            "scan_quota": scan_quota,
            "robot_actions": robot_actions,
        }

        return actions, total_log_prob, value, new_hidden

    def evaluate_actions(
        self,
        global_obs: torch.Tensor,
        robot_obs: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        robot_mask: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        hidden: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log prob and entropy for given actions.

        Returns:
            (log_probs, entropy, values)
        """
        action_params, value, _ = self.forward(
            global_obs, robot_obs, robot_mask, dt, hidden
        )

        global_p = action_params["global"]

        # Slice budgets
        slice_dist = Normal(global_p["slice_mean"], global_p["slice_std"])
        slice_log_prob = slice_dist.log_prob(actions["slice_budgets"]).sum(-1)
        slice_entropy = slice_dist.entropy().sum(-1)

        # Congestion mode
        mode_dist = Categorical(logits=global_p["mode_logits"])
        mode_log_prob = mode_dist.log_prob(actions["congestion_mode"])
        mode_entropy = mode_dist.entropy()

        # Scan quota
        scan_dist = Normal(global_p["scan_mean"], global_p["scan_std"])
        scan_log_prob = scan_dist.log_prob(actions["scan_quota"]).sum(-1)
        scan_entropy = scan_dist.entropy().sum(-1)

        # Robot actions
        robot_log_probs = []
        robot_entropies = []
        for i, robot_params in enumerate(action_params["robots"]):
            if i < len(actions.get("robot_actions", [])):
                ra = actions["robot_actions"][i]
                action_dist = Categorical(logits=robot_params["action_logits"])
                candidate_dist = Categorical(logits=robot_params["candidate_logits"])

                robot_log_probs.append(
                    action_dist.log_prob(ra["action"]) +
                    candidate_dist.log_prob(ra["candidate"])
                )
                robot_entropies.append(
                    action_dist.entropy() + candidate_dist.entropy()
                )

        # Aggregate
        total_log_prob = (
            slice_log_prob +
            mode_log_prob +
            scan_log_prob +
            (sum(robot_log_probs) / max(1, len(robot_log_probs)) if robot_log_probs else 0)
        )

        total_entropy = (
            slice_entropy +
            mode_entropy +
            scan_entropy +
            (sum(robot_entropies) / max(1, len(robot_entropies)) if robot_entropies else 0)
        )

        return total_log_prob, total_entropy, value


class PPOTrainer:
    """
    PPO training loop for Airtime OS policy.
    """

    def __init__(
        self,
        policy: PPOLNNPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: (T,) rewards
            values: (T,) value estimates
            dones: (T,) done flags
            next_value: Final value estimate

        Returns:
            (advantages, returns)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam
            )

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        PPO update step.

        Args:
            rollout: Dict containing:
                - global_obs, robot_obs, robot_mask, dt
                - actions, log_probs, values, rewards, dones

        Returns:
            Training metrics
        """
        # Compute advantages
        with torch.no_grad():
            # Get next value
            last_obs = {
                "global_obs": rollout["global_obs"][-1:],
                "robot_obs": rollout["robot_obs"][-1:],
                "robot_mask": rollout.get("robot_mask", None),
                "dt": rollout.get("dt", None),
            }
            if last_obs["robot_mask"] is not None:
                last_obs["robot_mask"] = last_obs["robot_mask"][-1:]
            if last_obs["dt"] is not None:
                last_obs["dt"] = last_obs["dt"][-1:]

            _, _, next_value, _ = self.policy.get_action(
                last_obs["global_obs"],
                last_obs["robot_obs"],
                last_obs["robot_mask"],
                last_obs["dt"],
                deterministic=True,
            )

        advantages, returns = self.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            next_value,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_samples = len(rollout["rewards"])
        indices = np.arange(total_samples)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        n_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, total_samples, self.minibatch_size):
                end = min(start + self.minibatch_size, total_samples)
                mb_indices = indices[start:end]

                # Get minibatch
                mb_global_obs = rollout["global_obs"][mb_indices]
                mb_robot_obs = rollout["robot_obs"][mb_indices]
                mb_actions = {
                    k: v[mb_indices] if isinstance(v, torch.Tensor) else [v[i] for i in mb_indices]
                    for k, v in rollout["actions"].items()
                }
                mb_old_log_probs = rollout["log_probs"][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Evaluate actions
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    mb_global_obs,
                    mb_robot_obs,
                    mb_actions,
                )

                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, mb_returns)

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy.mean()
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_fraction"] += clip_frac
                n_updates += 1

        # Average metrics
        for k in metrics:
            metrics[k] /= max(1, n_updates)

        return metrics
