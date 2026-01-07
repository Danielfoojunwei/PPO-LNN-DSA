"""
PPO-LSTM Baseline for Dynamic Spectrum Access

Implements PPO with LSTM backbone as baseline comparison for PPO-LNN.
Following the architecture from standard PPO-LSTM implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class LSTMEncoder(nn.Module):
    """
    LSTM-based encoder for sequential state processing.

    Unlike LNN/LTC which uses continuous-time dynamics,
    LSTM uses discrete gated updates.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, input_size) or (batch, seq, input_size)
            hidden: Previous hidden state (h, c)

        Returns:
            (output, new_hidden)
        """
        batch_size = x.size(0)
        device = x.device

        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)

        # Project input
        x = self.input_proj(x)

        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # LSTM forward
        lstm_out, new_hidden = self.lstm(x, hidden)

        # Take last output
        output = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Project output
        output = self.output_proj(output)

        return output, new_hidden


class PPOLSTMPolicy(nn.Module):
    """
    PPO Policy with LSTM backbone for Dynamic Spectrum Access.

    Architecture mirrors PPOLNNPolicy but uses LSTM instead of LNN/LTC.
    This serves as the baseline for comparison.
    """

    def __init__(
        self,
        state_dim: int = 42,
        action_dim: int = 20,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # LSTM encoder
        self.encoder = LSTMEncoder(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
        )

        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states."""
        return self.encoder.init_hidden(batch_size, device)

    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            state: State tensor (batch, state_dim)
            hidden: Previous hidden state

        Returns:
            (action_logits, value, new_hidden)
        """
        # Encode state
        encoded, new_hidden = self.encoder(state, hidden)

        # Get action logits and value
        action_logits = self.actor(encoded)
        value = self.critic(encoded).squeeze(-1)

        return action_logits, value, new_hidden

    def get_action(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample action from policy.

        Returns:
            (action, log_prob, value, new_hidden)
        """
        action_logits, value, new_hidden = self.forward(state, hidden)

        # Create distribution
        dist = Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value, new_hidden

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log prob and entropy for given actions.

        Returns:
            (log_probs, entropy, values)
        """
        action_logits, values, _ = self.forward(states, hidden)

        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values


class PPOLSTMTrainer:
    """
    PPO training loop for LSTM policy.
    """

    def __init__(
        self,
        policy: PPOLSTMPolicy,
        lr: float = 3e-4,
        gamma: float = 0.95,
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
        """Compute Generalized Advantage Estimation."""
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

    def update(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """PPO update step."""
        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]

        # Compute advantages
        with torch.no_grad():
            _, next_value, _ = self.policy.forward(states[-1:])
            next_value = next_value.squeeze()

        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_samples = len(rewards)
        indices = np.arange(total_samples)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        n_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, total_samples, self.minibatch_size):
                end = min(start + self.minibatch_size, total_samples)
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Evaluate actions
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    mb_states, mb_actions
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

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["approx_kl"] += approx_kl
                n_updates += 1

        # Average metrics
        for k in metrics:
            metrics[k] /= max(1, n_updates)

        return metrics


class PPOLSTMAgent:
    """
    Complete PPO-LSTM agent for DSA benchmark.

    Wraps policy and trainer with episode management.
    """

    def __init__(
        self,
        state_dim: int = 42,
        action_dim: int = 20,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.95,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create policy
        self.policy = PPOLSTMPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
        ).to(self.device)

        # Create trainer
        self.trainer = PPOLSTMTrainer(
            self.policy,
            lr=lr,
            gamma=gamma,
        )

        # Hidden state
        self.hidden = None

        # Rollout buffer
        self.buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
        }

        # Stats
        self.total_steps = 0

    def reset_hidden(self):
        """Reset hidden state for new episode."""
        self.hidden = None

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select action given state.

        Returns:
            (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, self.hidden = self.policy.get_action(
                state_tensor, self.hidden, deterministic
            )

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ):
        """Store transition in buffer."""
        self.buffer["states"].append(torch.FloatTensor(state))
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["values"].append(value)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        """Update policy using collected rollout."""
        if len(self.buffer["states"]) == 0:
            return {"loss": 0.0}

        # Convert to tensors
        rollout = {
            "states": torch.stack(self.buffer["states"]).to(self.device),
            "actions": torch.tensor(self.buffer["actions"]).to(self.device),
            "log_probs": torch.tensor(self.buffer["log_probs"]).to(self.device),
            "values": torch.tensor(self.buffer["values"]).to(self.device),
            "rewards": torch.tensor(self.buffer["rewards"], dtype=torch.float32).to(self.device),
            "dones": torch.tensor(self.buffer["dones"], dtype=torch.float32).to(self.device),
        }

        # Update
        metrics = self.trainer.update(rollout)

        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []

        return metrics

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
