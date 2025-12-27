"""
PPO-LFM Agent Implementation

Proximal Policy Optimization with Liquid Foundation Models for
dynamic spectrum access in federated learning scenarios.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from models.actor_critic import LFMActorCritic


class RolloutBuffer:
    """
    Buffer for storing trajectories during PPO rollout.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def add(self, state, action, log_prob, value, reward, done):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            last_value: Value estimate for the last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        advantages = []
        gae = 0

        # Convert to tensors
        values = torch.cat(self.values)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        # Append last value
        values = torch.cat([values, last_value.unsqueeze(0)])

        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        self.advantages = torch.stack(advantages)
        self.returns = self.advantages + values[:-1]

    def get(self):
        """Get all data from the buffer."""
        return {
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'log_probs': torch.stack(self.log_probs),
            'values': torch.cat(self.values),
            'advantages': self.advantages,
            'returns': self.returns
        }

    def clear(self):
        """Clear the buffer."""
        self.__init__()


class PPOLFMAgent:
    """
    PPO Agent using Liquid Foundation Models.

    This agent replaces the traditional PPO-LSTM with PPO-LFM for
    improved temporal modeling and efficiency.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_lfm_layers=2,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cpu'
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension for LFM
            num_lfm_layers: Number of LFM layers
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on (cpu/cuda)
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Initialize actor-critic network with LFM
        self.policy = LFMActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_lfm_layers=num_lfm_layers
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Tracking metrics
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)

    def select_action(self, state, deterministic=False):
        """
        Select action given state.

        Args:
            state: State tensor [seq_len, state_dim]
            deterministic: Whether to select action deterministically

        Returns:
            action: Selected action (int)
            log_prob: Log probability of action
            value: State value estimate
        """
        # Add batch dimension if needed
        if len(state.shape) == 2:
            state = state.unsqueeze(0)  # [1, seq_len, state_dim]

        state = state.to(self.device)

        with torch.no_grad():
            action, log_prob, value, _ = self.policy(state, deterministic)

        return action.item(), log_prob, value

    def store_transition(self, state, action, log_prob, value, reward, done):
        """Store a transition in the rollout buffer."""
        # Convert to tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)

        self.buffer.add(state, action, log_prob, value, reward, done)

    def update(self, n_epochs=10, batch_size=64):
        """
        Update the policy using PPO.

        Args:
            n_epochs: Number of optimization epochs
            batch_size: Mini-batch size for updates

        Returns:
            Dictionary of training metrics
        """
        # Get last value for advantage computation
        last_state = self.buffer.states[-1].unsqueeze(0).to(self.device)
        last_value = self.policy.get_value(last_state)

        # Compute advantages
        self.buffer.compute_advantages(last_value, self.gamma, self.gae_lambda)

        # Get all data
        data = self.buffer.get()
        states = data['states'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        advantages = data['advantages'].to(self.device)
        returns = data['returns'].to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # Multiple epochs of optimization
        for epoch in range(n_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions with current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                # Compute ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        # Return metrics
        metrics = {
            'total_loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }

        return metrics

    def save(self, path):
        """Save the policy network."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps
        }, path)

    def load(self, path):
        """Load the policy network."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)

    def get_state_dict(self):
        """Get state dict for federated learning."""
        return self.policy.state_dict()

    def set_state_dict(self, state_dict):
        """Set state dict from federated aggregation."""
        self.policy.load_state_dict(state_dict)


# Import at the end to avoid circular dependency
import torch.nn.functional as F
