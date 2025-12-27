"""
PPO-LSTM Baseline Implementation

This implements the traditional PPO-LSTM approach for comparison with PPO-LFM.
Based on the architecture from the research paper on hierarchical federated
deep reinforcement learning for dynamic spectrum access.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque


class LSTMActorCritic(nn.Module):
    """
    Traditional Actor-Critic network using LSTM for temporal modeling.

    This is the baseline architecture that PPO-LFM replaces.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_lstm_layers=2
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension for LSTM
            num_lstm_layers: Number of LSTM layers
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # Input projection (optional, helps with feature extraction)
        self.input_proj = nn.Linear(state_dim, hidden_dim)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, hidden_state=None, deterministic=False):
        """
        Forward pass through LSTM actor-critic.

        Args:
            state: State tensor [batch, seq_len, state_dim]
            hidden_state: Previous LSTM hidden state (optional)
            deterministic: Whether to select actions deterministically

        Returns:
            action, log_prob, value, entropy, hidden_state
        """
        batch_size = state.size(0)

        # Project input
        x = self.input_proj(state)  # [batch, seq_len, hidden_dim]

        # LSTM encoding
        if hidden_state is None:
            # Initialize hidden state
            h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(state.device)
            c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(state.device)
            hidden_state = (h0, c0)

        lstm_out, hidden_state = self.lstm(x, hidden_state)

        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]

        # Actor: Compute action probabilities
        action_logits = self.actor(last_output)
        action_dist = Categorical(logits=action_logits)

        # Sample or select deterministic action
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # Critic: Estimate state value
        value = self.critic(last_output).squeeze(-1)

        return action, log_prob, value, entropy, hidden_state

    def evaluate_actions(self, state, action, hidden_state=None):
        """
        Evaluate actions for PPO training.

        Args:
            state: State tensor [batch, seq_len, state_dim]
            action: Action tensor [batch]
            hidden_state: LSTM hidden state (optional)

        Returns:
            log_prob, value, entropy
        """
        batch_size = state.size(0)

        # Project input
        x = self.input_proj(state)

        # LSTM encoding
        if hidden_state is None:
            h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(state.device)
            c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(state.device)
            hidden_state = (h0, c0)

        lstm_out, _ = self.lstm(x, hidden_state)
        last_output = lstm_out[:, -1, :]

        # Actor evaluation
        action_logits = self.actor(last_output)
        action_dist = Categorical(logits=action_logits)

        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # Critic evaluation
        value = self.critic(last_output).squeeze(-1)

        return log_prob, value, entropy

    def get_value(self, state, hidden_state=None):
        """Get value estimate for a state."""
        with torch.no_grad():
            batch_size = state.size(0)

            x = self.input_proj(state)

            if hidden_state is None:
                h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(state.device)
                c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(state.device)
                hidden_state = (h0, c0)

            lstm_out, _ = self.lstm(x, hidden_state)
            last_output = lstm_out[:, -1, :]

            value = self.critic(last_output).squeeze(-1)

        return value


class RolloutBufferLSTM:
    """
    Buffer for storing trajectories during PPO rollout (LSTM version).
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
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0

        values = torch.cat(self.values)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        values = torch.cat([values, last_value.unsqueeze(0)])

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


class PPOLSTMAgent:
    """
    PPO Agent using LSTM (baseline for comparison with PPO-LFM).
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_lstm_layers=2,
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
            hidden_dim: Hidden dimension for LSTM
            num_lstm_layers: Number of LSTM layers
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
            device: Device to run on
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Initialize LSTM actor-critic
        self.policy = LSTMActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBufferLSTM()

        # Tracking
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)

    def select_action(self, state, deterministic=False):
        """Select action given state."""
        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        state = state.to(self.device)

        with torch.no_grad():
            action, log_prob, value, _, _ = self.policy(state, deterministic=deterministic)

        return action.item(), log_prob, value

    def store_transition(self, state, action, log_prob, value, reward, done):
        """Store a transition in the rollout buffer."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)

        self.buffer.add(state, action, log_prob, value, reward, done)

    def update(self, n_epochs=10, batch_size=64):
        """Update the policy using PPO."""
        last_state = self.buffer.states[-1].unsqueeze(0).to(self.device)
        last_value = self.policy.get_value(last_state)

        self.buffer.compute_advantages(last_value, self.gamma, self.gae_lambda)

        data = self.buffer.get()
        states = data['states'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        advantages = data['advantages'].to(self.device)
        returns = data['returns'].to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(n_epochs):
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                ratio = torch.exp(log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.buffer.clear()

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
