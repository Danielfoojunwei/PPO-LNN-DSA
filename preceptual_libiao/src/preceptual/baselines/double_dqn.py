"""
Double DQN Baseline for Dynamic Spectrum Access

Implements Double Deep Q-Network as baseline comparison following
Bowen Shen's methodology from:
"Dynamic spectrum access for Internet-of-Things with hierarchical
federated deep reinforcement learning" (Ad Hoc Networks, 2023)

Benchmark Parameters from paper:
- 10 Secondary Users (SUs), 20 channels
- Learning rate: 0.9 (we use 0.001 for deep learning)
- Discount factor (gamma): 0.95
- Batch size: 50
- Experience replay buffer: 1000
- Epsilon decay for exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class DoubleDQNConfig:
    """Configuration matching Bowen Shen's paper parameters."""
    # Network
    state_dim: int = 42
    action_dim: int = 20  # Number of channels
    hidden_dim: int = 128

    # Learning parameters (adapted from paper)
    learning_rate: float = 0.001  # Paper uses 0.9 for tabular, we use 0.001 for DNN
    gamma: float = 0.95  # Discount factor from paper

    # Replay buffer (from paper)
    buffer_size: int = 1000
    batch_size: int = 50

    # Exploration (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1  # Paper uses 0.1 for random action probability
    epsilon_decay: float = 0.995

    # Target network
    target_update_freq: int = 100

    # Training
    num_episodes: int = 1000
    max_steps_per_episode: int = 500


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network for Double DQN."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.

    Separates state value and advantage streams for better learning
    in environments where actions don't always affect state value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class DoubleDQNAgent:
    """
    Double DQN Agent for Dynamic Spectrum Access.

    Implements Double DQN to address overestimation bias in standard DQN.
    Uses separate networks for action selection and evaluation.
    """

    def __init__(
        self,
        config: Optional[DoubleDQNConfig] = None,
        use_dueling: bool = True,
        device: str = "cpu"
    ):
        self.config = config or DoubleDQNConfig()
        self.device = torch.device(device)

        # Networks
        NetworkClass = DuelingQNetwork if use_dueling else QNetwork
        self.q_network = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        self.target_network = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )

        # Replay buffer
        self.buffer = ReplayBuffer(self.config.buffer_size)

        # Exploration
        self.epsilon = self.config.epsilon_start

        # Training state
        self.total_steps = 0
        self.update_count = 0

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using epsilon-greedy policy.

        Returns:
            (action, q_value, exploration_prob)
        """
        if not deterministic and random.random() < self.epsilon:
            action = random.randint(0, self.config.action_dim - 1)
            q_value = 0.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=-1).item()
                q_value = q_values[0, action].item()

        return action, q_value, self.epsilon

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        """
        Perform Double DQN update.

        Double DQN key insight:
        - Use online network to SELECT best action
        - Use target network to EVALUATE that action

        Q_target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
        """
        if len(self.buffer) < self.config.batch_size:
            return {"loss": 0.0, "mean_q": 0.0}

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(dim=-1)

            # Evaluate using target network
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            # Compute target
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        # Loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1

        # Update target network
        if self.update_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

        return {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
            "epsilon": self.epsilon,
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "update_count": self.update_count,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]
        self.update_count = checkpoint["update_count"]


class RandomAgent:
    """Random baseline agent."""

    def __init__(self, action_dim: int = 20):
        self.action_dim = action_dim
        self.total_steps = 0

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        action = random.randint(0, self.action_dim - 1)
        return action, 0.0, 1.0

    def store_transition(self, *args):
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        return {"loss": 0.0, "mean_q": 0.0}


class GreedyAgent:
    """Greedy baseline that always selects best recent channel."""

    def __init__(self, action_dim: int = 20):
        self.action_dim = action_dim
        self.channel_success = np.zeros(action_dim)
        self.channel_attempts = np.ones(action_dim)  # Avoid div by zero
        self.total_steps = 0

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        success_rate = self.channel_success / self.channel_attempts
        action = success_rate.argmax()
        return int(action), success_rate[action], 0.0

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.channel_attempts[action] += 1
        if reward > 0:
            self.channel_success[action] += 1
        # Decay old statistics
        self.channel_success *= 0.99
        self.channel_attempts *= 0.99
        self.channel_attempts = np.maximum(self.channel_attempts, 1.0)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        return {"loss": 0.0, "mean_q": 0.0}
