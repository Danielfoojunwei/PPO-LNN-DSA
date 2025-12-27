"""
Federated Learning Client (IoT Device)

Implements local training for IoT devices in the hierarchical
federated learning framework.
"""

import torch
import numpy as np
from models.ppo_lfm_agent import PPOLFMAgent
from environment.spectrum_env import DynamicSpectrumAccessEnv


class FederatedClient:
    """
    Federated learning client that runs on IoT devices.

    Each client:
    1. Maintains local PPO-LFM agent
    2. Trains on local spectrum access episodes
    3. Sends model updates to edge server
    """
    def __init__(
        self,
        client_id,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_lfm_layers=2,
        num_channels=10,
        num_devices=20,
        device='cpu'
    ):
        """
        Args:
            client_id: Unique identifier for this client
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension for LFM
            num_lfm_layers: Number of LFM layers
            num_channels: Number of spectrum channels
            num_devices: Total number of devices in the network
            device: Device to run on
        """
        self.client_id = client_id
        self.device = device

        # Initialize PPO-LFM agent
        self.agent = PPOLFMAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_lfm_layers=num_lfm_layers,
            device=device
        )

        # Initialize environment for this client
        self.env = DynamicSpectrumAccessEnv(
            num_channels=num_channels,
            num_devices=num_devices,
            device_id=client_id % num_devices  # Map client to device
        )

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_rounds = 0

    def local_train(self, num_episodes=10, max_steps=100, update_interval=2048):
        """
        Perform local training for specified number of episodes.

        Args:
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            update_interval: Steps between PPO updates

        Returns:
            metrics: Dictionary of training metrics
        """
        total_steps = 0
        episode_rewards_list = []
        episode_lengths_list = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < max_steps:
                # Convert state to tensor
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                # Select action
                action, log_prob, value = self.agent.select_action(state_tensor)

                # Take step in environment
                next_state, reward, done, info = self.env.step(action)

                # Store transition
                self.agent.store_transition(
                    state, action, log_prob, value, reward, done
                )

                # Update counters
                episode_reward += reward
                episode_length += 1
                total_steps += 1

                # PPO update at regular intervals
                if total_steps % update_interval == 0:
                    update_metrics = self.agent.update()

                # Move to next state
                state = next_state

            episode_rewards_list.append(episode_reward)
            episode_lengths_list.append(episode_length)

        # Final update if there are remaining transitions
        if len(self.agent.buffer.states) > 0:
            update_metrics = self.agent.update()

        # Store statistics
        self.episode_rewards.extend(episode_rewards_list)
        self.episode_lengths.extend(episode_lengths_list)
        self.training_rounds += 1

        # Compute metrics
        metrics = {
            'client_id': self.client_id,
            'mean_reward': np.mean(episode_rewards_list),
            'std_reward': np.std(episode_rewards_list),
            'mean_length': np.mean(episode_lengths_list),
            'total_steps': total_steps
        }

        return metrics

    def get_model_update(self):
        """
        Get model parameters for federated aggregation.

        Returns:
            state_dict: Model state dictionary
        """
        return self.agent.get_state_dict()

    def set_model_parameters(self, state_dict):
        """
        Update model parameters from aggregated global model.

        Args:
            state_dict: Model state dictionary
        """
        self.agent.set_state_dict(state_dict)

    def evaluate(self, num_episodes=10):
        """
        Evaluate the current policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            metrics: Evaluation metrics
        """
        episode_rewards = []
        success_rates = []
        collision_rates = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            total_successes = 0
            total_collisions = 0

            while not done and episode_length < 100:
                # Select action deterministically
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, _, _ = self.agent.select_action(state_tensor, deterministic=True)

                # Take step
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                if info['success']:
                    total_successes += 1
                if info['collision']:
                    total_collisions += 1

                state = next_state

            episode_rewards.append(episode_reward)
            success_rates.append(total_successes / episode_length)
            collision_rates.append(total_collisions / episode_length)

        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_success_rate': np.mean(success_rates),
            'mean_collision_rate': np.mean(collision_rates)
        }

        return metrics

    def save(self, path):
        """Save client state."""
        self.agent.save(path)

    def load(self, path):
        """Load client state."""
        self.agent.load(path)
