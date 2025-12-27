"""
Dynamic Spectrum Access Environment for IoT

Simulates a dynamic spectrum access scenario where IoT devices
must select channels while minimizing interference and maximizing throughput.
"""

import numpy as np
import torch


class DynamicSpectrumAccessEnv:
    """
    Environment for dynamic spectrum access in IoT networks.

    State: Channel quality indicators, interference levels, historical usage
    Action: Select a channel for transmission
    Reward: Based on successful transmission, throughput, and interference
    """
    def __init__(
        self,
        num_channels=10,
        num_devices=20,
        seq_length=10,
        max_steps=100,
        interference_threshold=0.5,
        device_id=0
    ):
        """
        Args:
            num_channels: Number of available spectrum channels
            num_devices: Number of IoT devices sharing the spectrum
            seq_length: Length of observation sequence (for temporal modeling)
            max_steps: Maximum steps per episode
            interference_threshold: Threshold for interference penalty
            device_id: ID of the device (for multi-agent scenarios)
        """
        self.num_channels = num_channels
        self.num_devices = num_devices
        self.seq_length = seq_length
        self.max_steps = max_steps
        self.interference_threshold = interference_threshold
        self.device_id = device_id

        # State dimensions: [channel_quality, interference, historical_usage]
        self.state_dim = num_channels * 3

        # Action space: select one of the channels
        self.action_dim = num_channels

        # Environment state
        self.current_step = 0
        self.channel_quality = np.zeros(num_channels)
        self.interference = np.zeros(num_channels)
        self.channel_usage_history = np.zeros((seq_length, num_channels))
        self.device_actions = np.zeros((num_devices, num_channels))  # Track other devices

        # Statistics
        self.total_throughput = 0
        self.total_collisions = 0
        self.successful_transmissions = 0

    def reset(self):
        """
        Reset the environment to initial state.

        Returns:
            state: Initial state observation [seq_length, state_dim]
        """
        self.current_step = 0

        # Initialize channel quality (varies by location/channel)
        self.channel_quality = np.random.uniform(0.5, 1.0, self.num_channels)

        # Initialize interference (background noise)
        self.interference = np.random.uniform(0.0, 0.3, self.num_channels)

        # Initialize history
        self.channel_usage_history = np.zeros((self.seq_length, self.num_channels))

        # Reset device actions
        self.device_actions = np.zeros((self.num_devices, num_channels))

        # Reset statistics
        self.total_throughput = 0
        self.total_collisions = 0
        self.successful_transmissions = 0

        return self._get_observation()

    def step(self, action):
        """
        Execute action and return next state, reward, done flag.

        Args:
            action: Channel selection (0 to num_channels-1)

        Returns:
            state: Next state observation
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        assert 0 <= action < self.num_channels, f"Invalid action: {action}"

        # Update step counter
        self.current_step += 1

        # Simulate other devices' actions (simplified)
        self._simulate_other_devices(action)

        # Compute interference on selected channel
        channel_interference = self._compute_interference(action)

        # Compute reward
        reward, success, collision = self._compute_reward(action, channel_interference)

        # Update channel quality (channel conditions change over time)
        self._update_channel_quality()

        # Update usage history
        self._update_history(action)

        # Update statistics
        self.total_throughput += reward
        if collision:
            self.total_collisions += 1
        if success:
            self.successful_transmissions += 1

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Get next observation
        next_state = self._get_observation()

        # Info dictionary
        info = {
            'success': success,
            'collision': collision,
            'interference': channel_interference,
            'channel_quality': self.channel_quality[action],
            'total_throughput': self.total_throughput,
            'collision_rate': self.total_collisions / (self.current_step + 1e-6)
        }

        return next_state, reward, done, info

    def _get_observation(self):
        """
        Get current observation (state).

        Returns:
            state: [seq_length, state_dim] numpy array
        """
        # Construct state: [channel_quality, interference, history]
        # Each timestep in sequence has: [quality (N), interference (N), usage (N)]
        state_sequence = []

        for t in range(self.seq_length):
            # Channel quality at this timestep
            quality = self.channel_quality.copy()

            # Current interference
            interference = self.interference.copy()

            # Historical usage at this timestep
            if t < len(self.channel_usage_history):
                usage = self.channel_usage_history[-(t+1)]
            else:
                usage = np.zeros(self.num_channels)

            # Concatenate features
            timestep_features = np.concatenate([quality, interference, usage])
            state_sequence.append(timestep_features)

        # Stack into sequence: [seq_length, state_dim]
        state = np.stack(state_sequence)

        return state

    def _simulate_other_devices(self, my_action):
        """
        Simulate actions of other devices in the network.

        Args:
            my_action: Action taken by this device
        """
        # Reset device actions
        self.device_actions = np.zeros((self.num_devices, self.num_channels))

        # This device's action
        self.device_actions[self.device_id, my_action] = 1

        # Other devices use simple strategies (random, greedy, etc.)
        for device_id in range(self.num_devices):
            if device_id == self.device_id:
                continue

            # Strategy: Select channel with lowest interference (greedy)
            # Add some randomness
            if np.random.random() < 0.3:
                # Random selection
                action = np.random.randint(0, self.num_channels)
            else:
                # Greedy selection
                action = np.argmin(self.interference)

            self.device_actions[device_id, action] = 1

    def _compute_interference(self, action):
        """
        Compute interference on the selected channel.

        Args:
            action: Selected channel

        Returns:
            interference: Total interference level [0, 1]
        """
        # Base interference
        base_interference = self.interference[action]

        # Interference from other devices on the same channel
        num_devices_on_channel = self.device_actions[:, action].sum()

        # More devices = more interference
        device_interference = (num_devices_on_channel - 1) * 0.2  # Collision interference

        # Total interference
        total_interference = min(1.0, base_interference + device_interference)

        return total_interference

    def _compute_reward(self, action, interference):
        """
        Compute reward for the action.

        Args:
            action: Selected channel
            interference: Interference level on the channel

        Returns:
            reward: Reward value
            success: Whether transmission was successful
            collision: Whether collision occurred
        """
        # Channel quality
        quality = self.channel_quality[action]

        # Check for collision (multiple devices on same channel)
        num_devices_on_channel = self.device_actions[:, action].sum()
        collision = num_devices_on_channel > 1

        # Transmission success probability
        success_prob = quality * (1 - interference)

        # Determine success
        success = np.random.random() < success_prob and not collision

        # Reward components
        if success:
            # Successful transmission: high reward based on quality
            reward = quality * 10.0
        elif collision:
            # Collision: negative reward
            reward = -5.0
        else:
            # Failed transmission (poor quality/interference): small negative reward
            reward = -1.0

        # Penalty for high interference
        if interference > self.interference_threshold:
            reward -= (interference - self.interference_threshold) * 2.0

        return reward, success, collision

    def _update_channel_quality(self):
        """
        Update channel quality over time (simulates fading, mobility, etc.).
        """
        # Add some temporal variation (Markov process)
        noise = np.random.normal(0, 0.05, self.num_channels)
        self.channel_quality = np.clip(
            0.8 * self.channel_quality + 0.2 * np.random.uniform(0.5, 1.0, self.num_channels) + noise,
            0.1, 1.0
        )

        # Update interference (dynamic environment)
        noise = np.random.normal(0, 0.02, self.num_channels)
        self.interference = np.clip(
            0.9 * self.interference + 0.1 * np.random.uniform(0.0, 0.3, self.num_channels) + noise,
            0.0, 0.8
        )

    def _update_history(self, action):
        """
        Update channel usage history.

        Args:
            action: Channel that was selected
        """
        # Create one-hot encoding of action
        usage = np.zeros(self.num_channels)
        usage[action] = 1.0

        # Shift history and add new usage
        self.channel_usage_history = np.roll(self.channel_usage_history, 1, axis=0)
        self.channel_usage_history[0] = usage

    def get_state_dim(self):
        """Return state dimension."""
        return self.state_dim

    def get_action_dim(self):
        """Return action dimension."""
        return self.action_dim

    def render(self):
        """Render the environment (optional, for debugging)."""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Channel Quality: {self.channel_quality}")
        print(f"Interference: {self.interference}")
        print(f"Devices per channel: {self.device_actions.sum(axis=0)}")
        print(f"Total Throughput: {self.total_throughput:.2f}")
        print(f"Collision Rate: {self.total_collisions / (self.current_step + 1e-6):.2%}")
        print(f"Success Rate: {self.successful_transmissions / (self.current_step + 1e-6):.2%}")
        print("-" * 50)


def make_env(num_channels=10, num_devices=20, device_id=0):
    """
    Factory function to create environment.

    Args:
        num_channels: Number of spectrum channels
        num_devices: Number of IoT devices
        device_id: ID of the device

    Returns:
        env: DynamicSpectrumAccessEnv instance
    """
    return DynamicSpectrumAccessEnv(
        num_channels=num_channels,
        num_devices=num_devices,
        device_id=device_id
    )
