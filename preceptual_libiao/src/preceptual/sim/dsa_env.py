"""
Dynamic Spectrum Access Environment

Implements DSA environment following Bowen Shen's methodology:
"Dynamic spectrum access for Internet-of-Things with hierarchical
federated deep reinforcement learning" (Ad Hoc Networks, 2023)

Key features:
- Multi-user channel access simulation
- Primary User (PU) and Secondary User (SU) model
- Collision detection and success tracking
- Reward based on successful transmission

Simulation Parameters (from paper):
- 10 SUs, 20 channels
- 5 idle channels, 10 PU-occupied, 5 SU-occupied
- Reward = 1 for successful access, 0 otherwise
- Collision when accessing occupied channel
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class ChannelState(Enum):
    """Channel occupancy state."""
    IDLE = 0
    PU_OCCUPIED = 1
    SU_OCCUPIED = 2


@dataclass
class DSAConfig:
    """
    Configuration for DSA environment.

    Default values from Bowen Shen's paper.
    """
    # Users and channels
    num_secondary_users: int = 10  # Number of SUs (robots)
    num_channels: int = 20  # Total channels

    # Channel distribution (from paper)
    num_idle_channels: int = 5
    num_pu_channels: int = 10
    num_su_channels: int = 5

    # Dynamics
    pu_activity_prob: float = 0.3  # Probability PU becomes active
    pu_idle_prob: float = 0.2  # Probability PU becomes idle

    # Episode
    max_steps: int = 500

    # State representation
    observation_history: int = 5  # Number of past observations to include

    # Rewards (from paper formulation)
    reward_success: float = 1.0  # Successful transmission
    reward_collision: float = 0.0  # Collision (sometimes -1 in variants)
    reward_idle: float = 0.0  # No transmission attempt

    # Throughput model
    base_throughput: float = 1.0  # Throughput per successful transmission

    # Noise in observations
    sensing_error_prob: float = 0.05  # Probability of incorrect sensing


@dataclass
class SUState:
    """State of a secondary user."""
    su_id: int
    current_channel: int = -1  # -1 means not assigned
    last_action: int = 0
    consecutive_successes: int = 0
    consecutive_collisions: int = 0
    total_successes: int = 0
    total_collisions: int = 0
    total_throughput: float = 0.0


class DSAEnvironment:
    """
    Dynamic Spectrum Access Environment.

    Simulates multi-user DSA with:
    - Primary users with stochastic activity
    - Secondary users competing for channels
    - Collision detection
    - Success/throughput rewards

    Follows gymnasium-style interface.
    """

    def __init__(self, config: Optional[DSAConfig] = None):
        self.config = config or DSAConfig()

        # Validate config
        assert (self.config.num_idle_channels +
                self.config.num_pu_channels +
                self.config.num_su_channels) == self.config.num_channels

        # State
        self.channels: List[ChannelState] = []
        self.pu_channels: List[int] = []  # Channels allocated to PUs
        self.sus: Dict[int, SUState] = {}

        # Episode tracking
        self.step_count = 0
        self.episode_successes = 0
        self.episode_collisions = 0
        self.episode_throughput = 0.0

        # History for observation
        self.channel_history: List[List[int]] = []

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize channel allocation."""
        cfg = self.config

        # Assign channels to different states
        all_channels = list(range(cfg.num_channels))
        random.shuffle(all_channels)

        # PU channels
        self.pu_channels = all_channels[:cfg.num_pu_channels]

        # Initialize channel states
        self.channels = [ChannelState.IDLE] * cfg.num_channels
        for ch in self.pu_channels:
            if random.random() < 0.5:  # 50% start active
                self.channels[ch] = ChannelState.PU_OCCUPIED

        # Initialize SUs
        self.sus = {}
        for i in range(cfg.num_secondary_users):
            self.sus[i] = SUState(su_id=i)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.

        Returns:
            (observation, info)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._initialize()

        self.step_count = 0
        self.episode_successes = 0
        self.episode_collisions = 0
        self.episode_throughput = 0.0
        self.channel_history = []

        obs = self._get_observation()
        info = {"channel_states": [c.value for c in self.channels]}

        return obs, info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute actions for all SUs.

        Args:
            actions: Array of channel selections for each SU (num_sus,)
                    action = channel index to access, or -1 for no action

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        cfg = self.config

        # Track which channels are accessed this step
        channel_access_count: Dict[int, List[int]] = {i: [] for i in range(cfg.num_channels)}

        # Process each SU's action
        step_successes = 0
        step_collisions = 0
        step_throughput = 0.0

        su_results = {}

        for su_id, action in enumerate(actions):
            action = int(action)
            su = self.sus[su_id]
            su.last_action = action

            if action < 0 or action >= cfg.num_channels:
                # No action or invalid
                su_results[su_id] = {"success": False, "collision": False, "idle": True}
                continue

            channel_access_count[action].append(su_id)

        # Resolve collisions and successes
        for channel, accessing_sus in channel_access_count.items():
            if len(accessing_sus) == 0:
                continue

            channel_state = self.channels[channel]

            # Check for collision with PU
            if channel_state == ChannelState.PU_OCCUPIED:
                # All SUs accessing this channel collide with PU
                for su_id in accessing_sus:
                    su = self.sus[su_id]
                    su.total_collisions += 1
                    su.consecutive_collisions += 1
                    su.consecutive_successes = 0
                    su_results[su_id] = {"success": False, "collision": True, "idle": False}
                    step_collisions += 1

            elif len(accessing_sus) > 1:
                # Multiple SUs accessing same channel - collision between SUs
                for su_id in accessing_sus:
                    su = self.sus[su_id]
                    su.total_collisions += 1
                    su.consecutive_collisions += 1
                    su.consecutive_successes = 0
                    su_results[su_id] = {"success": False, "collision": True, "idle": False}
                    step_collisions += 1

            else:
                # Single SU accessing idle/available channel - success!
                su_id = accessing_sus[0]
                su = self.sus[su_id]
                su.total_successes += 1
                su.consecutive_successes += 1
                su.consecutive_collisions = 0
                su.current_channel = channel

                # Calculate throughput
                throughput = cfg.base_throughput
                su.total_throughput += throughput
                step_throughput += throughput

                su_results[su_id] = {"success": True, "collision": False, "idle": False}
                step_successes += 1

                # Mark channel as SU occupied
                self.channels[channel] = ChannelState.SU_OCCUPIED

        # Fill in results for SUs that took no action
        for su_id in range(cfg.num_secondary_users):
            if su_id not in su_results:
                su_results[su_id] = {"success": False, "collision": False, "idle": True}

        # Update PU activity (stochastic)
        self._update_pu_activity()

        # Clear SU occupancy for next step (SUs need to re-access)
        for i in range(cfg.num_channels):
            if self.channels[i] == ChannelState.SU_OCCUPIED:
                self.channels[i] = ChannelState.IDLE

        # Update episode stats
        self.episode_successes += step_successes
        self.episode_collisions += step_collisions
        self.episode_throughput += step_throughput

        # Calculate reward (following Shen's formulation)
        # Reward = sum of successful transmissions
        reward = float(step_successes) * cfg.reward_success

        # Get observation
        obs = self._get_observation()

        # Check termination
        terminated = False
        truncated = self.step_count >= cfg.max_steps

        # Build info
        info = {
            "step_successes": step_successes,
            "step_collisions": step_collisions,
            "step_throughput": step_throughput,
            "episode_successes": self.episode_successes,
            "episode_collisions": self.episode_collisions,
            "episode_throughput": self.episode_throughput,
            "su_results": su_results,
            "channel_states": [c.value for c in self.channels],
        }

        return obs, reward, terminated, truncated, info

    def _update_pu_activity(self) -> None:
        """Update PU channel occupancy stochastically."""
        cfg = self.config

        for ch in self.pu_channels:
            if self.channels[ch] == ChannelState.PU_OCCUPIED:
                # PU might become idle
                if random.random() < cfg.pu_idle_prob:
                    self.channels[ch] = ChannelState.IDLE
            else:
                # PU might become active
                if random.random() < cfg.pu_activity_prob:
                    self.channels[ch] = ChannelState.PU_OCCUPIED

    def _get_observation(self) -> np.ndarray:
        """
        Get observation for all SUs.

        Observation includes:
        - Channel sensing results (with possible errors)
        - SU's own state (last action, success history)
        - Aggregate statistics

        Returns:
            Flattened observation array
        """
        cfg = self.config

        # Channel sensing (with possible errors)
        channel_obs = np.zeros(cfg.num_channels, dtype=np.float32)
        for i, state in enumerate(self.channels):
            if state == ChannelState.IDLE:
                sensed = 0.0
            else:
                sensed = 1.0

            # Add sensing error
            if random.random() < cfg.sensing_error_prob:
                sensed = 1.0 - sensed

            channel_obs[i] = sensed

        # Store in history
        self.channel_history.append(channel_obs.tolist())
        if len(self.channel_history) > cfg.observation_history:
            self.channel_history.pop(0)

        # Build observation vector
        obs_parts = [channel_obs]

        # History (pad if needed)
        for i in range(cfg.observation_history):
            if i < len(self.channel_history):
                obs_parts.append(np.array(self.channel_history[-(i+1)], dtype=np.float32))
            else:
                obs_parts.append(np.zeros(cfg.num_channels, dtype=np.float32))

        # Aggregate SU statistics
        total_successes = sum(su.total_successes for su in self.sus.values())
        total_collisions = sum(su.total_collisions for su in self.sus.values())
        total_attempts = total_successes + total_collisions

        stats = np.array([
            total_successes / max(1, self.step_count * cfg.num_secondary_users),  # Success rate
            total_collisions / max(1, self.step_count * cfg.num_secondary_users),  # Collision rate
            self.step_count / cfg.max_steps,  # Episode progress
        ], dtype=np.float32)
        obs_parts.append(stats)

        return np.concatenate(obs_parts)

    def get_state_dim(self) -> int:
        """Get observation dimension."""
        cfg = self.config
        return (
            cfg.num_channels * (1 + cfg.observation_history) +  # Current + history
            3  # Stats
        )

    def get_action_dim(self) -> int:
        """Get action dimension (number of channels)."""
        return self.config.num_channels

    def get_metrics(self) -> Dict[str, float]:
        """Get episode metrics."""
        cfg = self.config
        total_attempts = self.episode_successes + self.episode_collisions

        return {
            "success_rate": self.episode_successes / max(1, total_attempts),
            "collision_rate": self.episode_collisions / max(1, total_attempts),
            "throughput": self.episode_throughput,
            "avg_throughput_per_step": self.episode_throughput / max(1, self.step_count),
            "total_successes": self.episode_successes,
            "total_collisions": self.episode_collisions,
        }


class SingleAgentDSAWrapper:
    """
    Wrapper to treat DSA as single-agent environment.

    Converts multi-SU environment to single-agent by:
    - Using single action to select channel for all SUs
    - Or using single action per SU in round-robin
    """

    def __init__(self, config: Optional[DSAConfig] = None, mode: str = "shared"):
        """
        Args:
            config: DSA configuration
            mode:
                "shared" - All SUs use same channel selection
                "sequential" - Each step controls one SU
        """
        self.env = DSAEnvironment(config)
        self.mode = mode
        self.current_su = 0

        # Adjust state dim for single agent
        self._state_dim = self.env.get_state_dim()
        self._action_dim = self.env.get_action_dim()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        self.current_su = 0
        return self.env.reset(seed)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take single action.

        Args:
            action: Channel index to access

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        cfg = self.env.config

        if self.mode == "shared":
            # All SUs try the same channel - will cause collision!
            # Instead, distribute SUs across channels based on action
            actions = np.full(cfg.num_secondary_users, -1)

            # Assign SUs to channels around the selected one
            for i in range(cfg.num_secondary_users):
                channel = (action + i) % cfg.num_channels
                actions[i] = channel

        else:  # sequential
            # Only current SU acts
            actions = np.full(cfg.num_secondary_users, -1)
            actions[self.current_su] = action
            self.current_su = (self.current_su + 1) % cfg.num_secondary_users

        obs, reward, terminated, truncated, info = self.env.step(actions)

        # Add success/collision to info for easy access
        info["success"] = info["step_successes"] > 0
        info["collision"] = info["step_collisions"] > 0

        return obs, reward, terminated, truncated, info

    def get_state_dim(self) -> int:
        return self._state_dim

    def get_action_dim(self) -> int:
        return self._action_dim

    def get_metrics(self) -> Dict[str, float]:
        return self.env.get_metrics()
