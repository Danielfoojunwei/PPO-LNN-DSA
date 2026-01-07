"""
DSA Environment with Variable Time Steps (dt-aware)

This environment tests the key advantage of LTC/LFM networks:
the ability to handle variable time intervals between decisions.

Real-world scenarios where dt varies:
1. Network congestion → longer processing delays
2. Priority interrupts → shorter decision intervals
3. Sensor failures → irregular observation timing
4. Multi-rate control → different subsystems at different rates

This benchmark specifically tests how well agents adapt to:
- Regular dt (easy): constant time steps
- Variable dt (medium): random time steps
- Burst dt (hard): alternating fast/slow periods
- Adversarial dt (very hard): dt correlates with channel state
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class DTPattern(Enum):
    """Time step patterns for testing dt-awareness."""
    CONSTANT = "constant"      # Fixed dt (baseline)
    UNIFORM = "uniform"        # Uniform random dt
    EXPONENTIAL = "exponential"  # Exponential distribution (bursty)
    BIMODAL = "bimodal"        # Two distinct modes (fast/slow)
    ADVERSARIAL = "adversarial"  # dt correlates with optimal action
    INCREASING = "increasing"   # dt grows over episode
    OSCILLATING = "oscillating"  # Sinusoidal dt pattern


@dataclass
class VariableDTConfig:
    """Configuration for variable-dt DSA environment."""
    # Users and channels
    num_secondary_users: int = 10
    num_channels: int = 20
    num_idle_channels: int = 5
    num_pu_channels: int = 10
    num_su_channels: int = 5

    # Episode
    max_steps: int = 200

    # Time step configuration
    dt_pattern: DTPattern = DTPattern.UNIFORM
    dt_base: float = 1.0        # Base time step
    dt_min: float = 0.1         # Minimum dt
    dt_max: float = 5.0         # Maximum dt
    dt_noise: float = 0.2       # Noise level for patterns

    # PU dynamics (affected by dt!)
    pu_transition_rate: float = 0.3  # Transitions per unit time

    # Rewards
    reward_success: float = 1.0
    reward_collision: float = 0.0

    # Observation
    include_dt_in_obs: bool = True  # Include dt as observation feature


class VariableDTDSAEnvironment:
    """
    DSA Environment with Variable Time Steps.

    Key difference from standard DSA:
    - dt varies between steps
    - PU dynamics depend on elapsed time (not step count)
    - Success probability may depend on timing
    - Agents must adapt their strategy to dt

    This environment specifically tests:
    1. Can the agent handle irregular timing?
    2. Does the agent learn to predict based on elapsed time?
    3. How does performance degrade with dt variability?
    """

    def __init__(self, config: Optional[VariableDTConfig] = None):
        self.config = config or VariableDTConfig()
        self._validate_config()

        # State
        self.channels: List[int] = []  # 0=idle, 1=PU, 2=SU
        self.pu_channels: List[int] = []
        self.step_count = 0
        self.total_time = 0.0

        # dt tracking
        self.current_dt = self.config.dt_base
        self.dt_history: List[float] = []

        # Metrics
        self.episode_successes = 0
        self.episode_collisions = 0
        self.episode_throughput = 0.0

        # For adversarial dt pattern
        self.best_channel = 0

        self._initialize()

    def _validate_config(self):
        cfg = self.config
        assert cfg.num_idle_channels + cfg.num_pu_channels + cfg.num_su_channels == cfg.num_channels

    def _initialize(self):
        """Initialize channel allocation."""
        cfg = self.config

        # Assign channels
        all_channels = list(range(cfg.num_channels))
        random.shuffle(all_channels)
        self.pu_channels = all_channels[:cfg.num_pu_channels]

        # Initialize states
        self.channels = [0] * cfg.num_channels  # 0=idle
        for ch in self.pu_channels:
            if random.random() < 0.5:
                self.channels[ch] = 1  # PU occupied

        self.best_channel = self._find_best_channel()

    def _find_best_channel(self) -> int:
        """Find the current best channel (for adversarial dt)."""
        idle_channels = [i for i, s in enumerate(self.channels) if s == 0]
        return random.choice(idle_channels) if idle_channels else 0

    def _generate_dt(self) -> float:
        """Generate next time step based on pattern."""
        cfg = self.config
        pattern = cfg.dt_pattern

        if pattern == DTPattern.CONSTANT:
            dt = cfg.dt_base

        elif pattern == DTPattern.UNIFORM:
            dt = random.uniform(cfg.dt_min, cfg.dt_max)

        elif pattern == DTPattern.EXPONENTIAL:
            # Exponential distribution (bursty traffic)
            dt = np.random.exponential(cfg.dt_base)
            dt = np.clip(dt, cfg.dt_min, cfg.dt_max)

        elif pattern == DTPattern.BIMODAL:
            # Two modes: fast (0.2x) or slow (3x)
            if random.random() < 0.5:
                dt = cfg.dt_base * 0.2
            else:
                dt = cfg.dt_base * 3.0
            dt = np.clip(dt, cfg.dt_min, cfg.dt_max)

        elif pattern == DTPattern.ADVERSARIAL:
            # dt is long when best channel is obvious, short when ambiguous
            idle_count = sum(1 for s in self.channels if s == 0)
            ambiguity = 1.0 - abs(idle_count / cfg.num_channels - 0.5) * 2
            dt = cfg.dt_min + (cfg.dt_max - cfg.dt_min) * ambiguity

        elif pattern == DTPattern.INCREASING:
            # dt grows linearly over episode
            progress = self.step_count / cfg.max_steps
            dt = cfg.dt_min + (cfg.dt_max - cfg.dt_min) * progress

        elif pattern == DTPattern.OSCILLATING:
            # Sinusoidal pattern
            phase = 2 * np.pi * self.step_count / 50  # Period of 50 steps
            dt = cfg.dt_base + (cfg.dt_max - cfg.dt_min) / 2 * np.sin(phase)
            dt = np.clip(dt, cfg.dt_min, cfg.dt_max)

        else:
            dt = cfg.dt_base

        # Add noise
        dt += random.gauss(0, cfg.dt_noise * cfg.dt_base)
        dt = np.clip(dt, cfg.dt_min, cfg.dt_max)

        return float(dt)

    def _update_pu_dynamics(self, dt: float):
        """Update PU occupancy based on elapsed time."""
        cfg = self.config

        # Probability of transition depends on dt
        # Longer dt = more likely to have changed
        transition_prob = 1 - np.exp(-cfg.pu_transition_rate * dt)

        for ch in self.pu_channels:
            if random.random() < transition_prob:
                # Toggle state
                self.channels[ch] = 1 - self.channels[ch]

        self.best_channel = self._find_best_channel()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._initialize()
        self.step_count = 0
        self.total_time = 0.0
        self.current_dt = self.config.dt_base
        self.dt_history = []

        self.episode_successes = 0
        self.episode_collisions = 0
        self.episode_throughput = 0.0

        obs = self._get_observation()
        return obs, {"dt": self.current_dt}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action with variable dt.

        The key insight: dt affects:
        1. How much the environment has changed since last observation
        2. The "staleness" of the agent's information
        3. PU transition probabilities
        """
        cfg = self.config

        # Get current dt and update for next step
        dt = self.current_dt
        self.dt_history.append(dt)
        self.total_time += dt
        self.step_count += 1

        # Update PU dynamics based on dt
        self._update_pu_dynamics(dt)

        # Process action
        action = int(action) % cfg.num_channels

        # Check outcome
        if self.channels[action] == 1:
            # Collision with PU
            reward = cfg.reward_collision
            success = False
            collision = True
            self.episode_collisions += 1
        else:
            # Success!
            reward = cfg.reward_success
            success = True
            collision = False
            self.episode_successes += 1
            self.episode_throughput += 1.0

            # Mark as SU occupied briefly
            self.channels[action] = 2

        # Clear SU occupation for next step
        for i in range(cfg.num_channels):
            if self.channels[i] == 2:
                self.channels[i] = 0

        # Generate next dt
        self.current_dt = self._generate_dt()

        # Get observation
        obs = self._get_observation()

        # Check termination
        terminated = False
        truncated = self.step_count >= cfg.max_steps

        info = {
            "success": success,
            "collision": collision,
            "dt": dt,
            "next_dt": self.current_dt,
            "total_time": self.total_time,
            "dt_mean": np.mean(self.dt_history),
            "dt_std": np.std(self.dt_history) if len(self.dt_history) > 1 else 0,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get observation including dt information."""
        cfg = self.config

        # Channel states
        channel_obs = np.array([float(s == 1) for s in self.channels], dtype=np.float32)

        # Statistics
        stats = np.array([
            self.episode_successes / max(1, self.step_count),  # Success rate
            self.episode_collisions / max(1, self.step_count),  # Collision rate
            self.step_count / cfg.max_steps,  # Progress
        ], dtype=np.float32)

        if cfg.include_dt_in_obs:
            # Include dt information (crucial for dt-aware models)
            dt_features = np.array([
                self.current_dt / cfg.dt_max,  # Normalized current dt
                np.mean(self.dt_history[-10:]) / cfg.dt_max if self.dt_history else 0.5,  # Recent mean
                np.std(self.dt_history[-10:]) / cfg.dt_max if len(self.dt_history) > 1 else 0,  # Recent std
                self.total_time / (cfg.max_steps * cfg.dt_base),  # Normalized total time
            ], dtype=np.float32)
            return np.concatenate([channel_obs, stats, dt_features])
        else:
            return np.concatenate([channel_obs, stats])

    def get_state_dim(self) -> int:
        cfg = self.config
        base_dim = cfg.num_channels + 3  # channels + stats
        if cfg.include_dt_in_obs:
            base_dim += 4  # dt features
        return base_dim

    def get_action_dim(self) -> int:
        return self.config.num_channels

    def get_metrics(self) -> Dict[str, float]:
        """Get episode metrics including dt statistics."""
        total = self.episode_successes + self.episode_collisions
        return {
            "success_rate": self.episode_successes / max(1, total),
            "collision_rate": self.episode_collisions / max(1, total),
            "throughput": self.episode_throughput,
            "total_time": self.total_time,
            "dt_mean": np.mean(self.dt_history) if self.dt_history else 0,
            "dt_std": np.std(self.dt_history) if len(self.dt_history) > 1 else 0,
            "dt_min": min(self.dt_history) if self.dt_history else 0,
            "dt_max": max(self.dt_history) if self.dt_history else 0,
        }


def create_dt_test_suite() -> Dict[str, VariableDTConfig]:
    """Create a suite of test configurations for dt-awareness evaluation."""
    return {
        "constant": VariableDTConfig(
            dt_pattern=DTPattern.CONSTANT,
            dt_base=1.0,
        ),
        "uniform": VariableDTConfig(
            dt_pattern=DTPattern.UNIFORM,
            dt_min=0.1,
            dt_max=5.0,
        ),
        "exponential": VariableDTConfig(
            dt_pattern=DTPattern.EXPONENTIAL,
            dt_base=1.0,
            dt_min=0.1,
            dt_max=5.0,
        ),
        "bimodal": VariableDTConfig(
            dt_pattern=DTPattern.BIMODAL,
            dt_base=1.0,
        ),
        "adversarial": VariableDTConfig(
            dt_pattern=DTPattern.ADVERSARIAL,
            dt_min=0.1,
            dt_max=5.0,
        ),
        "increasing": VariableDTConfig(
            dt_pattern=DTPattern.INCREASING,
            dt_min=0.1,
            dt_max=5.0,
        ),
        "oscillating": VariableDTConfig(
            dt_pattern=DTPattern.OSCILLATING,
            dt_base=1.0,
            dt_min=0.1,
            dt_max=3.0,
        ),
    }
