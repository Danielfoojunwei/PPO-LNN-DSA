"""
Realistic WiFi Standards Benchmark with Real-World Constraints

This benchmark addresses practical deployment limitations:
1. End-to-end capability constraints (MLO not always available)
2. Cross-layer/system objectives (fleet fairness, admission control)
3. Irregular observability (variable-rate telemetry, dropped samples)
4. Multi-technology coexistence (DSA with non-802.11 interference)
5. Policy/governance constraints (blacklists, time-of-day rules)
6. Training/implementation limits (reality drift, short training)

These scenarios reflect real-world situations where protocol-based approaches
face practical limitations and adaptive RL agents may excel.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
from collections import deque
import sys
import os

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preceptual.baselines.ppo_lstm import PPOLSTMAgent
from preceptual.baselines.ppo_lnn import PPOLNNAgent, PPOLTCAgent
from preceptual.baselines.double_dqn import DoubleDQNAgent, DoubleDQNConfig


class RealisticScenario(Enum):
    """Real-world constraint scenarios"""
    MIXED_CLIENT_ESTATE = "mixed_client_estate"           # WiFi 7 MLO not available on all clients
    DRIVER_INSTABILITY = "driver_instability"             # MLO present but unstable
    SPECTRUM_CONTENTION = "spectrum_contention"           # Cannot get multiple clean links
    FLEET_FAIRNESS = "fleet_fairness"                     # Multi-robot/flow fairness objective
    QOS_ADMISSION_CONTROL = "qos_admission_control"       # Per-app QoS prioritization
    GLOBAL_CHANNEL_PLAN = "global_channel_plan"           # Multi-cell coordination
    VARIABLE_TELEMETRY = "variable_telemetry"             # Dropped/bursty telemetry
    MOBILITY_EVENTS = "mobility_events"                   # Rapid mobility with variable sensing
    LTE_U_COEXISTENCE = "lte_u_coexistence"               # LTE-LAA/LTE-U in unlicensed band
    RADAR_DFS = "radar_dfs"                               # DFS radar detection events
    BLUETOOTH_INTERFERENCE = "bluetooth_interference"     # BT/WiFi coexistence
    CHANNEL_BLACKLIST = "channel_blacklist"               # Mandatory channel restrictions
    TIME_OF_DAY_POLICY = "time_of_day_policy"             # Time-based RF policies
    SAFETY_PRIORITY = "safety_priority"                   # Mission-critical traffic priority
    DOMAIN_DRIFT = "domain_drift"                         # Reality differs from training
    SHORT_TRAINING = "short_training"                     # Limited training episodes


@dataclass
class RealisticConfig:
    """Configuration for realistic constraint scenarios"""
    num_channels: int = 20
    num_stations: int = 10
    state_dim: int = 42
    action_dim: int = 20

    # MLO constraints
    mlo_capable_ratio: float = 1.0          # Fraction of clients supporting MLO
    mlo_stability: float = 1.0               # Probability MLO works when attempted
    available_links: int = 3                 # Number of clean links available

    # Collision and timing
    collision_prob_base: float = 0.2
    slot_time: float = 9e-6
    sifs: float = 16e-6
    difs: float = 34e-6

    # Telemetry/observability
    telemetry_drop_rate: float = 0.0         # Probability of dropped telemetry
    scan_interval_variance: float = 0.0      # Variance in scan timing
    observation_delay_max: float = 0.0       # Max observation delay (steps)

    # Multi-technology interference
    external_interference_prob: float = 0.0  # Non-802.11 interference
    radar_event_prob: float = 0.0            # DFS radar detection probability

    # Policy constraints
    blacklisted_channels: List[int] = field(default_factory=list)
    restricted_hours: Tuple[int, int] = (0, 24)  # Hours when restrictions apply
    safety_traffic_ratio: float = 0.0        # Ratio of safety-critical traffic

    # Fleet/fairness
    num_flows: int = 1                       # Number of competing flows
    fairness_weight: float = 0.0             # Weight for Jain's fairness index

    # Training constraints
    domain_drift_rate: float = 0.0           # Rate of environment parameter drift
    max_training_episodes: int = 1000


class RealisticWiFiEnvironment:
    """
    WiFi environment with realistic real-world constraints.

    Models practical deployment challenges that limit theoretical
    protocol performance and create opportunities for adaptive policies.
    """

    def __init__(self, config: RealisticConfig, scenario: RealisticScenario):
        self.config = config
        self.scenario = scenario
        self.current_step = 0
        self.episode_step = 0
        self.current_hour = 12  # Simulated time of day

        # Apply scenario-specific configuration
        self._configure_scenario()

        # State tracking
        self.channel_states = np.zeros(config.num_channels)
        self.link_states = np.ones(config.available_links)  # MLO link availability
        self.flow_throughputs = np.zeros(config.num_flows)
        self.qos_queues = {
            'VO': deque(maxlen=100),  # Voice
            'VI': deque(maxlen=100),  # Video
            'BE': deque(maxlen=100),  # Best effort
            'BK': deque(maxlen=100),  # Background
        }

        # Observation tracking for variable telemetry
        self.last_observation_step = 0
        self.pending_observations = []

        # Domain drift tracking
        self.original_collision_prob = config.collision_prob_base
        self.drift_direction = 1

        self.reset()

    def _configure_scenario(self):
        """Apply scenario-specific constraints"""
        cfg = self.config

        if self.scenario == RealisticScenario.MIXED_CLIENT_ESTATE:
            # Only 40% of clients support WiFi 7 MLO
            cfg.mlo_capable_ratio = 0.4
            cfg.collision_prob_base = 0.25

        elif self.scenario == RealisticScenario.DRIVER_INSTABILITY:
            # MLO available but unstable - 30% failure rate
            cfg.mlo_stability = 0.7
            cfg.collision_prob_base = 0.2

        elif self.scenario == RealisticScenario.SPECTRUM_CONTENTION:
            # Only 1 clean link available instead of 3
            cfg.available_links = 1
            cfg.collision_prob_base = 0.35

        elif self.scenario == RealisticScenario.FLEET_FAIRNESS:
            # 10 competing robot flows, fairness matters
            cfg.num_flows = 10
            cfg.fairness_weight = 0.5
            cfg.collision_prob_base = 0.3

        elif self.scenario == RealisticScenario.QOS_ADMISSION_CONTROL:
            # High load with strict QoS requirements
            cfg.num_stations = 30
            cfg.collision_prob_base = 0.35
            cfg.safety_traffic_ratio = 0.2

        elif self.scenario == RealisticScenario.GLOBAL_CHANNEL_PLAN:
            # Multi-cell with interference between cells
            cfg.num_stations = 40
            cfg.collision_prob_base = 0.3
            cfg.external_interference_prob = 0.15

        elif self.scenario == RealisticScenario.VARIABLE_TELEMETRY:
            # 20% telemetry drop, high variance in timing
            cfg.telemetry_drop_rate = 0.2
            cfg.scan_interval_variance = 0.5
            cfg.observation_delay_max = 3

        elif self.scenario == RealisticScenario.MOBILITY_EVENTS:
            # Rapid mobility with sensing gaps
            cfg.telemetry_drop_rate = 0.15
            cfg.scan_interval_variance = 0.8
            cfg.collision_prob_base = 0.3

        elif self.scenario == RealisticScenario.LTE_U_COEXISTENCE:
            # LTE-LAA interference in 5GHz band
            cfg.external_interference_prob = 0.25
            cfg.collision_prob_base = 0.3

        elif self.scenario == RealisticScenario.RADAR_DFS:
            # DFS radar events requiring channel evacuation
            cfg.radar_event_prob = 0.05
            cfg.blacklisted_channels = []  # Dynamic based on radar

        elif self.scenario == RealisticScenario.BLUETOOTH_INTERFERENCE:
            # BT interference in 2.4GHz channels
            cfg.external_interference_prob = 0.2
            cfg.collision_prob_base = 0.28

        elif self.scenario == RealisticScenario.CHANNEL_BLACKLIST:
            # 5 channels permanently blacklisted
            cfg.blacklisted_channels = [2, 5, 8, 12, 17]
            cfg.collision_prob_base = 0.25

        elif self.scenario == RealisticScenario.TIME_OF_DAY_POLICY:
            # Restricted operation during certain hours
            cfg.restricted_hours = (9, 17)  # Business hours restricted
            cfg.collision_prob_base = 0.25

        elif self.scenario == RealisticScenario.SAFETY_PRIORITY:
            # 30% safety-critical traffic that must be prioritized
            cfg.safety_traffic_ratio = 0.3
            cfg.num_stations = 25
            cfg.collision_prob_base = 0.3

        elif self.scenario == RealisticScenario.DOMAIN_DRIFT:
            # Environment parameters drift over time
            cfg.domain_drift_rate = 0.002
            cfg.collision_prob_base = 0.2

        elif self.scenario == RealisticScenario.SHORT_TRAINING:
            # Limited training budget
            cfg.max_training_episodes = 200
            cfg.collision_prob_base = 0.25

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.episode_step = 0
        self.current_hour = np.random.randint(0, 24)

        # Reset channel states with some randomness
        self.channel_states = np.random.uniform(0.3, 0.8, self.config.num_channels)

        # Reset MLO link availability
        self.link_states = np.ones(self.config.available_links)
        if self.scenario in [RealisticScenario.DRIVER_INSTABILITY,
                             RealisticScenario.SPECTRUM_CONTENTION]:
            # Some links may start unavailable
            for i in range(len(self.link_states)):
                if np.random.random() > self.config.mlo_stability:
                    self.link_states[i] = 0

        # Reset flow tracking
        self.flow_throughputs = np.zeros(self.config.num_flows)

        # Reset blacklist for radar scenarios
        if self.scenario == RealisticScenario.RADAR_DFS:
            self.config.blacklisted_channels = []

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current observation with realistic constraints"""
        # Check for telemetry drop
        if np.random.random() < self.config.telemetry_drop_rate:
            # Return stale observation
            if hasattr(self, '_last_observation'):
                return self._last_observation

        # Add observation delay
        if self.config.observation_delay_max > 0:
            delay = np.random.randint(0, int(self.config.observation_delay_max) + 1)
            # Use delayed channel states (simplified - just add noise)
            if delay > 0:
                noise = np.random.normal(0, 0.1 * delay, self.config.num_channels)
                delayed_channels = np.clip(self.channel_states + noise, 0, 1)
            else:
                delayed_channels = self.channel_states
        else:
            delayed_channels = self.channel_states

        # Build observation vector
        obs = np.zeros(self.config.state_dim)
        obs[:self.config.num_channels] = delayed_channels

        # Add MLO link states
        link_start = self.config.num_channels
        obs[link_start:link_start + self.config.available_links] = self.link_states

        # Add interference indicators
        obs[link_start + 3] = self.config.external_interference_prob

        # Add time-of-day encoding (normalized hour)
        obs[link_start + 4] = self.current_hour / 24.0

        # Add blacklist mask
        blacklist_mask = np.zeros(min(10, self.config.num_channels))
        for ch in self.config.blacklisted_channels[:10]:
            if ch < len(blacklist_mask):
                blacklist_mask[ch] = 1.0
        obs[link_start + 5:link_start + 15] = blacklist_mask

        # Normalize
        obs = np.clip(obs, 0, 1)

        self._last_observation = obs
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action with realistic constraints"""
        self.episode_step += 1
        self.current_step += 1

        # Update time of day
        if self.episode_step % 50 == 0:
            self.current_hour = (self.current_hour + 1) % 24

        # Apply domain drift
        if self.scenario == RealisticScenario.DOMAIN_DRIFT:
            self._apply_domain_drift()

        # Check radar events (DFS)
        if self.scenario == RealisticScenario.RADAR_DFS:
            self._check_radar_events(action)

        # Check policy constraints
        policy_penalty = self._check_policy_constraints(action)

        # Calculate base success probability
        success_prob = self._calculate_success_prob(action)

        # Apply MLO constraints
        mlo_factor = self._apply_mlo_constraints()

        # Apply external interference
        interference_factor = self._apply_interference()

        # Final success probability
        final_success_prob = success_prob * mlo_factor * interference_factor

        # Determine outcome
        success = np.random.random() < final_success_prob

        # Calculate reward with multi-objective considerations
        reward = self._calculate_reward(success, action, policy_penalty)

        # Update channel states
        self._update_channel_states(action, success)

        # Update flow throughputs for fairness tracking
        if self.config.num_flows > 1:
            flow_idx = self.episode_step % self.config.num_flows
            self.flow_throughputs[flow_idx] += 1.0 if success else 0.0

        # Check done condition
        done = self.episode_step >= 500

        # Build info dict with detailed metrics
        info = {
            'success': success,
            'success_prob': final_success_prob,
            'mlo_factor': mlo_factor,
            'interference_factor': interference_factor,
            'policy_penalty': policy_penalty,
            'channel': action,
            'hour': self.current_hour,
            'links_available': np.sum(self.link_states),
            'collision_prob': self.config.collision_prob_base,
        }

        # Add fairness metric
        if self.config.num_flows > 1:
            info['jains_fairness'] = self._calculate_jains_fairness()

        # Add QoS metrics
        info['qos_vo_latency'] = np.random.exponential(5) if success else np.random.exponential(50)
        info['qos_vi_latency'] = np.random.exponential(15) if success else np.random.exponential(80)
        info['qos_be_latency'] = np.random.exponential(30) if success else np.random.exponential(150)

        obs = self._get_observation()
        return obs, reward, done, info

    def _apply_domain_drift(self):
        """Apply gradual domain drift to environment parameters"""
        drift = self.config.domain_drift_rate * self.drift_direction
        self.config.collision_prob_base += drift

        # Bound collision probability and reverse direction at bounds
        if self.config.collision_prob_base > 0.5:
            self.config.collision_prob_base = 0.5
            self.drift_direction = -1
        elif self.config.collision_prob_base < 0.1:
            self.config.collision_prob_base = 0.1
            self.drift_direction = 1

    def _check_radar_events(self, action: int):
        """Check for radar detection events requiring channel evacuation"""
        if np.random.random() < self.config.radar_event_prob:
            # Radar detected on action channel - must evacuate
            if action not in self.config.blacklisted_channels:
                self.config.blacklisted_channels.append(action)
                # Channel unavailable for 30 minutes (simulated as 100 steps)

    def _check_policy_constraints(self, action: int) -> float:
        """Check policy/governance constraints and return penalty"""
        penalty = 0.0

        # Blacklist violation
        if action in self.config.blacklisted_channels:
            penalty += 0.5

        # Time-of-day restrictions
        start_hour, end_hour = self.config.restricted_hours
        if start_hour <= self.current_hour < end_hour:
            if self.scenario == RealisticScenario.TIME_OF_DAY_POLICY:
                # During restricted hours, certain channels have higher penalty
                if action in [0, 1, 2, 3, 4]:  # Low channels restricted
                    penalty += 0.3

        return penalty

    def _calculate_success_prob(self, action: int) -> float:
        """Calculate base success probability for action"""
        if action >= self.config.num_channels:
            return 0.0

        channel_quality = self.channel_states[action]
        collision_prob = self.config.collision_prob_base

        # Adjust for number of stations
        station_factor = 1.0 - (self.config.num_stations / 100.0) * 0.3

        success_prob = channel_quality * (1 - collision_prob) * station_factor
        return np.clip(success_prob, 0.1, 0.95)

    def _apply_mlo_constraints(self) -> float:
        """Apply MLO capability constraints and return factor"""
        if self.scenario not in [RealisticScenario.MIXED_CLIENT_ESTATE,
                                  RealisticScenario.DRIVER_INSTABILITY,
                                  RealisticScenario.SPECTRUM_CONTENTION]:
            return 1.0

        # Check if this transmission uses MLO
        if np.random.random() > self.config.mlo_capable_ratio:
            # Client doesn't support MLO - no MLO benefit
            return 0.85

        # Check MLO stability
        if np.random.random() > self.config.mlo_stability:
            # MLO failed - fall back to single link
            return 0.8

        # Check available links
        active_links = np.sum(self.link_states)
        if active_links < 2:
            # Can't do meaningful aggregation with 1 link
            return 0.9

        # MLO working - full benefit
        return 1.0

    def _apply_interference(self) -> float:
        """Apply external interference effects"""
        if self.config.external_interference_prob == 0:
            return 1.0

        if np.random.random() < self.config.external_interference_prob:
            # External interference occurred
            if self.scenario == RealisticScenario.LTE_U_COEXISTENCE:
                return 0.6  # LTE-U is aggressive
            elif self.scenario == RealisticScenario.BLUETOOTH_INTERFERENCE:
                return 0.75  # BT is less impactful
            else:
                return 0.7

        return 1.0

    def _calculate_reward(self, success: bool, action: int, policy_penalty: float) -> float:
        """Calculate multi-objective reward"""
        base_reward = 1.0 if success else -0.5

        # Apply policy penalty
        reward = base_reward - policy_penalty

        # Add fairness component
        if self.config.fairness_weight > 0 and self.config.num_flows > 1:
            fairness = self._calculate_jains_fairness()
            reward += self.config.fairness_weight * (fairness - 0.5)

        # Add safety traffic priority
        if self.config.safety_traffic_ratio > 0:
            if np.random.random() < self.config.safety_traffic_ratio:
                # This was safety-critical traffic
                reward *= 2.0 if success else 1.5  # Extra penalty for failure

        return reward

    def _calculate_jains_fairness(self) -> float:
        """Calculate Jain's fairness index across flows"""
        if np.sum(self.flow_throughputs) == 0:
            return 1.0

        n = len(self.flow_throughputs)
        sum_x = np.sum(self.flow_throughputs)
        sum_x_sq = np.sum(self.flow_throughputs ** 2)

        if sum_x_sq == 0:
            return 1.0

        fairness = (sum_x ** 2) / (n * sum_x_sq)
        return fairness

    def _update_channel_states(self, action: int, success: bool):
        """Update channel states based on action outcome"""
        # Gradual state evolution
        decay = 0.98
        self.channel_states *= decay
        self.channel_states += np.random.uniform(0, 0.02, self.config.num_channels)

        # Update selected channel based on outcome
        if success:
            self.channel_states[action] = min(1.0, self.channel_states[action] + 0.1)
        else:
            self.channel_states[action] = max(0.1, self.channel_states[action] - 0.15)

        # Update MLO link states
        if self.scenario in [RealisticScenario.DRIVER_INSTABILITY]:
            for i in range(len(self.link_states)):
                if np.random.random() < 0.05:  # 5% chance of link state change
                    self.link_states[i] = 1.0 if self.link_states[i] == 0 else 0.0

        # Clip states
        self.channel_states = np.clip(self.channel_states, 0.1, 1.0)


class RealisticWiFiStandardAgent:
    """
    WiFi standard agent with realistic capability constraints.

    Models how WiFi 7, WiFi 6E, and 802.11k/v/r actually behave
    when facing real-world deployment limitations.
    """

    def __init__(self, standard: str, config: RealisticConfig, scenario: RealisticScenario):
        self.standard = standard
        self.config = config
        self.scenario = scenario
        self.hidden_state = None

        # Track performance degradation factors
        self.capability_factor = self._calculate_capability_factor()

    def _calculate_capability_factor(self) -> float:
        """Calculate how much the standard's capabilities are degraded"""
        factor = 1.0

        if self.standard == "wifi7":
            if self.scenario == RealisticScenario.MIXED_CLIENT_ESTATE:
                # Only 40% of clients can use WiFi 7 features
                factor *= 0.4 + 0.6 * 0.7  # Weighted average
            elif self.scenario == RealisticScenario.DRIVER_INSTABILITY:
                factor *= 0.7  # 30% MLO failures
            elif self.scenario == RealisticScenario.SPECTRUM_CONTENTION:
                factor *= 0.5  # Can't use MLO effectively
            elif self.scenario == RealisticScenario.DOMAIN_DRIFT:
                factor *= 0.85  # Static protocol can't adapt

        elif self.standard == "wifi6e":
            if self.scenario == RealisticScenario.LTE_U_COEXISTENCE:
                factor *= 0.8  # 5GHz affected
            elif self.scenario == RealisticScenario.SPECTRUM_CONTENTION:
                factor *= 0.9  # Still has 6GHz option

        elif self.standard == "80211kvr":
            if self.scenario == RealisticScenario.MOBILITY_EVENTS:
                factor *= 1.1  # Actually helps here
            elif self.scenario == RealisticScenario.DOMAIN_DRIFT:
                factor *= 0.8  # Static roaming decisions

        return factor

    def select_action(self, state: np.ndarray) -> int:
        """Select channel based on standard-specific strategy"""
        num_channels = self.config.num_channels
        channel_states = state[:num_channels]

        # Check blacklist
        valid_channels = [i for i in range(num_channels)
                         if i not in self.config.blacklisted_channels]

        if not valid_channels:
            valid_channels = list(range(num_channels))

        if self.standard == "wifi7":
            return self._wifi7_selection(channel_states, valid_channels)
        elif self.standard == "wifi6e":
            return self._wifi6e_selection(channel_states, valid_channels)
        elif self.standard == "80211kvr":
            return self._80211kvr_selection(channel_states, valid_channels)
        else:
            # Random fallback
            return np.random.choice(valid_channels)

    def _wifi7_selection(self, channel_states: np.ndarray, valid_channels: List[int]) -> int:
        """WiFi 7 channel selection with MLO awareness"""
        # WiFi 7 tries to find channels for multi-link aggregation
        # But in constrained scenarios, this may not work well

        if self.scenario in [RealisticScenario.SPECTRUM_CONTENTION,
                             RealisticScenario.MIXED_CLIENT_ESTATE]:
            # Can't rely on MLO - fall back to single best channel
            valid_states = [(ch, channel_states[ch]) for ch in valid_channels]
            valid_states.sort(key=lambda x: x[1], reverse=True)
            return valid_states[0][0]

        # Normal WiFi 7: try to select from best channels across bands
        # Simulated band grouping
        band_groups = [
            valid_channels[:7],   # 2.4 GHz
            valid_channels[7:14], # 5 GHz
            valid_channels[14:],  # 6 GHz
        ]

        best_per_band = []
        for band in band_groups:
            if band:
                band_ch = [(ch, channel_states[ch]) for ch in band if ch in valid_channels]
                if band_ch:
                    best_per_band.append(max(band_ch, key=lambda x: x[1]))

        if best_per_band:
            # Select best channel (can't actually do MLO aggregation in action space)
            return max(best_per_band, key=lambda x: x[1])[0]

        return np.random.choice(valid_channels)

    def _wifi6e_selection(self, channel_states: np.ndarray, valid_channels: List[int]) -> int:
        """WiFi 6E channel selection preferring 6GHz"""
        # Prefer 6GHz channels (simulated as higher numbered channels)
        six_ghz_channels = [ch for ch in valid_channels if ch >= 14]

        if six_ghz_channels and self.scenario != RealisticScenario.SPECTRUM_CONTENTION:
            # Prefer 6GHz
            six_ghz_states = [(ch, channel_states[ch]) for ch in six_ghz_channels]
            return max(six_ghz_states, key=lambda x: x[1])[0]

        # Fall back to best available
        valid_states = [(ch, channel_states[ch]) for ch in valid_channels]
        return max(valid_states, key=lambda x: x[1])[0]

    def _80211kvr_selection(self, channel_states: np.ndarray, valid_channels: List[int]) -> int:
        """802.11k/v/r channel selection with roaming awareness"""
        # Uses radio resource measurement but doesn't adapt quickly
        # Good for mobility but not for rapid changes

        if not hasattr(self, '_rrm_cache'):
            self._rrm_cache = {}
            self._rrm_age = {}

        # Update RRM cache occasionally (not every step - realistic)
        current_time = getattr(self, '_step_count', 0)
        self._step_count = current_time + 1

        # RRM updates every ~20 steps
        if current_time % 20 == 0:
            for ch in valid_channels:
                self._rrm_cache[ch] = channel_states[ch]
                self._rrm_age[ch] = current_time

        # Select based on cached RRM data
        if self._rrm_cache:
            cached_valid = [(ch, self._rrm_cache.get(ch, 0.5))
                           for ch in valid_channels]
            return max(cached_valid, key=lambda x: x[1])[0]

        return np.random.choice(valid_channels)

    def reset(self):
        """Reset agent state"""
        self.hidden_state = None
        if hasattr(self, '_rrm_cache'):
            self._rrm_cache = {}
            self._rrm_age = {}
        self._step_count = 0


def run_realistic_benchmark():
    """Run comprehensive benchmark with realistic constraints"""

    print("=" * 80)
    print("REALISTIC WIFI STANDARDS BENCHMARK")
    print("Real-World Constraint Scenarios")
    print("=" * 80)

    # Define scenario groups
    scenario_groups = {
        "MLO Capability Constraints": [
            RealisticScenario.MIXED_CLIENT_ESTATE,
            RealisticScenario.DRIVER_INSTABILITY,
            RealisticScenario.SPECTRUM_CONTENTION,
        ],
        "Cross-Layer System Objectives": [
            RealisticScenario.FLEET_FAIRNESS,
            RealisticScenario.QOS_ADMISSION_CONTROL,
            RealisticScenario.GLOBAL_CHANNEL_PLAN,
        ],
        "Variable-Rate Telemetry": [
            RealisticScenario.VARIABLE_TELEMETRY,
            RealisticScenario.MOBILITY_EVENTS,
        ],
        "Multi-Technology Coexistence": [
            RealisticScenario.LTE_U_COEXISTENCE,
            RealisticScenario.RADAR_DFS,
            RealisticScenario.BLUETOOTH_INTERFERENCE,
        ],
        "Policy and Governance": [
            RealisticScenario.CHANNEL_BLACKLIST,
            RealisticScenario.TIME_OF_DAY_POLICY,
            RealisticScenario.SAFETY_PRIORITY,
        ],
        "Training and Drift": [
            RealisticScenario.DOMAIN_DRIFT,
            RealisticScenario.SHORT_TRAINING,
        ],
    }

    all_results = {}

    # Run each scenario
    for group_name, scenarios in scenario_groups.items():
        print(f"\n{'='*80}")
        print(f"SCENARIO GROUP: {group_name}")
        print("=" * 80)

        for scenario in scenarios:
            print(f"\n--- Scenario: {scenario.value} ---")

            config = RealisticConfig()
            env = RealisticWiFiEnvironment(config, scenario)

            # Initialize agents
            ddqn_config = DoubleDQNConfig(state_dim=config.state_dim, action_dim=config.action_dim)
            agents = {
                'WiFi_6E': RealisticWiFiStandardAgent('wifi6e', config, scenario),
                '802.11k/v/r': RealisticWiFiStandardAgent('80211kvr', config, scenario),
                'WiFi_7': RealisticWiFiStandardAgent('wifi7', config, scenario),
                'PPO-LSTM': PPOLSTMAgent(config.state_dim, config.action_dim),
                'DDQN': DoubleDQNAgent(config=ddqn_config),
                'PPO-LTC': PPOLTCAgent(config.state_dim, config.action_dim),
            }

            scenario_results = {}

            # Determine training episodes for this scenario
            if scenario == RealisticScenario.SHORT_TRAINING:
                train_episodes = 20
            else:
                train_episodes = 50  # Further reduced for faster benchmarking

            # Evaluate each agent
            for agent_name, agent in agents.items():
                print(f"  Evaluating {agent_name}...", end=" ", flush=True)

                # Training phase for RL agents
                if isinstance(agent, (PPOLSTMAgent, PPOLTCAgent, DoubleDQNAgent)):
                    for ep in range(train_episodes):
                        state = env.reset()
                        episode_reward = 0

                        for step in range(200):  # Reduced episode length for faster training
                            # Get action with dt awareness for LTC
                            if isinstance(agent, PPOLTCAgent):
                                dt = 1.0 + np.random.uniform(-0.3, 0.3) * config.scan_interval_variance
                                action, log_prob, value = agent.select_action(state, dt=dt)
                            elif isinstance(agent, PPOLSTMAgent):
                                action, log_prob, value = agent.select_action(state)
                            else:
                                action, q_val, _ = agent.select_action(state)
                                log_prob, value = 0.0, q_val

                            next_state, reward, done, info = env.step(action)

                            # Store experience for learning
                            if isinstance(agent, DoubleDQNAgent):
                                agent.store_transition(state, action, reward, next_state, done)
                                if step % 4 == 0:
                                    agent.update()
                            elif isinstance(agent, (PPOLSTMAgent, PPOLTCAgent)):
                                if isinstance(agent, PPOLTCAgent):
                                    agent.store_transition(state, action, log_prob, value, reward, done, dt=dt)
                                else:
                                    agent.store_transition(state, action, log_prob, value, reward, done)

                            state = next_state
                            episode_reward += reward

                            if done:
                                break

                        # Update PPO agents at end of episode
                        if isinstance(agent, (PPOLSTMAgent, PPOLTCAgent)):
                            agent.update()

                # Evaluation phase
                eval_metrics = {
                    'successes': 0,
                    'total_steps': 0,
                    'total_reward': 0,
                    'latencies': [],
                    'fairness_scores': [],
                    'policy_penalties': 0,
                    'qos_vo': [],
                    'qos_vi': [],
                    'qos_be': [],
                }

                num_eval_episodes = 10  # Reduced for faster evaluation

                for ep in range(num_eval_episodes):
                    state = env.reset()

                    for step in range(200):  # Reduced episode length
                        if isinstance(agent, RealisticWiFiStandardAgent):
                            action = agent.select_action(state)
                        elif isinstance(agent, PPOLTCAgent):
                            dt = 1.0 + np.random.uniform(-0.3, 0.3) * config.scan_interval_variance
                            action, _, _ = agent.select_action(state, dt=dt, deterministic=True)
                        elif isinstance(agent, PPOLSTMAgent):
                            action, _, _ = agent.select_action(state, deterministic=True)
                        else:
                            action, _, _ = agent.select_action(state, deterministic=True)

                        next_state, reward, done, info = env.step(action)

                        eval_metrics['total_steps'] += 1
                        eval_metrics['total_reward'] += reward
                        if info['success']:
                            eval_metrics['successes'] += 1

                        eval_metrics['policy_penalties'] += info['policy_penalty']
                        eval_metrics['qos_vo'].append(info['qos_vo_latency'])
                        eval_metrics['qos_vi'].append(info['qos_vi_latency'])
                        eval_metrics['qos_be'].append(info['qos_be_latency'])

                        if 'jains_fairness' in info:
                            eval_metrics['fairness_scores'].append(info['jains_fairness'])

                        # Estimate latency from success/collision
                        latency = 5.0 if info['success'] else 25.0 + np.random.exponential(10)
                        eval_metrics['latencies'].append(latency)

                        state = next_state
                        if done:
                            break

                    # Reset standard agents
                    if isinstance(agent, RealisticWiFiStandardAgent):
                        agent.reset()

                # Calculate final metrics
                success_rate = eval_metrics['successes'] / eval_metrics['total_steps'] * 100
                avg_latency = np.mean(eval_metrics['latencies'])
                p95_latency = np.percentile(eval_metrics['latencies'], 95)
                avg_reward = eval_metrics['total_reward'] / num_eval_episodes

                qos_score = (
                    0.4 * (1.0 / (1.0 + np.mean(eval_metrics['qos_vo']) / 10)) +
                    0.3 * (1.0 / (1.0 + np.mean(eval_metrics['qos_vi']) / 30)) +
                    0.3 * (1.0 / (1.0 + np.mean(eval_metrics['qos_be']) / 100))
                )

                fairness = np.mean(eval_metrics['fairness_scores']) if eval_metrics['fairness_scores'] else 1.0

                scenario_results[agent_name] = {
                    'success_rate': success_rate,
                    'avg_latency': avg_latency,
                    'p95_latency': p95_latency,
                    'qos_score': qos_score,
                    'fairness': fairness,
                    'avg_reward': avg_reward,
                    'policy_penalties': eval_metrics['policy_penalties'],
                    'qos_vo_avg': np.mean(eval_metrics['qos_vo']),
                    'qos_vi_avg': np.mean(eval_metrics['qos_vi']),
                    'qos_be_avg': np.mean(eval_metrics['qos_be']),
                }

                print(f"Success: {success_rate:.1f}%, Latency: {avg_latency:.1f}ms, QoS: {qos_score:.3f}")

            all_results[scenario.value] = scenario_results

            # Print scenario summary
            print(f"\n  Scenario Summary ({scenario.value}):")
            print(f"  {'Agent':<15} {'Success%':>10} {'Latency':>10} {'P95 Lat':>10} {'QoS':>8} {'Fairness':>10}")
            print(f"  {'-'*65}")

            # Find winner
            winner = max(scenario_results.items(),
                        key=lambda x: x[1]['success_rate'] * 0.4 + x[1]['qos_score'] * 100 * 0.3 + (100 - x[1]['avg_latency']) * 0.3)

            for agent_name, metrics in scenario_results.items():
                marker = " ***" if agent_name == winner[0] else ""
                print(f"  {agent_name:<15} {metrics['success_rate']:>9.1f}% {metrics['avg_latency']:>9.1f}ms "
                      f"{metrics['p95_latency']:>9.1f}ms {metrics['qos_score']:>7.3f} {metrics['fairness']:>9.3f}{marker}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: WINNERS BY SCENARIO GROUP")
    print("=" * 80)

    group_winners = {}
    for group_name, scenarios in scenario_groups.items():
        print(f"\n{group_name}:")
        for scenario in scenarios:
            results = all_results[scenario.value]
            # Composite score: success_rate * 0.4 + qos * 0.3 + (100-latency) * 0.3
            winner = max(results.items(),
                        key=lambda x: x[1]['success_rate'] * 0.4 + x[1]['qos_score'] * 100 * 0.3 + max(0, 100 - x[1]['avg_latency']) * 0.3)
            print(f"  {scenario.value:<25}: {winner[0]} ({winner[1]['success_rate']:.1f}%)")

            if group_name not in group_winners:
                group_winners[group_name] = {}
            group_winners[group_name][scenario.value] = winner[0]

    # Count overall wins
    print("\n" + "=" * 80)
    print("OVERALL WIN COUNTS")
    print("=" * 80)

    win_counts = {}
    for group_results in group_winners.values():
        for winner in group_results.values():
            win_counts[winner] = win_counts.get(winner, 0) + 1

    for agent, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {agent}: {wins} scenario wins")

    # Save results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS TABLE")
    print("=" * 80)

    # Create summary table
    print(f"\n{'Scenario':<25} | {'WiFi_6E':>10} | {'802.11k/v/r':>12} | {'WiFi_7':>10} | {'PPO-LSTM':>10} | {'DDQN':>10} | {'PPO-LTC':>10}")
    print("-" * 105)

    for scenario_name, results in all_results.items():
        row = f"{scenario_name:<25}"
        for agent in ['WiFi_6E', '802.11k/v/r', 'WiFi_7', 'PPO-LSTM', 'DDQN', 'PPO-LTC']:
            rate = results[agent]['success_rate']
            row += f" | {rate:>9.1f}%"
        print(row)

    return all_results


if __name__ == "__main__":
    results = run_realistic_benchmark()
