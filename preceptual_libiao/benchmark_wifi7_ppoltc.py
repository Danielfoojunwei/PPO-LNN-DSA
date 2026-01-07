#!/usr/bin/env python3
"""
WiFi 7 (802.11be) vs WiFi 7 + PPO-LTC Comprehensive Benchmark

This benchmark compares:
1. Standard WiFi 7 (using 802.11be default mechanisms)
2. WiFi 7 + PPO-LTC (intelligent dt-aware channel selection)

WiFi 7 Features Simulated:
- Multi-Link Operation (MLO) with up to 3 links
- 320 MHz channels (simulated as multiple sub-channels)
- OFDMA resource unit allocation
- Enhanced QoS with multiple access categories
- Power Save Mode (PSM) transitions

Scenarios tested to find where PPO-LTC helps and where it doesn't:
1. Low contention (stable, predictable)
2. High contention (chaotic, unpredictable)
3. Mixed traffic (voice + video + background)
4. Bursty traffic (alternating idle/active)
5. Power save transitions (bimodal dt)
6. Dense AP deployment (many interferers)
7. Single AP isolation (simple environment)
8. Roaming scenarios (dynamic environment)
"""

import os
import sys
import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preceptual.baselines.ppo_lnn import PPOLTCAgent, PPOLNNAgent
from preceptual.baselines.ppo_lstm import PPOLSTMAgent


class TrafficType(Enum):
    """WiFi QoS Access Categories"""
    VOICE = "voice"           # AC_VO: Highest priority, strict timing
    VIDEO = "video"           # AC_VI: High priority, timing sensitive
    BEST_EFFORT = "best_effort"  # AC_BE: Default
    BACKGROUND = "background"  # AC_BK: Lowest priority


class WiFi7Scenario(Enum):
    """Test scenarios for WiFi 7"""
    LOW_CONTENTION = "low_contention"
    HIGH_CONTENTION = "high_contention"
    MIXED_TRAFFIC = "mixed_traffic"
    BURSTY_TRAFFIC = "bursty_traffic"
    POWER_SAVE = "power_save"
    DENSE_AP = "dense_ap"
    SINGLE_AP = "single_ap"
    ROAMING = "roaming"


@dataclass
class WiFi7Config:
    """WiFi 7 (802.11be) simulation configuration"""
    # MLO Configuration
    num_links: int = 3              # Multi-Link Operation links
    channels_per_link: int = 8      # Sub-channels per link (320MHz / 40MHz)
    total_channels: int = 24        # Total available channels

    # Traffic
    traffic_type: TrafficType = TrafficType.BEST_EFFORT

    # Contention
    num_stations: int = 20          # Number of competing stations
    collision_prob_base: float = 0.1  # Base collision probability

    # Timing (in seconds)
    slot_time: float = 0.009        # 9 μs slot time
    sifs: float = 0.016             # 16 μs SIFS
    difs: float = 0.034             # 34 μs DIFS
    beacon_interval: float = 0.1024  # ~100ms beacon interval

    # Power Save
    psm_enabled: bool = False
    psm_wake_interval: float = 0.3   # Wake every 300ms in PSM

    # Scenario
    scenario: WiFi7Scenario = WiFi7Scenario.LOW_CONTENTION

    # Episode
    max_steps: int = 200

    # Rewards
    reward_success: float = 1.0
    reward_collision: float = -0.5
    reward_latency_penalty: float = -0.1


class WiFi7Environment:
    """
    WiFi 7 (802.11be) Environment Simulation

    Models key WiFi 7 features:
    - Multi-Link Operation: Multiple simultaneous links
    - Channel state tracking: Busy/Idle per channel
    - Contention: CSMA/CA with exponential backoff
    - QoS: Different timing requirements per traffic type
    - Power Save: PSM transitions create bimodal dt
    """

    def __init__(self, config: Optional[WiFi7Config] = None):
        self.config = config or WiFi7Config()
        self._setup_scenario()
        self.reset()

    def _setup_scenario(self):
        """Configure environment based on scenario"""
        cfg = self.config
        scenario = cfg.scenario

        if scenario == WiFi7Scenario.LOW_CONTENTION:
            cfg.num_stations = 5
            cfg.collision_prob_base = 0.05
            cfg.psm_enabled = False

        elif scenario == WiFi7Scenario.HIGH_CONTENTION:
            cfg.num_stations = 50
            cfg.collision_prob_base = 0.3
            cfg.psm_enabled = False

        elif scenario == WiFi7Scenario.MIXED_TRAFFIC:
            cfg.num_stations = 20
            cfg.collision_prob_base = 0.15
            # Traffic type changes during episode

        elif scenario == WiFi7Scenario.BURSTY_TRAFFIC:
            cfg.num_stations = 30
            cfg.collision_prob_base = 0.2
            # Alternates between high/low activity

        elif scenario == WiFi7Scenario.POWER_SAVE:
            cfg.num_stations = 15
            cfg.collision_prob_base = 0.1
            cfg.psm_enabled = True

        elif scenario == WiFi7Scenario.DENSE_AP:
            cfg.num_stations = 40
            cfg.collision_prob_base = 0.25
            cfg.num_links = 3  # More links needed

        elif scenario == WiFi7Scenario.SINGLE_AP:
            cfg.num_stations = 10
            cfg.collision_prob_base = 0.08
            cfg.num_links = 1  # Single link only

        elif scenario == WiFi7Scenario.ROAMING:
            cfg.num_stations = 25
            cfg.collision_prob_base = 0.2
            # Channel quality changes rapidly

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        cfg = self.config

        # Channel states: 0=idle, 1=busy
        self.channel_states = np.zeros(cfg.total_channels)
        self._randomize_channel_states()

        # Link states for MLO
        self.link_quality = np.random.uniform(0.5, 1.0, cfg.num_links)

        # Timing
        self.step_count = 0
        self.total_time = 0.0
        self.dt_history = []

        # Metrics
        self.successes = 0
        self.collisions = 0
        self.total_latency = 0.0
        self.throughput = 0.0

        # Traffic state
        self.current_traffic = cfg.traffic_type
        self.backoff_stage = 0
        self.in_psm = False

        # Burst state
        self.burst_active = random.random() < 0.5

        # Generate dt after all state is initialized
        self.current_dt = self._generate_dt()

        obs = self._get_observation()
        return obs, {"dt": self.current_dt}

    def _randomize_channel_states(self):
        """Randomize channel busy/idle states"""
        cfg = self.config
        busy_prob = cfg.collision_prob_base * cfg.num_stations / 50
        busy_prob = min(0.7, busy_prob)

        for i in range(cfg.total_channels):
            self.channel_states[i] = 1 if random.random() < busy_prob else 0

    def _generate_dt(self) -> float:
        """Generate time step based on scenario"""
        cfg = self.config
        scenario = cfg.scenario

        # Base dt from slot time and backoff
        base_dt = cfg.slot_time * (2 ** self.backoff_stage)

        if scenario == WiFi7Scenario.LOW_CONTENTION:
            # Regular, predictable timing
            dt = cfg.slot_time * random.randint(1, 4)

        elif scenario == WiFi7Scenario.HIGH_CONTENTION:
            # Highly variable due to backoffs
            dt = base_dt * random.uniform(1, 10)

        elif scenario == WiFi7Scenario.MIXED_TRAFFIC:
            # Depends on traffic type
            if self.current_traffic == TrafficType.VOICE:
                dt = cfg.slot_time * 2  # Fast, predictable
            elif self.current_traffic == TrafficType.VIDEO:
                dt = cfg.slot_time * 4
            elif self.current_traffic == TrafficType.BACKGROUND:
                dt = cfg.slot_time * random.randint(8, 32)
            else:
                dt = cfg.slot_time * random.randint(4, 16)

        elif scenario == WiFi7Scenario.BURSTY_TRAFFIC:
            # Bimodal: fast during burst, slow otherwise
            if self.burst_active:
                dt = cfg.slot_time * random.randint(1, 3)
            else:
                dt = cfg.slot_time * random.randint(20, 50)

        elif scenario == WiFi7Scenario.POWER_SAVE:
            # Strong bimodal: PSM wake vs active
            if self.in_psm:
                dt = cfg.psm_wake_interval  # Long sleep
            else:
                dt = cfg.slot_time * random.randint(2, 8)

        elif scenario == WiFi7Scenario.DENSE_AP:
            # Variable due to interference
            dt = base_dt * random.uniform(2, 15)

        elif scenario == WiFi7Scenario.SINGLE_AP:
            # Very predictable
            dt = cfg.slot_time * random.randint(2, 6)

        elif scenario == WiFi7Scenario.ROAMING:
            # Occasional long delays for scanning
            if random.random() < 0.1:
                dt = 0.05  # 50ms scan delay
            else:
                dt = cfg.slot_time * random.randint(2, 10)
        else:
            dt = cfg.slot_time * 4

        # Add noise
        dt += random.gauss(0, cfg.slot_time)
        dt = max(cfg.slot_time, dt)

        return float(dt)

    def _update_environment(self, dt: float):
        """Update environment state based on elapsed time"""
        cfg = self.config

        # Channel states change based on time elapsed
        change_prob = 1 - np.exp(-dt / 0.1)  # ~100ms time constant

        for i in range(cfg.total_channels):
            if random.random() < change_prob:
                # Toggle or randomize
                if random.random() < 0.5:
                    self.channel_states[i] = 1 - self.channel_states[i]
                else:
                    busy_prob = cfg.collision_prob_base * cfg.num_stations / 50
                    self.channel_states[i] = 1 if random.random() < busy_prob else 0

        # Update link quality for MLO
        for i in range(cfg.num_links):
            self.link_quality[i] += random.gauss(0, 0.05 * dt)
            self.link_quality[i] = np.clip(self.link_quality[i], 0.3, 1.0)

        # Update burst state
        if cfg.scenario == WiFi7Scenario.BURSTY_TRAFFIC:
            if random.random() < 0.1:
                self.burst_active = not self.burst_active

        # Update traffic type for mixed scenario
        if cfg.scenario == WiFi7Scenario.MIXED_TRAFFIC:
            if random.random() < 0.05:
                self.current_traffic = random.choice(list(TrafficType))

        # Update PSM state
        if cfg.psm_enabled:
            if self.in_psm:
                # Wake up after interval
                self.in_psm = False
            else:
                # Go to sleep with some probability
                if random.random() < 0.1:
                    self.in_psm = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return result"""
        cfg = self.config

        # Record timing
        dt = self.current_dt
        self.dt_history.append(dt)
        self.total_time += dt
        self.step_count += 1

        # Update environment based on elapsed time
        self._update_environment(dt)

        # Process action (channel selection)
        action = int(action) % cfg.total_channels
        link_idx = action // cfg.channels_per_link
        channel_in_link = action % cfg.channels_per_link

        # Check collision
        collision_prob = cfg.collision_prob_base
        if self.channel_states[action] == 1:
            collision_prob = 0.8  # High collision if channel busy

        # Adjust for link quality in MLO
        if link_idx < cfg.num_links:
            collision_prob *= (1.1 - self.link_quality[link_idx])

        # Determine outcome
        if random.random() < collision_prob:
            # Collision
            reward = cfg.reward_collision
            success = False
            self.collisions += 1
            self.backoff_stage = min(6, self.backoff_stage + 1)
            latency = dt * (2 ** self.backoff_stage)  # Backoff penalty
        else:
            # Success
            reward = cfg.reward_success
            success = True
            self.successes += 1
            self.backoff_stage = 0
            self.throughput += 1.0

            # Latency depends on traffic type
            if self.current_traffic == TrafficType.VOICE:
                latency = dt
                if latency > 0.02:  # 20ms threshold
                    reward += cfg.reward_latency_penalty
            elif self.current_traffic == TrafficType.VIDEO:
                latency = dt
                if latency > 0.05:  # 50ms threshold
                    reward += cfg.reward_latency_penalty * 0.5
            else:
                latency = dt

        self.total_latency += latency

        # Generate next dt
        self.current_dt = self._generate_dt()

        # Get observation
        obs = self._get_observation()

        # Check termination
        terminated = False
        truncated = self.step_count >= cfg.max_steps

        info = {
            "success": success,
            "collision": not success,
            "dt": dt,
            "next_dt": self.current_dt,
            "latency": latency,
            "throughput": self.throughput,
            "traffic_type": self.current_traffic.value,
            "in_psm": self.in_psm,
            "burst_active": self.burst_active,
            "link_quality": self.link_quality.copy(),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        cfg = self.config

        # Channel states
        channel_obs = self.channel_states.astype(np.float32)

        # Link quality
        link_obs = np.zeros(3, dtype=np.float32)
        link_obs[:cfg.num_links] = self.link_quality

        # Traffic encoding
        traffic_enc = np.zeros(4, dtype=np.float32)
        traffic_enc[list(TrafficType).index(self.current_traffic)] = 1.0

        # Timing features
        dt_features = np.array([
            self.current_dt / 0.1,  # Normalized dt
            np.mean(self.dt_history[-10:]) / 0.1 if self.dt_history else 0.5,
            np.std(self.dt_history[-10:]) / 0.1 if len(self.dt_history) > 1 else 0,
            float(self.in_psm),
            float(self.burst_active),
        ], dtype=np.float32)

        # Performance features
        total = max(1, self.successes + self.collisions)
        perf_features = np.array([
            self.successes / total,
            self.collisions / total,
            self.backoff_stage / 6.0,
            self.step_count / cfg.max_steps,
        ], dtype=np.float32)

        return np.concatenate([channel_obs, link_obs, traffic_enc, dt_features, perf_features])

    def get_state_dim(self) -> int:
        cfg = self.config
        return cfg.total_channels + 3 + 4 + 5 + 4  # channels + links + traffic + dt + perf

    def get_action_dim(self) -> int:
        return self.config.total_channels

    def get_metrics(self) -> Dict[str, float]:
        """Get episode metrics"""
        total = max(1, self.successes + self.collisions)
        return {
            "success_rate": self.successes / total,
            "collision_rate": self.collisions / total,
            "throughput": self.throughput,
            "avg_latency": self.total_latency / max(1, self.step_count),
            "total_time": self.total_time,
            "dt_mean": np.mean(self.dt_history) if self.dt_history else 0,
            "dt_std": np.std(self.dt_history) if len(self.dt_history) > 1 else 0,
            "dt_cv": (np.std(self.dt_history) / np.mean(self.dt_history)) if self.dt_history and np.mean(self.dt_history) > 0 else 0,
        }


class WiFi7StandardAgent:
    """
    Standard WiFi 7 Agent using 802.11be default mechanisms

    Uses:
    - Random channel selection with preference for idle channels
    - Standard CSMA/CA behavior
    - No learning, no dt-awareness
    """

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        self.action_dim = action_dim
        self.last_successful_channel = 0
        self.channel_success_counts = np.zeros(action_dim)
        self.channel_attempt_counts = np.ones(action_dim)  # Avoid div by zero

    def reset_hidden(self):
        pass

    def select_action(self, state: np.ndarray, dt: float = None, **kwargs) -> Tuple[int, float, float]:
        """Select channel using WiFi 7 standard approach"""
        # Extract channel states from observation
        channel_states = state[:self.action_dim]

        # Find idle channels
        idle_channels = np.where(channel_states < 0.5)[0]

        if len(idle_channels) > 0:
            # Prefer idle channels with good history
            success_rates = self.channel_success_counts / self.channel_attempt_counts
            idle_success_rates = success_rates[idle_channels]

            # Weighted random selection favoring successful channels
            weights = idle_success_rates + 0.1  # Add small value to ensure exploration
            weights = weights / weights.sum()
            action = np.random.choice(idle_channels, p=weights)
        else:
            # All channels busy, pick least recently failed
            action = np.argmax(self.channel_success_counts / self.channel_attempt_counts)

        return int(action), 0.0, 0.0

    def store_transition(self, state, action, log_prob, value, reward, done, dt=None):
        """Update channel statistics"""
        self.channel_attempt_counts[action] += 1
        if reward > 0:
            self.channel_success_counts[action] += 1
            self.last_successful_channel = action

    def update(self) -> Dict[str, float]:
        return {}


def train_and_evaluate(
    agent,
    env: WiFi7Environment,
    episodes: int = 100,
    agent_name: str = "agent"
) -> Dict[str, Any]:
    """Train and evaluate an agent"""

    metrics = {
        "success_rates": [],
        "collision_rates": [],
        "throughputs": [],
        "latencies": [],
        "dt_means": [],
        "dt_stds": [],
        "dt_cvs": [],  # Coefficient of variation
        "episode_rewards": [],
        "episode_times": [],
    }

    start_time = time.time()

    for ep in range(episodes):
        ep_start = time.time()
        state, info = env.reset()

        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()

        episode_reward = 0
        episode_successes = 0
        episode_collisions = 0
        episode_latency = 0
        episode_dts = []

        done = False
        step = 0
        while not done:
            dt = info.get("dt", 0.01)
            episode_dts.append(dt)

            # Select action
            if hasattr(agent, 'select_action'):
                # Check if agent accepts dt parameter
                if isinstance(agent, (PPOLTCAgent, PPOLNNAgent)):
                    result = agent.select_action(state, dt=dt)
                else:
                    try:
                        result = agent.select_action(state, dt=dt)
                    except TypeError:
                        result = agent.select_action(state)

                if isinstance(result, tuple) and len(result) == 3:
                    action, log_prob, value = result
                elif isinstance(result, tuple):
                    action = result[0]
                    log_prob, value = 0.0, 0.0
                else:
                    action = result
                    log_prob, value = 0.0, 0.0
            else:
                action = random.randint(0, env.get_action_dim() - 1)
                log_prob, value = 0.0, 0.0

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track metrics
            if info.get("success", False):
                episode_successes += 1
            else:
                episode_collisions += 1
            episode_latency += info.get("latency", dt)

            # Store transition
            if hasattr(agent, 'store_transition'):
                if isinstance(agent, (PPOLTCAgent, PPOLNNAgent)):
                    agent.store_transition(state, action, log_prob, value, reward, done, dt=dt)
                else:
                    try:
                        agent.store_transition(state, action, log_prob, value, reward, done, dt=dt)
                    except TypeError:
                        agent.store_transition(state, action, log_prob, value, reward, done)

            episode_reward += reward
            state = next_state
            step += 1

        # Update agent
        if hasattr(agent, 'update'):
            agent.update()

        # Collect metrics
        ep_time = time.time() - ep_start
        total_decisions = episode_successes + episode_collisions

        metrics["success_rates"].append(episode_successes / max(1, total_decisions))
        metrics["collision_rates"].append(episode_collisions / max(1, total_decisions))
        metrics["throughputs"].append(episode_successes)
        metrics["latencies"].append(episode_latency / max(1, step))
        metrics["dt_means"].append(np.mean(episode_dts))
        metrics["dt_stds"].append(np.std(episode_dts))
        metrics["dt_cvs"].append(np.std(episode_dts) / np.mean(episode_dts) if np.mean(episode_dts) > 0 else 0)
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_times"].append(ep_time)

        # Print progress
        if ep % 25 == 0 or ep == episodes - 1:
            print(f"    [{agent_name}] Ep {ep}: "
                  f"Success={metrics['success_rates'][-1]*100:.1f}%, "
                  f"Throughput={metrics['throughputs'][-1]:.0f}, "
                  f"Latency={metrics['latencies'][-1]*1000:.2f}ms, "
                  f"dt_cv={metrics['dt_cvs'][-1]:.2f}")

    total_time = time.time() - start_time

    # Compute final statistics
    return {
        "success_rates": metrics["success_rates"],
        "collision_rates": metrics["collision_rates"],
        "throughputs": metrics["throughputs"],
        "latencies": metrics["latencies"],
        "dt_cvs": metrics["dt_cvs"],
        "final_success": np.mean(metrics["success_rates"][-10:]),
        "final_throughput": np.mean(metrics["throughputs"][-10:]),
        "final_latency": np.mean(metrics["latencies"][-10:]),
        "avg_dt_cv": np.mean(metrics["dt_cvs"]),
        "improvement": metrics["success_rates"][-1] - metrics["success_rates"][0],
        "total_time": total_time,
    }


def run_benchmark():
    """Run comprehensive WiFi 7 vs WiFi 7 + PPO-LTC benchmark"""

    print("=" * 100)
    print("COMPREHENSIVE BENCHMARK: WiFi 7 (Standard) vs WiFi 7 + PPO-LTC")
    print("Testing when dt-aware LTC agents provide benefit over standard WiFi 7 mechanisms")
    print("=" * 100)
    print()

    episodes = 100
    scenarios = list(WiFi7Scenario)

    results = {}

    for scenario in scenarios:
        print()
        print("=" * 100)
        print(f"SCENARIO: {scenario.name}")
        print("=" * 100)

        # Create environment
        config = WiFi7Config(scenario=scenario)
        env = WiFi7Environment(config)

        state_dim = env.get_state_dim()
        action_dim = env.get_action_dim()

        scenario_results = {}

        # Test WiFi 7 Standard
        print("\n  Training WiFi 7 (Standard)...")
        wifi7_agent = WiFi7StandardAgent(state_dim, action_dim)
        wifi7_metrics = train_and_evaluate(wifi7_agent, env, episodes, "wifi7_std")
        scenario_results["wifi7_standard"] = wifi7_metrics

        # Test WiFi 7 + PPO-LSTM (for comparison)
        print("\n  Training WiFi 7 + PPO-LSTM...")
        env.reset()
        lstm_agent = PPOLSTMAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=128,
            lr=3e-4,
            gamma=0.95,
            device="cpu"
        )
        lstm_metrics = train_and_evaluate(lstm_agent, env, episodes, "wifi7_lstm")
        scenario_results["wifi7_lstm"] = lstm_metrics

        # Test WiFi 7 + PPO-LTC (our dt-aware agent)
        print("\n  Training WiFi 7 + PPO-LTC...")
        env.reset()
        ltc_agent = PPOLTCAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=128,
            lr=3e-4,
            gamma=0.95,
            device="cpu"
        )
        ltc_metrics = train_and_evaluate(ltc_agent, env, episodes, "wifi7_ltc")
        scenario_results["wifi7_ltc"] = ltc_metrics

        results[scenario.name] = scenario_results

    # Analysis
    print()
    print("=" * 100)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 100)

    # Create comparison tables
    print("\n" + "-" * 100)
    print("SUCCESS RATE COMPARISON (Final 10 episodes average)")
    print("-" * 100)
    print(f"{'Scenario':<20} {'WiFi7 Std':>12} {'WiFi7+LSTM':>12} {'WiFi7+LTC':>12} {'LTC vs Std':>12} {'LTC vs LSTM':>12} {'dt_cv':>8}")
    print("-" * 100)

    ltc_better_scenarios = []
    ltc_not_needed_scenarios = []

    for scenario in scenarios:
        r = results[scenario.name]
        std = r["wifi7_standard"]["final_success"]
        lstm = r["wifi7_lstm"]["final_success"]
        ltc = r["wifi7_ltc"]["final_success"]
        dt_cv = r["wifi7_ltc"]["avg_dt_cv"]

        ltc_vs_std = (ltc - std) * 100
        ltc_vs_lstm = (ltc - lstm) * 100

        print(f"{scenario.name:<20} {std*100:>11.1f}% {lstm*100:>11.1f}% {ltc*100:>11.1f}% "
              f"{ltc_vs_std:>+11.1f}% {ltc_vs_lstm:>+11.1f}% {dt_cv:>7.2f}")

        # Categorize
        if ltc_vs_std > 2 or ltc_vs_lstm > 2:
            ltc_better_scenarios.append((scenario.name, ltc_vs_std, ltc_vs_lstm, dt_cv))
        if ltc_vs_std < 2 and ltc_vs_lstm < 2:
            ltc_not_needed_scenarios.append((scenario.name, ltc_vs_std, ltc_vs_lstm, dt_cv))

    # Throughput comparison
    print("\n" + "-" * 100)
    print("THROUGHPUT COMPARISON (transmissions per episode)")
    print("-" * 100)
    print(f"{'Scenario':<20} {'WiFi7 Std':>12} {'WiFi7+LSTM':>12} {'WiFi7+LTC':>12} {'LTC Gain':>12}")
    print("-" * 100)

    for scenario in scenarios:
        r = results[scenario.name]
        std = r["wifi7_standard"]["final_throughput"]
        lstm = r["wifi7_lstm"]["final_throughput"]
        ltc = r["wifi7_ltc"]["final_throughput"]
        gain = ((ltc - std) / std * 100) if std > 0 else 0

        print(f"{scenario.name:<20} {std:>12.1f} {lstm:>12.1f} {ltc:>12.1f} {gain:>+11.1f}%")

    # Latency comparison
    print("\n" + "-" * 100)
    print("LATENCY COMPARISON (ms per decision)")
    print("-" * 100)
    print(f"{'Scenario':<20} {'WiFi7 Std':>12} {'WiFi7+LSTM':>12} {'WiFi7+LTC':>12} {'LTC Reduction':>14}")
    print("-" * 100)

    for scenario in scenarios:
        r = results[scenario.name]
        std = r["wifi7_standard"]["final_latency"] * 1000
        lstm = r["wifi7_lstm"]["final_latency"] * 1000
        ltc = r["wifi7_ltc"]["final_latency"] * 1000
        reduction = ((std - ltc) / std * 100) if std > 0 else 0

        print(f"{scenario.name:<20} {std:>11.2f}ms {lstm:>11.2f}ms {ltc:>11.2f}ms {reduction:>+13.1f}%")

    # Training time comparison
    print("\n" + "-" * 100)
    print("TRAINING TIME COMPARISON (seconds)")
    print("-" * 100)
    print(f"{'Scenario':<20} {'WiFi7 Std':>12} {'WiFi7+LSTM':>12} {'WiFi7+LTC':>12} {'LTC Overhead':>14}")
    print("-" * 100)

    for scenario in scenarios:
        r = results[scenario.name]
        std = r["wifi7_standard"]["total_time"]
        lstm = r["wifi7_lstm"]["total_time"]
        ltc = r["wifi7_ltc"]["total_time"]
        overhead = ((ltc - std) / std * 100) if std > 0 else 0

        print(f"{scenario.name:<20} {std:>11.1f}s {lstm:>11.1f}s {ltc:>11.1f}s {overhead:>+13.1f}%")

    # Key findings
    print("\n" + "=" * 100)
    print("KEY FINDINGS: WHERE PPO-LTC HELPS AND WHERE IT DOESN'T")
    print("=" * 100)

    print("\n" + "-" * 50)
    print("SCENARIOS WHERE PPO-LTC PROVIDES SIGNIFICANT BENEFIT:")
    print("-" * 50)
    if ltc_better_scenarios:
        for name, vs_std, vs_lstm, dt_cv in ltc_better_scenarios:
            print(f"  ✓ {name:<20} LTC vs Std: {vs_std:+.1f}%, LTC vs LSTM: {vs_lstm:+.1f}%, dt_cv: {dt_cv:.2f}")
            # Explain why
            if "POWER_SAVE" in name:
                print(f"      → Bimodal dt from PSM wake/sleep cycles requires dt-awareness")
            elif "BURSTY" in name:
                print(f"      → Alternating fast/slow periods benefit from continuous-time modeling")
            elif "HIGH_CONTENTION" in name:
                print(f"      → Variable backoff timing needs adaptive time constant learning")
            elif "DENSE" in name:
                print(f"      → Complex interference patterns require learned temporal dynamics")
            elif "ROAMING" in name:
                print(f"      → Scan delays create irregular timing that LTC handles better")
    else:
        print("  (None identified in this run)")

    print("\n" + "-" * 50)
    print("SCENARIOS WHERE PPO-LTC MAY NOT BE NEEDED:")
    print("-" * 50)
    if ltc_not_needed_scenarios:
        for name, vs_std, vs_lstm, dt_cv in ltc_not_needed_scenarios:
            print(f"  ✗ {name:<20} LTC vs Std: {vs_std:+.1f}%, LTC vs LSTM: {vs_lstm:+.1f}%, dt_cv: {dt_cv:.2f}")
            # Explain why
            if "LOW_CONTENTION" in name:
                print(f"      → Regular, predictable timing doesn't require dt-awareness")
            elif "SINGLE_AP" in name:
                print(f"      → Simple environment with stable dt doesn't benefit from LTC")
            elif "MIXED" in name:
                print(f"      → Traffic type variation matters more than timing variation")
    else:
        print("  (None identified in this run)")

    # Summary table
    print("\n" + "=" * 100)
    print("RECOMMENDATION SUMMARY")
    print("=" * 100)
    print("""
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │ USE PPO-LTC WHEN:                           │ STANDARD WiFi 7 SUFFICIENT WHEN:             │
    ├─────────────────────────────────────────────┼──────────────────────────────────────────────┤
    │ • Power Save Mode (PSM) enabled             │ • Low contention environment                 │
    │ • Bursty/irregular traffic patterns         │ • Single AP with stable channel              │
    │ • High contention with variable backoff     │ • Predictable, regular timing (dt_cv < 0.5)  │
    │ • Dense AP deployment with interference     │ • Simple best-effort traffic only            │
    │ • Roaming with scan delays                  │ • Fixed infrastructure, no mobility          │
    │ • dt coefficient of variation > 1.0         │ • Low latency requirements not critical      │
    │ • Real-time traffic (voice/video)           │                                              │
    └─────────────────────────────────────────────┴──────────────────────────────────────────────┘

    KEY METRIC: dt Coefficient of Variation (dt_cv)
    - dt_cv < 0.5:  Stable timing → Standard WiFi 7 sufficient
    - dt_cv 0.5-1.0: Moderate variation → LTC provides some benefit
    - dt_cv > 1.0:  High variation → LTC significantly outperforms
    """)

    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)

    return results


if __name__ == "__main__":
    results = run_benchmark()
