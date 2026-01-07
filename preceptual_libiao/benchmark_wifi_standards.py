#!/usr/bin/env python3
"""
Extended WiFi Standards Benchmark: WiFi 6E, 802.11k/v/r vs WiFi 7 vs PPO Agents

This benchmark compares:
1. WiFi 6E (802.11ax in 6GHz band) - More channels, less interference
2. 802.11k/v/r (Fast Roaming suite) - Optimized roaming with neighbor reports
3. WiFi 7 Standard (802.11be)
4. WiFi 7 + PPO-LSTM
5. WiFi 7 + DDQN
6. WiFi 7 + PPO-LTC

Standards Explained:
- WiFi 6E: Extension of WiFi 6 to 6GHz band (1200MHz spectrum, 59 new channels)
- 802.11k: Radio Resource Measurement - neighbor AP reports for faster scanning
- 802.11v: BSS Transition Management - AP-assisted roaming decisions
- 802.11r: Fast BSS Transition - pre-authentication for seamless handoffs
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
from preceptual.baselines.double_dqn import DoubleDQNAgent, DoubleDQNConfig


class WiFiStandard(Enum):
    """WiFi Standards being compared"""
    WIFI_6E = "wifi_6e"           # 802.11ax in 6GHz
    FAST_ROAMING = "802.11k/v/r"  # Fast roaming suite
    WIFI_7 = "wifi_7"             # 802.11be


class TrafficType(Enum):
    """WiFi QoS Access Categories"""
    VOICE = "voice"
    VIDEO = "video"
    BEST_EFFORT = "best_effort"
    BACKGROUND = "background"


class WiFi7Scenario(Enum):
    """Test scenarios"""
    LOW_CONTENTION = "low_contention"
    HIGH_CONTENTION = "high_contention"
    CHANNEL_SWITCHING = "channel_switching"
    MLO_AGGREGATION = "mlo_aggregation"
    POWER_SAVE = "power_save"
    DENSE_AP = "dense_ap"
    SINGLE_AP = "single_ap"
    ROAMING = "roaming"


@dataclass
class WiFiConfig:
    """WiFi configuration for different standards"""
    standard: WiFiStandard = WiFiStandard.WIFI_7
    scenario: WiFi7Scenario = WiFi7Scenario.LOW_CONTENTION

    # Channel Configuration
    num_links: int = 3
    channels_per_link: int = 8
    total_channels: int = 24

    # Traffic
    traffic_type: TrafficType = TrafficType.BEST_EFFORT

    # Contention
    num_stations: int = 20
    collision_prob_base: float = 0.1

    # Timing (seconds)
    slot_time: float = 0.009
    sifs: float = 0.016
    difs: float = 0.034
    beacon_interval: float = 0.1024

    # Power Save
    psm_enabled: bool = False
    psm_wake_interval: float = 0.3

    # Standard-specific
    fast_roaming_enabled: bool = False  # 802.11k/v/r
    neighbor_report_available: bool = False  # 802.11k
    preauth_enabled: bool = False  # 802.11r
    wifi6e_mode: bool = False  # 6GHz band

    # Episode
    max_steps: int = 200

    # Rewards
    reward_success: float = 1.0
    reward_collision: float = -0.5
    reward_latency_penalty: float = -0.1


def configure_standard(config: WiFiConfig):
    """Apply standard-specific configuration"""
    std = config.standard

    if std == WiFiStandard.WIFI_6E:
        # WiFi 6E: 6GHz band with 59 channels, less interference
        config.total_channels = 59  # More channels in 6GHz
        config.channels_per_link = 20
        config.collision_prob_base *= 0.6  # Less congested band
        config.num_links = 1  # No MLO in WiFi 6E
        config.wifi6e_mode = True
        config.slot_time = 0.009  # Same slot time

    elif std == WiFiStandard.FAST_ROAMING:
        # 802.11k/v/r: Fast roaming optimization
        config.fast_roaming_enabled = True
        config.neighbor_report_available = True  # 802.11k
        config.preauth_enabled = True  # 802.11r
        config.num_links = 1  # Standard single link
        config.total_channels = 24

    elif std == WiFiStandard.WIFI_7:
        # WiFi 7: Full features
        config.num_links = 3
        config.total_channels = 24
        config.channels_per_link = 8


def configure_scenario(config: WiFiConfig):
    """Configure scenario-specific parameters"""
    scenario = config.scenario

    if scenario == WiFi7Scenario.LOW_CONTENTION:
        config.num_stations = 5
        config.collision_prob_base = 0.05
        config.psm_enabled = False

    elif scenario == WiFi7Scenario.HIGH_CONTENTION:
        config.num_stations = 50
        config.collision_prob_base = 0.3
        config.psm_enabled = False

    elif scenario == WiFi7Scenario.CHANNEL_SWITCHING:
        config.num_stations = 25
        config.collision_prob_base = 0.15

    elif scenario == WiFi7Scenario.MLO_AGGREGATION:
        config.num_stations = 30
        config.collision_prob_base = 0.2

    elif scenario == WiFi7Scenario.POWER_SAVE:
        config.num_stations = 15
        config.collision_prob_base = 0.1
        config.psm_enabled = True

    elif scenario == WiFi7Scenario.DENSE_AP:
        config.num_stations = 40
        config.collision_prob_base = 0.25
        config.num_links = 3

    elif scenario == WiFi7Scenario.SINGLE_AP:
        config.num_stations = 10
        config.collision_prob_base = 0.08
        config.num_links = 1

    elif scenario == WiFi7Scenario.ROAMING:
        config.num_stations = 25
        config.collision_prob_base = 0.2


class WiFiEnvironment:
    """WiFi Environment supporting multiple standards"""

    def __init__(self, config: Optional[WiFiConfig] = None):
        self.config = config or WiFiConfig()
        configure_standard(self.config)
        configure_scenario(self.config)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        cfg = self.config

        # Channel states
        self.channel_states = np.zeros(cfg.total_channels)
        self._randomize_channel_states()

        # Link states
        self.link_quality = np.random.uniform(0.5, 1.0, max(1, cfg.num_links))

        # Timing
        self.step_count = 0
        self.total_time = 0.0
        self.dt_history = []

        # Metrics
        self.successes = 0
        self.collisions = 0
        self.total_latency = 0.0
        self.throughput = 0.0

        # State
        self.current_traffic = cfg.traffic_type
        self.backoff_stage = 0
        self.in_psm = False
        self.burst_active = random.random() < 0.5

        # Roaming state
        self.roaming_scan_active = False
        self.neighbor_cache = []  # For 802.11k
        self.preauth_done = False  # For 802.11r

        self.current_dt = self._generate_dt()

        obs = self._get_observation()
        return obs, {"dt": self.current_dt}

    def _randomize_channel_states(self):
        cfg = self.config
        busy_prob = cfg.collision_prob_base * cfg.num_stations / 50
        busy_prob = min(0.7, busy_prob)

        for i in range(cfg.total_channels):
            self.channel_states[i] = 1 if random.random() < busy_prob else 0

    def _generate_dt(self) -> float:
        cfg = self.config
        scenario = cfg.scenario

        base_dt = cfg.slot_time * (2 ** self.backoff_stage)

        # Standard-specific timing adjustments
        if cfg.standard == WiFiStandard.WIFI_6E:
            # WiFi 6E: Generally more predictable in 6GHz
            if scenario == WiFi7Scenario.ROAMING:
                if random.random() < 0.08:  # Less scanning needed
                    dt = 0.04  # 40ms scan (faster than WiFi 7)
                else:
                    dt = cfg.slot_time * random.randint(2, 8)
            else:
                dt = cfg.slot_time * random.randint(2, 6)

        elif cfg.standard == WiFiStandard.FAST_ROAMING:
            # 802.11k/v/r: Much faster roaming
            if scenario == WiFi7Scenario.ROAMING:
                if random.random() < 0.05:  # Rare scanning due to neighbor reports
                    if cfg.preauth_enabled:
                        dt = 0.015  # 15ms fast transition with 802.11r
                    else:
                        dt = 0.025  # 25ms with neighbor report
                else:
                    dt = cfg.slot_time * random.randint(2, 8)
            elif scenario == WiFi7Scenario.POWER_SAVE:
                if self.in_psm:
                    dt = cfg.psm_wake_interval
                else:
                    dt = cfg.slot_time * random.randint(2, 8)
            else:
                dt = cfg.slot_time * random.randint(2, 8)

        else:  # WiFi 7
            if scenario == WiFi7Scenario.LOW_CONTENTION:
                dt = cfg.slot_time * random.randint(1, 4)
            elif scenario == WiFi7Scenario.HIGH_CONTENTION:
                dt = base_dt * random.uniform(1, 10)
            elif scenario == WiFi7Scenario.CHANNEL_SWITCHING:
                if random.random() < 0.15:
                    dt = 0.02  # Channel switch delay
                else:
                    dt = cfg.slot_time * random.randint(2, 8)
            elif scenario == WiFi7Scenario.MLO_AGGREGATION:
                dt = cfg.slot_time * random.randint(1, 6)
            elif scenario == WiFi7Scenario.POWER_SAVE:
                if self.in_psm:
                    dt = cfg.psm_wake_interval
                else:
                    dt = cfg.slot_time * random.randint(2, 8)
            elif scenario == WiFi7Scenario.DENSE_AP:
                dt = base_dt * random.uniform(2, 15)
            elif scenario == WiFi7Scenario.SINGLE_AP:
                dt = cfg.slot_time * random.randint(2, 6)
            elif scenario == WiFi7Scenario.ROAMING:
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
        cfg = self.config

        change_prob = 1 - np.exp(-dt / 0.1)

        for i in range(cfg.total_channels):
            if random.random() < change_prob:
                if random.random() < 0.5:
                    self.channel_states[i] = 1 - self.channel_states[i]
                else:
                    busy_prob = cfg.collision_prob_base * cfg.num_stations / 50
                    self.channel_states[i] = 1 if random.random() < busy_prob else 0

        # Update link quality
        for i in range(len(self.link_quality)):
            self.link_quality[i] += random.gauss(0, 0.05 * dt)
            self.link_quality[i] = np.clip(self.link_quality[i], 0.3, 1.0)

        # Update PSM state
        if cfg.psm_enabled:
            if self.in_psm:
                self.in_psm = False
            else:
                if random.random() < 0.1:
                    self.in_psm = True

        # Update burst state
        if cfg.scenario == WiFi7Scenario.CHANNEL_SWITCHING:
            if random.random() < 0.1:
                self.burst_active = not self.burst_active

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        cfg = self.config

        dt = self.current_dt
        self.dt_history.append(dt)
        self.total_time += dt
        self.step_count += 1

        self._update_environment(dt)

        action = int(action) % cfg.total_channels

        # Calculate collision probability
        collision_prob = cfg.collision_prob_base
        if self.channel_states[action] == 1:
            collision_prob = 0.8

        # Standard-specific collision adjustments
        if cfg.standard == WiFiStandard.WIFI_6E:
            collision_prob *= 0.7  # Less congested 6GHz band
        elif cfg.standard == WiFiStandard.FAST_ROAMING:
            if cfg.scenario == WiFi7Scenario.ROAMING:
                collision_prob *= 0.8  # Better AP selection

        # Adjust for link quality
        if cfg.num_links > 0:
            link_idx = action // max(1, cfg.channels_per_link)
            if link_idx < len(self.link_quality):
                collision_prob *= (1.1 - self.link_quality[link_idx])

        # Determine outcome
        if random.random() < collision_prob:
            reward = cfg.reward_collision
            success = False
            self.collisions += 1
            self.backoff_stage = min(6, self.backoff_stage + 1)
            latency = dt * (2 ** self.backoff_stage)
        else:
            reward = cfg.reward_success
            success = True
            self.successes += 1
            self.backoff_stage = 0
            self.throughput += 1.0
            latency = dt

            if self.current_traffic == TrafficType.VOICE:
                if latency > 0.02:
                    reward += cfg.reward_latency_penalty
            elif self.current_traffic == TrafficType.VIDEO:
                if latency > 0.05:
                    reward += cfg.reward_latency_penalty * 0.5

        self.total_latency += latency
        self.current_dt = self._generate_dt()

        obs = self._get_observation()
        terminated = False
        truncated = self.step_count >= cfg.max_steps

        info = {
            "success": success,
            "collision": not success,
            "dt": dt,
            "next_dt": self.current_dt,
            "latency": latency,
            "throughput": self.throughput,
            "standard": cfg.standard.value,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        cfg = self.config

        # Pad or truncate channel states to fixed size
        channel_obs = np.zeros(60, dtype=np.float32)  # Max channels
        channel_obs[:min(60, cfg.total_channels)] = self.channel_states[:min(60, cfg.total_channels)]

        # Link quality (padded)
        link_obs = np.zeros(3, dtype=np.float32)
        link_obs[:min(3, len(self.link_quality))] = self.link_quality[:min(3, len(self.link_quality))]

        # Traffic encoding
        traffic_enc = np.zeros(4, dtype=np.float32)
        traffic_enc[list(TrafficType).index(self.current_traffic)] = 1.0

        # Timing features
        dt_features = np.array([
            self.current_dt / 0.1,
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

        # Standard encoding
        std_enc = np.zeros(3, dtype=np.float32)
        std_enc[list(WiFiStandard).index(cfg.standard)] = 1.0

        return np.concatenate([channel_obs, link_obs, traffic_enc, dt_features, perf_features, std_enc])

    def get_state_dim(self) -> int:
        return 60 + 3 + 4 + 5 + 4 + 3  # channels + links + traffic + dt + perf + standard

    def get_action_dim(self) -> int:
        return 60  # Max channels (actions)

    def get_metrics(self) -> Dict[str, float]:
        total = max(1, self.successes + self.collisions)
        return {
            "success_rate": self.successes / total,
            "collision_rate": self.collisions / total,
            "throughput": self.throughput,
            "avg_latency": self.total_latency / max(1, self.step_count),
            "total_time": self.total_time,
            "dt_mean": np.mean(self.dt_history) if self.dt_history else 0,
            "dt_std": np.std(self.dt_history) if len(self.dt_history) > 1 else 0,
            "dt_cv": (np.std(self.dt_history) / np.mean(self.dt_history))
                     if self.dt_history and np.mean(self.dt_history) > 0 else 0,
        }


class WiFiStandardAgent:
    """Standard WiFi agent (no learning)"""

    def __init__(self, state_dim: int, action_dim: int, standard: WiFiStandard = WiFiStandard.WIFI_7):
        self.action_dim = action_dim
        self.standard = standard
        self.channel_success_counts = np.zeros(action_dim)
        self.channel_attempt_counts = np.ones(action_dim)

    def reset_hidden(self):
        pass

    def select_action(self, state: np.ndarray, dt: float = None, **kwargs) -> Tuple[int, float, float]:
        # Extract channel states
        channel_states = state[:60]

        # Find idle channels
        idle_channels = np.where(channel_states < 0.5)[0]

        if len(idle_channels) > 0:
            success_rates = self.channel_success_counts / self.channel_attempt_counts
            idle_success_rates = success_rates[idle_channels]
            weights = idle_success_rates + 0.1
            weights = weights / weights.sum()
            action = np.random.choice(idle_channels, p=weights)
        else:
            action = np.argmax(self.channel_success_counts / self.channel_attempt_counts)

        return int(action), 0.0, 0.0

    def store_transition(self, state, action, log_prob, value, reward, done, dt=None):
        self.channel_attempt_counts[action] += 1
        if reward > 0:
            self.channel_success_counts[action] += 1

    def update(self) -> Dict[str, float]:
        return {}


def train_and_evaluate(
    agent,
    env: WiFiEnvironment,
    episodes: int = 100,
    agent_name: str = "agent"
) -> Dict[str, Any]:
    """Train and evaluate an agent"""

    metrics = {
        "success_rates": [],
        "collision_rates": [],
        "throughputs": [],
        "latencies": [],
        "dt_cvs": [],
        "episode_rewards": [],
    }

    start_time = time.time()

    for ep in range(episodes):
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

            # Limit action to valid range
            action = min(action, env.config.total_channels - 1)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info.get("success", False):
                episode_successes += 1
            else:
                episode_collisions += 1
            episode_latency += info.get("latency", dt)

            # Store transition
            if hasattr(agent, 'store_transition'):
                if isinstance(agent, DoubleDQNAgent):
                    agent.store_transition(state, action, reward, next_state, done)
                elif isinstance(agent, (PPOLTCAgent, PPOLNNAgent)):
                    agent.store_transition(state, action, log_prob, value, reward, done, dt=dt)
                else:
                    try:
                        agent.store_transition(state, action, log_prob, value, reward, done, dt=dt)
                    except TypeError:
                        try:
                            agent.store_transition(state, action, log_prob, value, reward, done)
                        except TypeError:
                            agent.store_transition(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state
            step += 1

        if hasattr(agent, 'update'):
            agent.update()

        total_decisions = episode_successes + episode_collisions

        metrics["success_rates"].append(episode_successes / max(1, total_decisions))
        metrics["collision_rates"].append(episode_collisions / max(1, total_decisions))
        metrics["throughputs"].append(episode_successes)
        metrics["latencies"].append(episode_latency / max(1, step))
        metrics["dt_cvs"].append(np.std(episode_dts) / np.mean(episode_dts) if np.mean(episode_dts) > 0 else 0)
        metrics["episode_rewards"].append(episode_reward)

        if ep % 25 == 0 or ep == episodes - 1:
            print(f"    [{agent_name}] Ep {ep}: "
                  f"Success={metrics['success_rates'][-1]*100:.1f}%, "
                  f"Throughput={metrics['throughputs'][-1]:.0f}, "
                  f"Latency={metrics['latencies'][-1]*1000:.2f}ms, "
                  f"dt_cv={metrics['dt_cvs'][-1]:.2f}")

    total_time = time.time() - start_time

    return {
        "success_rates": metrics["success_rates"],
        "final_success": np.mean(metrics["success_rates"][-10:]),
        "final_throughput": np.mean(metrics["throughputs"][-10:]),
        "final_latency": np.mean(metrics["latencies"][-10:]),
        "avg_dt_cv": np.mean(metrics["dt_cvs"]),
        "total_time": total_time,
    }


def run_extended_benchmark():
    """Run benchmark comparing WiFi 6E, 802.11k/v/r, WiFi 7, and RL agents"""

    print("=" * 120)
    print("EXTENDED WIFI STANDARDS BENCHMARK")
    print("Comparing: WiFi 6E | 802.11k/v/r | WiFi 7 | PPO-LSTM | DDQN | PPO-LTC")
    print("=" * 120)
    print()

    episodes = 100
    scenarios = list(WiFi7Scenario)

    all_results = {}

    for scenario in scenarios:
        print()
        print("=" * 100)
        print(f"SCENARIO: {scenario.name}")
        print("=" * 100)

        scenario_results = {}

        # Test each standard
        for standard in WiFiStandard:
            print(f"\n  Testing {standard.value}...")
            config = WiFiConfig(standard=standard, scenario=scenario)
            env = WiFiEnvironment(config)
            state_dim = env.get_state_dim()
            action_dim = env.get_action_dim()

            agent = WiFiStandardAgent(state_dim, action_dim, standard)
            metrics = train_and_evaluate(agent, env, episodes, standard.value)
            scenario_results[standard.value] = metrics

        # Test PPO-LSTM on WiFi 7
        print("\n  Testing WiFi 7 + PPO-LSTM...")
        config = WiFiConfig(standard=WiFiStandard.WIFI_7, scenario=scenario)
        env = WiFiEnvironment(config)
        state_dim = env.get_state_dim()
        action_dim = env.get_action_dim()

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

        # Test DDQN on WiFi 7
        print("\n  Testing WiFi 7 + DDQN...")
        env.reset()
        ddqn_config = DoubleDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=3e-4,
            gamma=0.95,
            buffer_size=2000,
            batch_size=64,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            target_update_freq=50,
        )
        ddqn_agent = DoubleDQNAgent(config=ddqn_config, use_dueling=True, device="cpu")
        ddqn_metrics = train_and_evaluate(ddqn_agent, env, episodes, "wifi7_ddqn")
        scenario_results["wifi7_ddqn"] = ddqn_metrics

        # Test PPO-LTC on WiFi 7
        print("\n  Testing WiFi 7 + PPO-LTC...")
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

        all_results[scenario.name] = scenario_results

        # Print scenario summary
        print(f"\n{scenario.name} FINAL:")
        print(f"  WiFi 6E:      {scenario_results['wifi_6e']['final_success']*100:.1f}%")
        print(f"  802.11k/v/r:  {scenario_results['802.11k/v/r']['final_success']*100:.1f}%")
        print(f"  WiFi 7 Std:   {scenario_results['wifi_7']['final_success']*100:.1f}%")
        print(f"  PPO-LSTM:     {scenario_results['wifi7_lstm']['final_success']*100:.1f}%")
        print(f"  DDQN:         {scenario_results['wifi7_ddqn']['final_success']*100:.1f}%")
        print(f"  PPO-LTC:      {scenario_results['wifi7_ltc']['final_success']*100:.1f}%")
        print(f"  dt_cv:        {scenario_results['wifi7_ltc']['avg_dt_cv']:.2f}")

    # Final summary table
    print("\n" + "=" * 140)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 140)
    print(f"{'Scenario':<20} {'WiFi6E':>9} {'802.11kvr':>10} {'WiFi7':>9} {'LSTM':>9} {'DDQN':>9} {'LTC':>9} {'dt_cv':>7}")
    print("-" * 140)

    for scenario in scenarios:
        r = all_results[scenario.name]
        print(f"{scenario.name:<20} "
              f"{r['wifi_6e']['final_success']*100:>8.1f}% "
              f"{r['802.11k/v/r']['final_success']*100:>9.1f}% "
              f"{r['wifi_7']['final_success']*100:>8.1f}% "
              f"{r['wifi7_lstm']['final_success']*100:>8.1f}% "
              f"{r['wifi7_ddqn']['final_success']*100:>8.1f}% "
              f"{r['wifi7_ltc']['final_success']*100:>8.1f}% "
              f"{r['wifi7_ltc']['avg_dt_cv']:>6.2f}")

    print("\n" + "=" * 140)
    print("BENCHMARK COMPLETE")
    print("=" * 140)

    return all_results


if __name__ == "__main__":
    results = run_extended_benchmark()
