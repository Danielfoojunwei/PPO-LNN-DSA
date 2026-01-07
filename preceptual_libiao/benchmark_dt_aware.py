#!/usr/bin/env python3
"""
Comprehensive dt-Aware Benchmark: PPO-LSTM vs PPO-LNN (LFM)

This benchmark specifically tests the advantages of Liquid Neural Networks
over traditional LSTM in handling variable time steps.

Based on:
- Liquid.ai LFM: https://github.com/kyegomez/LFM
- ncps library: https://github.com/mlech26l/ncps
- Paper: "Liquid Time-constant Networks" (https://arxiv.org/abs/2006.04439)

Key metrics for dt-awareness:
1. Performance degradation with dt variability
2. Adaptation speed to changing dt patterns
3. Temporal consistency (same state, different dt → appropriate response)
4. Time constant distribution (learned τ values)

Test scenarios:
- Constant dt (baseline)
- Uniform random dt
- Exponential (bursty) dt
- Bimodal dt (fast/slow)
- Adversarial dt (correlated with optimal action)
- Increasing dt (episode progression)
- Oscillating dt (periodic)
"""

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preceptual.sim.dsa_variable_dt import (
    VariableDTDSAEnvironment,
    VariableDTConfig,
    DTPattern,
    create_dt_test_suite,
)
from preceptual.baselines.ppo_lstm import PPOLSTMAgent
from preceptual.baselines.ppo_lfm import PPOLFMAgent
from preceptual.baselines.double_dqn import RandomAgent


@dataclass
class DTAwareMetrics:
    """Comprehensive metrics for dt-awareness evaluation."""
    agent_name: str
    dt_pattern: str
    trial: int

    # Performance
    success_rates: List[float] = field(default_factory=list)
    collision_rates: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    # dt statistics
    dt_means: List[float] = field(default_factory=list)
    dt_stds: List[float] = field(default_factory=list)

    # Timing
    forward_times_ms: List[float] = field(default_factory=list)

    # Final metrics
    final_success_rate: float = 0.0
    final_collision_rate: float = 0.0

    # dt-awareness specific
    dt_correlation: float = 0.0  # Correlation between dt and success
    performance_stability: float = 0.0  # Std of success rate

    # Time constants (for LFM only)
    tau_mean: float = 0.0
    tau_std: float = 0.0

    def compute_final(self, window: int = 50):
        if len(self.success_rates) >= window:
            self.final_success_rate = np.mean(self.success_rates[-window:])
            self.final_collision_rate = np.mean(self.collision_rates[-window:])
            self.performance_stability = np.std(self.success_rates[-window:])

    def compute_dt_correlation(self):
        """Compute correlation between dt and success rate."""
        if len(self.dt_means) > 10 and len(self.success_rates) > 10:
            # Ensure same length
            min_len = min(len(self.dt_means), len(self.success_rates))
            dt_arr = np.array(self.dt_means[:min_len])
            sr_arr = np.array(self.success_rates[:min_len])
            if np.std(dt_arr) > 0 and np.std(sr_arr) > 0:
                self.dt_correlation = float(np.corrcoef(dt_arr, sr_arr)[0, 1])

    def to_dict(self) -> Dict:
        return {
            "agent_name": self.agent_name,
            "dt_pattern": self.dt_pattern,
            "trial": self.trial,
            "final_success_rate": self.final_success_rate,
            "final_collision_rate": self.final_collision_rate,
            "performance_stability": self.performance_stability,
            "dt_correlation": self.dt_correlation,
            "tau_mean": self.tau_mean,
            "tau_std": self.tau_std,
            "mean_forward_time_ms": np.mean(self.forward_times_ms) if self.forward_times_ms else 0,
            "num_episodes": len(self.success_rates),
        }


def train_agent_dt_aware(
    agent_type: str,
    env: VariableDTDSAEnvironment,
    dt_pattern: str,
    num_episodes: int,
    trial: int,
    seed: int,
    device: str,
) -> DTAwareMetrics:
    """Train agent with dt-aware environment."""
    metrics = DTAwareMetrics(
        agent_name=agent_type,
        dt_pattern=dt_pattern,
        trial=trial,
    )

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # Create agent
    if agent_type == "random":
        agent = RandomAgent(action_dim=action_dim)
    elif agent_type == "ppo_lstm":
        agent = PPOLSTMAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=128,
            num_lstm_layers=2,
            gamma=0.95,
            device=device,
        )
    elif agent_type == "ppo_lfm":
        agent = PPOLFMAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=128,
            num_layers=2,
            gamma=0.95,
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    for episode in range(num_episodes):
        state, info = env.reset(seed=seed + trial * 10000 + episode)
        current_dt = info.get("dt", 1.0)

        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()

        episode_reward = 0

        while True:
            # Select action with timing
            t0 = time.time()

            if agent_type == "random":
                action, log_prob, value = agent.select_action(state)
            elif agent_type == "ppo_lstm":
                action, log_prob, value = agent.select_action(state)
            elif agent_type == "ppo_lfm":
                # PPO-LFM uses dt in forward pass!
                action, log_prob, value = agent.select_action(state, dt=current_dt)

            forward_time = (time.time() - t0) * 1000
            metrics.forward_times_ms.append(forward_time)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            if hasattr(agent, 'store_transition'):
                if agent_type == "ppo_lfm":
                    agent.store_transition(state, action, log_prob, value, reward, done, dt=current_dt)
                elif agent_type == "ppo_lstm":
                    agent.store_transition(state, action, log_prob, value, reward, done)

            episode_reward += reward
            state = next_state
            current_dt = info.get("next_dt", 1.0)

            if done:
                break

        # Update agent
        if hasattr(agent, 'update'):
            agent.update()

        # Record metrics
        env_metrics = env.get_metrics()
        metrics.success_rates.append(env_metrics["success_rate"])
        metrics.collision_rates.append(env_metrics["collision_rate"])
        metrics.rewards.append(episode_reward)
        metrics.dt_means.append(env_metrics["dt_mean"])
        metrics.dt_stds.append(env_metrics["dt_std"])

        if episode % 50 == 0:
            print(f"    [{agent_type}] Ep {episode}: "
                  f"Success={env_metrics['success_rate']:.2%}, "
                  f"dt_mean={env_metrics['dt_mean']:.2f}")

    # Finalize metrics
    metrics.compute_final()
    metrics.compute_dt_correlation()

    # Get tau statistics for LFM
    if agent_type == "ppo_lfm" and hasattr(agent, 'get_tau_statistics'):
        tau_stats = agent.get_tau_statistics()
        metrics.tau_mean = tau_stats.get("tau_mean", 0)
        metrics.tau_std = tau_stats.get("tau_std", 0)

    return metrics


def run_dt_aware_benchmark(
    num_episodes: int = 200,
    num_trials: int = 2,
    seed: int = 42,
    device: str = "cpu",
    results_dir: str = "./benchmark_results",
    patterns: Optional[List[str]] = None,
):
    """Run comprehensive dt-aware benchmark."""
    print("=" * 90)
    print("DT-AWARE BENCHMARK: PPO-LSTM vs PPO-LNN (LFM)")
    print("Testing Liquid Neural Network advantages in variable time-step environments")
    print("=" * 90)

    # Get test configurations
    test_suite = create_dt_test_suite()

    if patterns:
        test_suite = {k: v for k, v in test_suite.items() if k in patterns}

    print(f"\nTest Patterns: {list(test_suite.keys())}")
    print(f"Episodes: {num_episodes}")
    print(f"Trials: {num_trials}")
    print("=" * 90)

    os.makedirs(results_dir, exist_ok=True)

    all_results = {
        "config": {
            "num_episodes": num_episodes,
            "num_trials": num_trials,
            "seed": seed,
            "patterns": list(test_suite.keys()),
        },
        "results": {},
    }

    agent_types = ["random", "ppo_lstm", "ppo_lfm"]

    # Run benchmark for each dt pattern
    for pattern_name, config in test_suite.items():
        print(f"\n{'='*90}")
        print(f"DT PATTERN: {pattern_name.upper()}")
        print(f"{'='*90}")

        pattern_results = {}

        for agent_type in agent_types:
            print(f"\n  Training {agent_type}...")

            agent_trials = []

            for trial in range(num_trials):
                print(f"    Trial {trial + 1}/{num_trials}")

                env = VariableDTDSAEnvironment(config)

                metrics = train_agent_dt_aware(
                    agent_type=agent_type,
                    env=env,
                    dt_pattern=pattern_name,
                    num_episodes=num_episodes,
                    trial=trial,
                    seed=seed,
                    device=device,
                )

                agent_trials.append(metrics.to_dict())

            pattern_results[agent_type] = agent_trials

        all_results["results"][pattern_name] = pattern_results

    # Print summary
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY BY DT PATTERN")
    print("=" * 90)

    for pattern_name in all_results["results"]:
        print(f"\n{pattern_name.upper()}:")
        print(f"  {'Agent':<12} {'Success Rate':<18} {'Stability':<15} {'dt-Corr':<12} {'Forward(ms)':<12} {'τ mean':<10}")
        print("  " + "-" * 85)

        pattern_data = all_results["results"][pattern_name]

        for agent_type in agent_types:
            trials = pattern_data[agent_type]

            success = np.mean([t["final_success_rate"] for t in trials])
            success_std = np.std([t["final_success_rate"] for t in trials])
            stability = np.mean([t["performance_stability"] for t in trials])
            dt_corr = np.mean([t["dt_correlation"] for t in trials])
            forward = np.mean([t["mean_forward_time_ms"] for t in trials])
            tau = np.mean([t["tau_mean"] for t in trials])

            tau_str = f"{tau:.2f}" if agent_type == "ppo_lfm" else "N/A"

            print(f"  {agent_type:<12} "
                  f"{success:>6.2%} ± {success_std:<6.2%} "
                  f"{stability:>8.4f}      "
                  f"{dt_corr:>8.3f}    "
                  f"{forward:>8.3f}    "
                  f"{tau_str:<10}")

    # Compute dt-awareness advantage
    print("\n" + "=" * 90)
    print("DT-AWARENESS ANALYSIS: PPO-LFM vs PPO-LSTM")
    print("=" * 90)

    print(f"\n{'Pattern':<15} {'LSTM Success':<15} {'LFM Success':<15} {'Improvement':<15} {'LFM τ mean':<12}")
    print("-" * 75)

    for pattern_name in all_results["results"]:
        pattern_data = all_results["results"][pattern_name]

        lstm_success = np.mean([t["final_success_rate"] for t in pattern_data["ppo_lstm"]])
        lfm_success = np.mean([t["final_success_rate"] for t in pattern_data["ppo_lfm"]])
        improvement = (lfm_success - lstm_success) / max(0.001, lstm_success) * 100
        tau_mean = np.mean([t["tau_mean"] for t in pattern_data["ppo_lfm"]])

        print(f"{pattern_name:<15} {lstm_success:>10.2%}     {lfm_success:>10.2%}     {improvement:>+10.1f}%     {tau_mean:>8.2f}")

    # Overall summary
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)

    # Average across all patterns
    lstm_avg = np.mean([
        np.mean([t["final_success_rate"] for t in all_results["results"][p]["ppo_lstm"]])
        for p in all_results["results"]
    ])
    lfm_avg = np.mean([
        np.mean([t["final_success_rate"] for t in all_results["results"][p]["ppo_lfm"]])
        for p in all_results["results"]
    ])

    print(f"\nAverage Success Rate:")
    print(f"  PPO-LSTM: {lstm_avg:.2%}")
    print(f"  PPO-LFM:  {lfm_avg:.2%}")
    print(f"  Overall Improvement: {(lfm_avg - lstm_avg) / max(0.001, lstm_avg) * 100:+.1f}%")

    # Variable dt advantage (exclude constant)
    variable_patterns = [p for p in all_results["results"] if p != "constant"]
    if variable_patterns:
        lstm_var = np.mean([
            np.mean([t["final_success_rate"] for t in all_results["results"][p]["ppo_lstm"]])
            for p in variable_patterns
        ])
        lfm_var = np.mean([
            np.mean([t["final_success_rate"] for t in all_results["results"][p]["ppo_lfm"]])
            for p in variable_patterns
        ])

        print(f"\nVariable dt Patterns Only:")
        print(f"  PPO-LSTM: {lstm_var:.2%}")
        print(f"  PPO-LFM:  {lfm_var:.2%}")
        print(f"  Variable dt Advantage: {(lfm_var - lstm_var) / max(0.001, lstm_var) * 100:+.1f}%")

    # Save results
    results_path = os.path.join(results_dir, "dt_aware_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="dt-Aware Benchmark: PPO-LSTM vs PPO-LNN (LFM)"
    )
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-dir", type=str, default="./benchmark_results")
    parser.add_argument("--patterns", type=str, nargs="*",
                        help="Specific patterns to test (default: all)")
    args = parser.parse_args()

    run_dt_aware_benchmark(
        num_episodes=args.episodes,
        num_trials=args.trials,
        seed=args.seed,
        device=args.device,
        results_dir=args.results_dir,
        patterns=args.patterns,
    )


if __name__ == "__main__":
    main()
