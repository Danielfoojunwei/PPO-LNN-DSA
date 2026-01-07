#!/usr/bin/env python3
"""
Comprehensive Benchmark: PPO-LSTM vs PPO-LNN (ncps CfC) vs PPO-LFM

This benchmark compares:
1. Random baseline
2. PPO-LSTM (discrete-time, no dt-awareness)
3. PPO-LNN (ncps CfC - simplified dt-aware architecture)
4. PPO-LFM (complex LFM with MoE/attention - optional)

Tests across multiple dt patterns to evaluate dt-awareness advantages.
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preceptual.baselines.ppo_lstm import PPOLSTMAgent
from preceptual.baselines.ppo_lnn import PPOLNNAgent
from preceptual.sim.dsa_variable_dt import (
    VariableDTDSAEnvironment as VariableDTDSAEnv,
    DTPattern,
    VariableDTConfig,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    episodes: int = 100
    trials: int = 3
    hidden_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.95
    device: str = "cpu"


class RandomAgentWrapper:
    """Random agent for baseline comparison."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.total_steps = 0

    def reset_hidden(self):
        pass

    def select_action(self, state, dt=None, deterministic=False):
        action = np.random.randint(0, self.action_dim)
        return action, 0.0, 0.0

    def store_transition(self, *args, **kwargs):
        self.total_steps += 1

    def update(self):
        return {"loss": 0.0}


def train_agent(
    agent,
    env: VariableDTDSAEnv,
    episodes: int,
    agent_name: str,
    print_freq: int = 50,
) -> Dict[str, List[float]]:
    """
    Train an agent and collect metrics.

    Returns:
        Dictionary with success_rates, rewards, dt_means, etc.
    """
    metrics = {
        "success_rates": [],
        "episode_rewards": [],
        "dt_means": [],
        "dt_stds": [],
        "episode_lengths": [],
        "training_time": [],
    }

    start_time = time.time()

    for ep in range(episodes):
        ep_start = time.time()
        state, info = env.reset()
        agent.reset_hidden()

        episode_reward = 0
        episode_dts = []
        episode_length = 0

        done = False
        while not done:
            dt = info.get("dt", 1.0)
            episode_dts.append(dt)

            # Select action
            if hasattr(agent, 'select_action'):
                if isinstance(agent, (PPOLNNAgent,)):
                    action, log_prob, value = agent.select_action(state, dt=dt)
                else:
                    action, log_prob, value = agent.select_action(state)
            else:
                action = agent.select_action(state)
                log_prob, value = 0.0, 0.0

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            if hasattr(agent, 'store_transition'):
                if isinstance(agent, (PPOLNNAgent,)):
                    agent.store_transition(state, action, log_prob, value, reward, done, dt=dt)
                else:
                    agent.store_transition(state, action, log_prob, value, reward, done)

            episode_reward += reward
            episode_length += 1
            state = next_state

        # Update agent
        if hasattr(agent, 'update'):
            agent.update()

        # Collect metrics
        success_rate = info.get("success_rate", 0.0)
        metrics["success_rates"].append(success_rate)
        metrics["episode_rewards"].append(episode_reward)
        metrics["dt_means"].append(np.mean(episode_dts))
        metrics["dt_stds"].append(np.std(episode_dts))
        metrics["episode_lengths"].append(episode_length)
        metrics["training_time"].append(time.time() - ep_start)

        # Print progress
        if ep == 0 or (ep + 1) % print_freq == 0:
            print(f"    [{agent_name}] Ep {ep}: Success={success_rate*100:.2f}%, "
                  f"dt_mean={np.mean(episode_dts):.2f}, "
                  f"time={time.time()-ep_start:.2f}s")

    metrics["total_time"] = time.time() - start_time
    return metrics


def run_benchmark(
    config: BenchmarkConfig,
    patterns: List[DTPattern],
    include_lfm: bool = False,
) -> Dict:
    """
    Run comprehensive benchmark across multiple dt patterns.

    Returns:
        Dictionary with all results.
    """
    results = {
        "config": vars(config),
        "patterns": {},
    }

    print("=" * 90)
    print("COMPREHENSIVE BENCHMARK: PPO-LSTM vs PPO-LNN (ncps CfC)")
    print("Testing Liquid Neural Network advantages in variable time-step environments")
    print("=" * 90)
    print()
    print(f"Episodes: {config.episodes}")
    print(f"Trials: {config.trials}")
    print(f"Patterns: {[p.name for p in patterns]}")
    print("=" * 90)

    for pattern in patterns:
        print()
        print("=" * 90)
        print(f"DT PATTERN: {pattern.name}")
        print("=" * 90)

        pattern_results = {
            "random": [],
            "ppo_lstm": [],
            "ppo_lnn": [],
        }
        if include_lfm:
            pattern_results["ppo_lfm"] = []

        for trial in range(config.trials):
            print(f"\n  Trial {trial + 1}/{config.trials}")

            # Create environment
            env_config = VariableDTConfig(
                num_channels=20,
                num_idle_channels=5,
                num_pu_channels=10,
                num_su_channels=5,
                max_steps=200,
                dt_pattern=pattern,
                dt_min=0.5,
                dt_max=5.0,
            )
            env = VariableDTDSAEnv(config=env_config)
            state_dim = env.get_state_dim()
            action_dim = env.get_action_dim()

            # Random agent
            print("\n    Training Random...")
            random_agent = RandomAgentWrapper(action_dim)
            random_metrics = train_agent(
                random_agent, env, config.episodes, "random"
            )
            pattern_results["random"].append(random_metrics)

            # PPO-LSTM
            print("\n    Training PPO-LSTM...")
            lstm_agent = PPOLSTMAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_size=config.hidden_size,
                lr=config.lr,
                gamma=config.gamma,
                device=config.device,
            )
            lstm_metrics = train_agent(
                lstm_agent, env, config.episodes, "ppo_lstm"
            )
            pattern_results["ppo_lstm"].append(lstm_metrics)

            # PPO-LNN (ncps CfC)
            print("\n    Training PPO-LNN (CfC)...")
            lnn_agent = PPOLNNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_size=config.hidden_size,
                lr=config.lr,
                gamma=config.gamma,
                use_cfc=True,  # Use CfC (faster than LTC)
                use_ncp_wiring=False,  # Use fully connected (simpler)
                device=config.device,
            )
            lnn_metrics = train_agent(
                lnn_agent, env, config.episodes, "ppo_lnn"
            )
            pattern_results["ppo_lnn"].append(lnn_metrics)

            # PPO-LFM (optional - complex architecture)
            if include_lfm:
                print("\n    Training PPO-LFM...")
                from preceptual.baselines.ppo_lfm import PPOLFMAgent
                lfm_agent = PPOLFMAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_size=config.hidden_size,
                    lr=config.lr,
                    gamma=config.gamma,
                    device=config.device,
                )
                lfm_metrics = train_agent(
                    lfm_agent, env, config.episodes, "ppo_lfm"
                )
                pattern_results["ppo_lfm"].append(lfm_metrics)

        results["patterns"][pattern.name] = pattern_results

    return results


def analyze_results(results: Dict) -> Dict:
    """
    Analyze benchmark results and compute summary statistics.
    """
    analysis = {
        "summary": {},
        "per_pattern": {},
    }

    print()
    print("=" * 90)
    print("RESULTS ANALYSIS")
    print("=" * 90)

    agents = ["random", "ppo_lstm", "ppo_lnn"]
    if "ppo_lfm" in results["patterns"].get(list(results["patterns"].keys())[0], {}):
        agents.append("ppo_lfm")

    for pattern_name, pattern_results in results["patterns"].items():
        print(f"\n--- Pattern: {pattern_name} ---")

        pattern_analysis = {}

        for agent_name in agents:
            if agent_name not in pattern_results:
                continue

            trials = pattern_results[agent_name]

            # Aggregate metrics across trials
            final_success_rates = [t["success_rates"][-1] for t in trials]
            mean_success_rates = [np.mean(t["success_rates"]) for t in trials]
            total_times = [t["total_time"] for t in trials]

            # Learning curve analysis
            all_success_curves = [t["success_rates"] for t in trials]
            mean_curve = np.mean(all_success_curves, axis=0)
            std_curve = np.std(all_success_curves, axis=0)

            # Improvement from start to end
            improvement = mean_curve[-1] - mean_curve[0]

            pattern_analysis[agent_name] = {
                "final_success_mean": np.mean(final_success_rates),
                "final_success_std": np.std(final_success_rates),
                "avg_success_mean": np.mean(mean_success_rates),
                "total_time_mean": np.mean(total_times),
                "improvement": improvement,
                "learning_curve_mean": mean_curve.tolist(),
                "learning_curve_std": std_curve.tolist(),
            }

            print(f"  {agent_name:12s}: "
                  f"Final={np.mean(final_success_rates)*100:.2f}% ± {np.std(final_success_rates)*100:.2f}%, "
                  f"Improvement={improvement*100:+.2f}%, "
                  f"Time={np.mean(total_times):.1f}s")

        analysis["per_pattern"][pattern_name] = pattern_analysis

    # Overall summary
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)

    for agent_name in agents:
        all_finals = []
        all_improvements = []

        for pattern_name, pattern_analysis in analysis["per_pattern"].items():
            if agent_name in pattern_analysis:
                all_finals.append(pattern_analysis[agent_name]["final_success_mean"])
                all_improvements.append(pattern_analysis[agent_name]["improvement"])

        if all_finals:
            analysis["summary"][agent_name] = {
                "avg_final_success": np.mean(all_finals),
                "avg_improvement": np.mean(all_improvements),
            }

            print(f"  {agent_name:12s}: "
                  f"Avg Final={np.mean(all_finals)*100:.2f}%, "
                  f"Avg Improvement={np.mean(all_improvements)*100:+.2f}%")

    # Comparison: LNN vs LSTM
    if "ppo_lnn" in analysis["summary"] and "ppo_lstm" in analysis["summary"]:
        lnn_final = analysis["summary"]["ppo_lnn"]["avg_final_success"]
        lstm_final = analysis["summary"]["ppo_lstm"]["avg_final_success"]
        delta = lnn_final - lstm_final

        print()
        print(f"  PPO-LNN vs PPO-LSTM: {delta*100:+.2f}% "
              f"({'LNN better' if delta > 0 else 'LSTM better'})")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Comprehensive PPO benchmark")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per trial")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--include-lfm", action="store_true", help="Include PPO-LFM (slow)")
    parser.add_argument("--patterns", nargs="+", default=["constant", "uniform", "bimodal"],
                        help="DT patterns to test")
    args = parser.parse_args()

    # Map pattern names to DTPattern enum
    pattern_map = {
        "constant": DTPattern.CONSTANT,
        "uniform": DTPattern.UNIFORM,
        "exponential": DTPattern.EXPONENTIAL,
        "bimodal": DTPattern.BIMODAL,
        "adversarial": DTPattern.ADVERSARIAL,
        "increasing": DTPattern.INCREASING,
        "oscillating": DTPattern.OSCILLATING,
    }

    patterns = [pattern_map[p] for p in args.patterns if p in pattern_map]

    config = BenchmarkConfig(
        episodes=args.episodes,
        trials=args.trials,
        hidden_size=args.hidden_size,
        lr=args.lr,
    )

    # Run benchmark
    results = run_benchmark(config, patterns, include_lfm=args.include_lfm)

    # Analyze results
    analysis = analyze_results(results)

    print()
    print("=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
