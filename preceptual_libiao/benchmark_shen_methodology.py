#!/usr/bin/env python3
"""
Benchmark: PPO-LNN-DSA vs Double DQN

Following Bowen Shen's methodology from:
"Dynamic spectrum access for Internet-of-Things with hierarchical
federated deep reinforcement learning" (Ad Hoc Networks 149, 2023)

Benchmark Parameters (from paper):
- 10 Secondary Users (SUs) / 20 channels
- Learning rate: adjusted for deep learning
- Discount factor (gamma): 0.95
- Batch size: 50
- Experience replay: 1000
- Epsilon: 0.1 (random action probability)

Metrics (from paper):
1. Total reward (cumulative)
2. Collision rate (cumulative collisions)
3. Success rate (access accuracy)
4. Throughput (sum throughput)
5. Convergence time (rounds to reach threshold)
6. Computation time (forward pass latency)

Baselines compared:
- Random access
- Greedy
- Standard DQN
- Double DQN (Dueling)
- PPO-LNN (ours)
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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preceptual.sim.fleet_world import FleetWorld, FleetWorldConfig
from preceptual.rl.ppo_lnn import PPOLNNPolicy, PPOTrainer
from preceptual.baselines.double_dqn import (
    DoubleDQNAgent,
    DoubleDQNConfig,
    RandomAgent,
    GreedyAgent,
)


@dataclass
class BenchmarkConfig:
    """
    Benchmark configuration following Bowen Shen's paper parameters.
    """
    # Environment (adapted from paper: 10 SUs, 20 channels)
    num_robots: int = 10  # Secondary Users in paper
    num_aps: int = 5  # Access points (each with multiple channels)
    num_channels: int = 20  # Total channels

    # Training parameters (from paper)
    num_episodes: int = 1000  # Paper uses ~5000 steps × 20 iterations
    max_steps_per_episode: int = 500
    eval_interval: int = 50
    eval_episodes: int = 20

    # Learning (from paper)
    gamma: float = 0.95  # Discount factor from paper
    batch_size: int = 50  # From paper
    buffer_size: int = 1000  # Experience pool from paper
    epsilon: float = 0.1  # Random selection probability from paper

    # Seeds for reproducibility
    seed: int = 42
    num_trials: int = 5  # Multiple trials for statistical significance

    # Output
    results_dir: str = "./benchmark_results"
    save_models: bool = True

    # Device
    device: str = "auto"


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    episode: int
    total_reward: float
    success_rate: float
    collision_rate: float
    throughput: float
    steps: int
    forward_time_ms: float


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics following Shen's paper."""
    agent_name: str
    trial: int

    # Core metrics (from paper)
    episode_rewards: List[float] = field(default_factory=list)
    cumulative_rewards: List[float] = field(default_factory=list)
    success_rates: List[float] = field(default_factory=list)
    collision_rates: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)

    # Convergence
    convergence_episode: int = -1
    convergence_threshold: float = 0.8

    # Timing
    forward_times_ms: List[float] = field(default_factory=list)
    update_times_ms: List[float] = field(default_factory=list)
    total_training_time_s: float = 0.0

    # Final performance (last 50 episodes average)
    final_reward: float = 0.0
    final_success_rate: float = 0.0
    final_collision_rate: float = 0.0
    final_throughput: float = 0.0

    def compute_final_metrics(self, window: int = 50):
        """Compute final metrics from last N episodes."""
        if len(self.episode_rewards) >= window:
            self.final_reward = np.mean(self.episode_rewards[-window:])
            self.final_success_rate = np.mean(self.success_rates[-window:])
            self.final_collision_rate = np.mean(self.collision_rates[-window:])
            self.final_throughput = np.mean(self.throughputs[-window:])

    def find_convergence(self, threshold: float = 0.8, window: int = 20):
        """Find episode where performance converges."""
        self.convergence_threshold = threshold
        for i in range(window, len(self.success_rates)):
            if np.mean(self.success_rates[i-window:i]) >= threshold:
                self.convergence_episode = i - window
                return
        self.convergence_episode = -1  # Did not converge

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "trial": self.trial,
            "final_reward": self.final_reward,
            "final_success_rate": self.final_success_rate,
            "final_collision_rate": self.final_collision_rate,
            "final_throughput": self.final_throughput,
            "convergence_episode": self.convergence_episode,
            "mean_forward_time_ms": np.mean(self.forward_times_ms) if self.forward_times_ms else 0,
            "total_training_time_s": self.total_training_time_s,
            "num_episodes": len(self.episode_rewards),
        }


class DSAEnvironmentWrapper:
    """
    Wrapper to make FleetWorld compatible with DSA benchmark.

    Maps the complex fleet control to channel selection actions
    for fair comparison with DQN baselines.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        # Create fleet world
        fleet_config = FleetWorldConfig(
            num_robots=config.num_robots,
            num_aps=config.num_aps,
            tick_interval_seconds=0.5,
            max_episode_seconds=config.max_steps_per_episode * 0.5,
        )
        self.env = FleetWorld(fleet_config)

        # State/action dimensions
        self.state_dim = 42  # Global observation size
        self.action_dim = config.num_channels

        # Track metrics
        self.episode_successes = 0
        self.episode_collisions = 0
        self.episode_throughput = 0.0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment."""
        obs, info = self.env.reset(seed=seed)
        self.episode_successes = 0
        self.episode_collisions = 0
        self.episode_throughput = 0.0
        return obs[:self.state_dim]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.

        Maps discrete channel selection to fleet control action.
        """
        # Convert discrete action to fleet action
        # Action represents channel selection for load balancing
        target_channel = action % 44  # Map to valid channel range

        # Build action dict for fleet world
        fleet_action = {
            "slice_budgets": [0.3, 0.25, 0.2, 0.15, 0.05, 0.05],  # Default budgets
            "congestion_mode": 0,  # Normal mode
            "switch_actions": [],  # Derived from channel selection
            "scan_robots": [],
        }

        # Find robots that could benefit from switching to target channel
        for i, robot in enumerate(self.env.rcs.robots.values()):
            if i >= 2:  # Limit switches per step
                break
            if robot.channel != target_channel:
                # Check if switch would improve balance
                current_load = self.env.lbap.get_channel_load(robot.channel)
                target_load = self.env.lbap.get_channel_load(target_channel)
                if target_load < current_load - 2:
                    fleet_action["switch_actions"].append(
                        (robot.robot_id, robot.ap_ip, target_channel)
                    )

        # Execute step
        obs, reward, terminated, truncated, info = self.env.step(fleet_action)
        done = terminated or truncated

        # Extract metrics
        metrics = info.get("metrics", {})

        # Success: switch worked or system stable
        success = metrics.get("switch_success_rate", 0.5) > 0.5
        if success:
            self.episode_successes += 1

        # Collision: safety violation or failed switch
        collision = metrics.get("safety_violations", 0) > 0
        if collision:
            self.episode_collisions += 1

        # Throughput proxy: inverse of drops
        slice_a_drops = metrics.get("slice_a_drops", 0)
        slice_b_drops = metrics.get("slice_b_drops", 0)
        self.episode_throughput += max(0, 1.0 - (slice_a_drops + slice_b_drops) * 0.1)

        info_out = {
            "success": success,
            "collision": collision,
            "throughput": self.episode_throughput,
            "raw_metrics": metrics,
        }

        return obs[:self.state_dim], reward, done, info_out

    def get_episode_stats(self, steps: int) -> Dict[str, float]:
        """Get episode statistics."""
        return {
            "success_rate": self.episode_successes / max(1, steps),
            "collision_rate": self.episode_collisions / max(1, steps),
            "throughput": self.episode_throughput,
        }


def train_double_dqn(
    env: DSAEnvironmentWrapper,
    config: BenchmarkConfig,
    trial: int,
) -> BenchmarkMetrics:
    """Train Double DQN agent."""
    metrics = BenchmarkMetrics(agent_name="Double-DQN", trial=trial)

    # Create agent with paper parameters
    dqn_config = DoubleDQNConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        gamma=config.gamma,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        epsilon_end=config.epsilon,
    )

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DoubleDQNAgent(dqn_config, use_dueling=True, device=device)

    cumulative_reward = 0.0
    start_time = time.time()

    for episode in range(config.num_episodes):
        state = env.reset(seed=config.seed + trial * 1000 + episode)
        episode_reward = 0.0
        steps = 0

        for step in range(config.max_steps_per_episode):
            # Select action with timing
            t0 = time.time()
            action, q_val, epsilon = agent.select_action(state)
            forward_time = (time.time() - t0) * 1000
            metrics.forward_times_ms.append(forward_time)

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Update with timing
            t0 = time.time()
            update_metrics = agent.update()
            update_time = (time.time() - t0) * 1000
            if update_metrics.get("loss", 0) > 0:
                metrics.update_times_ms.append(update_time)

            episode_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        # Record episode metrics
        cumulative_reward += episode_reward
        stats = env.get_episode_stats(steps)

        metrics.episode_rewards.append(episode_reward)
        metrics.cumulative_rewards.append(cumulative_reward)
        metrics.success_rates.append(stats["success_rate"])
        metrics.collision_rates.append(stats["collision_rate"])
        metrics.throughputs.append(stats["throughput"])

        if episode % 100 == 0:
            print(f"  [Double-DQN] Episode {episode}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Success={stats['success_rate']:.2%}, "
                  f"Epsilon={agent.epsilon:.3f}")

    metrics.total_training_time_s = time.time() - start_time
    metrics.compute_final_metrics()
    metrics.find_convergence()

    return metrics


def train_ppo_lnn(
    env: DSAEnvironmentWrapper,
    config: BenchmarkConfig,
    trial: int,
) -> BenchmarkMetrics:
    """Train PPO-LNN agent."""
    metrics = BenchmarkMetrics(agent_name="PPO-LNN", trial=trial)

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create PPO-LNN policy
    policy = PPOLNNPolicy(
        global_obs_size=env.state_dim,
        robot_obs_size=14,
        hidden_size=128,
        num_lnn_layers=2,
        max_robots=config.num_robots,
        top_k_robots=min(10, config.num_robots),
    ).to(device)

    trainer = PPOTrainer(
        policy,
        lr=3e-4,
        gamma=config.gamma,
        ppo_epochs=4,
        minibatch_size=config.batch_size,
    )

    cumulative_reward = 0.0
    start_time = time.time()

    # Rollout buffer
    rollout = {
        "global_obs": [],
        "robot_obs": [],
        "actions": {"slice_budgets": [], "congestion_mode": [], "scan_quota": [], "robot_actions": []},
        "log_probs": [],
        "values": [],
        "rewards": [],
        "dones": [],
    }

    hidden = None

    for episode in range(config.num_episodes):
        state = env.reset(seed=config.seed + trial * 1000 + episode)
        episode_reward = 0.0
        steps = 0

        for step in range(config.max_steps_per_episode):
            # Prepare observation
            global_obs = torch.FloatTensor(state).unsqueeze(0).to(device)
            robot_obs = torch.zeros(1, config.num_robots, 14).to(device)
            dt = torch.tensor([0.5]).to(device)

            # Select action with timing
            t0 = time.time()
            with torch.no_grad():
                actions, log_prob, value, hidden = policy.get_action(
                    global_obs, robot_obs, dt=dt, hidden=hidden
                )
            forward_time = (time.time() - t0) * 1000
            metrics.forward_times_ms.append(forward_time)

            # Map PPO action to discrete channel for environment
            slice_budgets = actions["slice_budgets"][0].cpu().numpy()
            channel_action = int(np.argmax(slice_budgets) * (env.action_dim / 6))
            channel_action = min(channel_action, env.action_dim - 1)

            # Step environment
            next_state, reward, done, info = env.step(channel_action)

            # Store rollout
            rollout["global_obs"].append(global_obs.cpu())
            rollout["robot_obs"].append(robot_obs.cpu())
            rollout["actions"]["slice_budgets"].append(actions["slice_budgets"].cpu())
            rollout["actions"]["congestion_mode"].append(actions["congestion_mode"].cpu())
            rollout["actions"]["scan_quota"].append(actions["scan_quota"].cpu())
            rollout["log_probs"].append(log_prob.cpu().squeeze())
            rollout["values"].append(value.cpu().squeeze())
            rollout["rewards"].append(reward)
            rollout["dones"].append(done)

            episode_reward += reward
            steps += 1
            state = next_state

            # PPO update every N steps
            if len(rollout["rewards"]) >= 128:
                # Convert to tensors
                update_rollout = {
                    "global_obs": torch.cat(rollout["global_obs"]),
                    "robot_obs": torch.cat(rollout["robot_obs"]),
                    "actions": {
                        "slice_budgets": torch.cat(rollout["actions"]["slice_budgets"]),
                        "congestion_mode": torch.cat(rollout["actions"]["congestion_mode"]),
                        "scan_quota": torch.cat(rollout["actions"]["scan_quota"]),
                    },
                    "log_probs": torch.tensor([lp.item() if lp.dim() == 0 else lp[0].item() for lp in rollout["log_probs"]]),
                    "values": torch.tensor([v.item() if v.dim() == 0 else v[0].item() for v in rollout["values"]]),
                    "rewards": torch.tensor(rollout["rewards"]),
                    "dones": torch.tensor(rollout["dones"]),
                }

                t0 = time.time()
                trainer.update(update_rollout)
                update_time = (time.time() - t0) * 1000
                metrics.update_times_ms.append(update_time)

                # Clear rollout
                for k in rollout:
                    if isinstance(rollout[k], dict):
                        for kk in rollout[k]:
                            rollout[k][kk] = []
                    else:
                        rollout[k] = []

            if done:
                break

        # Record episode metrics
        cumulative_reward += episode_reward
        stats = env.get_episode_stats(steps)

        metrics.episode_rewards.append(episode_reward)
        metrics.cumulative_rewards.append(cumulative_reward)
        metrics.success_rates.append(stats["success_rate"])
        metrics.collision_rates.append(stats["collision_rate"])
        metrics.throughputs.append(stats["throughput"])

        if episode % 100 == 0:
            print(f"  [PPO-LNN] Episode {episode}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Success={stats['success_rate']:.2%}")

    metrics.total_training_time_s = time.time() - start_time
    metrics.compute_final_metrics()
    metrics.find_convergence()

    return metrics


def train_baseline(
    agent_class,
    agent_name: str,
    env: DSAEnvironmentWrapper,
    config: BenchmarkConfig,
    trial: int,
) -> BenchmarkMetrics:
    """Train baseline agent (Random or Greedy)."""
    metrics = BenchmarkMetrics(agent_name=agent_name, trial=trial)

    agent = agent_class(action_dim=env.action_dim)

    cumulative_reward = 0.0
    start_time = time.time()

    for episode in range(config.num_episodes):
        state = env.reset(seed=config.seed + trial * 1000 + episode)
        episode_reward = 0.0
        steps = 0

        for step in range(config.max_steps_per_episode):
            t0 = time.time()
            action, _, _ = agent.select_action(state)
            forward_time = (time.time() - t0) * 1000
            metrics.forward_times_ms.append(forward_time)

            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            episode_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        cumulative_reward += episode_reward
        stats = env.get_episode_stats(steps)

        metrics.episode_rewards.append(episode_reward)
        metrics.cumulative_rewards.append(cumulative_reward)
        metrics.success_rates.append(stats["success_rate"])
        metrics.collision_rates.append(stats["collision_rate"])
        metrics.throughputs.append(stats["throughput"])

        if episode % 100 == 0:
            print(f"  [{agent_name}] Episode {episode}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Success={stats['success_rate']:.2%}")

    metrics.total_training_time_s = time.time() - start_time
    metrics.compute_final_metrics()
    metrics.find_convergence()

    return metrics


def run_benchmark(config: BenchmarkConfig):
    """Run complete benchmark comparison."""
    print("=" * 80)
    print("BENCHMARK: PPO-LNN-DSA vs Double DQN (Bowen Shen Methodology)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Robots/SUs: {config.num_robots}")
    print(f"  APs: {config.num_aps}")
    print(f"  Channels: {config.num_channels}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Trials: {config.num_trials}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Batch Size: {config.batch_size}")
    print("=" * 80)

    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)

    all_results = {
        "config": asdict(config),
        "agents": {},
    }

    agents_to_test = [
        ("Random", lambda env, cfg, t: train_baseline(RandomAgent, "Random", env, cfg, t)),
        ("Greedy", lambda env, cfg, t: train_baseline(GreedyAgent, "Greedy", env, cfg, t)),
        ("Double-DQN", train_double_dqn),
        ("PPO-LNN", train_ppo_lnn),
    ]

    for agent_name, train_fn in agents_to_test:
        print(f"\n{'='*80}")
        print(f"Training {agent_name}")
        print("=" * 80)

        agent_results = []

        for trial in range(config.num_trials):
            print(f"\n--- Trial {trial + 1}/{config.num_trials} ---")

            # Create fresh environment
            env = DSAEnvironmentWrapper(config)

            # Train agent
            metrics = train_fn(env, config, trial)
            agent_results.append(metrics.to_dict())

            print(f"  Final Reward: {metrics.final_reward:.2f}")
            print(f"  Final Success Rate: {metrics.final_success_rate:.2%}")
            print(f"  Final Collision Rate: {metrics.final_collision_rate:.2%}")
            print(f"  Convergence Episode: {metrics.convergence_episode}")
            print(f"  Mean Forward Time: {np.mean(metrics.forward_times_ms):.3f} ms")

        all_results["agents"][agent_name] = agent_results

    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (Mean ± Std across trials)")
    print("=" * 80)

    print(f"\n{'Agent':<15} {'Reward':<20} {'Success Rate':<20} {'Collision Rate':<20} {'Convergence':<15} {'Forward (ms)':<15}")
    print("-" * 105)

    for agent_name in all_results["agents"]:
        results = all_results["agents"][agent_name]

        reward_mean = np.mean([r["final_reward"] for r in results])
        reward_std = np.std([r["final_reward"] for r in results])

        success_mean = np.mean([r["final_success_rate"] for r in results])
        success_std = np.std([r["final_success_rate"] for r in results])

        collision_mean = np.mean([r["final_collision_rate"] for r in results])
        collision_std = np.std([r["final_collision_rate"] for r in results])

        conv_episodes = [r["convergence_episode"] for r in results if r["convergence_episode"] > 0]
        conv_str = f"{np.mean(conv_episodes):.0f}" if conv_episodes else "N/A"

        forward_mean = np.mean([r["mean_forward_time_ms"] for r in results])

        print(f"{agent_name:<15} "
              f"{reward_mean:>7.2f} ± {reward_std:<7.2f} "
              f"{success_mean:>7.2%} ± {success_std:<7.2%} "
              f"{collision_mean:>7.2%} ± {collision_std:<7.2%} "
              f"{conv_str:<15} "
              f"{forward_mean:>8.3f}")

    # Save results
    results_path = os.path.join(config.results_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Calculate improvements (PPO-LNN vs Double-DQN)
    if "PPO-LNN" in all_results["agents"] and "Double-DQN" in all_results["agents"]:
        ppo_results = all_results["agents"]["PPO-LNN"]
        dqn_results = all_results["agents"]["Double-DQN"]

        ppo_throughput = np.mean([r["final_reward"] for r in ppo_results])
        dqn_throughput = np.mean([r["final_reward"] for r in dqn_results])

        improvement = (ppo_throughput - dqn_throughput) / abs(dqn_throughput) * 100

        print(f"\n{'='*80}")
        print("COMPARISON: PPO-LNN vs Double-DQN")
        print(f"{'='*80}")
        print(f"Throughput Improvement: {improvement:+.1f}%")
        print(f"(Paper reports 6.7-9.1% improvement over baselines)")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PPO-LNN-DSA vs Double DQN (Bowen Shen Methodology)"
    )
    parser.add_argument("--num-robots", type=int, default=10, help="Number of robots/SUs")
    parser.add_argument("--num-aps", type=int, default=5, help="Number of APs")
    parser.add_argument("--num-channels", type=int, default=20, help="Number of channels")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--results-dir", type=str, default="./benchmark_results")
    args = parser.parse_args()

    config = BenchmarkConfig(
        num_robots=args.num_robots,
        num_aps=args.num_aps,
        num_channels=args.num_channels,
        num_episodes=args.episodes,
        num_trials=args.trials,
        seed=args.seed,
        device=args.device,
        results_dir=args.results_dir,
    )

    run_benchmark(config)


if __name__ == "__main__":
    main()
