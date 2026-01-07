#!/usr/bin/env python3
"""
Benchmark: PPO-LSTM vs PPO-LNN for Dynamic Spectrum Access

Following Bowen Shen's methodology from:
"Dynamic spectrum access for Internet-of-Things with hierarchical
federated deep reinforcement learning" (Ad Hoc Networks 149, 2023)

Key differences between PPO-LSTM and PPO-LNN:
- PPO-LSTM: Discrete-time LSTM cells with gated updates
- PPO-LNN: Continuous-time LTC cells with dt-aware dynamics

Benchmark Parameters (from paper):
- 10 Secondary Users (SUs), 20 channels
- 5 idle, 10 PU-occupied, 5 SU-occupied
- Gamma: 0.95, Batch size: 50, Buffer: 1000
- Reward = 1 for success, 0 otherwise

Metrics:
1. Success rate (primary metric)
2. Collision rate
3. Cumulative throughput
4. Convergence speed (episodes to 80% success)
5. Inference time (forward pass)
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

from preceptual.sim.dsa_env import DSAEnvironment, DSAConfig, SingleAgentDSAWrapper
from preceptual.baselines.ppo_lstm import PPOLSTMAgent
from preceptual.baselines.double_dqn import DoubleDQNAgent, DoubleDQNConfig, RandomAgent


class PPOLNNAgent:
    """
    PPO-LNN Agent wrapper for DSA benchmark.

    Uses the LNN/LTC backbone with continuous-time dynamics.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.95,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Import LNN components
        from preceptual.rl.lnn_ltc import LNNBackbone

        # Build network
        self.backbone = LNNBackbone(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(self.device)

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_dim),
        ).to(self.device)

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        ).to(self.device)

        # Optimizer
        params = list(self.backbone.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Training params
        self.gamma = gamma
        self.clip_epsilon = 0.2
        self.ppo_epochs = 4

        # State
        self.hidden = None
        self.last_dt = 0.5  # Default dt

        # Buffer
        self.buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
        }

        self.total_steps = 0

    def reset_hidden(self):
        """Reset hidden state."""
        self.hidden = None

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Select action given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dt_tensor = torch.tensor([self.last_dt]).to(self.device)

        with torch.no_grad():
            # Forward through backbone
            if self.hidden is None:
                self.hidden = self.backbone.init_hidden(1, self.device)

            encoded, self.hidden = self.backbone(state_tensor, self.hidden, dt_tensor)

            # Get action logits and value
            logits = self.actor(encoded)
            value = self.critic(encoded).squeeze(-1)

            # Sample action
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ):
        """Store transition in buffer."""
        self.buffer["states"].append(torch.FloatTensor(state))
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["values"].append(value)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        """Update policy."""
        if len(self.buffer["states"]) < 32:
            return {"loss": 0.0}

        # Convert to tensors
        states = torch.stack(self.buffer["states"]).to(self.device)
        actions = torch.tensor(self.buffer["actions"]).to(self.device)
        old_log_probs = torch.tensor(self.buffer["log_probs"]).to(self.device)
        values = torch.tensor(self.buffer["values"]).to(self.device)
        rewards = torch.tensor(self.buffer["rewards"], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.buffer["dones"], dtype=torch.float32).to(self.device)

        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

            td_error = rewards[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else 0) * (1 - dones[t]) - values[t]
            running_advantage = td_error + self.gamma * 0.95 * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_loss = 0
        dt_tensor = torch.tensor([self.last_dt]).to(self.device)

        for _ in range(self.ppo_epochs):
            # Forward pass
            hidden = self.backbone.init_hidden(len(states), self.device)
            encoded, _ = self.backbone(states, hidden, dt_tensor.expand(len(states)))

            logits = self.actor(encoded)
            new_values = self.critic(encoded).squeeze(-1)

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = torch.nn.functional.mse_loss(new_values, returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
                0.5
            )
            self.optimizer.step()

            total_loss += loss.item()

        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []

        return {"loss": total_loss / self.ppo_epochs}


@dataclass
class BenchmarkResult:
    """Results for a single agent trial."""
    agent_name: str
    trial: int

    # Episode metrics
    success_rates: List[float] = field(default_factory=list)
    collision_rates: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    # Timing
    forward_times_ms: List[float] = field(default_factory=list)
    update_times_ms: List[float] = field(default_factory=list)

    # Final metrics (average of last N episodes)
    final_success_rate: float = 0.0
    final_collision_rate: float = 0.0
    final_throughput: float = 0.0
    final_reward: float = 0.0

    # Convergence
    convergence_episode: int = -1
    convergence_threshold: float = 0.7

    # Total time
    total_time_s: float = 0.0

    def compute_final(self, window: int = 50):
        """Compute final metrics from last N episodes."""
        if len(self.success_rates) >= window:
            self.final_success_rate = np.mean(self.success_rates[-window:])
            self.final_collision_rate = np.mean(self.collision_rates[-window:])
            self.final_throughput = np.mean(self.throughputs[-window:])
            self.final_reward = np.mean(self.rewards[-window:])

    def find_convergence(self, threshold: float = 0.7, window: int = 20):
        """Find episode where success rate first exceeds threshold."""
        self.convergence_threshold = threshold
        for i in range(window, len(self.success_rates)):
            if np.mean(self.success_rates[i-window:i]) >= threshold:
                self.convergence_episode = i - window
                return
        self.convergence_episode = -1

    def to_dict(self) -> Dict:
        return {
            "agent_name": self.agent_name,
            "trial": self.trial,
            "final_success_rate": self.final_success_rate,
            "final_collision_rate": self.final_collision_rate,
            "final_throughput": self.final_throughput,
            "final_reward": self.final_reward,
            "convergence_episode": self.convergence_episode,
            "mean_forward_time_ms": np.mean(self.forward_times_ms) if self.forward_times_ms else 0,
            "mean_update_time_ms": np.mean(self.update_times_ms) if self.update_times_ms else 0,
            "total_time_s": self.total_time_s,
            "num_episodes": len(self.success_rates),
        }


def train_agent(
    agent_class,
    agent_name: str,
    env: SingleAgentDSAWrapper,
    num_episodes: int,
    max_steps: int,
    trial: int,
    seed: int,
    device: str,
    **agent_kwargs,
) -> BenchmarkResult:
    """Train an agent and collect metrics."""
    result = BenchmarkResult(agent_name=agent_name, trial=trial)

    # Create agent
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    if agent_class == "ppo_lstm":
        agent = PPOLSTMAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **agent_kwargs,
        )
    elif agent_class == "ppo_lnn":
        agent = PPOLNNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **agent_kwargs,
        )
    elif agent_class == "double_dqn":
        config = DoubleDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_kwargs,
        )
        agent = DoubleDQNAgent(config, device=device)
    elif agent_class == "random":
        agent = RandomAgent(action_dim=action_dim)
    else:
        raise ValueError(f"Unknown agent class: {agent_class}")

    start_time = time.time()

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + trial * 10000 + episode)

        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()

        episode_reward = 0
        episode_successes = 0
        episode_collisions = 0

        for step in range(max_steps):
            # Select action with timing
            t0 = time.time()
            if hasattr(agent, 'select_action'):
                if agent_class in ["ppo_lstm", "ppo_lnn"]:
                    action, log_prob, value = agent.select_action(state)
                else:
                    action, _, _ = agent.select_action(state)
                    log_prob, value = 0.0, 0.0
            else:
                action = np.random.randint(action_dim)
                log_prob, value = 0.0, 0.0

            forward_time = (time.time() - t0) * 1000
            result.forward_times_ms.append(forward_time)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            if info.get("success", False):
                episode_successes += 1
            if info.get("collision", False):
                episode_collisions += 1

            # Store transition
            if hasattr(agent, 'store_transition'):
                if agent_class in ["ppo_lstm", "ppo_lnn"]:
                    agent.store_transition(state, action, log_prob, value, reward, done)
                else:
                    agent.store_transition(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        # Update agent
        t0 = time.time()
        if hasattr(agent, 'update'):
            agent.update()
        update_time = (time.time() - t0) * 1000
        result.update_times_ms.append(update_time)

        # Get episode metrics
        metrics = env.get_metrics()
        result.success_rates.append(metrics["success_rate"])
        result.collision_rates.append(metrics["collision_rate"])
        result.throughputs.append(metrics["throughput"])
        result.rewards.append(episode_reward)

        if episode % 50 == 0:
            print(f"  [{agent_name}] Episode {episode}: "
                  f"Success={metrics['success_rate']:.2%}, "
                  f"Collision={metrics['collision_rate']:.2%}, "
                  f"Reward={episode_reward:.2f}")

    result.total_time_s = time.time() - start_time
    result.compute_final()
    result.find_convergence()

    return result


def run_benchmark(
    num_episodes: int = 500,
    max_steps: int = 200,
    num_trials: int = 3,
    seed: int = 42,
    device: str = "cpu",
    results_dir: str = "./benchmark_results",
):
    """Run complete benchmark."""
    print("=" * 80)
    print("BENCHMARK: PPO-LSTM vs PPO-LNN for Dynamic Spectrum Access")
    print("Following Bowen Shen's Methodology")
    print("=" * 80)

    # DSA environment config (from paper)
    dsa_config = DSAConfig(
        num_secondary_users=10,
        num_channels=20,
        num_idle_channels=5,
        num_pu_channels=10,
        num_su_channels=5,
        max_steps=max_steps,
    )

    print(f"\nConfiguration:")
    print(f"  Secondary Users: {dsa_config.num_secondary_users}")
    print(f"  Channels: {dsa_config.num_channels}")
    print(f"    Idle: {dsa_config.num_idle_channels}")
    print(f"    PU-occupied: {dsa_config.num_pu_channels}")
    print(f"    SU-occupied: {dsa_config.num_su_channels}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps/episode: {max_steps}")
    print(f"  Trials: {num_trials}")
    print("=" * 80)

    os.makedirs(results_dir, exist_ok=True)

    all_results = {
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "num_trials": num_trials,
            "seed": seed,
            "dsa_config": asdict(dsa_config),
        },
        "agents": {},
    }

    agents = [
        ("Random", "random", {}),
        ("Double-DQN", "double_dqn", {"gamma": 0.95, "batch_size": 50}),
        ("PPO-LSTM", "ppo_lstm", {"gamma": 0.95, "hidden_size": 128, "num_lstm_layers": 2}),
        ("PPO-LNN", "ppo_lnn", {"gamma": 0.95, "hidden_size": 128, "num_layers": 2}),
    ]

    for agent_name, agent_class, agent_kwargs in agents:
        print(f"\n{'='*80}")
        print(f"Training {agent_name}")
        print("=" * 80)

        agent_results = []

        for trial in range(num_trials):
            print(f"\n--- Trial {trial + 1}/{num_trials} ---")

            # Create fresh environment
            env = SingleAgentDSAWrapper(dsa_config, mode="shared")

            # Train
            result = train_agent(
                agent_class=agent_class,
                agent_name=agent_name,
                env=env,
                num_episodes=num_episodes,
                max_steps=max_steps,
                trial=trial,
                seed=seed,
                device=device,
                **agent_kwargs,
            )

            agent_results.append(result.to_dict())

            print(f"  Final Success Rate: {result.final_success_rate:.2%}")
            print(f"  Final Collision Rate: {result.final_collision_rate:.2%}")
            print(f"  Convergence Episode: {result.convergence_episode}")
            print(f"  Mean Forward Time: {np.mean(result.forward_times_ms):.3f} ms")

        all_results["agents"][agent_name] = agent_results

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Agent':<15} {'Success Rate':<20} {'Collision Rate':<20} {'Convergence':<15} {'Forward (ms)':<15}")
    print("-" * 85)

    for agent_name in all_results["agents"]:
        results = all_results["agents"][agent_name]

        success_mean = np.mean([r["final_success_rate"] for r in results])
        success_std = np.std([r["final_success_rate"] for r in results])

        collision_mean = np.mean([r["final_collision_rate"] for r in results])
        collision_std = np.std([r["final_collision_rate"] for r in results])

        conv_episodes = [r["convergence_episode"] for r in results if r["convergence_episode"] > 0]
        conv_str = f"{np.mean(conv_episodes):.0f}" if conv_episodes else "N/A"

        forward_mean = np.mean([r["mean_forward_time_ms"] for r in results])

        print(f"{agent_name:<15} "
              f"{success_mean:>7.2%} ± {success_std:<7.2%} "
              f"{collision_mean:>7.2%} ± {collision_std:<7.2%} "
              f"{conv_str:<15} "
              f"{forward_mean:>8.3f}")

    # Comparison
    if "PPO-LSTM" in all_results["agents"] and "PPO-LNN" in all_results["agents"]:
        lstm_results = all_results["agents"]["PPO-LSTM"]
        lnn_results = all_results["agents"]["PPO-LNN"]

        lstm_success = np.mean([r["final_success_rate"] for r in lstm_results])
        lnn_success = np.mean([r["final_success_rate"] for r in lnn_results])

        lstm_forward = np.mean([r["mean_forward_time_ms"] for r in lstm_results])
        lnn_forward = np.mean([r["mean_forward_time_ms"] for r in lnn_results])

        print(f"\n{'='*80}")
        print("PPO-LNN vs PPO-LSTM Comparison")
        print(f"{'='*80}")
        print(f"Success Rate Improvement: {(lnn_success - lstm_success) / max(0.001, lstm_success) * 100:+.1f}%")
        print(f"Forward Time Ratio: {lnn_forward / max(0.001, lstm_forward):.2f}x")

    # Save results
    results_path = os.path.join(results_dir, "ppo_lstm_vs_lnn_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PPO-LSTM vs PPO-LNN for DSA"
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-dir", type=str, default="./benchmark_results")
    args = parser.parse_args()

    run_benchmark(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        num_trials=args.trials,
        seed=args.seed,
        device=args.device,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
