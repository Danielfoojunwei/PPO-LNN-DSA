"""
Main Benchmark Script: PPO-LSTM vs PPO-LFM

Replicates experiments from the research paper and compares both approaches.
"""

import os
import sys
import torch
import numpy as np
import argparse
import time
from pathlib import Path

# Import PPO-LSTM (baseline)
from baseline.ppo_lstm_agent import PPOLSTMAgent

# Import PPO-LFM (our implementation)
from models.ppo_lfm_agent import PPOLFMAgent

# Import environment
from environment.spectrum_env import DynamicSpectrumAccessEnv

# Import federated learning components
from federated.client import FederatedClient
from federated.edge_server import EdgeServer
from federated.cloud_server import CloudServer

# Import utilities
from utils.config import load_config
from utils.metrics import Logger
from benchmark.metrics_tracker import BenchmarkMetrics, ComparisonMetrics


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_model(agent, env, num_episodes=50):
    """
    Evaluate a trained model.

    Args:
        agent: Trained agent (PPO-LSTM or PPO-LFM)
        env: Environment
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    success_rates = []
    collision_rates = []
    throughputs = []
    forward_times = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        total_successes = 0
        total_collisions = 0
        done = False

        while not done:
            # Measure forward pass time
            start_time = time.time()

            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, _, _ = agent.select_action(state_tensor, deterministic=True)

            forward_time = time.time() - start_time
            forward_times.append(forward_time)

            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if info['success']:
                total_successes += 1
            if info['collision']:
                total_collisions += 1

            state = next_state

        episode_rewards.append(episode_reward)
        success_rates.append(total_successes / episode_length if episode_length > 0 else 0)
        collision_rates.append(total_collisions / episode_length if episode_length > 0 else 0)
        throughputs.append(env.total_throughput)

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_success_rate': np.mean(success_rates),
        'std_success_rate': np.std(success_rates),
        'mean_collision_rate': np.mean(collision_rates),
        'std_collision_rate': np.std(collision_rates),
        'mean_throughput': np.mean(throughputs),
        'std_throughput': np.std(throughputs),
        'spectrum_efficiency': np.mean(success_rates) * np.mean(episode_rewards),
        'mean_forward_time': np.mean(forward_times),
    }


def train_single_agent(agent_type, config, metrics_tracker, logger):
    """
    Train a single agent (PPO-LSTM or PPO-LFM).

    Args:
        agent_type: 'lstm' or 'lfm'
        config: Configuration
        metrics_tracker: BenchmarkMetrics instance
        logger: Logger instance

    Returns:
        Trained agent
    """
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.log(f"\nTraining {agent_type.upper()} on device: {device}")

    # Initialize environment
    env = DynamicSpectrumAccessEnv(
        num_channels=config.env.num_channels,
        num_devices=config.env.num_devices,
        max_steps=config.env.max_steps
    )

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # Initialize agent
    if agent_type == 'lstm':
        agent = PPOLSTMAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.model.hidden_dim,
            num_lstm_layers=config.model.num_lstm_layers,
            lr=config.ppo.lr,
            gamma=config.ppo.gamma,
            device=device
        )
    else:  # lfm
        agent = PPOLFMAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.model.hidden_dim,
            num_lfm_layers=config.model.num_lfm_layers,
            lr=config.ppo.lr,
            gamma=config.ppo.gamma,
            device=device
        )

    # Training loop
    num_episodes = config.federated.num_rounds * config.federated.episodes_per_round
    num_rounds = config.federated.num_rounds
    episodes_per_round = config.federated.episodes_per_round

    round_num = 0
    episode_num = 0

    while round_num < num_rounds:
        round_num += 1
        round_start_time = time.time()

        round_rewards = []
        round_success_rates = []
        round_collision_rates = []
        round_throughputs = []

        # Train for episodes_per_round episodes
        for _ in range(episodes_per_round):
            episode_num += 1

            state = env.reset()
            episode_reward = 0
            episode_length = 0
            total_successes = 0
            total_collisions = 0
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # Measure forward time
                start_forward = time.time()
                action, log_prob, value = agent.select_action(state_tensor)
                forward_time = time.time() - start_forward
                metrics_tracker.add_timing(forward_time=forward_time)

                next_state, reward, done, info = env.step(action)

                agent.store_transition(state, action, log_prob, value, reward, done)

                episode_reward += reward
                episode_length += 1

                if info['success']:
                    total_successes += 1
                if info['collision']:
                    total_collisions += 1

                # Update at intervals
                if agent.total_steps % config.ppo.update_interval == 0 and len(agent.buffer.states) > 0:
                    start_update = time.time()
                    update_metrics = agent.update(
                        n_epochs=config.ppo.n_epochs,
                        batch_size=config.ppo.batch_size
                    )
                    update_time = time.time() - start_update
                    metrics_tracker.add_timing(update_time=update_time)

                agent.total_steps += 1
                state = next_state

            # Episode metrics
            round_rewards.append(episode_reward)
            round_success_rates.append(total_successes / episode_length if episode_length > 0 else 0)
            round_collision_rates.append(total_collisions / episode_length if episode_length > 0 else 0)
            round_throughputs.append(env.total_throughput)

        # Final update for remaining transitions
        if len(agent.buffer.states) > 0:
            agent.update()

        # Round metrics
        round_metrics = {
            'mean_reward': np.mean(round_rewards),
            'mean_success_rate': np.mean(round_success_rates),
            'mean_collision_rate': np.mean(round_collision_rates),
            'mean_throughput': np.mean(round_throughputs),
        }

        metrics_tracker.add_training_round(round_num, round_metrics)

        # Logging
        if round_num % config.log_interval == 0:
            logger.log(
                f"[{agent_type.upper()}] Round {round_num}/{num_rounds} - "
                f"Reward: {round_metrics['mean_reward']:.2f}, "
                f"Success: {round_metrics['mean_success_rate']:.2%}, "
                f"Collision: {round_metrics['mean_collision_rate']:.2%}"
            )

        # Evaluation
        if round_num % config.eval_interval == 0:
            eval_metrics = evaluate_model(agent, env, num_episodes=config.benchmark.eval_episodes)
            metrics_tracker.add_evaluation(eval_metrics)
            logger.log(
                f"[{agent_type.upper()}] Eval - "
                f"Reward: {eval_metrics['mean_reward']:.2f}, "
                f"Success: {eval_metrics['mean_success_rate']:.2%}"
            )

    return agent


def run_benchmark(config, trial_num=0):
    """
    Run complete benchmark comparing PPO-LSTM and PPO-LFM.

    Args:
        config: Configuration
        trial_num: Trial number (for multiple runs)

    Returns:
        tuple: (lstm_metrics, lfm_metrics)
    """
    # Create output directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Initialize logger
    log_file = os.path.join(config.log_dir, f'benchmark_trial_{trial_num}.log')
    logger = Logger(log_file=log_file, console=True)

    logger.log("=" * 100)
    logger.log(f"BENCHMARK TRIAL {trial_num + 1}")
    logger.log("=" * 100)
    logger.log(f"Configuration:")
    logger.log(f"  Channels: {config.env.num_channels}")
    logger.log(f"  Devices: {config.env.num_devices}")
    logger.log(f"  Rounds: {config.federated.num_rounds}")
    logger.log(f"  Episodes per round: {config.federated.episodes_per_round}")
    logger.log("=" * 100)

    # Initialize metrics trackers
    lstm_metrics = BenchmarkMetrics('PPO-LSTM')
    lfm_metrics = BenchmarkMetrics('PPO-LFM')

    # Train PPO-LSTM
    logger.log("\n" + "=" * 100)
    logger.log("TRAINING PPO-LSTM (Baseline)")
    logger.log("=" * 100)
    lstm_agent = train_single_agent('lstm', config, lstm_metrics, logger)

    # Train PPO-LFM
    logger.log("\n" + "=" * 100)
    logger.log("TRAINING PPO-LFM (Our Approach)")
    logger.log("=" * 100)
    lfm_agent = train_single_agent('lfm', config, lfm_metrics, logger)

    # Save metrics
    lstm_metrics.save_to_file(
        os.path.join(config.results_dir, f'lstm_metrics_trial_{trial_num}.json')
    )
    lfm_metrics.save_to_file(
        os.path.join(config.results_dir, f'lfm_metrics_trial_{trial_num}.json')
    )

    # Generate comparison
    logger.log("\n" + "=" * 100)
    logger.log("COMPARISON RESULTS")
    logger.log("=" * 100)

    comparison = ComparisonMetrics(lstm_metrics, lfm_metrics)
    comparison_table = comparison.generate_comparison_table()
    logger.log(comparison_table)

    comparison.save_comparison(
        os.path.join(config.results_dir, f'comparison_trial_{trial_num}.json')
    )

    return lstm_metrics, lfm_metrics


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(
        description='Benchmark PPO-LSTM vs PPO-LFM'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/benchmark.yaml',
        help='Path to config file'
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda/auto)')
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    # Override with command-line arguments
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device

    num_trials = args.trials

    # Run benchmark trials
    all_lstm_metrics = []
    all_lfm_metrics = []

    for trial in range(num_trials):
        # Set seed for this trial
        trial_seed = config.seed + trial
        set_seed(trial_seed)

        print(f"\n{'=' * 100}")
        print(f"RUNNING TRIAL {trial + 1}/{num_trials} (seed={trial_seed})")
        print(f"{'=' * 100}\n")

        lstm_metrics, lfm_metrics = run_benchmark(config, trial_num=trial)
        all_lstm_metrics.append(lstm_metrics)
        all_lfm_metrics.append(lfm_metrics)

    # Aggregate results across trials
    print(f"\n{'=' * 100}")
    print(f"AGGREGATE RESULTS ACROSS {num_trials} TRIAL(S)")
    print(f"{'=' * 100}\n")

    if num_trials > 1:
        # Compute mean and std across trials
        lstm_summaries = [m.get_summary() for m in all_lstm_metrics]
        lfm_summaries = [m.get_summary() for m in all_lfm_metrics]

        print(f"{'Metric':<40} {'PPO-LSTM':<25} {'PPO-LFM':<25}")
        print("-" * 90)

        metrics_to_show = [
            ('Final Success Rate', 'final_success_rate', '.2%'),
            ('Final Collision Rate', 'final_collision_rate', '.2%'),
            ('Final Reward', 'final_reward', '.2f'),
            ('Convergence Round', 'convergence_round', 'd'),
            ('Mean Forward Time (ms)', 'mean_forward_time', '.3f'),
        ]

        for name, key, fmt in metrics_to_show:
            lstm_vals = [s[key] for s in lstm_summaries]
            lfm_vals = [s[key] for s in lfm_summaries]

            lstm_mean = np.mean(lstm_vals)
            lstm_std = np.std(lstm_vals)
            lfm_mean = np.mean(lfm_vals)
            lfm_std = np.std(lfm_vals)

            if 'time' in key.lower() and key != 'convergence_round':
                lstm_mean *= 1000
                lstm_std *= 1000
                lfm_mean *= 1000
                lfm_std *= 1000

            print(f"{name:<40} {lstm_mean:{fmt}} ± {lstm_std:{fmt}}   {lfm_mean:{fmt}} ± {lfm_std:{fmt}}")

    print(f"\n{'=' * 100}")
    print("Benchmark complete! Results saved to:", config.results_dir)
    print(f"{'=' * 100}\n")


if __name__ == '__main__':
    main()
