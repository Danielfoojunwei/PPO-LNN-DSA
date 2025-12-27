"""
Baseline training script for single-agent PPO-LFM (no federated learning).

This serves as a baseline to compare against the federated approach.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path

from models.ppo_lfm_agent import PPOLFMAgent
from environment.spectrum_env import DynamicSpectrumAccessEnv
from utils.config import load_config, save_config
from utils.metrics import MetricsTracker, Logger, save_metrics_to_json


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_single_agent(config):
    """
    Train a single PPO-LFM agent (baseline, no federated learning).

    Args:
        config: Training configuration
    """
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Determine device
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Initialize environment
    env = DynamicSpectrumAccessEnv(
        num_channels=config.env.num_channels,
        num_devices=config.env.num_devices,
        max_steps=config.env.max_steps
    )

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # Initialize agent
    agent = PPOLFMAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.model.hidden_dim,
        num_lfm_layers=config.model.num_lfm_layers,
        lr=config.ppo.lr,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_epsilon=config.ppo.clip_epsilon,
        value_coef=config.ppo.value_coef,
        entropy_coef=config.ppo.entropy_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        device=device
    )

    # Initialize logger and metrics
    log_file = os.path.join(config.log_dir, 'training_single.log')
    logger = Logger(log_file=log_file, console=True)
    metrics_tracker = MetricsTracker(window_size=100)

    logger.log("=" * 80)
    logger.log("Single-Agent PPO-LFM for Dynamic Spectrum Access (Baseline)")
    logger.log("=" * 80)
    logger.log(f"Configuration:")
    logger.log(f"  Model: hidden_dim={config.model.hidden_dim}, layers={config.model.num_lfm_layers}")
    logger.log(f"  Environment: channels={config.env.num_channels}, devices={config.env.num_devices}")
    logger.log(f"  PPO: lr={config.ppo.lr}, gamma={config.ppo.gamma}")
    logger.log("=" * 80)

    # Training loop
    num_episodes = config.federated.num_rounds * config.federated.episodes_per_round
    episode_num = 0
    total_steps = 0

    while episode_num < num_episodes:
        episode_num += 1

        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Episode loop
        while not done:
            # Select action
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, log_prob, value = agent.select_action(state_tensor)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, log_prob, value, reward, done)

            # Update counters
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # PPO update at intervals
            if total_steps % config.ppo.update_interval == 0:
                update_metrics = agent.update(
                    n_epochs=config.ppo.n_epochs,
                    batch_size=config.ppo.batch_size
                )
                metrics_tracker.add('policy_loss', update_metrics['policy_loss'])
                metrics_tracker.add('value_loss', update_metrics['value_loss'])
                metrics_tracker.add('entropy', update_metrics['entropy'])

            state = next_state

        # Track episode metrics
        metrics_tracker.add('episode_reward', episode_reward)
        metrics_tracker.add('episode_length', episode_length)

        # Logging
        if episode_num % config.log_interval == 0:
            logger.log_metrics(episode_num, {
                'reward': metrics_tracker.get_mean('episode_reward', window=True),
                'length': metrics_tracker.get_mean('episode_length', window=True),
                'total_steps': total_steps
            })

        # Save checkpoint
        if episode_num % (config.save_interval * config.federated.episodes_per_round) == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'single_agent_episode_{episode_num}.pt'
            )
            agent.save(checkpoint_path)
            logger.log(f"Saved checkpoint: {checkpoint_path}")

        # Evaluation
        if episode_num % (config.eval_interval * config.federated.episodes_per_round) == 0:
            eval_metrics = evaluate_agent(agent, env, num_episodes=10)
            logger.log_metrics(episode_num, {
                'eval_reward': eval_metrics['mean_reward'],
                'eval_success_rate': eval_metrics['mean_success_rate'],
                'eval_collision_rate': eval_metrics['mean_collision_rate']
            })

    # Final checkpoint
    final_checkpoint = os.path.join(config.checkpoint_dir, 'single_agent_final.pt')
    agent.save(final_checkpoint)
    logger.log(f"\nSaved final model: {final_checkpoint}")

    logger.log("\n" + "=" * 80)
    logger.log("Training completed!")
    logger.log("=" * 80)


def evaluate_agent(agent, env, num_episodes=10):
    """
    Evaluate an agent.

    Args:
        agent: PPOLFMAgent
        env: Environment
        num_episodes: Number of evaluation episodes

    Returns:
        eval_metrics: Evaluation metrics
    """
    episode_rewards = []
    success_rates = []
    collision_rates = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        total_successes = 0
        total_collisions = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, _, _ = agent.select_action(state_tensor, deterministic=True)

            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if info['success']:
                total_successes += 1
            if info['collision']:
                total_collisions += 1

            state = next_state

        episode_rewards.append(episode_reward)
        success_rates.append(total_successes / episode_length)
        collision_rates.append(total_collisions / episode_length)

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_success_rate': np.mean(success_rates),
        'mean_collision_rate': np.mean(collision_rates)
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train single-agent PPO-LFM (baseline)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/baseline.yaml',
        help='Path to config file'
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda/auto)')
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = load_config()
        print("Using default config")

    # Override with command-line arguments
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device

    # Set random seed
    set_seed(config.seed)

    # Train
    train_single_agent(config)


if __name__ == '__main__':
    main()
