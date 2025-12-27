"""
Evaluation script for trained PPO-LFM models.
"""

import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from models.ppo_lfm_agent import PPOLFMAgent
from environment.spectrum_env import DynamicSpectrumAccessEnv
from utils.config import load_config
from utils.metrics import Logger


def evaluate_model(agent, env, num_episodes=100, render=False):
    """
    Evaluate a trained model.

    Args:
        agent: Trained PPOLFMAgent
        env: Environment
        num_episodes: Number of evaluation episodes
        render: Whether to render environment

    Returns:
        results: Dictionary of evaluation results
    """
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    collision_rates = []
    throughputs = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        total_successes = 0
        total_collisions = 0
        done = False

        while not done:
            # Select action deterministically
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, _, _ = agent.select_action(state_tensor, deterministic=True)

            # Take step
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if info['success']:
                total_successes += 1
            if info['collision']:
                total_collisions += 1

            if render:
                env.render()

            state = next_state

        # Collect metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_rates.append(total_successes / episode_length if episode_length > 0 else 0)
        collision_rates.append(total_collisions / episode_length if episode_length > 0 else 0)
        throughputs.append(env.total_throughput)

    # Compute statistics
    results = {
        'num_episodes': num_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_success_rate': np.mean(success_rates),
        'std_success_rate': np.std(success_rates),
        'mean_collision_rate': np.mean(collision_rates),
        'std_collision_rate': np.std(collision_rates),
        'mean_throughput': np.mean(throughputs),
        'std_throughput': np.std(throughputs)
    }

    return results


def plot_evaluation_results(results, save_path=None):
    """
    Plot evaluation results.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Rewards
    axes[0, 0].bar(['Mean', 'Min', 'Max'],
                    [results['mean_reward'], results['min_reward'], results['max_reward']])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_ylabel('Reward')

    # Success and Collision Rates
    rates = [results['mean_success_rate'], results['mean_collision_rate']]
    stds = [results['std_success_rate'], results['std_collision_rate']]
    x = np.arange(len(['Success Rate', 'Collision Rate']))
    axes[0, 1].bar(x, rates, yerr=stds, capsize=5)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Success Rate', 'Collision Rate'])
    axes[0, 1].set_title('Performance Metrics')
    axes[0, 1].set_ylabel('Rate')

    # Throughput
    axes[1, 0].bar(['Mean Throughput'], [results['mean_throughput']], yerr=[results['std_throughput']], capsize=5)
    axes[1, 0].set_title('Throughput')
    axes[1, 0].set_ylabel('Total Throughput')

    # Summary stats
    axes[1, 1].axis('off')
    summary_text = f"""
    Evaluation Summary
    {'=' * 30}
    Episodes: {results['num_episodes']}
    Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}
    Success Rate: {results['mean_success_rate']:.2%}
    Collision Rate: {results['mean_collision_rate']:.2%}
    Mean Throughput: {results['mean_throughput']:.2f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                     verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO-LFM model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--save-plot', type=str, default=None, help='Path to save plot')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = load_config()

    # Initialize environment
    env = DynamicSpectrumAccessEnv(
        num_channels=config.env.num_channels,
        num_devices=config.env.num_devices,
        max_steps=config.env.max_steps
    )

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # Initialize agent
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = PPOLFMAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.model.hidden_dim,
        num_lfm_layers=config.model.num_lfm_layers,
        device=device
    )

    # Load model
    print(f"Loading model from {args.model}")
    agent.load(args.model)
    print("Model loaded successfully")

    # Evaluate
    print(f"\nEvaluating for {args.num_episodes} episodes...")
    results = evaluate_model(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
        render=args.render
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Episodes: {results['num_episodes']}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f}")
    print(f"Success Rate: {results['mean_success_rate']:.2%} ± {results['std_success_rate']:.2%}")
    print(f"Collision Rate: {results['mean_collision_rate']:.2%} ± {results['std_collision_rate']:.2%}")
    print(f"Mean Throughput: {results['mean_throughput']:.2f} ± {results['std_throughput']:.2f}")
    print("=" * 80)

    # Plot if requested
    if args.plot:
        plot_evaluation_results(results, save_path=args.save_plot)


if __name__ == '__main__':
    main()
