"""
Visualization script for benchmark results.

Creates comprehensive plots comparing PPO-LSTM and PPO-LFM.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from benchmark.metrics_tracker import BenchmarkMetrics
import json


def plot_training_curves(lstm_metrics, lfm_metrics, save_path=None):
    """
    Plot training curves comparing PPO-LSTM and PPO-LFM.

    Args:
        lstm_metrics: BenchmarkMetrics for PPO-LSTM
        lfm_metrics: BenchmarkMetrics for PPO-LFM
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PPO-LSTM vs PPO-LFM Training Comparison', fontsize=16, fontweight='bold')

    # Define colors
    lstm_color = '#e74c3c'  # Red
    lfm_color = '#3498db'   # Blue

    # 1. Rewards
    ax = axes[0, 0]
    ax.plot(lstm_metrics.round_rewards, label='PPO-LSTM', color=lstm_color, linewidth=2)
    ax.plot(lfm_metrics.round_rewards, label='PPO-LFM', color=lfm_color, linewidth=2)
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Episode Reward over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Success Rate
    ax = axes[0, 1]
    ax.plot(lstm_metrics.round_success_rates, label='PPO-LSTM', color=lstm_color, linewidth=2)
    ax.plot(lfm_metrics.round_success_rates, label='PPO-LFM', color=lfm_color, linewidth=2)
    ax.axhline(y=0.8, color='green', linestyle='--', label='Target (80%)', alpha=0.5)
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Collision Rate
    ax = axes[0, 2]
    ax.plot(lstm_metrics.round_collision_rates, label='PPO-LSTM', color=lstm_color, linewidth=2)
    ax.plot(lfm_metrics.round_collision_rates, label='PPO-LFM', color=lfm_color, linewidth=2)
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Collision Rate')
    ax.set_title('Collision Rate over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Throughput
    ax = axes[1, 0]
    ax.plot(lstm_metrics.round_throughputs, label='PPO-LSTM', color=lstm_color, linewidth=2)
    ax.plot(lfm_metrics.round_throughputs, label='PPO-LFM', color=lfm_color, linewidth=2)
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Throughput')
    ax.set_title('Throughput over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Policy Loss
    ax = axes[1, 1]
    if lstm_metrics.policy_losses and lfm_metrics.policy_losses:
        ax.plot(lstm_metrics.policy_losses, label='PPO-LSTM', color=lstm_color, linewidth=2, alpha=0.7)
        ax.plot(lfm_metrics.policy_losses, label='PPO-LFM', color=lfm_color, linewidth=2, alpha=0.7)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Policy Loss')

    # 6. Value Loss
    ax = axes[1, 2]
    if lstm_metrics.value_losses and lfm_metrics.value_losses:
        ax.plot(lstm_metrics.value_losses, label='PPO-LSTM', color=lstm_color, linewidth=2, alpha=0.7)
        ax.plot(lfm_metrics.value_losses, label='PPO-LFM', color=lfm_color, linewidth=2, alpha=0.7)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Value Loss')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()


def plot_performance_comparison(lstm_metrics, lfm_metrics, save_path=None):
    """
    Create bar chart comparing final performance metrics.

    Args:
        lstm_metrics: BenchmarkMetrics for PPO-LSTM
        lfm_metrics: BenchmarkMetrics for PPO-LFM
        save_path: Path to save the plot
    """
    lstm_summary = lstm_metrics.get_summary()
    lfm_summary = lfm_metrics.get_summary()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPO-LSTM vs PPO-LFM Performance Comparison', fontsize=16, fontweight='bold')

    lstm_color = '#e74c3c'
    lfm_color = '#3498db'

    # 1. Success vs Collision Rate
    ax = axes[0, 0]
    metrics = ['Success Rate', 'Collision Rate']
    lstm_vals = [lstm_summary['final_success_rate'], lstm_summary['final_collision_rate']]
    lfm_vals = [lfm_summary['final_success_rate'], lfm_summary['final_collision_rate']]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, lstm_vals, width, label='PPO-LSTM', color=lstm_color)
    ax.bar(x + width/2, lfm_vals, width, label='PPO-LFM', color=lfm_color)

    ax.set_ylabel('Rate')
    ax.set_title('Success vs Collision Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(lstm_vals):
        ax.text(i - width/2, v + 0.01, f'{v:.2%}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(lfm_vals):
        ax.text(i + width/2, v + 0.01, f'{v:.2%}', ha='center', va='bottom', fontsize=9)

    # 2. Reward and Throughput
    ax = axes[0, 1]
    metrics = ['Final Reward', 'Throughput']
    lstm_vals = [lstm_summary['final_reward'], lstm_summary['final_throughput'] / 100]
    lfm_vals = [lfm_summary['final_reward'], lfm_summary['final_throughput'] / 100]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, lstm_vals, width, label='PPO-LSTM', color=lstm_color)
    ax.bar(x + width/2, lfm_vals, width, label='PPO-LFM', color=lfm_color)

    ax.set_ylabel('Value')
    ax.set_title('Reward and Throughput (normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Convergence Speed
    ax = axes[1, 0]
    metrics = ['Convergence Round']
    lstm_vals = [lstm_summary['convergence_round']]
    lfm_vals = [lfm_summary['convergence_round']]

    x = np.arange(len(metrics))
    bars1 = ax.bar(x - width/2, lstm_vals, width, label='PPO-LSTM', color=lstm_color)
    bars2 = ax.bar(x + width/2, lfm_vals, width, label='PPO-LFM', color=lfm_color)

    ax.set_ylabel('Rounds to Convergence')
    ax.set_title('Convergence Speed (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

    # Add improvement percentage
    improvement = ((lstm_vals[0] - lfm_vals[0]) / lstm_vals[0]) * 100
    ax.text(0, max(lstm_vals[0], lfm_vals[0]) * 1.1,
            f'{improvement:.1f}% faster', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Computational Efficiency
    ax = axes[1, 1]
    metrics = ['Forward Pass\n(ms)', 'Training Time\n(s)']
    lstm_vals = [lstm_summary['mean_forward_time'] * 1000,
                 lstm_summary['total_training_time'] / 10]  # Normalized
    lfm_vals = [lfm_summary['mean_forward_time'] * 1000,
                lfm_summary['total_training_time'] / 10]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, lstm_vals, width, label='PPO-LSTM', color=lstm_color)
    ax.bar(x + width/2, lfm_vals, width, label='PPO-LFM', color=lfm_color)

    ax.set_ylabel('Time')
    ax.set_title('Computational Efficiency (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison to {save_path}")

    plt.show()


def plot_improvement_summary(lstm_metrics, lfm_metrics, save_path=None):
    """
    Create a summary plot showing percentage improvements.

    Args:
        lstm_metrics: BenchmarkMetrics for PPO-LSTM
        lfm_metrics: BenchmarkMetrics for PPO-LFM
        save_path: Path to save the plot
    """
    from benchmark.metrics_tracker import ComparisonMetrics

    comparison = ComparisonMetrics(lstm_metrics, lfm_metrics)
    improvements = comparison.compute_improvements()

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('PPO-LFM Improvements over PPO-LSTM', fontsize=16, fontweight='bold')

    metrics = [
        ('Success Rate', improvements['success_rate_improvement']),
        ('Throughput', improvements['throughput_improvement']),
        ('Reward', improvements['reward_improvement']),
        ('Convergence Speed', improvements['convergence_speedup']),
        ('Forward Pass Time', improvements['forward_time_improvement']),
        ('Collision Rate', improvements['collision_rate_improvement']),
    ]

    names = [m[0] for m in metrics]
    values = [m[1] for m in metrics]

    # Color bars based on positive/negative
    colors = ['green' if v > 0 else 'red' for v in values]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Improvement (%)')
    ax.set_title('Percentage Improvements (positive = better)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        x_pos = value + (2 if value > 0 else -2)
        ax.text(x_pos, i, f'{value:+.1f}%',
                ha='left' if value > 0 else 'right',
                va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved improvement summary to {save_path}")

    plt.show()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing results')
    parser.add_argument('--trial', type=int, default=0,
                        help='Trial number to visualize')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save plots')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics
    lstm_path = os.path.join(args.results_dir, f'lstm_metrics_trial_{args.trial}.json')
    lfm_path = os.path.join(args.results_dir, f'lfm_metrics_trial_{args.trial}.json')

    if not os.path.exists(lstm_path) or not os.path.exists(lfm_path):
        print(f"Error: Metrics files not found in {args.results_dir}")
        print(f"Looking for:")
        print(f"  - {lstm_path}")
        print(f"  - {lfm_path}")
        return

    print(f"Loading metrics from trial {args.trial}...")
    lstm_metrics = BenchmarkMetrics.load_from_file(lstm_path)
    lfm_metrics = BenchmarkMetrics.load_from_file(lfm_path)

    print("Creating visualizations...")

    # Generate plots
    plot_training_curves(
        lstm_metrics, lfm_metrics,
        save_path=os.path.join(args.output_dir, f'training_curves_trial_{args.trial}.png')
    )

    plot_performance_comparison(
        lstm_metrics, lfm_metrics,
        save_path=os.path.join(args.output_dir, f'performance_comparison_trial_{args.trial}.png')
    )

    plot_improvement_summary(
        lstm_metrics, lfm_metrics,
        save_path=os.path.join(args.output_dir, f'improvement_summary_trial_{args.trial}.png')
    )

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()
