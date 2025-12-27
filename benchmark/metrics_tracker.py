"""
Comprehensive metrics tracking for benchmarking PPO-LSTM vs PPO-LFM.
"""

import numpy as np
import time
from collections import defaultdict
import json
import os


class BenchmarkMetrics:
    """
    Tracks detailed metrics for comparing PPO-LSTM and PPO-LFM.
    Includes metrics from the research paper.
    """
    def __init__(self, model_name):
        """
        Args:
            model_name: Name of the model (e.g., 'PPO-LSTM' or 'PPO-LFM')
        """
        self.model_name = model_name
        self.reset()

    def reset(self):
        """Reset all metrics."""
        # Training metrics (per round)
        self.round_rewards = []
        self.round_success_rates = []
        self.round_collision_rates = []
        self.round_throughputs = []
        self.round_times = []

        # Loss metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []

        # Evaluation metrics (periodic)
        self.eval_rewards = []
        self.eval_success_rates = []
        self.eval_collision_rates = []
        self.eval_spectrum_efficiency = []

        # Communication metrics (for federated learning)
        self.communication_rounds = []
        self.model_sizes = []

        # Computational metrics
        self.forward_pass_times = []
        self.update_times = []
        self.memory_usage = []

        # Convergence metrics
        self.convergence_round = None
        self.convergence_threshold = 0.8  # 80% success rate

        # Start time
        self.start_time = time.time()

    def add_training_round(self, round_num, metrics):
        """
        Add metrics from a training round.

        Args:
            round_num: Round number
            metrics: Dictionary with training metrics
        """
        self.round_rewards.append(metrics.get('mean_reward', 0))
        self.round_success_rates.append(metrics.get('mean_success_rate', 0))
        self.round_collision_rates.append(metrics.get('mean_collision_rate', 0))
        self.round_throughputs.append(metrics.get('mean_throughput', 0))

        if 'policy_loss' in metrics:
            self.policy_losses.append(metrics['policy_loss'])
        if 'value_loss' in metrics:
            self.value_losses.append(metrics['value_loss'])
        if 'entropy' in metrics:
            self.entropies.append(metrics['entropy'])

        # Check convergence
        if (self.convergence_round is None and
            metrics.get('mean_success_rate', 0) >= self.convergence_threshold):
            self.convergence_round = round_num

    def add_evaluation(self, eval_metrics):
        """Add evaluation metrics."""
        self.eval_rewards.append(eval_metrics.get('mean_reward', 0))
        self.eval_success_rates.append(eval_metrics.get('mean_success_rate', 0))
        self.eval_collision_rates.append(eval_metrics.get('mean_collision_rate', 0))
        self.eval_spectrum_efficiency.append(eval_metrics.get('spectrum_efficiency', 0))

    def add_timing(self, forward_time=None, update_time=None):
        """Add timing metrics."""
        if forward_time is not None:
            self.forward_pass_times.append(forward_time)
        if update_time is not None:
            self.update_times.append(update_time)

    def add_communication_round(self, round_num, model_size):
        """Add communication metrics."""
        self.communication_rounds.append(round_num)
        self.model_sizes.append(model_size)

    def get_summary(self):
        """
        Get summary statistics.

        Returns:
            Dictionary with summary metrics
        """
        total_time = time.time() - self.start_time

        summary = {
            'model_name': self.model_name,

            # Performance metrics
            'final_reward': self.round_rewards[-1] if self.round_rewards else 0,
            'max_reward': max(self.round_rewards) if self.round_rewards else 0,
            'mean_reward': np.mean(self.round_rewards) if self.round_rewards else 0,
            'final_success_rate': self.round_success_rates[-1] if self.round_success_rates else 0,
            'mean_success_rate': np.mean(self.round_success_rates) if self.round_success_rates else 0,
            'final_collision_rate': self.round_collision_rates[-1] if self.round_collision_rates else 0,
            'mean_collision_rate': np.mean(self.round_collision_rates) if self.round_collision_rates else 0,
            'final_throughput': self.round_throughputs[-1] if self.round_throughputs else 0,
            'mean_throughput': np.mean(self.round_throughputs) if self.round_throughputs else 0,

            # Convergence metrics
            'convergence_round': self.convergence_round or len(self.round_rewards),
            'converged': self.convergence_round is not None,

            # Evaluation metrics (if available)
            'eval_mean_reward': np.mean(self.eval_rewards) if self.eval_rewards else 0,
            'eval_mean_success_rate': np.mean(self.eval_success_rates) if self.eval_success_rates else 0,
            'eval_mean_collision_rate': np.mean(self.eval_collision_rates) if self.eval_collision_rates else 0,
            'eval_mean_spectrum_efficiency': np.mean(self.eval_spectrum_efficiency) if self.eval_spectrum_efficiency else 0,

            # Training efficiency
            'total_training_time': total_time,
            'time_per_round': total_time / len(self.round_rewards) if self.round_rewards else 0,
            'mean_forward_time': np.mean(self.forward_pass_times) if self.forward_pass_times else 0,
            'mean_update_time': np.mean(self.update_times) if self.update_times else 0,

            # Communication efficiency
            'total_rounds': len(self.round_rewards),
            'mean_model_size': np.mean(self.model_sizes) if self.model_sizes else 0,

            # Learning stability
            'reward_std': np.std(self.round_rewards) if self.round_rewards else 0,
            'success_rate_std': np.std(self.round_success_rates) if self.round_success_rates else 0,
        }

        return summary

    def save_to_file(self, filepath):
        """Save detailed metrics to JSON file."""
        data = {
            'model_name': self.model_name,
            'summary': self.get_summary(),
            'detailed': {
                'round_rewards': self.round_rewards,
                'round_success_rates': self.round_success_rates,
                'round_collision_rates': self.round_collision_rates,
                'round_throughputs': self.round_throughputs,
                'policy_losses': self.policy_losses,
                'value_losses': self.value_losses,
                'entropies': self.entropies,
                'eval_rewards': self.eval_rewards,
                'eval_success_rates': self.eval_success_rates,
                'eval_collision_rates': self.eval_collision_rates,
                'forward_pass_times': self.forward_pass_times,
                'update_times': self.update_times,
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_from_file(filepath):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        metrics = BenchmarkMetrics(data['model_name'])
        metrics.round_rewards = data['detailed']['round_rewards']
        metrics.round_success_rates = data['detailed']['round_success_rates']
        metrics.round_collision_rates = data['detailed']['round_collision_rates']
        metrics.round_throughputs = data['detailed']['round_throughputs']
        metrics.policy_losses = data['detailed']['policy_losses']
        metrics.value_losses = data['detailed']['value_losses']
        metrics.entropies = data['detailed']['entropies']
        metrics.eval_rewards = data['detailed']['eval_rewards']
        metrics.eval_success_rates = data['detailed']['eval_success_rates']
        metrics.eval_collision_rates = data['detailed']['eval_collision_rates']

        return metrics


class ComparisonMetrics:
    """
    Compare metrics between PPO-LSTM and PPO-LFM.
    """
    def __init__(self, lstm_metrics, lfm_metrics):
        """
        Args:
            lstm_metrics: BenchmarkMetrics for PPO-LSTM
            lfm_metrics: BenchmarkMetrics for PPO-LFM
        """
        self.lstm = lstm_metrics
        self.lfm = lfm_metrics

    def compute_improvements(self):
        """
        Compute improvement percentages of LFM over LSTM.

        Returns:
            Dictionary with improvement metrics
        """
        lstm_summary = self.lstm.get_summary()
        lfm_summary = self.lfm.get_summary()

        def pct_improvement(lfm_val, lstm_val, lower_is_better=False):
            """Compute percentage improvement."""
            if lstm_val == 0:
                return 0
            improvement = ((lfm_val - lstm_val) / lstm_val) * 100
            return -improvement if lower_is_better else improvement

        improvements = {
            'reward_improvement': pct_improvement(
                lfm_summary['final_reward'],
                lstm_summary['final_reward']
            ),
            'success_rate_improvement': pct_improvement(
                lfm_summary['final_success_rate'],
                lstm_summary['final_success_rate']
            ),
            'collision_rate_improvement': pct_improvement(
                lfm_summary['final_collision_rate'],
                lstm_summary['final_collision_rate'],
                lower_is_better=True
            ),
            'throughput_improvement': pct_improvement(
                lfm_summary['final_throughput'],
                lstm_summary['final_throughput']
            ),
            'convergence_speedup': pct_improvement(
                lstm_summary['convergence_round'],
                lfm_summary['convergence_round'],
                lower_is_better=True
            ),
            'forward_time_improvement': pct_improvement(
                lstm_summary['mean_forward_time'],
                lfm_summary['mean_forward_time'],
                lower_is_better=True
            ),
            'training_time_improvement': pct_improvement(
                lstm_summary['total_training_time'],
                lfm_summary['total_training_time'],
                lower_is_better=True
            ),
        }

        return improvements

    def generate_comparison_table(self):
        """
        Generate a comparison table.

        Returns:
            String with formatted table
        """
        lstm_summary = self.lstm.get_summary()
        lfm_summary = self.lfm.get_summary()
        improvements = self.compute_improvements()

        table = "=" * 100 + "\n"
        table += "PPO-LSTM vs PPO-LFM BENCHMARK COMPARISON\n"
        table += "=" * 100 + "\n\n"

        table += f"{'Metric':<40} {'PPO-LSTM':<20} {'PPO-LFM':<20} {'Improvement':<15}\n"
        table += "-" * 100 + "\n"

        metrics = [
            ('Final Reward', 'final_reward', 'reward_improvement', '.2f'),
            ('Success Rate', 'final_success_rate', 'success_rate_improvement', '.2%'),
            ('Collision Rate', 'final_collision_rate', 'collision_rate_improvement', '.2%'),
            ('Throughput', 'final_throughput', 'throughput_improvement', '.2f'),
            ('Convergence Round', 'convergence_round', 'convergence_speedup', 'd'),
            ('Forward Pass Time (ms)', 'mean_forward_time', 'forward_time_improvement', '.3f'),
            ('Total Training Time (s)', 'total_training_time', 'training_time_improvement', '.1f'),
        ]

        for name, key, imp_key, fmt in metrics:
            lstm_val = lstm_summary[key]
            lfm_val = lfm_summary[key]
            improvement = improvements[imp_key]

            if 'time' in key.lower() and key != 'convergence_round':
                # Convert times to ms
                lstm_val *= 1000
                lfm_val *= 1000
                if 'total' in key.lower():
                    lstm_val /= 1000  # back to seconds
                    lfm_val /= 1000

            table += f"{name:<40} {lstm_val:<20{fmt}} {lfm_val:<20{fmt}} {improvement:>+.1f}%\n"

        table += "=" * 100 + "\n"

        return table

    def save_comparison(self, filepath):
        """Save comparison to file."""
        comparison = {
            'lstm_summary': self.lstm.get_summary(),
            'lfm_summary': self.lfm.get_summary(),
            'improvements': self.compute_improvements(),
            'table': self.generate_comparison_table()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)

        # Also save human-readable table
        table_path = filepath.replace('.json', '_table.txt')
        with open(table_path, 'w') as f:
            f.write(self.generate_comparison_table())
