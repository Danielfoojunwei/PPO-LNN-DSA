"""
Metrics tracking and logging utilities.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List
import json


class MetricsTracker:
    """
    Track and compute training metrics.
    """
    def __init__(self, window_size=100):
        """
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.moving_averages = defaultdict(lambda: deque(maxlen=window_size))

    def add(self, metric_name: str, value: float):
        """
        Add a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.metrics[metric_name].append(value)
        self.moving_averages[metric_name].append(value)

    def get_mean(self, metric_name: str, window: bool = False) -> float:
        """
        Get mean of a metric.

        Args:
            metric_name: Name of the metric
            window: If True, return moving average

        Returns:
            Mean value
        """
        if window:
            values = list(self.moving_averages[metric_name])
        else:
            values = self.metrics[metric_name]

        return np.mean(values) if values else 0.0

    def get_std(self, metric_name: str, window: bool = False) -> float:
        """
        Get standard deviation of a metric.

        Args:
            metric_name: Name of the metric
            window: If True, use moving window

        Returns:
            Standard deviation
        """
        if window:
            values = list(self.moving_averages[metric_name])
        else:
            values = self.metrics[metric_name]

        return np.std(values) if values else 0.0

    def get_last(self, metric_name: str) -> float:
        """Get last value of a metric."""
        values = self.metrics[metric_name]
        return values[-1] if values else 0.0

    def get_summary(self, window: bool = True) -> Dict[str, float]:
        """
        Get summary of all metrics.

        Args:
            window: If True, use moving averages

        Returns:
            Dictionary of metric summaries
        """
        summary = {}
        for metric_name in self.metrics.keys():
            summary[f"{metric_name}_mean"] = self.get_mean(metric_name, window)
            summary[f"{metric_name}_std"] = self.get_std(metric_name, window)
            summary[f"{metric_name}_last"] = self.get_last(metric_name)
        return summary

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.moving_averages.clear()


class Logger:
    """
    Simple logger for training progress.
    """
    def __init__(self, log_file=None, console=True):
        """
        Args:
            log_file: Path to log file (optional)
            console: Whether to print to console
        """
        self.log_file = log_file
        self.console = console

    def log(self, message: str):
        """Log a message."""
        if self.console:
            print(message)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')

    def log_metrics(self, round_num: int, metrics: Dict[str, float]):
        """
        Log metrics for a training round.

        Args:
            round_num: Round number
            metrics: Dictionary of metrics
        """
        message = f"Round {round_num}:"
        for key, value in metrics.items():
            if isinstance(value, float):
                message += f" {key}={value:.4f}"
            else:
                message += f" {key}={value}"
        self.log(message)

    def log_dict(self, data: Dict, prefix: str = ""):
        """
        Log a dictionary.

        Args:
            data: Dictionary to log
            prefix: Prefix for log message
        """
        message = prefix
        for key, value in data.items():
            if isinstance(value, (int, float)):
                message += f" {key}={value:.4f}" if isinstance(value, float) else f" {key}={value}"
        self.log(message)


def save_metrics_to_json(metrics: Dict, path: str):
    """
    Save metrics to JSON file.

    Args:
        metrics: Metrics dictionary
        path: Path to save JSON
    """
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return obj

    converted_metrics = convert(metrics)

    with open(path, 'w') as f:
        json.dump(converted_metrics, f, indent=2)


def load_metrics_from_json(path: str) -> Dict:
    """
    Load metrics from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Metrics dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)
