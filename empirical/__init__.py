"""Empirical benchmark package for dynamic spectrum access experiments."""

from empirical.envs import EmpiricalDSAConfig, EmpiricalSpectrumEnv, GreedyOccupancyHeuristic, RandomHeuristic, make_empirical_suite
from empirical.models import MODEL_REGISTRY, build_model, count_parameters
from empirical.ppo import PPOAgent, PPOHyperParams, evaluate_agent, train_agent
from empirical.federated import FederatedConfig, run_flat_federated, run_hierarchical_federated, summarize_federated_results

__all__ = [
    "EmpiricalDSAConfig",
    "EmpiricalSpectrumEnv",
    "GreedyOccupancyHeuristic",
    "RandomHeuristic",
    "make_empirical_suite",
    "MODEL_REGISTRY",
    "build_model",
    "count_parameters",
    "PPOAgent",
    "PPOHyperParams",
    "evaluate_agent",
    "train_agent",
    "FederatedConfig",
    "run_flat_federated",
    "run_hierarchical_federated",
    "summarize_federated_results",
]
