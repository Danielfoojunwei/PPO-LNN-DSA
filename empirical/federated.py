"""Federated training utilities for the empirical DSA benchmark.

This module implements executable centralized-adjacent, flat federated, and
hierarchical federated training over the same reinforcement-learning benchmark
used by the single-agent suite. The goal is not to overclaim real-world
deployment fidelity, but to provide a truthful communication-performance
benchmark for the repository's federated title and README claims.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from empirical.envs import EmpiricalDSAConfig, EmpiricalSpectrumEnv
from empirical.ppo import PPOAgent, PPOHyperParams, evaluate_agent


@dataclass
class FederatedConfig:
    """Configuration for flat and hierarchical federated PPO experiments."""

    model_name: str = "ppo_ltc_lfm"
    num_clients: int = 6
    num_edges: int = 2
    clients_per_round: int = 6
    rounds: int = 6
    local_timesteps: int = 1500
    eval_episodes: int = 8
    device: str = "cpu"
    seed: int = 42

    def to_dict(self) -> Dict[str, int | str]:
        return asdict(self)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_client_config(base_config: EmpiricalDSAConfig, client_id: int, seed: int) -> EmpiricalDSAConfig:
    cfg = deepcopy(base_config)
    cfg.seed = seed
    cfg.num_background_users = max(1, base_config.num_background_users + ((client_id % 3) - 1))
    cfg.non_stationarity_strength = min(0.6, base_config.non_stationarity_strength + 0.05 * (client_id % 2))
    cfg.interference_scale = min(2.0, base_config.interference_scale + 0.05 * ((client_id + 1) % 3))
    return cfg


def _env_factory(config: EmpiricalDSAConfig):
    def _factory() -> EmpiricalSpectrumEnv:
        return EmpiricalSpectrumEnv(deepcopy(config))

    return _factory


def _parameter_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    total = 0
    for tensor in state_dict.values():
        total += tensor.numel() * tensor.element_size()
    return int(total)


def _average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("Cannot average an empty state_dict list")
    avg = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].detach().float().cpu() for sd in state_dicts], dim=0)
        avg[key] = stacked.mean(dim=0)
    return avg


def _client_agent_from_global(
    model_name: str,
    global_agent: PPOAgent,
    obs_dim: int,
    seq_len: int,
    action_dim: int,
    device: str,
    hyperparams: PPOHyperParams,
) -> PPOAgent:
    client_agent = PPOAgent(
        model_name=model_name,
        obs_dim=obs_dim,
        seq_len=seq_len,
        action_dim=action_dim,
        device=device,
        hyperparams=hyperparams,
    )
    client_agent.set_state_dict(global_agent.get_state_dict())
    return client_agent


def _train_local_agent(agent: PPOAgent, env_factory, total_timesteps: int) -> Dict[str, float]:
    metrics: List[Dict[str, float]] = []
    while agent.total_env_steps < total_timesteps:
        env = env_factory()
        state, _ = env.reset()
        done = False
        while not done and agent.total_env_steps < total_timesteps:
            action, log_prob, value, _ = agent.act(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, log_prob, value, reward, done, info)
            state = next_state
            if len(agent.buffer) >= agent.hparams.rollout_steps:
                agent.update(next_state)
        if len(agent.buffer) > 0:
            agent.update(state)
        metrics.append(env.get_metrics())
    summary = {}
    if metrics:
        for key in metrics[0].keys():
            values = np.array([m[key] for m in metrics], dtype=np.float64)
            summary[f"local_mean_{key}"] = float(values.mean())
    return summary


def build_global_agent(model_name: str, base_config: EmpiricalDSAConfig, device: str, hyperparams: PPOHyperParams) -> PPOAgent:
    env = EmpiricalSpectrumEnv(deepcopy(base_config))
    state, _ = env.reset(seed=base_config.seed)
    seq_len, obs_dim = state["obs"].shape
    action_dim = env.get_action_dim()
    return PPOAgent(
        model_name=model_name,
        obs_dim=obs_dim,
        seq_len=seq_len,
        action_dim=action_dim,
        device=device,
        hyperparams=hyperparams,
    )


def run_flat_federated(
    base_config: EmpiricalDSAConfig,
    fed_config: FederatedConfig,
    hyperparams: PPOHyperParams | None = None,
) -> Dict[str, object]:
    """Run a flat federated PPO experiment with synchronous FedAvg."""

    hyperparams = hyperparams or PPOHyperParams()
    _set_seed(fed_config.seed)
    global_agent = build_global_agent(fed_config.model_name, base_config, fed_config.device, hyperparams)

    env = EmpiricalSpectrumEnv(deepcopy(base_config))
    state, _ = env.reset(seed=base_config.seed)
    seq_len, obs_dim = state["obs"].shape
    action_dim = env.get_action_dim()

    global_state = global_agent.get_state_dict()
    param_bytes = _parameter_bytes(global_state)
    round_history: List[Dict[str, float]] = []
    communication_bytes = 0

    for round_idx in range(fed_config.rounds):
        client_states = []
        participating_clients = list(range(min(fed_config.clients_per_round, fed_config.num_clients)))
        local_summaries = []
        for client_id in participating_clients:
            client_seed = fed_config.seed + 1000 * round_idx + client_id
            client_cfg = _build_client_config(base_config, client_id, client_seed)
            client_agent = _client_agent_from_global(
                model_name=fed_config.model_name,
                global_agent=global_agent,
                obs_dim=obs_dim,
                seq_len=seq_len,
                action_dim=action_dim,
                device=fed_config.device,
                hyperparams=hyperparams,
            )
            local_summary = _train_local_agent(
                client_agent,
                _env_factory(client_cfg),
                total_timesteps=fed_config.local_timesteps,
            )
            client_states.append(client_agent.get_state_dict())
            local_summaries.append(local_summary)
            communication_bytes += 2 * param_bytes

        global_state = _average_state_dicts(client_states)
        global_agent.set_state_dict(global_state)

        eval_summary = evaluate_agent(global_agent, _env_factory(deepcopy(base_config)), num_episodes=fed_config.eval_episodes)
        round_record = {
            "round": float(round_idx + 1),
            "communication_bytes": float(communication_bytes),
            "communication_megabytes": float(communication_bytes / (1024 ** 2)),
        }
        if local_summaries:
            numeric_keys = sorted(local_summaries[0].keys())
            for key in numeric_keys:
                round_record[key] = float(np.mean([x[key] for x in local_summaries]))
        round_record.update(eval_summary)
        round_history.append(round_record)

    final_eval = evaluate_agent(global_agent, _env_factory(deepcopy(base_config)), num_episodes=fed_config.eval_episodes)
    return {
        "mode": "flat_federated",
        "federated_config": fed_config.to_dict(),
        "parameter_bytes": param_bytes,
        "parameter_count": global_agent.parameter_count,
        "total_communication_bytes": communication_bytes,
        "round_history": round_history,
        "final_eval": final_eval,
    }


def run_hierarchical_federated(
    base_config: EmpiricalDSAConfig,
    fed_config: FederatedConfig,
    hyperparams: PPOHyperParams | None = None,
) -> Dict[str, object]:
    """Run a two-level hierarchical federated PPO experiment."""

    hyperparams = hyperparams or PPOHyperParams()
    _set_seed(fed_config.seed)
    global_agent = build_global_agent(fed_config.model_name, base_config, fed_config.device, hyperparams)

    env = EmpiricalSpectrumEnv(deepcopy(base_config))
    state, _ = env.reset(seed=base_config.seed)
    seq_len, obs_dim = state["obs"].shape
    action_dim = env.get_action_dim()

    param_bytes = _parameter_bytes(global_agent.get_state_dict())
    communication_bytes = 0
    round_history: List[Dict[str, float]] = []
    clients = list(range(fed_config.num_clients))
    edges = {edge_id: [] for edge_id in range(fed_config.num_edges)}
    for idx, client_id in enumerate(clients):
        edges[idx % fed_config.num_edges].append(client_id)

    for round_idx in range(fed_config.rounds):
        edge_states = []
        edge_local_stats = []
        for edge_id, edge_clients in edges.items():
            client_states = []
            local_summaries = []
            for client_id in edge_clients:
                client_seed = fed_config.seed + 10000 * round_idx + 100 * edge_id + client_id
                client_cfg = _build_client_config(base_config, client_id, client_seed)
                client_agent = _client_agent_from_global(
                    model_name=fed_config.model_name,
                    global_agent=global_agent,
                    obs_dim=obs_dim,
                    seq_len=seq_len,
                    action_dim=action_dim,
                    device=fed_config.device,
                    hyperparams=hyperparams,
                )
                local_summary = _train_local_agent(
                    client_agent,
                    _env_factory(client_cfg),
                    total_timesteps=fed_config.local_timesteps,
                )
                client_states.append(client_agent.get_state_dict())
                local_summaries.append(local_summary)
                communication_bytes += param_bytes  # client -> edge

            edge_state = _average_state_dicts(client_states)
            edge_states.append(edge_state)
            communication_bytes += param_bytes  # edge -> global
            edge_local_stats.extend(local_summaries)

        global_state = _average_state_dicts(edge_states)
        global_agent.set_state_dict(global_state)
        communication_bytes += fed_config.num_edges * param_bytes  # global -> edges
        communication_bytes += fed_config.num_clients * param_bytes  # edges -> clients sync proxy

        eval_summary = evaluate_agent(global_agent, _env_factory(deepcopy(base_config)), num_episodes=fed_config.eval_episodes)
        round_record = {
            "round": float(round_idx + 1),
            "communication_bytes": float(communication_bytes),
            "communication_megabytes": float(communication_bytes / (1024 ** 2)),
        }
        if edge_local_stats:
            numeric_keys = sorted(edge_local_stats[0].keys())
            for key in numeric_keys:
                round_record[key] = float(np.mean([x[key] for x in edge_local_stats]))
        round_record.update(eval_summary)
        round_history.append(round_record)

    final_eval = evaluate_agent(global_agent, _env_factory(deepcopy(base_config)), num_episodes=fed_config.eval_episodes)
    return {
        "mode": "hierarchical_federated",
        "federated_config": fed_config.to_dict(),
        "parameter_bytes": param_bytes,
        "parameter_count": global_agent.parameter_count,
        "total_communication_bytes": communication_bytes,
        "round_history": round_history,
        "final_eval": final_eval,
    }


def summarize_federated_results(result: Dict[str, object]) -> Dict[str, float]:
    final_eval = result["final_eval"]
    return {
        "mean_episode_reward": float(final_eval.get("mean_episode_reward", 0.0)),
        "mean_spectrum_utilization": float(final_eval.get("mean_spectrum_utilization", 0.0)),
        "mean_collision_rate": float(final_eval.get("mean_collision_rate", 0.0)),
        "mean_wall_clock_episode_seconds": float(final_eval.get("mean_wall_clock_episode_seconds", 0.0)),
        "communication_megabytes": float(result.get("total_communication_bytes", 0.0) / (1024 ** 2)),
    }
