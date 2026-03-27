"""Run truthful federated PPO benchmarks for dynamic spectrum access.

This script benchmarks three executable modes on the same empirical DSA task:
1. Centralized PPO training
2. Flat federated PPO with synchronous FedAvg
3. Hierarchical federated PPO with edge aggregation

It produces structured result artifacts that can be cited honestly in the
repository documentation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from empirical import (
    FederatedConfig,
    PPOAgent,
    PPOHyperParams,
    EmpiricalSpectrumEnv,
    evaluate_agent,
    make_empirical_suite,
    run_flat_federated,
    run_hierarchical_federated,
    summarize_federated_results,
    train_agent,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    subdirs = {
        "root": base_dir,
        "raw_training_logs": base_dir / "raw_training_logs",
        "episode_metrics": base_dir / "episode_metrics",
        "aggregate_tables": base_dir / "aggregate_tables",
        "config_snapshots": base_dir / "config_snapshots",
        "checkpoints": base_dir / "checkpoints",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def env_factory(config):
    def _factory():
        return EmpiricalSpectrumEnv(deepcopy(config))

    return _factory


def run_centralized(
    scenario_name: str,
    seed: int,
    model_name: str,
    device: str,
    total_timesteps: int,
    eval_every: int,
    eval_episodes: int,
    hyperparams: PPOHyperParams,
    result_dirs: Dict[str, Path],
) -> Dict[str, float]:
    suite = make_empirical_suite(seed=seed)
    config = deepcopy(suite[scenario_name])
    config.seed = seed
    set_global_seed(seed)

    env = EmpiricalSpectrumEnv(config)
    state, _ = env.reset(seed=seed)
    seq_len, obs_dim = state["obs"].shape
    action_dim = env.get_action_dim()

    agent = PPOAgent(
        model_name=model_name,
        obs_dim=obs_dim,
        seq_len=seq_len,
        action_dim=action_dim,
        device=device,
        hyperparams=hyperparams,
    )

    wall_start = time.perf_counter()
    histories = train_agent(
        agent=agent,
        env_factory=env_factory(config),
        total_timesteps=total_timesteps,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
    )
    train_seconds = time.perf_counter() - wall_start
    final_eval = evaluate_agent(agent, env_factory(config), num_episodes=eval_episodes)

    run_stub = f"{scenario_name}__centralized__seed{seed}"
    pd.DataFrame(histories["train_history"]).to_csv(result_dirs["raw_training_logs"] / f"{run_stub}_train.csv", index=False)
    pd.DataFrame(histories["eval_history"] or [{"step": float(agent.total_env_steps), **final_eval}]).to_csv(
        result_dirs["raw_training_logs"] / f"{run_stub}_eval.csv", index=False
    )
    pd.DataFrame(histories["update_history"]).to_csv(result_dirs["raw_training_logs"] / f"{run_stub}_update.csv", index=False)
    agent.save(str(result_dirs["checkpoints"] / f"{run_stub}.pt"))

    payload = {
        "scenario": scenario_name,
        "mode": "centralized",
        "seed": seed,
        "model_name": model_name,
        "parameter_count": agent.parameter_count,
        "train_steps": agent.total_env_steps,
        "communication_megabytes": 0.0,
        "wall_clock_training_time": float(train_seconds),
        "final_eval": final_eval,
    }
    with open(result_dirs["episode_metrics"] / f"{run_stub}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "scenario": scenario_name,
        "mode": "centralized",
        "seed": seed,
        "parameter_count": agent.parameter_count,
        "train_steps": agent.total_env_steps,
        "communication_megabytes": 0.0,
        "wall_clock_training_time": float(train_seconds),
    }
    for k, v in final_eval.items():
        summary[k] = float(v)
    return summary


def run_federated_mode(
    mode: str,
    scenario_name: str,
    seed: int,
    model_name: str,
    device: str,
    rounds: int,
    local_timesteps: int,
    eval_episodes: int,
    num_clients: int,
    num_edges: int,
    hyperparams: PPOHyperParams,
    result_dirs: Dict[str, Path],
) -> Dict[str, float]:
    suite = make_empirical_suite(seed=seed)
    config = deepcopy(suite[scenario_name])
    config.seed = seed
    set_global_seed(seed)

    fed_config = FederatedConfig(
        model_name=model_name,
        num_clients=num_clients,
        num_edges=num_edges,
        clients_per_round=num_clients,
        rounds=rounds,
        local_timesteps=local_timesteps,
        eval_episodes=eval_episodes,
        device=device,
        seed=seed,
    )

    wall_start = time.perf_counter()
    if mode == "flat_federated":
        result = run_flat_federated(base_config=config, fed_config=fed_config, hyperparams=hyperparams)
    elif mode == "hierarchical_federated":
        result = run_hierarchical_federated(base_config=config, fed_config=fed_config, hyperparams=hyperparams)
    else:
        raise ValueError(f"Unsupported federated mode: {mode}")
    train_seconds = time.perf_counter() - wall_start

    run_stub = f"{scenario_name}__{mode}__seed{seed}"
    pd.DataFrame(result["round_history"]).to_csv(result_dirs["raw_training_logs"] / f"{run_stub}_rounds.csv", index=False)
    with open(result_dirs["episode_metrics"] / f"{run_stub}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    summary = {
        "scenario": scenario_name,
        "mode": mode,
        "seed": seed,
        "parameter_count": float(result.get("parameter_count", 0.0)),
        "parameter_megabytes": float(result.get("parameter_bytes", 0.0) / (1024 ** 2)),
        "train_steps": float(rounds * local_timesteps),
        "communication_megabytes": float(result.get("total_communication_bytes", 0.0) / (1024 ** 2)),
        "wall_clock_training_time": float(train_seconds),
    }
    summary.update(summarize_federated_results(result))
    return summary


def aggregate(results_df: pd.DataFrame, result_dirs: Dict[str, Path]) -> None:
    rows: List[Dict[str, float]] = []
    metric_cols = [c for c in results_df.columns if c not in {"scenario", "mode", "seed"}]
    for (scenario, mode), frame in results_df.groupby(["scenario", "mode"], dropna=False):
        row = {"scenario": scenario, "mode": mode, "num_seeds": int(frame["seed"].nunique())}
        for col in metric_cols:
            row[f"mean_{col}"] = float(frame[col].mean())
            row[f"std_{col}"] = float(frame[col].std(ddof=0))
        rows.append(row)
    aggregate_df = pd.DataFrame(rows).sort_values(["scenario", "mode"]).reset_index(drop=True)
    aggregate_df.to_csv(result_dirs["aggregate_tables"] / "federated_aggregate_results.csv", index=False)
    results_df.to_csv(result_dirs["aggregate_tables"] / "federated_all_runs.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated empirical DSA benchmarks.")
    parser.add_argument("--results-dir", default="results/federated")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="ppo_ltc_lfm")
    parser.add_argument("--scenario", default="non_stationary")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 23, 41, 89])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--local-timesteps", type=int, default=1500)
    parser.add_argument("--centralized-timesteps", type=int, default=9000)
    parser.add_argument("--eval-every", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument("--num-edges", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = ensure_dirs(Path(args.results_dir))
    hyperparams = PPOHyperParams(
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )

    with open(result_dirs["config_snapshots"] / "federated_suite.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "scenario": args.scenario,
                "model": args.model,
                "seeds": args.seeds,
                "rounds": args.rounds,
                "local_timesteps": args.local_timesteps,
                "centralized_timesteps": args.centralized_timesteps,
                "eval_every": args.eval_every,
                "eval_episodes": args.eval_episodes,
                "num_clients": args.num_clients,
                "num_edges": args.num_edges,
                "hyperparams": hyperparams.to_dict(),
            },
            f,
            sort_keys=False,
        )

    summaries: List[Dict[str, float]] = []
    for seed in args.seeds:
        print(f"[run] scenario={args.scenario} mode=centralized seed={seed}", flush=True)
        summaries.append(
            run_centralized(
                scenario_name=args.scenario,
                seed=seed,
                model_name=args.model,
                device=args.device,
                total_timesteps=args.centralized_timesteps,
                eval_every=args.eval_every,
                eval_episodes=args.eval_episodes,
                hyperparams=hyperparams,
                result_dirs=result_dirs,
            )
        )
        print(f"[run] scenario={args.scenario} mode=flat_federated seed={seed}", flush=True)
        summaries.append(
            run_federated_mode(
                mode="flat_federated",
                scenario_name=args.scenario,
                seed=seed,
                model_name=args.model,
                device=args.device,
                rounds=args.rounds,
                local_timesteps=args.local_timesteps,
                eval_episodes=args.eval_episodes,
                num_clients=args.num_clients,
                num_edges=args.num_edges,
                hyperparams=hyperparams,
                result_dirs=result_dirs,
            )
        )
        print(f"[run] scenario={args.scenario} mode=hierarchical_federated seed={seed}", flush=True)
        summaries.append(
            run_federated_mode(
                mode="hierarchical_federated",
                scenario_name=args.scenario,
                seed=seed,
                model_name=args.model,
                device=args.device,
                rounds=args.rounds,
                local_timesteps=args.local_timesteps,
                eval_episodes=args.eval_episodes,
                num_clients=args.num_clients,
                num_edges=args.num_edges,
                hyperparams=hyperparams,
                result_dirs=result_dirs,
            )
        )
        pd.DataFrame(summaries).to_csv(result_dirs["aggregate_tables"] / "federated_all_runs_live.csv", index=False)

    results_df = pd.DataFrame(summaries).sort_values(["scenario", "mode", "seed"]).reset_index(drop=True)
    aggregate(results_df, result_dirs)
    with open(result_dirs["episode_metrics"] / "federated_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": args.scenario,
                "model": args.model,
                "seeds": args.seeds,
                "modes": ["centralized", "flat_federated", "hierarchical_federated"],
            },
            f,
            indent=2,
        )
    print("Federated empirical benchmark suite completed.")


if __name__ == "__main__":
    main()
