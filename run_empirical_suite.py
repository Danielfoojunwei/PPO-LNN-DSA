"""Run real empirical benchmarks for dynamic spectrum access.

This script executes reproducible training and evaluation for heuristic and PPO
baselines across multiple dynamic spectrum access scenarios. It generates raw
training logs, per-run metrics, aggregate tables, plots, and configuration
snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import yaml

from empirical import (
    EmpiricalDSAConfig,
    EmpiricalSpectrumEnv,
    GreedyOccupancyHeuristic,
    PPOAgent,
    PPOHyperParams,
    RandomHeuristic,
    evaluate_agent,
    make_empirical_suite,
    train_agent,
)


DEFAULT_TRAINING_STEPS = 6000
DEFAULT_EVAL_EPISODES = 12
DEFAULT_EVAL_EVERY = 1500
DEFAULT_SEEDS = [7, 19, 23, 41, 89]


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


def ensure_result_dirs(base_dir: Path) -> Dict[str, Path]:
    subdirs = {
        "root": base_dir,
        "raw_training_logs": base_dir / "raw_training_logs",
        "episode_metrics": base_dir / "episode_metrics",
        "aggregate_tables": base_dir / "aggregate_tables",
        "plots": base_dir / "plots",
        "config_snapshots": base_dir / "config_snapshots",
        "checkpoints": base_dir / "checkpoints",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def env_factory_from_config(config: EmpiricalDSAConfig) -> Callable[[], EmpiricalSpectrumEnv]:
    def _factory() -> EmpiricalSpectrumEnv:
        return EmpiricalSpectrumEnv(deepcopy(config))

    return _factory


def evaluate_heuristic(agent_name: str, heuristic, env_factory, num_episodes: int) -> Dict[str, float]:
    metrics: List[Dict[str, float]] = []
    for _ in range(num_episodes):
        env = env_factory()
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        step_forward_times: List[float] = []
        while not done:
            t0 = time.perf_counter()
            action = heuristic.act(state)
            step_forward_times.append(time.perf_counter() - t0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        episode_metrics = env.get_metrics()
        episode_metrics.update(
            {
                "episode_reward": float(total_reward),
                "episode_length": float(env.step_count),
                "mean_forward_time": float(np.mean(step_forward_times) if step_forward_times else 0.0),
            }
        )
        metrics.append(episode_metrics)

    summary: Dict[str, float] = {"model": agent_name}
    for key in sorted(metrics[0].keys()):
        values = np.array([m[key] for m in metrics], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std(ddof=0))
    return summary


def run_single_experiment(
    model_name: str,
    scenario_name: str,
    seed: int,
    total_steps: int,
    eval_every: int,
    eval_episodes: int,
    device: str,
    result_dirs: Dict[str, Path],
    hyperparams: PPOHyperParams,
) -> Dict[str, float]:
    suite = make_empirical_suite(seed=seed)
    config = deepcopy(suite[scenario_name])
    config.seed = seed
    set_global_seed(seed)

    env = EmpiricalSpectrumEnv(config)
    state, _ = env.reset(seed=seed)
    seq_len, obs_dim = state["obs"].shape
    action_dim = env.get_action_dim()
    env_factory = env_factory_from_config(config)

    if model_name == "random_policy":
        summary = evaluate_heuristic(model_name, RandomHeuristic(action_dim=action_dim, seed=seed), env_factory, eval_episodes)
        summary.update({"seed": seed, "scenario": scenario_name, "parameter_count": 0, "train_steps": 0})
        return summary

    if model_name == "greedy_heuristic":
        summary = evaluate_heuristic(model_name, GreedyOccupancyHeuristic(num_channels=action_dim), env_factory, eval_episodes)
        summary.update({"seed": seed, "scenario": scenario_name, "parameter_count": 0, "train_steps": 0})
        return summary

    train_start = time.perf_counter()
    agent = PPOAgent(
        model_name=model_name,
        obs_dim=obs_dim,
        seq_len=seq_len,
        action_dim=action_dim,
        device=device,
        hyperparams=hyperparams,
    )
    histories = train_agent(
        agent=agent,
        env_factory=env_factory,
        total_timesteps=total_steps,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
    )
    final_eval = evaluate_agent(agent, env_factory, num_episodes=eval_episodes)
    train_seconds = time.perf_counter() - train_start

    train_df = pd.DataFrame(histories["train_history"])
    if histories["eval_history"]:
        eval_df = pd.DataFrame(histories["eval_history"])
    else:
        eval_df = pd.DataFrame([{"step": float(agent.total_env_steps), **final_eval}])
    update_df = pd.DataFrame(histories["update_history"])

    run_stub = f"{scenario_name}__{model_name}__seed{seed}"
    train_df.to_csv(result_dirs["raw_training_logs"] / f"{run_stub}_train.csv", index=False)
    eval_df.to_csv(result_dirs["raw_training_logs"] / f"{run_stub}_eval.csv", index=False)
    update_df.to_csv(result_dirs["raw_training_logs"] / f"{run_stub}_update.csv", index=False)

    metrics_payload = {
        "scenario": scenario_name,
        "model": model_name,
        "seed": seed,
        "hyperparams": hyperparams.to_dict(),
        "environment": config.to_dict(),
        "parameter_count": agent.parameter_count,
        "total_env_steps": agent.total_env_steps,
        "wall_clock_training_time": train_seconds,
        "train_history_last": histories["train_history"][-1] if histories["train_history"] else {},
        "final_eval": final_eval,
        "update_history_last": histories["update_history"][-1] if histories["update_history"] else {},
    }
    with open(result_dirs["episode_metrics"] / f"{run_stub}.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    checkpoint_path = result_dirs["checkpoints"] / f"{run_stub}.pt"
    agent.save(str(checkpoint_path))

    summary = {
        "scenario": scenario_name,
        "model": model_name,
        "seed": seed,
        "parameter_count": agent.parameter_count,
        "train_steps": agent.total_env_steps,
        "wall_clock_training_time": float(train_seconds),
    }
    summary.update(final_eval)
    return summary


def aggregate_results(results_df: pd.DataFrame, result_dirs: Dict[str, Path]) -> None:
    metric_cols = [
        c
        for c in results_df.columns
        if c not in {"scenario", "model", "seed"}
    ]
    grouped = results_df.groupby(["scenario", "model"], dropna=False)
    rows = []
    for (scenario, model), frame in grouped:
        row = {"scenario": scenario, "model": model, "num_seeds": int(frame["seed"].nunique())}
        for col in metric_cols:
            row[f"mean_{col}"] = float(frame[col].mean())
            row[f"std_{col}"] = float(frame[col].std(ddof=0))
        rows.append(row)
    aggregate_df = pd.DataFrame(rows).sort_values(["scenario", "model"]).reset_index(drop=True)
    aggregate_df.to_csv(result_dirs["aggregate_tables"] / "aggregate_results.csv", index=False)

    key_metrics = [
        "mean_mean_episode_reward",
        "mean_mean_spectrum_utilization",
        "mean_mean_collision_rate",
        "mean_wall_clock_training_time",
        "mean_parameter_count",
    ]
    available = [c for c in key_metrics if c in aggregate_df.columns]
    if available:
        aggregate_df[["scenario", "model", *available]].to_csv(
            result_dirs["aggregate_tables"] / "paper_table_core_metrics.csv",
            index=False,
        )


def build_plots(result_dirs: Dict[str, Path], results_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    aggregate_path = result_dirs["aggregate_tables"] / "aggregate_results.csv"
    agg = pd.read_csv(aggregate_path)

    plt.figure(figsize=(14, 6))
    sns.barplot(data=agg, x="scenario", y="mean_mean_episode_reward", hue="model")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean Episode Reward")
    plt.xlabel("Scenario")
    plt.tight_layout()
    plt.savefig(result_dirs["plots"] / "reward_by_scenario.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 6))
    sns.barplot(data=agg, x="scenario", y="mean_mean_collision_rate", hue="model")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean Collision Rate")
    plt.xlabel("Scenario")
    plt.tight_layout()
    plt.savefig(result_dirs["plots"] / "collision_by_scenario.png", dpi=180)
    plt.close()

    train_logs = sorted(result_dirs["raw_training_logs"].glob("*_train.csv"))
    if train_logs:
        all_train = []
        for csv_path in train_logs:
            df = pd.read_csv(csv_path)
            stem = csv_path.stem.replace("_train", "")
            parts = stem.split("__")
            if len(parts) >= 3:
                df["scenario"] = parts[0]
                df["model"] = parts[1]
                df["seed"] = parts[2].replace("seed", "")
            all_train.append(df)
        train_df = pd.concat(all_train, ignore_index=True)
        if "train_step" in train_df.columns:
            plt.figure(figsize=(14, 7))
            sns.lineplot(data=train_df, x="train_step", y="episode_reward", hue="model", style="scenario", errorbar=None)
            plt.ylabel("Episode Reward")
            plt.xlabel("Training Step")
            plt.tight_layout()
            plt.savefig(result_dirs["plots"] / "training_curves.png", dpi=180)
            plt.close()

    results_df.to_csv(result_dirs["aggregate_tables"] / "all_run_summaries.csv", index=False)


def save_config_snapshot(args, hyperparams: PPOHyperParams, result_dirs: Dict[str, Path]) -> None:
    snapshot = {
        "models": args.models,
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "training_steps": args.training_steps,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "device": args.device,
        "hyperparams": hyperparams.to_dict(),
        "suite_preview": {name: cfg.to_dict() for name, cfg in make_empirical_suite(seed=args.seeds[0]).items() if name in args.scenarios},
    }
    with open(result_dirs["config_snapshots"] / "empirical_suite.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the empirical DSA benchmark suite.")
    parser.add_argument("--results-dir", default="results", help="Directory where artifacts will be saved.")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or cuda.")
    parser.add_argument("--training-steps", type=int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "random_policy",
            "greedy_heuristic",
            "ppo_mlp",
            "ppo_lstm",
            "ppo_gru",
            "ppo_ltc",
            "ppo_lfm",
            "ppo_ltc_lfm",
        ],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "stationary",
            "non_stationary",
            "interference_heavy",
            "varying_users_small",
            "varying_users_large",
            "noisy_partial",
            "variable_dt",
        ],
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = ensure_result_dirs(Path(args.results_dir))
    hyperparams = PPOHyperParams(
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )
    save_config_snapshot(args, hyperparams, result_dirs)

    summaries: List[Dict[str, float]] = []
    for scenario in args.scenarios:
        for model in args.models:
            for seed in args.seeds:
                print(f"[run] scenario={scenario} model={model} seed={seed}", flush=True)
                summary = run_single_experiment(
                    model_name=model,
                    scenario_name=scenario,
                    seed=seed,
                    total_steps=args.training_steps,
                    eval_every=args.eval_every,
                    eval_episodes=args.eval_episodes,
                    device=args.device,
                    result_dirs=result_dirs,
                    hyperparams=hyperparams,
                )
                summaries.append(summary)
                pd.DataFrame(summaries).to_csv(result_dirs["aggregate_tables"] / "all_run_summaries_live.csv", index=False)

    results_df = pd.DataFrame(summaries).sort_values(["scenario", "model", "seed"]).reset_index(drop=True)
    aggregate_results(results_df, result_dirs)
    build_plots(result_dirs, results_df)
    with open(result_dirs["episode_metrics"] / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_runs": len(summaries),
                "models": args.models,
                "scenarios": args.scenarios,
                "seeds": args.seeds,
                "training_steps": args.training_steps,
            },
            f,
            indent=2,
        )
    print("Empirical benchmark suite completed.")


if __name__ == "__main__":
    main()
