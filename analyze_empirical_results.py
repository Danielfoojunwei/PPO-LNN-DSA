"""Build paper-ready summaries from completed empirical benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
AGG = RESULTS / "aggregate_tables" / "aggregate_results.csv"
FED_AGG = RESULTS / "federated" / "aggregate_tables" / "federated_aggregate_results.csv"
PLOTS = RESULTS / "plots"
TABLES = RESULTS / "aggregate_tables"


CORE_MODELS = [
    "greedy_heuristic",
    "ppo_mlp",
    "ppo_lstm",
    "ppo_gru",
    "ppo_ltc",
    "ppo_lfm",
    "ppo_ltc_lfm",
]


PRETTY = {
    "greedy_heuristic": "Greedy heuristic",
    "ppo_mlp": "PPO-MLP",
    "ppo_lstm": "PPO-LSTM",
    "ppo_gru": "PPO-GRU",
    "ppo_ltc": "PPO-LTC",
    "ppo_lfm": "PPO-LFM",
    "ppo_ltc_lfm": "PPO-LTC-LFM",
    "centralized": "Centralized",
    "flat_federated": "Flat federated",
    "hierarchical_federated": "Hierarchical federated",
}


SCENARIO_ORDER = [
    "stationary",
    "non_stationary",
    "interference_heavy",
    "varying_users_small",
    "varying_users_large",
    "noisy_partial",
    "variable_dt",
]


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 180


def fmt(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"



def safe_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]



def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = pd.read_csv(AGG)
    fed = pd.read_csv(FED_AGG)
    agg = agg[agg["model"].isin(CORE_MODELS)].copy()
    agg["model_pretty"] = agg["model"].map(PRETTY)
    agg["scenario"] = pd.Categorical(agg["scenario"], categories=SCENARIO_ORDER, ordered=True)
    agg = agg.sort_values(["scenario", "model_pretty"]).reset_index(drop=True)

    fed["mode_pretty"] = fed["mode"].map(PRETTY)
    return agg, fed



def build_main_table(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            {
                "Scenario": r["scenario"],
                "Model": r["model_pretty"],
                "Reward (mean ± std)": fmt(r["mean_mean_episode_reward"], r["std_mean_episode_reward"]),
                "Utilization (mean ± std)": fmt(r["mean_mean_spectrum_utilization"], r["std_mean_spectrum_utilization"]),
                "Collision rate (mean ± std)": fmt(r["mean_mean_collision_rate"], r["std_mean_collision_rate"]),
                "Training time, s": round(float(r["mean_wall_clock_training_time"]), 3),
                "Parameters": int(r["mean_parameter_count"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "paper_table_main.csv", index=False)
    return out



def build_overall_ranking(agg: pd.DataFrame) -> pd.DataFrame:
    ranking = (
        agg.groupby("model_pretty", as_index=False)
        .agg(
            avg_reward=("mean_mean_episode_reward", "mean"),
            avg_utilization=("mean_mean_spectrum_utilization", "mean"),
            avg_collision_rate=("mean_mean_collision_rate", "mean"),
            avg_train_time_s=("mean_wall_clock_training_time", "mean"),
            parameters=("mean_parameter_count", "mean"),
        )
        .sort_values("avg_reward", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(TABLES / "overall_model_ranking.csv", index=False)
    return ranking



def build_ablation_table(agg: pd.DataFrame) -> pd.DataFrame:
    subset = agg[agg["model"].isin(["ppo_mlp", "ppo_ltc", "ppo_lfm", "ppo_ltc_lfm"])].copy()
    rows = []
    for scenario, frame in subset.groupby("scenario", sort=False):
        pivot = frame.set_index("model")
        base_reward = float(pivot.loc["ppo_mlp", "mean_mean_episode_reward"])
        base_collision = float(pivot.loc["ppo_mlp", "mean_mean_collision_rate"])
        for model in ["ppo_ltc", "ppo_lfm", "ppo_ltc_lfm"]:
            rows.append(
                {
                    "scenario": scenario,
                    "variant": PRETTY[model],
                    "reward": float(pivot.loc[model, "mean_mean_episode_reward"]),
                    "reward_gain_vs_mlp": float(pivot.loc[model, "mean_mean_episode_reward"] - base_reward),
                    "utilization": float(pivot.loc[model, "mean_mean_spectrum_utilization"]),
                    "collision_rate": float(pivot.loc[model, "mean_mean_collision_rate"]),
                    "collision_change_vs_mlp": float(pivot.loc[model, "mean_mean_collision_rate"] - base_collision),
                    "train_time_s": float(pivot.loc[model, "mean_wall_clock_training_time"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "ablation_ltc_lfm.csv", index=False)
    return out



def build_winner_table(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, frame in agg.groupby("scenario", sort=False):
        reward_winner = frame.sort_values("mean_mean_episode_reward", ascending=False).iloc[0]
        collision_winner = frame.sort_values("mean_mean_collision_rate", ascending=True).iloc[0]
        util_winner = frame.sort_values("mean_mean_spectrum_utilization", ascending=False).iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "best_reward_model": reward_winner["model_pretty"],
                "best_reward": float(reward_winner["mean_mean_episode_reward"]),
                "best_utilization_model": util_winner["model_pretty"],
                "best_utilization": float(util_winner["mean_mean_spectrum_utilization"]),
                "lowest_collision_model": collision_winner["model_pretty"],
                "lowest_collision_rate": float(collision_winner["mean_mean_collision_rate"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "scenario_winners.csv", index=False)
    return out



def build_federated_table(fed: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scenario",
        "mode_pretty",
        "mean_mean_episode_reward",
        "std_mean_episode_reward",
        "mean_mean_spectrum_utilization",
        "std_mean_spectrum_utilization",
        "mean_mean_collision_rate",
        "std_mean_collision_rate",
        "mean_communication_megabytes",
        "std_communication_megabytes",
        "mean_wall_clock_training_time",
        "std_wall_clock_training_time",
    ]
    cols = safe_cols(fed, cols)
    out = fed[cols].copy()
    if "mean_mean_episode_reward" in out.columns:
        out["Reward (mean ± std)"] = out.apply(
            lambda r: fmt(r["mean_mean_episode_reward"], r.get("std_mean_episode_reward", 0.0)), axis=1
        )
    if "mean_mean_spectrum_utilization" in out.columns:
        out["Utilization (mean ± std)"] = out.apply(
            lambda r: fmt(r["mean_mean_spectrum_utilization"], r.get("std_mean_spectrum_utilization", 0.0)), axis=1
        )
    if "mean_mean_collision_rate" in out.columns:
        out["Collision rate (mean ± std)"] = out.apply(
            lambda r: fmt(r["mean_mean_collision_rate"], r.get("std_mean_collision_rate", 0.0)), axis=1
        )
    out.to_csv(TABLES / "federated_tradeoff_table.csv", index=False)
    return out



def plot_reward_heatmap(agg: pd.DataFrame) -> None:
    pivot = agg.pivot(index="model_pretty", columns="scenario", values="mean_mean_episode_reward")
    pivot = pivot[[s for s in SCENARIO_ORDER if s in pivot.columns]]
    plt.figure(figsize=(12, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.xlabel("Scenario")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(PLOTS / "reward_heatmap.png")
    plt.close()



def plot_ablation(ablation: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ablation, x="scenario", y="reward_gain_vs_mlp", hue="variant")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Reward gain vs PPO-MLP")
    plt.xlabel("Scenario")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / "ablation_reward_gain_vs_mlp.png")
    plt.close()



def plot_federated_tradeoff(fed: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=fed,
        x="mean_communication_megabytes",
        y="mean_mean_episode_reward",
        hue="mode_pretty",
        style="mode_pretty",
        s=120,
    )
    for _, row in fed.iterrows():
        plt.text(
            float(row["mean_communication_megabytes"]) + 0.15,
            float(row["mean_mean_episode_reward"]) + 0.1,
            row["mode_pretty"],
            fontsize=9,
        )
    plt.xlabel("Communication volume (MB)")
    plt.ylabel("Mean episode reward")
    plt.tight_layout()
    plt.savefig(PLOTS / "federated_reward_vs_communication.png")
    plt.close()



def main() -> None:
    agg, fed = load_tables()
    build_main_table(agg)
    build_overall_ranking(agg)
    ablation = build_ablation_table(agg)
    build_winner_table(agg)
    build_federated_table(fed)
    plot_reward_heatmap(agg)
    plot_ablation(ablation)
    plot_federated_tradeoff(fed)


if __name__ == "__main__":
    main()
