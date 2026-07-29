"""Figures.  Every figure draws its uncertainty.

A bar chart of means with no error bars is a claim without evidence, so nothing
here plots a point estimate on its own.  Figures are rendered with the ``Agg``
backend so they work headless in CI, and every figure contains every policy that
ran -- the same no-filter rule as the tables.
"""

from __future__ import annotations

import pathlib
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dsa.analysis.tables import PRIMARY_METRIC  # noqa: E402

__all__ = ["build_all_figures"]

FIGURE_DPI = 140


def _finish(fig: plt.Figure, path: pathlib.Path) -> pathlib.Path:
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


def _err(low: Sequence[float], high: Sequence[float], mid: Sequence[float]) -> np.ndarray:
    """Asymmetric error bars from a confidence interval, clipped at zero."""
    mid = np.asarray(mid, dtype=np.float64)
    low = np.nan_to_num(np.asarray(low, dtype=np.float64), nan=0.0)
    high = np.nan_to_num(np.asarray(high, dtype=np.float64), nan=0.0)
    return np.vstack([np.maximum(mid - low, 0.0), np.maximum(high - mid, 0.0)])


def _plot_overall_ranking(table: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path | None:
    if table.empty:
        return None
    t = table.sort_values(PRIMARY_METRIC)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 2))
    y = np.arange(len(t))
    ax.barh(y, t[PRIMARY_METRIC], color="#4C72B0", alpha=0.85)
    ax.errorbar(
        t[PRIMARY_METRIC],
        y,
        xerr=_err(t.get(f"{PRIMARY_METRIC}_ci_low"), t.get(f"{PRIMARY_METRIC}_ci_high"), t[PRIMARY_METRIC]),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(t["model"])
    ax.set_xlabel(f"{PRIMARY_METRIC} (mean over scenarios, 95% bootstrap CI)")
    ax.set_title("Overall ranking -- every policy that ran, baselines included")
    ax.grid(axis="x", alpha=0.3)
    return _finish(fig, out_dir / "overall_ranking.png")


def _plot_per_scenario(table: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path | None:
    if table.empty:
        return None
    scenarios = sorted(table["scenario"].unique())
    models = sorted(table["model"].unique())
    ncols = min(3, len(scenarios))
    nrows = int(np.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    for i, scenario in enumerate(scenarios):
        ax = axes[i // ncols][i % ncols]
        chunk = table[table["scenario"] == scenario].set_index("model").reindex(models)
        y = np.arange(len(models))
        ax.barh(y, chunk[PRIMARY_METRIC], color="#55A868", alpha=0.85)
        ax.errorbar(
            chunk[PRIMARY_METRIC],
            y,
            xerr=_err(
                chunk.get(f"{PRIMARY_METRIC}_ci_low"),
                chunk.get(f"{PRIMARY_METRIC}_ci_high"),
                chunk[PRIMARY_METRIC],
            ),
            fmt="none",
            ecolor="black",
            capsize=2,
            linewidth=1.0,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_title(scenario, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    for j in range(len(scenarios), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{PRIMARY_METRIC} by scenario (95% bootstrap CI over seeds)", y=1.0)
    return _finish(fig, out_dir / "per_scenario.png")


def _plot_comparisons(table: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path | None:
    if table.empty or "mean_difference" not in table.columns:
        return None
    t = table.sort_values(["family", "mean_difference"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(t) + 2.5))
    y = np.arange(len(t))
    # matplotlib takes a single ecolor per call, so the two verdict classes are
    # drawn as two series: grey where the interval covers zero, red where it does
    # not.  The colour is the verdict, not the sign of the point estimate.
    crosses_zero = (t["verdict"] == "no_detectable_difference").to_numpy()
    for mask, colour, label in (
        (crosses_zero, "#999999", "interval contains zero"),
        (~crosses_zero, "#C44E52", "interval excludes zero"),
    ):
        if not mask.any():
            continue
        sub = t[mask]
        ax.errorbar(
            sub["mean_difference"],
            y[mask],
            xerr=_err(sub["ci_low"], sub["ci_high"], sub["mean_difference"]),
            fmt="o",
            markersize=3,
            color=colour,
            ecolor=colour,
            capsize=2,
            linewidth=1.0,
            linestyle="none",
            label=label,
        )
    ax.legend(fontsize=7, loc="best")
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"[{r.family.split('_')[0]}] {r.label_a} - {r.label_b} @ {r.scenario}" for r in t.itertuples()],
        fontsize=6,
    )
    ax.set_xlabel("paired mean difference in mean_eval_return (95% bootstrap CI)")
    ax.set_title("All pre-registered comparisons; grey = interval contains zero")
    ax.grid(axis="x", alpha=0.3)
    return _finish(fig, out_dir / "comparisons.png")


def _plot_learning_check(table: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path | None:
    if table.empty:
        return None
    t = table[table["trainable"].astype(bool)] if "trainable" in table else table
    if t.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(t) + 2.5))
    y = np.arange(len(t))
    ax.errorbar(
        t["mean_improvement"],
        y,
        xerr=_err(t["ci_low"], t["ci_high"], t["mean_improvement"]),
        fmt="o",
        markersize=3,
        color="black",
        capsize=2,
        linewidth=1.0,
        linestyle="none",
    )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.model} @ {r.scenario}" for r in t.itertuples()], fontsize=6)
    ax.set_xlabel("trained minus untrained mean_eval_return (95% bootstrap CI)")
    ax.set_title("F5 learning check: did training beat initialisation?")
    ax.grid(axis="x", alpha=0.3)
    return _finish(fig, out_dir / "learning_check.png")


def _plot_federated(table: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path | None:
    if table.empty or "cloud_megabytes" not in table.columns:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(table))
    ax1.bar(x, table[PRIMARY_METRIC], color="#4C72B0", alpha=0.85)
    ax1.errorbar(
        x,
        table[PRIMARY_METRIC],
        yerr=_err(
            table.get(f"{PRIMARY_METRIC}_ci_low"),
            table.get(f"{PRIMARY_METRIC}_ci_high"),
            table[PRIMARY_METRIC],
        ),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(table["arm"], rotation=15, fontsize=8)
    ax1.set_ylabel(PRIMARY_METRIC)
    ax1.set_title("Federated arms: task performance")
    ax1.grid(axis="y", alpha=0.3)

    width = 0.38
    ax2.bar(x - width / 2, table["cloud_megabytes"], width, label="cloud (wide-area)", color="#C44E52")
    if "edge_local_megabytes" in table.columns:
        ax2.bar(x + width / 2, table["edge_local_megabytes"], width, label="edge-local", color="#55A868")
    ax2.set_xticks(x)
    ax2.set_xticklabels(table["arm"], rotation=15, fontsize=8)
    ax2.set_ylabel("megabytes (measured from tensors)")
    ax2.set_title("Federated arms: measured communication")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    return _finish(fig, out_dir / "federated.png")


def build_all_figures(
    tables: Mapping[str, pd.DataFrame],
    out_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Render every figure that has data.  Returns the paths actually written."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = (
        ("overall_ranking", _plot_overall_ranking),
        ("main_table", _plot_per_scenario),
        ("comparisons", _plot_comparisons),
        ("learning_check", _plot_learning_check),
        ("federated_table", _plot_federated),
    )
    written: list[pathlib.Path] = []
    for key, builder in builders:
        table = tables.get(key)
        if table is None:
            continue
        path = builder(table, out_dir)
        if path is not None:
            written.append(path)
    return sorted(written)
