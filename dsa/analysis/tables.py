"""Table construction.  Every table contains every policy that ran.

The previous version of this repository defined a module-level ``CORE_MODELS``
tuple that omitted ``random_policy`` and applied it as an ``isin`` filter to the
four tables the README cited.  Restored, that baseline ranked 4th of 7 -- ahead
of three of the PPO variants.  There is no filter in this module.  The only
subsetting that happens anywhere is the **pre-registered family membership** in
``configs/preregistration.yaml``, which selects which *comparisons are tested*,
never which *rows are displayed*.

Uncertainty is mandatory.  Every aggregate column is emitted with a standard
deviation, a standard error and a bootstrap confidence interval over seeds.  A
bare mean is not a result.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from dsa.analysis.stats import (
    ANALYSIS_SEED,
    CI_LEVEL,
    NO_DIFFERENCE_PHRASE,
    PairedResult,
    compare_paired,
    holm_bonferroni,
    paired_bootstrap_ci,
)
from dsa.seeding import derive_seed

__all__ = [
    "PRIMARY_METRIC",
    "TABLE_NAMES",
    "build_all_tables",
    "build_claims",
    "expand_families",
    "summarise_by",
]

#: The single scalar every confirmatory comparison is computed on.
PRIMARY_METRIC = "mean_eval_return"

#: Extra per-cell metrics carried into the descriptive tables, each with its own
#: uncertainty columns.
SECONDARY_METRICS = (
    "mean_success_rate",
    "mean_collision_rate",
    "mean_spectrum_utilization",
    "mean_policy_entropy_nats",
    "mean_action_histogram_entropy",
)

TABLE_NAMES = (
    "main_table",
    "overall_ranking",
    "comparisons",
    "ablation",
    "federated_table",
    "learning_check",
    "parameter_budget",
)

#: Display precision for stored tables.  Spec 4.5: three decimals in prose, and
#: never 17 significant digits in a CSV.
DECIMALS = 6


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _round_floats(df: pd.DataFrame, decimals: int = DECIMALS) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def _require_columns(df: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{what} is missing required column(s): {missing}")


def _mean_ci(values: np.ndarray, tag: str) -> tuple[float, float]:
    """Bootstrap interval for a mean across seeds, seeded from ``tag``."""
    if values.size < 2:
        v = float(values[0]) if values.size else float("nan")
        return v, v
    return paired_bootstrap_ci(values, level=CI_LEVEL, seed=derive_seed(ANALYSIS_SEED, "boot", tag))


def summarise_by(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
    tag_prefix: str,
) -> pd.DataFrame:
    """Group and summarise, attaching uncertainty to every metric.

    For each metric ``m`` emits ``m`` (mean over seeds), ``m_std``, ``m_sem``,
    ``m_ci_low`` and ``m_ci_high``.  No metric is ever emitted without them.
    """
    group_columns = list(group_columns)
    rows: list[dict[str, Any]] = []
    for keys, chunk in df.groupby(group_columns, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(group_columns, keys))
        row["n_seeds"] = int(len(chunk))
        for metric in metrics:
            if metric not in chunk.columns:
                continue
            values = chunk[metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[metric] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_sem"] = float("nan")
                row[f"{metric}_ci_low"] = float("nan")
                row[f"{metric}_ci_high"] = float("nan")
                continue
            tag = f"{tag_prefix}|{'|'.join(str(k) for k in keys)}|{metric}"
            lo, hi = _mean_ci(values, tag)
            std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = (
                std / float(np.sqrt(values.size)) if values.size > 1 else float("nan")
            )
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _paired_vectors(
    df: pd.DataFrame,
    scenario: str,
    label_a: str,
    label_b: str,
    column_a: str,
    column_b: str,
    key_column: str = "model",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract two seed-aligned vectors, or raise if the pairing is broken.

    ``label_a``/``label_b`` may name the same arm (the trained-vs-untrained case)
    in which case ``column_a`` and ``column_b`` differ instead.
    """
    scoped = df[df["scenario"] == scenario]
    a = scoped[scoped[key_column] == label_a]
    b = scoped[scoped[key_column] == label_b]
    if a.empty or b.empty:
        return np.array([]), np.array([]), []

    a = a.set_index("seed").sort_index()
    b = b.set_index("seed").sort_index()
    shared = sorted(set(a.index) & set(b.index))
    if not shared:
        return np.array([]), np.array([]), []

    va = a.loc[shared, column_a].to_numpy(dtype=np.float64)
    vb = b.loc[shared, column_b].to_numpy(dtype=np.float64)
    return va, vb, [int(s) for s in shared]


# --------------------------------------------------------------------------- #
# Pre-registered family expansion
# --------------------------------------------------------------------------- #


def _resolve_arms(spec: Any, all_models: Sequence[str]) -> list[str]:
    if spec == "all_primary_models":
        return list(all_models)
    if isinstance(spec, str):
        return [spec]
    return [str(x) for x in spec]


def _resolve_scenarios(spec: Any, primary: Sequence[str]) -> list[str]:
    if spec in (None, "primary"):
        return list(primary)
    if isinstance(spec, str):
        return [spec]
    return [str(x) for x in spec]


def expand_families(
    prereg: Mapping[str, Any],
    primary_models: Sequence[str],
    primary_scenarios: Sequence[str],
) -> list[dict[str, Any]]:
    """Expand ``configs/preregistration.yaml`` into concrete comparison specs.

    Returns one dict per comparison with keys ``family``, ``comparison_id``,
    ``kind``, ``scenario``, ``arm_a``, ``arm_b``.  Expansion is a pure function
    of the pre-registration file, so the set of tested comparisons is fixed
    **before** any result is read -- which is what makes the Holm correction
    honest rather than a post-hoc count.
    """
    families = prereg.get("families") or {}
    out: list[dict[str, Any]] = []

    for family_name in sorted(families):
        block = families[family_name] or {}
        kind = str(block.get("kind", "model_vs_model"))

        explicit = block.get("comparisons")
        if explicit:
            for entry in explicit:
                out.append(
                    {
                        "family": family_name,
                        "comparison_id": str(entry["id"]),
                        "kind": str(entry.get("kind", kind)),
                        "scenario": str(entry.get("scenario", "pooled")),
                        "arm_a": str(entry["arm_a"]),
                        "arm_b": str(entry["arm_b"]),
                        "holm": bool(block.get("holm", True)),
                    }
                )
            continue

        scenarios = _resolve_scenarios(block.get("scenarios"), primary_scenarios)
        arms_a = _resolve_arms(block.get("arms_a", []), primary_models)
        arms_b = _resolve_arms(block.get("arms_b", []), primary_models)

        if kind == "trained_vs_untrained":
            for scenario in scenarios:
                for arm in arms_a:
                    out.append(
                        {
                            "family": family_name,
                            "comparison_id": f"{family_name}:{arm}:trained_vs_untrained:{scenario}",
                            "kind": kind,
                            "scenario": scenario,
                            "arm_a": arm,
                            "arm_b": arm,
                            "holm": bool(block.get("holm", True)),
                        }
                    )
            continue

        for scenario in scenarios:
            for arm_a in arms_a:
                for arm_b in arms_b:
                    if arm_a == arm_b:
                        continue
                    out.append(
                        {
                            "family": family_name,
                            "comparison_id": f"{family_name}:{arm_a}_vs_{arm_b}:{scenario}",
                            "kind": kind,
                            "scenario": scenario,
                            "arm_a": arm_a,
                            "arm_b": arm_b,
                            "holm": bool(block.get("holm", True)),
                        }
                    )
    return out


def _run_comparison(
    spec: Mapping[str, Any],
    all_runs: pd.DataFrame,
    fed_runs: pd.DataFrame,
) -> PairedResult | None:
    kind = spec["kind"]
    family = spec["family"]
    cid = spec["comparison_id"]
    scenario = spec["scenario"]

    if kind == "trained_vs_untrained":
        arm = spec["arm_a"]
        scoped = all_runs[(all_runs["scenario"] == scenario) & (all_runs["model"] == arm)]
        if scoped.empty:
            return None
        scoped = scoped.sort_values("seed")
        a = scoped["mean_eval_return_trained"].to_numpy(dtype=np.float64)
        b = scoped["mean_eval_return_untrained"].to_numpy(dtype=np.float64)
        return compare_paired(a, b, cid, family, f"{arm}(trained)", f"{arm}(untrained)", scenario)

    if kind == "federated_arm":
        if fed_runs is None or fed_runs.empty:
            return None
        if "arm" not in fed_runs.columns or PRIMARY_METRIC not in fed_runs.columns:
            return None
        # The federated study is single-scenario by design; pair on `arm` alone
        # and label the scenario uniformly so the comparison is well defined even
        # if a scenario column is absent from the CSV.
        scoped = fed_runs.assign(scenario="federated")
        a, b, shared = _paired_vectors(
            scoped,
            scenario="federated",
            label_a=spec["arm_a"],
            label_b=spec["arm_b"],
            column_a=PRIMARY_METRIC,
            column_b=PRIMARY_METRIC,
            key_column="arm",
        )
        if not shared:
            return None
        return compare_paired(a, b, cid, family, spec["arm_a"], spec["arm_b"], "federated")

    a, b, shared = _paired_vectors(
        all_runs, scenario, spec["arm_a"], spec["arm_b"], PRIMARY_METRIC, PRIMARY_METRIC
    )
    if not shared:
        return None
    return compare_paired(a, b, cid, family, spec["arm_a"], spec["arm_b"], scenario)


# --------------------------------------------------------------------------- #
# Individual tables
# --------------------------------------------------------------------------- #


def _parameter_budget(all_runs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in (
            "model",
            "policy_type",
            "family",
            "dt_aware",
            "cell_kinds",
            "hidden_dim",
            "num_heads",
            "parameter_count",
            "param_error_frac",
        )
        if c in all_runs.columns
    ]
    table = all_runs[cols].drop_duplicates(subset=["model"]).sort_values("model")
    if "parameter_count" in table.columns:
        neural = table[table["parameter_count"] > 0]["parameter_count"]
        table["param_spread_vs_min"] = (
            table["parameter_count"] / float(neural.min()) if len(neural) else float("nan")
        )
    return table.reset_index(drop=True)


def _main_table(all_runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [PRIMARY_METRIC, "mean_eval_return_untrained", "mean_eval_return_trained"]
    metrics += [m for m in SECONDARY_METRICS if m in all_runs.columns]
    table = summarise_by(all_runs, ["scenario", "model"], metrics, "main")
    table["rank_in_scenario"] = (
        table.groupby("scenario")[PRIMARY_METRIC].rank(ascending=False, method="min").astype(int)
    )
    return table.sort_values(["scenario", "rank_in_scenario", "model"]).reset_index(drop=True)


def _overall_ranking(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Pool over scenarios by first averaging within a scenario.

    Averaging the per-scenario means (rather than every cell) keeps scenarios
    equally weighted even if one of them has a missing cell.
    """
    per_cell = all_runs.groupby(["model", "scenario"], sort=True)[PRIMARY_METRIC].mean().reset_index()
    rows: list[dict[str, Any]] = []
    for model, chunk in per_cell.groupby("model", sort=True):
        values = chunk[PRIMARY_METRIC].to_numpy(dtype=np.float64)
        lo, hi = _mean_ci(values, f"rank|{model}")
        std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        rows.append(
            {
                "model": model,
                "n_scenarios": int(values.size),
                PRIMARY_METRIC: float(values.mean()),
                f"{PRIMARY_METRIC}_std": std,
                f"{PRIMARY_METRIC}_sem": (
                    std / float(np.sqrt(values.size)) if values.size > 1 else float("nan")
                ),
                f"{PRIMARY_METRIC}_ci_low": lo,
                f"{PRIMARY_METRIC}_ci_high": hi,
                "best_scenario": str(chunk.loc[chunk[PRIMARY_METRIC].idxmax(), "scenario"]),
                "worst_scenario": str(chunk.loc[chunk[PRIMARY_METRIC].idxmin(), "scenario"]),
            }
        )
    table = pd.DataFrame(rows)
    table["rank"] = table[PRIMARY_METRIC].rank(ascending=False, method="min").astype(int)
    return table.sort_values("rank").reset_index(drop=True)


def _learning_check(all_runs: pd.DataFrame, results: Sequence[PairedResult]) -> pd.DataFrame:
    """Trained vs untrained for every policy, tested where testable.

    Heuristic baselines have no initialisation to improve on, so their rows carry
    ``trainable=False`` and no p-value.  They are still **present**: a table that
    silently omits the policies it cannot test is how the previous version of
    this repository lost its random baseline.
    """
    by_id = {r.comparison_id: r for r in results}
    rows: list[dict[str, Any]] = []
    for (scenario, model), chunk in all_runs.groupby(["scenario", "model"], sort=True):
        trained = chunk["mean_eval_return_trained"].to_numpy(dtype=np.float64)
        untrained = chunk["mean_eval_return_untrained"].to_numpy(dtype=np.float64)
        trainable = bool(chunk["is_trainable"].iloc[0]) if "is_trainable" in chunk else True
        diff = trained - untrained
        row: dict[str, Any] = {
            "scenario": scenario,
            "model": model,
            "trainable": trainable,
            "n_seeds": int(len(chunk)),
            "mean_eval_return_untrained": float(untrained.mean()),
            "mean_eval_return_trained": float(trained.mean()),
            "mean_improvement": float(diff.mean()),
            "std_improvement": float(diff.std(ddof=1)) if diff.size > 1 else float("nan"),
        }
        match = [
            r
            for cid, r in by_id.items()
            if r.scenario == scenario and r.label_a == f"{model}(trained)"
        ]
        result = match[0] if match else None
        row.update(
            {
                "ci_low": result.ci_low if result else float("nan"),
                "ci_high": result.ci_high if result else float("nan"),
                "p_value": result.p_value if result else float("nan"),
                "holm_adjusted_p": (result.holm_adjusted_p if result else float("nan")),
                "significant": (result.significant if result else None),
                "verdict": (
                    result.verdict
                    if result
                    else ("not_applicable" if not trainable else "not_tested")
                ),
                "learned": bool(result.significant and result.mean_difference > 0)
                if result and result.significant is not None
                else None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "model"]).reset_index(drop=True)


def _ablation(results: Sequence[PairedResult]) -> pd.DataFrame:
    """The architectural ablation: F3 (hybrid decomposition) and F4 (dt awareness).

    Every row carries an interval.  The previous ablation table reported gains to
    17 significant digits with no uncertainty column at all, and in 19 of 21 rows
    the magnitude of the reported "gain" was smaller than the baseline's own
    across-seed standard deviation.
    """
    keep = [r for r in results if r.family.startswith(("F3", "F4"))]
    if not keep:
        return pd.DataFrame(
            columns=[
                "family",
                "comparison_id",
                "scenario",
                "label_a",
                "label_b",
                "n_pairs",
                "mean_difference",
                "ci_low",
                "ci_high",
                "p_value",
                "holm_adjusted_p",
                "significant",
                "verdict",
                "description",
            ]
        )
    return pd.DataFrame([r.to_row() for r in keep]).sort_values(
        ["family", "scenario", "label_a", "label_b"]
    ).reset_index(drop=True)


def _federated_table(fed_runs: pd.DataFrame) -> pd.DataFrame:
    if fed_runs is None or fed_runs.empty:
        return pd.DataFrame(columns=["arm", "n_seeds", PRIMARY_METRIC])
    metrics = [
        m
        for m in (
            PRIMARY_METRIC,
            "cloud_megabytes",
            "edge_local_megabytes",
            "total_megabytes",
            "mean_success_rate",
            "mean_collision_rate",
        )
        if m in fed_runs.columns
    ]
    table = summarise_by(fed_runs, ["arm"], metrics, "fed")
    for col in ("train_steps", "declared_train_steps", "parameter_count", "num_transfers"):
        if col in fed_runs.columns:
            table[col] = table["arm"].map(fed_runs.groupby("arm")[col].first())
    if {"train_steps", "declared_train_steps"} <= set(table.columns):
        table["train_steps_match_declared"] = table["train_steps"] == table["declared_train_steps"]
    return table.sort_values(PRIMARY_METRIC, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build_all_tables(
    all_runs: pd.DataFrame,
    fed_runs: pd.DataFrame,
    prereg: Mapping[str, Any],
    out_dir: pathlib.Path,
) -> dict[str, pd.DataFrame]:
    """Build and write every table.  Returns them keyed by ``TABLE_NAMES``.

    ``out_dir`` is created if absent.  Nothing outside ``out_dir`` is touched --
    the analysis layer has no default write path into tracked evidence.
    """
    _require_columns(
        all_runs,
        ["scenario", "model", "seed", PRIMARY_METRIC],
        "all_runs",
    )
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for column in ("mean_eval_return_trained", "mean_eval_return_untrained"):
        if column not in all_runs.columns:
            all_runs = all_runs.assign(**{column: all_runs[PRIMARY_METRIC]})

    if "is_trainable" in all_runs.columns:
        trainable_rows = all_runs[all_runs["is_trainable"].astype(bool)]
    else:
        trainable_rows = all_runs
    primary_models = sorted(trainable_rows["model"].unique())
    primary_scenarios = sorted(all_runs["scenario"].unique())

    specs = expand_families(prereg, primary_models, primary_scenarios)
    raw: list[PairedResult] = []
    for spec in specs:
        result = _run_comparison(spec, all_runs, fed_runs)
        if result is not None:
            raw.append(result)

    corrected: list[PairedResult] = []
    by_family: dict[str, list[PairedResult]] = {}
    for r in raw:
        by_family.setdefault(r.family, []).append(r)
    holm_flags = {s["family"]: s.get("holm", True) for s in specs}
    for family, members in sorted(by_family.items()):
        if holm_flags.get(family, True):
            corrected.extend(holm_bonferroni(members))
        else:
            # A pre-registered singleton family is reported unadjusted; with one
            # member Holm is the identity anyway, so this is documentation.
            corrected.extend(
                [
                    dataclasses.replace(
                        r,
                        holm_adjusted_p=r.p_value,
                        significant=bool(r.p_value <= 0.05),
                    )
                    for r in members
                ]
            )

    comparisons = (
        pd.DataFrame([r.to_row() for r in corrected])
        .sort_values(["family", "scenario", "comparison_id"])
        .reset_index(drop=True)
        if corrected
        else pd.DataFrame()
    )

    tables: dict[str, pd.DataFrame] = {
        "parameter_budget": _parameter_budget(all_runs),
        "main_table": _main_table(all_runs),
        "overall_ranking": _overall_ranking(all_runs),
        "learning_check": _learning_check(all_runs, corrected),
        "ablation": _ablation(corrected),
        "comparisons": comparisons,
        "federated_table": _federated_table(fed_runs),
    }

    for name, table in tables.items():
        _round_floats(table).to_csv(out_dir / f"{name}.csv", index=False)

    return tables


# --------------------------------------------------------------------------- #
# Machine-readable claims
# --------------------------------------------------------------------------- #

_CLAIM_FIELDS = (
    "claim_id",
    "statement",
    "source_file",
    "source_column",
    "value",
    "ci_low",
    "ci_high",
    "p_value",
    "holm_adjusted_p",
    "verdict",
)


def _claim(
    claim_id: str,
    statement: str,
    source_file: str,
    source_column: str,
    value: float,
    ci_low: float | None = None,
    ci_high: float | None = None,
    p_value: float | None = None,
    holm_adjusted_p: float | None = None,
    verdict: str = "descriptive",
) -> dict[str, Any]:
    def n(x: float | None) -> float | None:
        if x is None:
            return None
        x = float(x)
        return None if (np.isnan(x) or np.isinf(x)) else round(x, DECIMALS)

    return {
        "claim_id": claim_id,
        "statement": statement,
        "source_file": source_file,
        "source_column": source_column,
        "value": n(value),
        "ci_low": n(ci_low),
        "ci_high": n(ci_high),
        "p_value": n(p_value),
        "holm_adjusted_p": n(holm_adjusted_p),
        "verdict": verdict,
    }


def build_claims(tables: Mapping[str, pd.DataFrame], prereg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Emit one strictly-typed record per citable claim.

    Every entry names the CSV and the column it was read from, so the auditor can
    re-derive it without running anything.  Prose that cites a number must cite a
    ``claim_id``; a number with no claim is a number with no provenance.
    """
    claims: list[dict[str, Any]] = []

    comparisons = tables.get("comparisons")
    if comparisons is not None and not comparisons.empty:
        primary_id = str(((prereg.get("primary_comparison") or {}).get("id", "P1")))
        hit = comparisons[comparisons["comparison_id"] == primary_id]
        if not hit.empty:
            row = hit.iloc[0]
            verdict = str(row["verdict"])
            phrase = (
                NO_DIFFERENCE_PHRASE
                if verdict == "no_detectable_difference"
                else f"a difference favouring {row['label_a'] if verdict == 'favours_a' else row['label_b']}"
            )
            claims.append(
                _claim(
                    f"CLAIM.PRIMARY.{primary_id}",
                    (
                        f"Pre-registered primary comparison {primary_id}: "
                        f"{row['label_a']} vs {row['label_b']} on {row['scenario']} shows {phrase}."
                    ),
                    "results/tables/comparisons.csv",
                    "mean_difference",
                    row["mean_difference"],
                    row["ci_low"],
                    row["ci_high"],
                    row["p_value"],
                    row["holm_adjusted_p"],
                    verdict,
                )
            )
        for _, row in comparisons.iterrows():
            claims.append(
                _claim(
                    f"CLAIM.CMP.{row['comparison_id']}",
                    str(row["description"]),
                    "results/tables/comparisons.csv",
                    "mean_difference",
                    row["mean_difference"],
                    row["ci_low"],
                    row["ci_high"],
                    row["p_value"],
                    row["holm_adjusted_p"],
                    str(row["verdict"]),
                )
            )

    ranking = tables.get("overall_ranking")
    if ranking is not None and not ranking.empty:
        for _, row in ranking.iterrows():
            claims.append(
                _claim(
                    f"CLAIM.RANK.{row['model']}",
                    (
                        f"{row['model']} ranks {int(row['rank'])} of {len(ranking)} by "
                        f"{PRIMARY_METRIC} pooled over {int(row['n_scenarios'])} scenarios."
                    ),
                    "results/tables/overall_ranking.csv",
                    PRIMARY_METRIC,
                    row[PRIMARY_METRIC],
                    row.get(f"{PRIMARY_METRIC}_ci_low"),
                    row.get(f"{PRIMARY_METRIC}_ci_high"),
                    verdict="descriptive",
                )
            )

    budget = tables.get("parameter_budget")
    if budget is not None and not budget.empty and "parameter_count" in budget.columns:
        neural = budget[budget["parameter_count"] > 0]
        if not neural.empty:
            spread = float(neural["parameter_count"].max() / neural["parameter_count"].min())
            claims.append(
                _claim(
                    "CLAIM.BUDGET.spread",
                    (
                        "Ratio of the largest to the smallest trainable parameter count across "
                        "the registry, the capacity-matching invariant."
                    ),
                    "results/tables/parameter_budget.csv",
                    "parameter_count",
                    spread,
                    verdict="descriptive",
                )
            )

    fed = tables.get("federated_table")
    if fed is not None and not fed.empty and "cloud_megabytes" in fed.columns:
        for _, row in fed.iterrows():
            claims.append(
                _claim(
                    f"CLAIM.FED.cloud_mb.{row['arm']}",
                    f"Wide-area (cloud) traffic measured for the {row['arm']} arm.",
                    "results/tables/federated_table.csv",
                    "cloud_megabytes",
                    row["cloud_megabytes"],
                    row.get("cloud_megabytes_ci_low"),
                    row.get("cloud_megabytes_ci_high"),
                    verdict="descriptive",
                )
            )

    learning = tables.get("learning_check")
    if learning is not None and not learning.empty:
        tested = learning[learning["trainable"] == True]  # noqa: E712
        if not tested.empty:
            n_learned = int((tested["learned"] == True).sum())  # noqa: E712
            claims.append(
                _claim(
                    "CLAIM.LEARN.count",
                    (
                        f"{n_learned} of {len(tested)} trainable (scenario, model) cells improved "
                        "significantly over their own initialisation after Holm correction."
                    ),
                    "results/tables/learning_check.csv",
                    "learned",
                    float(n_learned),
                    verdict="descriptive",
                )
            )

    seen: set[str] = set()
    for claim in claims:
        if claim["claim_id"] in seen:
            raise ValueError(f"duplicate claim_id {claim['claim_id']!r}")
        seen.add(claim["claim_id"])
        if set(claim) != set(_CLAIM_FIELDS):
            raise ValueError(f"claim {claim['claim_id']!r} has a non-conforming schema")
    return claims
