"""The evidence-curation regression test (defect D4).

The previous analysis script defined::

    CORE_MODELS = (...)          # random_policy deliberately absent

and applied it at line 74 to ``paper_table_main.csv``,
``overall_model_ranking.csv``, ``ablation_ltc_lfm.csv`` and
``scenario_winners.csv`` -- exactly the four tables the README cited.  Restored,
``random_policy`` ranked 4th of 7 at -67.007, ahead of ``ppo_lstm`` (-67.398),
``ppo_gru`` (-68.254) and ``ppo_ltc`` (-68.395).  The baseline had genuinely been
run; it was filtered out of the presentation.

This file makes that impossible in two independent ways: structurally (no such
list may exist in the analysis layer) and behaviourally (every generated table
must still contain every policy that ran).
"""

from __future__ import annotations

import ast
import pathlib
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from dsa.analysis.tables import build_all_tables
from dsa.envs.heuristics import BASELINE_KEYS
from dsa.models.registry import MODEL_REGISTRY, PRIMARY_MODELS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYSIS_SOURCES = sorted((REPO_ROOT / "dsa" / "analysis").glob("*.py")) + sorted(
    (REPO_ROOT / "scripts").glob("*.py")
)
ALL_POLICY_KEYS = set(MODEL_REGISTRY) | set(BASELINE_KEYS)


# --------------------------------------------------------------------------- #
# Structural: no hardcoded policy list may exist in the analysis layer
# --------------------------------------------------------------------------- #


def test_analysis_sources_exist():
    assert ANALYSIS_SOURCES, "no analysis sources found; the test would vacuously pass"


BANNED_NAMES = {"CORE_MODELS", "MAIN_MODELS", "PAPER_MODELS", "SHOWN_MODELS", "KEEP_MODELS"}


def test_no_core_models_style_constant():
    """No *identifier* that looks like a curated policy allowlist.

    Matched on the parsed syntax tree, not on the raw text, so that prose in a
    docstring explaining why such a constant is forbidden does not trip the
    check -- while an actual binding or reference does.
    """
    offenders: list[str] = []
    for path in ANALYSIS_SOURCES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: .{node.attr}")
    assert not offenders, "curated model allowlist found:\n" + "\n".join(offenders)


def test_no_isin_filtering_in_the_analysis_layer():
    """``DataFrame.isin`` is how the previous curation was applied."""
    offenders = []
    for path in ANALYSIS_SOURCES:
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if ".isin(" in stripped:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}: {stripped}")
    assert not offenders, "isin() filtering found:\n" + "\n".join(offenders)


def test_no_literal_list_of_three_or_more_registry_keys():
    """A list literal naming three or more policies is a filter waiting to happen.

    Parsed with ``ast`` rather than grepped, so a multi-line literal cannot hide.
    """
    offenders: list[str] = []
    for path in ANALYSIS_SOURCES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                continue
            names = [
                e.value
                for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            hits = [n for n in names if n in ALL_POLICY_KEYS]
            if len(hits) >= 3:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: literal naming {sorted(hits)}"
                )
    assert not offenders, "hardcoded policy list found:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# Behavioural: every table keeps every policy
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_runs() -> pd.DataFrame:
    """A complete matrix over every registry key and every baseline.

    Values are synthetic; the property under test is which ROWS survive into the
    tables, which does not depend on the numbers being real.
    """
    rng = np.random.default_rng(0)
    scenarios = ["stationary", "irregular_dt"]
    seeds = [7, 19, 23, 41]
    rows = []
    for scenario in scenarios:
        for model in sorted(ALL_POLICY_KEYS):
            trainable = model in MODEL_REGISTRY
            for seed in seeds:
                trained = float(rng.normal(-50, 5))
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "seed": seed,
                        "is_trainable": trainable,
                        "policy_type": "neural" if trainable else "heuristic",
                        "family": "liquid" if trainable else "heuristic",
                        "dt_aware": trainable,
                        "cell_kinds": "x+y" if trainable else "",
                        "hidden_dim": 55 if trainable else 0,
                        "num_heads": 0,
                        "parameter_count": 39_994 if trainable else 0,
                        "param_error_frac": 0.0,
                        "mean_eval_return": trained,
                        "mean_eval_return_trained": trained,
                        "mean_eval_return_untrained": trained - float(rng.normal(1, 1)),
                        "mean_success_rate": float(rng.uniform(0.2, 0.6)),
                        "mean_collision_rate": float(rng.uniform(0.2, 0.6)),
                        "mean_spectrum_utilization": float(rng.uniform(0.2, 0.6)),
                        "mean_policy_entropy_nats": float(rng.uniform(1.5, 2.0)),
                        "mean_action_histogram_entropy": float(rng.uniform(1.5, 2.0)),
                        "max_first_epoch_ratio_deviation": 1e-7,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def prereg() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "preregistration.yaml").read_text())


def test_every_table_with_a_model_column_keeps_every_policy(synthetic_runs, prereg, tmp_path):
    tables = build_all_tables(synthetic_runs, pd.DataFrame(), prereg, tmp_path)
    expected = set(synthetic_runs["model"].unique())
    assert expected >= set(BASELINE_KEYS) | set(PRIMARY_MODELS)

    checked = 0
    for path in sorted(tmp_path.glob("*.csv")):
        table = pd.read_csv(path)
        if "model" not in table.columns or table.empty:
            continue
        missing = expected - set(table["model"].astype(str).unique())
        assert not missing, f"{path.name} dropped policies: {sorted(missing)}"
        checked += 1
    assert checked >= 3, "expected several model-indexed tables to verify"
    assert set(tables) >= {"main_table", "overall_ranking", "learning_check", "parameter_budget"}


def test_baselines_are_ranked_alongside_models_not_appended(synthetic_runs, prereg, tmp_path):
    """Baselines compete in the same ranking, not in a separate section."""
    build_all_tables(synthetic_runs, pd.DataFrame(), prereg, tmp_path)
    ranking = pd.read_csv(tmp_path / "overall_ranking.csv")
    for key in BASELINE_KEYS:
        assert key in set(ranking["model"]), f"{key} absent from overall_ranking.csv"
    assert ranking["rank"].is_monotonic_increasing
    assert ranking["rank"].nunique() > 1


def test_a_dominant_baseline_can_reach_rank_one(prereg, tmp_path):
    """If a zero-parameter baseline is best, the table must say so.

    This is the exact situation the previous filter concealed.
    """
    rows = []
    for seed in [7, 19, 23, 41]:
        rows.append(_row("stationary", "random_policy", seed, 100.0, False))
        for model in PRIMARY_MODELS:
            rows.append(_row("stationary", model, seed, -50.0, True))
    build_all_tables(pd.DataFrame(rows), pd.DataFrame(), prereg, tmp_path)
    ranking = pd.read_csv(tmp_path / "overall_ranking.csv")
    assert ranking.iloc[0]["model"] == "random_policy"
    assert int(ranking.iloc[0]["rank"]) == 1


def _row(scenario: str, model: str, seed: int, value: float, trainable: bool) -> dict:
    return {
        "scenario": scenario,
        "model": model,
        "seed": seed,
        "is_trainable": trainable,
        "parameter_count": 39_994 if trainable else 0,
        "mean_eval_return": value,
        "mean_eval_return_trained": value,
        "mean_eval_return_untrained": value,
    }


def test_every_aggregate_metric_ships_with_uncertainty(synthetic_runs, prereg, tmp_path):
    """A bare mean is not a result; every metric column needs its interval."""
    build_all_tables(synthetic_runs, pd.DataFrame(), prereg, tmp_path)
    main = pd.read_csv(tmp_path / "main_table.csv")
    for metric in ("mean_eval_return", "mean_success_rate"):
        for suffix in ("_std", "_sem", "_ci_low", "_ci_high"):
            assert f"{metric}{suffix}" in main.columns, f"{metric} has no {suffix} column"


def test_comparisons_table_reports_every_required_uncertainty_field(synthetic_runs, prereg, tmp_path):
    build_all_tables(synthetic_runs, pd.DataFrame(), prereg, tmp_path)
    comparisons = pd.read_csv(tmp_path / "comparisons.csv")
    assert not comparisons.empty
    for column in ("n_pairs", "ci_low", "ci_high", "p_value", "p_method", "holm_adjusted_p", "verdict"):
        assert column in comparisons.columns


def test_tables_are_written_only_inside_the_requested_output_dir(synthetic_runs, prereg, tmp_path):
    """The analysis layer has no default write path into tracked evidence."""
    target = tmp_path / "nested" / "out"
    build_all_tables(synthetic_runs, pd.DataFrame(), prereg, target)
    assert sorted(p.name for p in target.glob("*.csv"))
    assert not list(tmp_path.glob("*.csv"))


# --------------------------------------------------------------------------- #
# Claim provenance
# --------------------------------------------------------------------------- #


def test_every_claim_resolves_to_a_real_file_and_column(synthetic_runs, prereg, tmp_path):
    """The auditor's workflow, as a test.

    Each entry in ``claims.json`` names a source CSV and a source column. Both
    must exist, or the claim cannot be re-derived and is not evidence.
    """
    from dsa.analysis.tables import build_claims

    tables = build_all_tables(synthetic_runs, pd.DataFrame(), prereg, tmp_path)
    claims = build_claims(tables, prereg)
    assert claims, "no claims produced"

    for claim in claims:
        name = pathlib.Path(claim["source_file"]).stem
        assert name in tables, f"{claim['claim_id']} cites unknown table {claim['source_file']}"
        table = tables[name]
        assert claim["source_column"] in table.columns, (
            f"{claim['claim_id']} cites column {claim['source_column']!r} "
            f"absent from {claim['source_file']}"
        )


def test_claim_schema_is_strict_and_ids_are_unique(synthetic_runs, prereg, tmp_path):
    from dsa.analysis.tables import build_claims

    tables = build_all_tables(synthetic_runs, pd.DataFrame(), prereg, tmp_path)
    claims = build_claims(tables, prereg)

    required = {
        "claim_id", "statement", "source_file", "source_column", "value",
        "ci_low", "ci_high", "p_value", "holm_adjusted_p", "verdict",
    }
    ids = set()
    for claim in claims:
        assert set(claim) == required, f"{claim['claim_id']} has fields {sorted(claim)}"
        assert claim["claim_id"] not in ids, f"duplicate claim_id {claim['claim_id']}"
        ids.add(claim["claim_id"])
        assert isinstance(claim["statement"], str) and claim["statement"].strip()
        assert claim["verdict"] in {
            "descriptive", "no_detectable_difference", "favours_a", "favours_b",
        }


def test_a_comparison_claim_carries_its_uncertainty(synthetic_runs, prereg, tmp_path):
    """Descriptive claims may omit p; comparison claims may not omit the interval."""
    from dsa.analysis.tables import build_claims

    tables = build_all_tables(synthetic_runs, pd.DataFrame(), prereg, tmp_path)
    comparison_claims = [c for c in build_claims(tables, prereg) if c["claim_id"].startswith("CLAIM.CMP.")]
    assert comparison_claims
    for claim in comparison_claims:
        assert claim["ci_low"] is not None and claim["ci_high"] is not None
        assert claim["p_value"] is not None
        assert claim["ci_low"] <= claim["ci_high"]


def test_no_detectable_difference_claims_never_use_gain_language(synthetic_runs, prereg, tmp_path):
    from dsa.analysis.tables import build_claims

    tables = build_all_tables(synthetic_runs, pd.DataFrame(), prereg, tmp_path)
    for claim in build_claims(tables, prereg):
        if claim["verdict"] != "no_detectable_difference":
            continue
        lowered = claim["statement"].lower()
        for forbidden in ("gain", "improvement", "outperform", "wins", "beats"):
            assert forbidden not in lowered, f"{claim['claim_id']}: {claim['statement']}"
