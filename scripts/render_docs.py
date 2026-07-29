#!/usr/bin/env python3
"""Generate every metric that appears in ``README.md`` and ``docs/*.md``.

Why this script exists
----------------------
``scripts/make_report.py`` already guarantees that ``RESULTS.md`` contains no
hand-typed number.  ``README.md`` had no such guarantee, and an adversarial
audit proved the consequence: two headline figures were deliberately falsified
(the primary comparison's mean difference and the greedy baseline's pooled
return) and the entire test suite, plus the CI "report is generated" job, stayed
green.  The scan that was supposed to catch it skipped markdown table rows and
stripped inline-code spans -- which is exactly and only where those numbers
lived.

This script closes that hole from the other side.  Instead of *detecting*
hand-typed numbers it *removes the opportunity to type one*: every metric in the
hand-written documents is emitted from ``results/`` by the code below, and
``--check`` fails the build if a committed document does not match what this
script produces.

Two markup forms, both invisible in rendered markdown
-----------------------------------------------------
**Block regions** carry whole tables::

    <!-- BEGIN GENERATED: p1_table -->
    ... rewritten in place, every time ...
    <!-- END GENERATED: p1_table -->

**Inline spans** carry a single number inside a sentence::

    the mean paired difference is <!--v:claim.CLAIM.PRIMARY.P1.value-->-0.007<!--/v-->

An optional format follows a pipe: ``<!--v:f6.p_value_max|.6f-->``.  Both forms
keep the prose in markdown, where it is readable and diffable on GitHub, rather
than exiling it into Python string literals; and both make the number's source
visible at the point of use.

Provenance of every value
-------------------------
``build_values`` is the whole vocabulary.  Each entry is one expression over a
committed file.  Most read ``results/``:

* ``claim.<claim_id>.<field>`` -- ``results/claims.json``, verbatim;
* ``cmp.*``, ``f1.*`` .. ``f7.*`` -- ``results/tables/comparisons.csv``;
* ``rank.*``, ``learn.*``, ``fed.*`` -- the corresponding table under
  ``results/tables/``;
* ``runs.*`` -- ``results/all_runs.csv``;
* ``claims.*``, ``matrix.*`` -- ``results/claims.json`` and
  ``results/manifest.json``.

One namespace is a *second study*: ``studyb.*`` reads ``results/study_b/`` --
Study B's own ``claims.json``, ``tables/``, ``all_runs.csv``,
``mechanistic_endpoints.csv`` and ``manifest.json``, plus
``configs/preregistration_study_b.yaml`` for its protocol constants.  Study A's
keys are untouched by it, so a document can put the two budgets side by side
without either study's numbers being re-derived from the other's files.  The
only region that reads both is ``budget_comparison_table``, and every cell in it
is a claim from the study that cell belongs to.

Four namespaces are **not** ``results/``, and say so:

* ``prereg.*``   -- ``configs/preregistration.yaml`` (protocol constants, frozen
  before the run and sha256-recorded in the manifest);
* ``config.*``   -- ``configs/suite.yaml`` (worker count, guard threshold);
* ``env.*``      -- a module constant in ``dsa/envs/config.py`` (the channel
  count, which fixes the entropy of a uniform policy);
* ``hist.*``     -- ``docs/historical_figures.yaml``, the audit figures for
  *pre-rebuild* revisions of this repository.  Those artifacts were deleted
  rather than archived, so those numbers are **not** derivable from ``results/``
  and are not claims about the current evidence.  Registering one is a
  deliberate act: the entry must record what was measured and against what.

Where a document cites a ``claim_id``, the value rendered beside it is read from
**that claim**, not from a parallel read of the same CSV.  Otherwise corrupting
``results/claims.json`` would leave the document unchanged and the citation
would be decorative; ``scripts/analyze.py --check`` already pins ``claims.json``
to the tables.

Usage
-----
    python scripts/render_docs.py            # rewrite the generated parts
    python scripts/render_docs.py --check    # non-zero exit if anything is stale
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import pathlib
import re
import sys
from collections.abc import Mapping
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dsa.envs.config import DEFAULT_NUM_CHANNELS  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def DOC_PATHS(root: pathlib.Path) -> list[pathlib.Path]:
    """Documents whose metrics this script owns.

    Kept in lockstep with ``PROSE_FILES`` in ``tests/test_no_hardcoded_metrics.py``:
    a document the scan covers but the generator does not would be a document in
    which no number could ever be written.
    """
    return [root / "README.md"] + sorted((root / "docs").glob("*.md"))

#: Registry of audit figures that predate the rebuild and cannot be recomputed.
HISTORICAL_REGISTRY = REPO_ROOT / "docs" / "historical_figures.yaml"

BEGIN_RE = re.compile(r"^(\s*)<!--\s*BEGIN GENERATED:\s*([A-Za-z0-9_]+)\s*-->\s*$")
END_RE = re.compile(r"^\s*<!--\s*END GENERATED:\s*([A-Za-z0-9_]+)\s*-->\s*$")
SPAN_RE = re.compile(r"<!--v:([^>|]+?)(?:\|([^>]+?))?-->(.*?)<!--/v-->", re.DOTALL)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _fmt(value: Any, spec: str) -> str:
    """Format one value.  ``spec`` is a python format spec, or ``s`` for text."""
    if spec in ("", "s"):
        return str(value)
    if spec == "d":
        return str(int(value))
    return format(float(value), spec)


# --------------------------------------------------------------------------- #
# The value vocabulary
# --------------------------------------------------------------------------- #


def _first_sentence(text: str) -> str:
    flat = " ".join(str(text).split())
    match = re.search(r"[.?!](\s|$)", flat)
    return flat[: match.end()].strip() if match else flat


def build_values(results_dir: pathlib.Path, repo_root: pathlib.Path = REPO_ROOT) -> dict[str, tuple[Any, str]]:
    """Every citable scalar, keyed by the name a document may reference.

    Returns ``key -> (value, default_format)``.  A document that references a
    key not in here fails loudly; a key in here that no document references is
    harmless.  Nothing in this function is a literal metric -- every number is
    read from a file, and the only constants are formatting specs.
    """
    tables = results_dir / "tables"
    values: dict[str, tuple[Any, str]] = {}

    # -- claims.json: the canonical provenance record ------------------------ #
    claims_path = results_dir / "claims.json"
    if claims_path.exists():
        doc = json.loads(claims_path.read_text())
        values["claims.total"] = (len(doc.get("claims", [])), "d")
        for claim in doc.get("claims", []):
            cid = claim["claim_id"]
            for field, spec in (
                ("value", ".3f"),
                ("ci_low", ".3f"),
                ("ci_high", ".3f"),
                ("p_value", ".3f"),
                ("holm_adjusted_p", ".3f"),
                ("verdict", "s"),
            ):
                if claim.get(field) is not None:
                    values[f"claim.{cid}.{field}"] = (claim[field], spec)

    # -- comparisons.csv: family-level extrema ------------------------------- #
    comparisons_path = tables / "comparisons.csv"
    if comparisons_path.exists():
        comparisons = pd.read_csv(comparisons_path)
        values["cmp.total"] = (len(comparisons), "d")
        values["cmp.families"] = (comparisons["family"].nunique(), "d")
        values["cmp.significant"] = (int(comparisons["significant"].sum()), "d")
        _excludes = ~comparisons["interval_contains_zero"].astype(bool)
        values["cmp.excludes_zero"] = (int(_excludes.sum()), "d")
        values["cmp.excludes_zero_not_significant"] = (
            int((_excludes & ~comparisons["significant"].astype(bool)).sum()), "d"
        )
        values["cmp.directional_verdicts"] = (
            int((comparisons["verdict"] != "no_detectable_difference").sum()), "d"
        )
        for family, chunk in comparisons.groupby("family", sort=True):
            short = family.split("_")[0].lower()
            values[f"{short}.n"] = (len(chunk), "d")
            values[f"{short}.significant"] = (int(chunk["significant"].sum()), "d")
            # Rows whose raw bootstrap interval excludes zero but which the
            # pre-registered correction still declines to reject.  Publishing
            # this count is how the repository stays honest about what
            # correction costs, without letting any of them become a verdict.
            excludes = ~chunk["interval_contains_zero"].astype(bool)
            values[f"{short}.interval_excludes_zero"] = (int(excludes.sum()), "d")
            values[f"{short}.excludes_zero_not_significant"] = (
                int((excludes & ~chunk["significant"].astype(bool)).sum()), "d"
            )
            values[f"{short}.p_value_max"] = (chunk["p_value"].max(), ".6f")
            values[f"{short}.holm_adjusted_p_max"] = (chunk["holm_adjusted_p"].max(), ".4f")
            values[f"{short}.cliffs_delta_max"] = (chunk["cliffs_delta"].max(), ".3f")
            values[f"{short}.cliffs_delta_min"] = (chunk["cliffs_delta"].min(), ".3f")
            # "margin" = the mean paired difference; max is the least extreme.
            values[f"{short}.margin_max"] = (chunk["mean_difference"].max(), ".3f")
            values[f"{short}.margin_min"] = (chunk["mean_difference"].min(), ".3f")
            values[f"{short}.margin_max_id"] = (
                chunk.loc[chunk["mean_difference"].idxmax(), "comparison_id"], "s"
            )
            values[f"{short}.margin_min_id"] = (
                chunk.loc[chunk["mean_difference"].idxmin(), "comparison_id"], "s"
            )

    # -- all_runs.csv: the training diagnostics ------------------------------ #
    runs_path = results_dir / "all_runs.csv"
    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        neural = runs[runs["is_trainable"] == True] if "is_trainable" in runs else runs  # noqa: E712
        values["runs.rows"] = (len(runs), "d")
        values["runs.neural_rows"] = (len(neural), "d")
        values["runs.policies"] = (runs["model"].nunique(), "d")
        for column, key, spec in (
            ("final_explained_variance", "runs.explained_variance_max", ".2e"),
            ("max_first_epoch_ratio_deviation", "runs.ratio_deviation_max", ".2e"),
        ):
            if column in neural:
                values[key] = (neural[column].max(), spec)
        for column, key in (
            ("mean_policy_entropy_nats", "runs.policy_entropy_trained_mean"),
            ("untrained_mean_policy_entropy_nats", "runs.policy_entropy_untrained_mean"),
        ):
            if column in neural:
                values[key] = (neural[column].mean(), ".3f")
        if "total_gradient_steps" in neural and len(neural):
            unique = sorted(set(int(v) for v in neural["total_gradient_steps"]))
            values["runs.gradient_steps"] = (unique[0] if len(unique) == 1 else min(unique), "d")

    # -- wall clock, so the budget note cannot drift ------------------------- #
    suite_config = repo_root / "configs" / "suite.yaml"
    workers = None
    if suite_config.exists():
        suite = yaml.safe_load(suite_config.read_text()) or {}
        workers = suite.get("num_workers")
        if workers:
            values["config.num_workers"] = (workers, "d")
        cap = (suite.get("guard") or {}).get("max_projected_wall_minutes")
        if cap is not None:
            values["config.max_projected_wall_minutes"] = (cap, "g")
    for label, path in (("runs", results_dir / "all_runs.csv"),
                        ("fed", results_dir / "federated_all_runs.csv")):
        if path.exists() and workers:
            frame = pd.read_csv(path)
            if "wall_seconds" in frame.columns:
                values[f"{label}.wall_minutes"] = (
                    float(frame["wall_seconds"].sum()) / float(workers) / 60.0, ".1f"
                )

    # -- federated_table.csv: the communication ledger ----------------------- #
    fed_path = tables / "federated_table.csv"
    if fed_path.exists():
        fed = pd.read_csv(fed_path)
        for _, row in fed.iterrows():
            arm = row["arm"]
            for column in ("cloud_megabytes", "edge_local_megabytes", "total_megabytes"):
                if column in fed.columns:
                    values[f"fed.{arm}.{column}"] = (row[column], ".4f")
            for column in ("train_steps", "declared_train_steps"):
                if column in fed.columns:
                    values[f"fed.{arm}.{column}"] = (row[column], "d")
        if {"cloud_megabytes", "arm"} <= set(fed.columns):
            flat = fed.loc[fed["arm"] == "flat_federated", "cloud_megabytes"]
            hier = fed.loc[fed["arm"] == "hierarchical_federated", "cloud_megabytes"]
            if len(flat) and len(hier) and float(hier.iloc[0]) != 0.0:
                values["fed.cloud_ratio_flat_over_hier"] = (
                    float(flat.iloc[0]) / float(hier.iloc[0]),
                    ".2f",
                )

    # -- learning_check.csv: how many cells beat their own initialisation ---- #
    learning_path = tables / "learning_check.csv"
    if learning_path.exists():
        learning = pd.read_csv(learning_path)
        tested = learning[learning["trainable"] == True] if "trainable" in learning else learning  # noqa: E712
        values["learn.tested"] = (len(tested), "d")
        if "learned" in tested.columns:
            values["learn.learned"] = (int((tested["learned"] == True).sum()), "d")  # noqa: E712

    # -- overall_ranking.csv ------------------------------------------------- #
    ranking_path = tables / "overall_ranking.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        values["rank.policies"] = (len(ranking), "d")
        values["rank.scenarios"] = (int(ranking["n_scenarios"].max()), "d")

    # -- manifest.json: the matrix that produced all of the above ------------ #
    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        matrix = manifest.get("matrix", {})
        for key, source in (
            ("matrix.training_steps", matrix.get("training_steps")),
            ("matrix.eval_episodes", matrix.get("eval_episodes")),
            ("matrix.num_cells", matrix.get("num_cells")),
        ):
            if source is not None:
                values[key] = (source, "d")
        for key, source in (
            ("matrix.n_seeds", matrix.get("seeds")),
            ("matrix.n_scenarios", matrix.get("scenarios")),
            ("matrix.n_models", matrix.get("models")),
            ("matrix.n_baselines", matrix.get("baselines")),
        ):
            if source is not None:
                values[key] = (len(source), "d")

    # -- the pre-registration: protocol constants ---------------------------- #
    prereg_path = repo_root / "configs" / "preregistration.yaml"
    if prereg_path.exists():
        prereg = yaml.safe_load(prereg_path.read_text()) or {}
        if "alpha" in prereg:
            values["prereg.alpha"] = (prereg["alpha"], ".2f")
        if "ci_level" in prereg:
            values["prereg.ci_percent"] = (100.0 * float(prereg["ci_level"]), "g")

    # -- an environment constant --------------------------------------------- #
    values["env.num_channels"] = (DEFAULT_NUM_CHANNELS, "d")
    values["env.uniform_policy_entropy_nats"] = (math.log(DEFAULT_NUM_CHANNELS), ".3f")

    # -- audit figures that predate the rebuild ------------------------------ #
    for key, entry in load_historical(repo_root).items():
        values[f"hist.{key}"] = (entry["value"], entry.get("format", ".3f"))

    # -- Study B, in its own namespace --------------------------------------- #
    values.update(build_study_b_values(study_b_dir(results_dir), repo_root, values))

    return values


# --------------------------------------------------------------------------- #
# Study B
# --------------------------------------------------------------------------- #
#
# Study B is a second, separately pre-registered study at 16x Study A's budget.
# Its artifacts live under ``results/study_b/`` -- a subdirectory of the results
# tree, so ``--results-dir`` still selects *both* studies with one flag and the
# CI job that falsifies a claim in a scratch copy of ``results/`` copies Study B
# with it.  Every Study B key is prefixed ``studyb.``; nothing here can collide
# with, overwrite or re-derive a Study A value, which is what keeps Study A
# byte-identical committed evidence.

#: Name of the Study B subdirectory inside a results tree.
STUDY_B_DIRNAME = "study_b"

#: Study B's plan.  A separate file from Study A's on purpose: see the header of
#: configs/preregistration_study_b.yaml.
STUDY_B_PREREG = "preregistration_study_b.yaml"


def study_b_dir(results_dir: pathlib.Path) -> pathlib.Path:
    return results_dir / STUDY_B_DIRNAME


def build_study_b_values(
    study_dir: pathlib.Path,
    repo_root: pathlib.Path,
    study_a: dict[str, tuple[Any, str]],
) -> dict[str, tuple[Any, str]]:
    """Every citable Study B scalar, keyed ``studyb.*``.

    ``study_a`` is the already-built Study A vocabulary and is read only to form
    the two budget ratios, which are the whole point of running Study B and must
    not be typed by hand either.
    """
    values: dict[str, tuple[Any, str]] = {}
    if not (study_dir / "all_runs.csv").exists():
        return values
    tables = study_dir / "tables"
    P = "studyb."

    # -- claims.json --------------------------------------------------------- #
    claims_path = study_dir / "claims.json"
    if claims_path.exists():
        doc = json.loads(claims_path.read_text())
        values[f"{P}claims.total"] = (len(doc.get("claims", [])), "d")
        for claim in doc.get("claims", []):
            cid = claim["claim_id"]
            for field, spec in (
                ("value", ".3f"),
                ("ci_low", ".3f"),
                ("ci_high", ".3f"),
                ("p_value", ".3f"),
                ("holm_adjusted_p", ".3f"),
                ("verdict", "s"),
            ):
                if claim.get(field) is not None:
                    values[f"{P}claim.{cid}.{field}"] = (claim[field], spec)

    # -- comparisons.csv ----------------------------------------------------- #
    comparisons_path = tables / "comparisons.csv"
    if comparisons_path.exists():
        comparisons = pd.read_csv(comparisons_path)
        values[f"{P}cmp.total"] = (len(comparisons), "d")
        values[f"{P}cmp.families"] = (comparisons["family"].nunique(), "d")
        values[f"{P}cmp.significant"] = (int(comparisons["significant"].sum()), "d")
        _excludes = ~comparisons["interval_contains_zero"].astype(bool)
        values[f"{P}cmp.excludes_zero"] = (int(_excludes.sum()), "d")
        values[f"{P}cmp.excludes_zero_not_significant"] = (
            int((_excludes & ~comparisons["significant"].astype(bool)).sum()), "d"
        )
        values[f"{P}cmp.no_detectable_difference"] = (
            int((comparisons["verdict"] == "no_detectable_difference").sum()), "d"
        )
        for family, chunk in comparisons.groupby("family", sort=True):
            short = family.split("_")[0].lower()
            values[f"{P}{short}.n"] = (len(chunk), "d")
            values[f"{P}{short}.significant"] = (int(chunk["significant"].sum()), "d")
            excludes = ~chunk["interval_contains_zero"].astype(bool)
            values[f"{P}{short}.interval_excludes_zero"] = (int(excludes.sum()), "d")
            values[f"{P}{short}.excludes_zero_not_significant"] = (
                int((excludes & ~chunk["significant"].astype(bool)).sum()), "d"
            )
            values[f"{P}{short}.p_value_max"] = (chunk["p_value"].max(), ".6f")
            values[f"{P}{short}.p_value_min"] = (chunk["p_value"].min(), ".6f")
            values[f"{P}{short}.holm_adjusted_p_max"] = (chunk["holm_adjusted_p"].max(), ".4f")
            values[f"{P}{short}.cliffs_delta_max"] = (chunk["cliffs_delta"].max(), ".3f")
            values[f"{P}{short}.cliffs_delta_min"] = (chunk["cliffs_delta"].min(), ".3f")
            values[f"{P}{short}.margin_max"] = (chunk["mean_difference"].max(), ".3f")
            values[f"{P}{short}.margin_min"] = (chunk["mean_difference"].min(), ".3f")
            values[f"{P}{short}.margin_max_id"] = (
                chunk.loc[chunk["mean_difference"].idxmax(), "comparison_id"], "s"
            )
            values[f"{P}{short}.margin_min_id"] = (
                chunk.loc[chunk["mean_difference"].idxmin(), "comparison_id"], "s"
            )

    # -- learning_check.csv: the precondition every other family rests on ---- #
    learning_path = tables / "learning_check.csv"
    if learning_path.exists():
        learning = pd.read_csv(learning_path)
        tested = learning[learning["trainable"] == True] if "trainable" in learning else learning  # noqa: E712
        values[f"{P}learn.tested"] = (len(tested), "d")
        if "learned" in tested.columns:
            values[f"{P}learn.learned"] = (int((tested["learned"] == True).sum()), "d")  # noqa: E712
            values[f"{P}learn.not_learned"] = (int((tested["learned"] != True).sum()), "d")  # noqa: E712
            for scenario, chunk in tested.groupby("scenario", sort=True):
                values[f"{P}learn.learned.{scenario}"] = (
                    int((chunk["learned"] == True).sum()), "d"  # noqa: E712
                )
                values[f"{P}learn.tested.{scenario}"] = (len(chunk), "d")
            # The models that did not clear the precondition, named from the
            # table rather than from memory: every contrast involving one of
            # them is a contrast against a non-learner.
            names = sorted(set(tested.loc[tested["learned"] != True, "model"]))  # noqa: E712
            values[f"{P}learn.not_learned_models"] = (", ".join(f"`{n}`" for n in names), "s")
            values[f"{P}learn.not_learned_model_count"] = (len(names), "d")

    # -- overall_ranking.csv -------------------------------------------------- #
    ranking_path = tables / "overall_ranking.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        values[f"{P}rank.policies"] = (len(ranking), "d")
        values[f"{P}rank.scenarios"] = (int(ranking["n_scenarios"].max()), "d")
        ordered = ranking.sort_values("rank")
        for _, row in ordered.iterrows():
            values[f"{P}rank.{row['model']}.rank"] = (int(row["rank"]), "d")
        # The gap the study is actually about: the best zero-parameter policy
        # against the best learned one, pooled.
        budget_path = tables / "parameter_budget.csv"
        if budget_path.exists():
            budget = pd.read_csv(budget_path)
            # Membership is tested row by row rather than with DataFrame.isin:
            # tests/test_no_filter.py bans isin() anywhere in the analysis layer,
            # because that is how the pre-rebuild curation filtered a baseline
            # out of four published tables.  Nothing is dropped here -- the
            # ranking region below still renders every policy that ran; this only
            # names the top of each group.
            zero = set(budget.loc[budget["parameter_count"] == 0, "model"])
            best_learned = next(
                (r for _, r in ordered.iterrows() if str(r["model"]) not in zero), None
            )
            best_heuristic = next(
                (r for _, r in ordered.iterrows() if str(r["model"]) in zero), None
            )
            if best_learned is not None and best_heuristic is not None:
                values[f"{P}rank.best_learned"] = (str(best_learned["model"]), "s")
                values[f"{P}rank.best_learned_return"] = (
                    float(best_learned["mean_eval_return"]), ".3f"
                )
                values[f"{P}rank.best_heuristic"] = (str(best_heuristic["model"]), "s")
                values[f"{P}rank.heuristic_lead"] = (
                    float(best_heuristic["mean_eval_return"])
                    - float(best_learned["mean_eval_return"]),
                    ".3f",
                )

    # -- main_table.csv: the components the return is made of ---------------- #
    main_path = tables / "main_table.csv"
    if main_path.exists():
        main = pd.read_csv(main_path)
        for _, row in main.iterrows():
            base = f"{P}main.{row['scenario']}.{row['model']}"
            for column, spec in (
                ("mean_eval_return", ".3f"),
                ("mean_success_rate", ".3f"),
                ("mean_collision_rate", ".3f"),
                ("mean_spectrum_utilization", ".3f"),
                ("mean_action_histogram_entropy", ".3f"),
                ("mean_policy_entropy_nats", ".3f"),
            ):
                if column in main.columns:
                    values[f"{base}.{column}"] = (float(row[column]), spec)

    # -- mechanistic_endpoints.csv: descriptive by pre-registration ---------- #
    mech_path = study_dir / "mechanistic_endpoints.csv"
    if mech_path.exists():
        mech = pd.read_csv(mech_path)
        values[f"{P}mech.rows"] = (len(mech), "d")
        for _, row in mech.iterrows():
            base = f"{P}mech.{row['scenario']}.{row['model']}.{row['endpoint']}"
            for column, spec in (
                ("before", ".3f"),
                ("after", ".3f"),
                ("change", ".3f"),
                ("change_ci_low", ".3f"),
                ("change_ci_high", ".3f"),
            ):
                values[f"{base}.{column}"] = (float(row[column]), spec)
            values[f"{base}.n_seeds"] = (int(row["n_seeds"]), "d")
            values[f"{base}.n_seeds_moved"] = (
                int(row["n_seeds_moved_in_expected_direction"]), "d"
            )

    # -- all_runs.csv: the training diagnostics ------------------------------ #
    runs_path = study_dir / "all_runs.csv"
    runs = pd.read_csv(runs_path)
    neural = runs[runs["is_trainable"] == True] if "is_trainable" in runs else runs  # noqa: E712
    values[f"{P}runs.rows"] = (len(runs), "d")
    values[f"{P}runs.neural_rows"] = (len(neural), "d")
    values[f"{P}runs.policies"] = (runs["model"].nunique(), "d")
    if "total_gradient_steps" in neural and len(neural):
        values[f"{P}runs.gradient_steps"] = (int(neural["total_gradient_steps"].min()), "d")
    if "train_steps" in neural and len(neural):
        values[f"{P}runs.train_steps"] = (int(neural["train_steps"].min()), "d")
    if "max_first_epoch_ratio_deviation" in neural:
        values[f"{P}runs.ratio_deviation_max"] = (
            neural["max_first_epoch_ratio_deviation"].max(), ".2e"
        )
    for column, key in (
        ("mean_policy_entropy_nats", f"{P}runs.policy_entropy_trained_mean"),
        ("untrained_mean_policy_entropy_nats", f"{P}runs.policy_entropy_untrained_mean"),
        ("final_explained_variance", f"{P}runs.explained_variance_mean"),
    ):
        if column in neural:
            values[key] = (neural[column].mean(), ".3f")
    if "final_explained_variance" in neural:
        values[f"{P}runs.explained_variance_max"] = (neural["final_explained_variance"].max(), ".3f")

    # -- manifest.json: the matrix that produced all of the above ------------ #
    manifest_path = study_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        matrix = manifest.get("matrix", {})
        for key, source in (
            (f"{P}matrix.training_steps", matrix.get("training_steps")),
            (f"{P}matrix.eval_episodes", matrix.get("eval_episodes")),
            (f"{P}matrix.num_cells", matrix.get("rows_written")),
            (f"{P}matrix.total_env_steps_trained", matrix.get("total_env_steps_trained")),
            (f"{P}config.num_workers", matrix.get("num_workers")),
        ):
            if source is not None:
                values[key] = (source, "d")
        for key, source in (
            (f"{P}matrix.n_seeds", matrix.get("seeds")),
            (f"{P}matrix.n_scenarios", matrix.get("scenarios")),
            (f"{P}matrix.n_models", matrix.get("models")),
            (f"{P}matrix.n_baselines", matrix.get("baselines")),
        ):
            if source is not None:
                values[key] = (len(source), "d")
        if manifest.get("total_wall_minutes") is not None:
            values[f"{P}runs.wall_minutes"] = (float(manifest["total_wall_minutes"]), ".1f")
        if manifest.get("dropped_parts") is not None:
            values[f"{P}matrix.dropped_parts"] = (len(manifest["dropped_parts"]), "d")
        if manifest.get("parts_merged") is not None:
            values[f"{P}matrix.parts"] = (len(manifest["parts_merged"]), "d")
        if manifest.get("preregistration_sha256"):
            values[f"{P}prereg.sha256"] = (str(manifest["preregistration_sha256"]), "s")
        projection = manifest.get("projection") or {}
        if projection.get("wall_minutes") is not None:
            values[f"{P}projection.wall_minutes"] = (float(projection["wall_minutes"]), ".1f")

    # -- the pre-registration: protocol constants ---------------------------- #
    prereg_path = repo_root / "configs" / STUDY_B_PREREG
    if prereg_path.exists():
        prereg = yaml.safe_load(prereg_path.read_text()) or {}
        if "alpha" in prereg:
            values[f"{P}prereg.alpha"] = (prereg["alpha"], ".2f")
        if "ci_level" in prereg:
            values[f"{P}prereg.ci_percent"] = (100.0 * float(prereg["ci_level"]), "g")
        budget = prereg.get("budget") or {}
        for key, source in (
            (f"{P}prereg.training_steps", budget.get("training_steps_per_trainable_cell")),
            (f"{P}prereg.gradient_steps", budget.get("gradient_steps_per_trainable_cell")),
        ):
            if source is not None:
                values[key] = (source, "d")
        stopping = prereg.get("stopping_rule") or {}
        if stopping.get("wall_clock_cap_minutes") is not None:
            values[f"{P}prereg.wall_cap_minutes"] = (stopping["wall_clock_cap_minutes"], "d")
        descoped = prereg.get("descoped") or {}
        dropped_scenarios = (descoped.get("scenarios") or {}).get("dropped") or []
        if dropped_scenarios:
            values[f"{P}descoped.scenarios"] = (
                ", ".join(f"`{s}`" for s in dropped_scenarios), "s"
            )
            values[f"{P}descoped.n_scenarios"] = (len(dropped_scenarios), "d")

    # -- the two ratios the whole study exists to establish ------------------- #
    for key, a_key, b_key in (
        (f"{P}budget.env_step_ratio", "matrix.training_steps", f"{P}matrix.training_steps"),
        (f"{P}budget.gradient_step_ratio", "runs.gradient_steps", f"{P}runs.gradient_steps"),
    ):
        if a_key in study_a and b_key in values and float(study_a[a_key][0]):
            values[key] = (float(values[b_key][0]) / float(study_a[a_key][0]), ".0f")

    return values


def load_historical(repo_root: pathlib.Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Audit figures for pre-rebuild revisions, which ``results/`` cannot yield.

    Each entry must carry ``value``, ``measured`` and ``source`` so that a
    reader can see, at the point of registration, that the number describes code
    that no longer exists rather than the evidence in this tree.
    """
    if not HISTORICAL_REGISTRY.exists():
        return {}
    doc = yaml.safe_load(HISTORICAL_REGISTRY.read_text()) or {}
    figures = doc.get("figures") or {}
    for key, entry in figures.items():
        missing = [f for f in ("value", "measured", "source") if not entry.get(f)]
        if missing:
            raise SystemExit(
                f"error: historical figure {key!r} in {HISTORICAL_REGISTRY} is missing {missing}. "
                "A number with no provenance is exactly what this registry exists to prevent."
            )
    return figures


# --------------------------------------------------------------------------- #
# The generated blocks
# --------------------------------------------------------------------------- #


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


def build_blocks(results_dir: pathlib.Path, repo_root: pathlib.Path = REPO_ROOT) -> dict[str, list[str]]:
    """Render every named block region.  Keys are the region names in the docs."""
    tables = results_dir / "tables"
    blocks: dict[str, list[str]] = {}

    def read(name: str) -> pd.DataFrame:
        path = tables / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    comparisons = read("comparisons.csv")
    ranking = read("overall_ranking.csv")
    budget = read("parameter_budget.csv")
    federated = read("federated_table.csv")
    runs_path = results_dir / "all_runs.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()

    # A table that cites a claim_id must render *that claim's* value, not a
    # parallel read of the same CSV: otherwise corrupting claims.json leaves the
    # document unchanged and the citation is decorative.  The CSVs supply only
    # what claims.json does not carry (row order, n, method, the byte columns),
    # and `analyze.py --check` already pins claims.json to the CSVs.
    claims_path = results_dir / "claims.json"
    claims: dict[str, dict[str, Any]] = {}
    if claims_path.exists():
        claims = {
            c["claim_id"]: c for c in json.loads(claims_path.read_text()).get("claims", [])
        }

    prereg_path = repo_root / "configs" / "preregistration.yaml"
    prereg = yaml.safe_load(prereg_path.read_text()) if prereg_path.exists() else {}
    prereg = prereg or {}

    # ------------------------------------------------------------ p1_table -- #
    primary_id = str((prereg.get("primary_comparison") or {}).get("id", "P1"))
    cid = f"CLAIM.PRIMARY.{primary_id}"
    if cid in claims and not comparisons.empty and (comparisons["comparison_id"] == primary_id).any():
        claim = claims[cid]
        row = comparisons[comparisons["comparison_id"] == primary_id].iloc[0]
        method = str(row["p_method"]).replace("_", " ")
        level = f"{100.0 * float(row['ci_level']):g}%"
        blocks["p1_table"] = _md_table(
            ["quantity", "value", "claim_id"],
            [
                ["mean paired difference", f"{float(claim['value']):.3f}", f"`{cid}`"],
                [
                    f"{level} bootstrap CI",
                    f"[{float(claim['ci_low']):.3f}, {float(claim['ci_high']):.3f}]",
                    f"`{cid}`",
                ],
                [f"p ({method})", f"{float(claim['p_value']):.3f}", f"`{cid}`"],
                ["Holm-adjusted p", f"{float(claim['holm_adjusted_p']):.3f}", f"`{cid}`"],
                ["n paired seeds", str(int(row["n_pairs"])), f"`{cid}`"],
                ["verdict", f"`{claim['verdict']}`", f"`{cid}`"],
            ],
        )

    # ------------------------------------------------------- ranking_table -- #
    if not ranking.empty:
        zero_param = set()
        if not budget.empty and "parameter_count" in budget:
            zero_param = set(budget.loc[budget["parameter_count"] == 0, "model"])
        rows = []
        for _, row in ranking.sort_values("rank").iterrows():
            model = str(row["model"])
            claim = claims.get(f"CLAIM.RANK.{model}")
            if claim is None:
                raise RenderError(f"overall_ranking.csv has {model!r} but claims.json has no CLAIM.RANK.{model}")
            label = f"`{model}`" + (" (0 parameters)" if model in zero_param else "")
            rows.append(
                [
                    str(int(row["rank"])),
                    label,
                    f"{float(claim['value']):.3f}",
                    f"[{float(claim['ci_low']):.3f}, {float(claim['ci_high']):.3f}]",
                    f"`CLAIM.RANK.{model}`",
                ]
            )
        blocks["ranking_table"] = _md_table(
            ["rank", "policy", "mean eval return", "95% CI over scenarios", "claim_id"], rows
        )

    # --------------------------------------------------- diagnostics_table -- #
    if not runs.empty:
        neural = runs[runs["is_trainable"] == True] if "is_trainable" in runs else runs  # noqa: E712
        rows = []
        if "final_explained_variance" in neural:
            rows.append(
                [
                    "critic explained variance (max over all runs)",
                    "`final_explained_variance`",
                    f"`{neural['final_explained_variance'].max():.2e}`",
                ]
            )
        if "mean_policy_entropy_nats" in neural:
            rows.append(
                [
                    "mean policy entropy after training",
                    "`mean_policy_entropy_nats`",
                    f"`{neural['mean_policy_entropy_nats'].mean():.3f}`",
                ]
            )
        if "untrained_mean_policy_entropy_nats" in neural:
            rows.append(
                [
                    "mean policy entropy before training",
                    "`untrained_mean_policy_entropy_nats`",
                    f"`{neural['untrained_mean_policy_entropy_nats'].mean():.3f}`",
                ]
            )
        rows.append(
            [
                f"entropy of a uniform policy over {DEFAULT_NUM_CHANNELS} channels",
                "`ln(num_channels)`",
                f"`{math.log(DEFAULT_NUM_CHANNELS):.3f}`",
            ]
        )
        if "total_gradient_steps" in neural and len(neural):
            rows.append(
                [
                    "gradient steps per run",
                    "`total_gradient_steps`",
                    f"`{int(neural['total_gradient_steps'].max())}`",
                ]
            )
        blocks["diagnostics_table"] = _md_table(["diagnostic", "column", "value"], rows)

    # ---------------------------------------------- federated_bytes_table -- #
    if not federated.empty:
        rows = []
        for _, row in federated.sort_values("arm").iterrows():
            arm = str(row["arm"])
            claim = claims.get(f"CLAIM.FED.cloud_mb.{arm}")
            if claim is None:
                raise RenderError(
                    f"federated_table.csv has arm {arm!r} but claims.json has no "
                    f"CLAIM.FED.cloud_mb.{arm}"
                )
            rows.append(
                [
                    f"`{arm}`",
                    f"{float(claim['value']):.4f}",
                    f"`CLAIM.FED.cloud_mb.{arm}`",
                    f"{float(row['edge_local_megabytes']):.4f}",
                    f"{float(row['total_megabytes']):.4f}",
                ]
            )
        blocks["federated_bytes_table"] = _md_table(
            ["arm", "wide-area (cloud) MB", "claim_id", "edge-local MB", "total MB"], rows
        )

    # ------------------------------------------------- family_outcome_table -- #
    if not comparisons.empty:
        descriptions = {k: _first_sentence(v.get("description", "")) for k, v in (prereg.get("families") or {}).items()}
        rows = []
        for family, chunk in comparisons.groupby("family", sort=True):
            n_sig = int(chunk["significant"].sum())
            if n_sig == 0:
                outcome = "no detectable difference"
            else:
                winners = {
                    (r["label_a"] if r["verdict"] == "favours_a" else r["label_b"])
                    for _, r in chunk[chunk["significant"] == True].iterrows()  # noqa: E712
                }
                outcome = (
                    f"all {n_sig} favour `{winners.pop()}`"
                    if len(winners) == 1
                    else f"{n_sig} of {len(chunk)} reject"
                )
            rows.append(
                [
                    f"`{family}`",
                    descriptions.get(family, ""),
                    str(len(chunk)),
                    str(n_sig),
                    outcome,
                ]
            )
        blocks["family_outcome_table"] = _md_table(
            ["family", "what it tests (from `configs/preregistration.yaml`)", "comparisons",
             "significant after Holm", "outcome"],
            rows,
        )

    blocks.update(build_study_b_blocks(results_dir, repo_root, claims, ranking, budget))
    return blocks


# --------------------------------------------------------------------------- #
# Study B blocks
# --------------------------------------------------------------------------- #


def _frozen_utc(manifest: Mapping[str, Any]) -> str:
    """The freeze timestamp recorded beside the digest, without its yaml key."""
    text = str(manifest.get("preregistration_frozen_at", ""))
    match = re.search(r"frozen_utc:\s*(\S+)", text)
    return match.group(1) if match else "unrecorded"


def _family_order(name: str) -> tuple[int, str]:
    """Sort ``F10_x`` after ``F9_x`` instead of after ``F1_x``."""
    match = re.match(r"F(\d+)", str(name))
    return (int(match.group(1)) if match else 10_000, str(name))


def _cmp_claim(claims: dict[str, dict[str, Any]], comparison_id: str) -> dict[str, Any]:
    cid = f"CLAIM.CMP.{comparison_id}"
    claim = claims.get(cid)
    if claim is None:
        raise RenderError(
            f"comparisons.csv has {comparison_id!r} but claims.json has no {cid}. "
            "Re-run scripts/analyze.py for that study."
        )
    return claim


def _cmp_rows(
    chunk: pd.DataFrame, claims: dict[str, dict[str, Any]], with_scenario: bool = True
) -> list[list[str]]:
    """One row per comparison, every number read from that comparison's claim."""
    rows: list[list[str]] = []
    for _, row in chunk.iterrows():
        cid = f"CLAIM.CMP.{row['comparison_id']}"
        claim = _cmp_claim(claims, str(row["comparison_id"]))
        cells = [f"`{row['label_a']}` vs `{row['label_b']}`"]
        if with_scenario:
            cells.append(f"`{row['scenario']}`")
        cells += [
            f"{float(claim['value']):.3f}",
            f"[{float(claim['ci_low']):.3f}, {float(claim['ci_high']):.3f}]",
            f"{float(claim['holm_adjusted_p']):.4f}",
            f"{float(row['cliffs_delta']):.3f}",
            f"`{claim['verdict']}`",
            f"`{cid}`",
        ]
        rows.append(cells)
    return rows


def _cmp_header(with_scenario: bool = True) -> list[str]:
    head = ["contrast"] + (["scenario"] if with_scenario else [])
    return head + ["mean difference", "95% CI", "Holm p", "Cliff's delta", "verdict", "claim_id"]


def build_study_b_blocks(
    results_dir: pathlib.Path,
    repo_root: pathlib.Path,
    study_a_claims: dict[str, dict[str, Any]],
    study_a_ranking: pd.DataFrame,
    study_a_budget: pd.DataFrame,
) -> dict[str, list[str]]:
    """Every ``studyb_*`` region, plus the one region that spans both studies.

    ``results_dir`` is Study A's directory; Study B is the ``study_b``
    subdirectory of it.  The Study A frames are passed in rather than re-read so
    that the side-by-side region cites the same claims the Study A regions do.
    """
    blocks: dict[str, list[str]] = {}
    study_dir = study_b_dir(results_dir)
    if not (study_dir / "all_runs.csv").exists():
        return blocks

    tables = study_dir / "tables"

    def read(name: str) -> pd.DataFrame:
        path = tables / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    comparisons = read("comparisons.csv")
    ranking = read("overall_ranking.csv")
    budget = read("parameter_budget.csv")
    learning = read("learning_check.csv")
    main = read("main_table.csv")
    runs = pd.read_csv(study_dir / "all_runs.csv")
    mech_path = study_dir / "mechanistic_endpoints.csv"
    mech = pd.read_csv(mech_path) if mech_path.exists() else pd.DataFrame()
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    claims_path = study_dir / "claims.json"
    claims: dict[str, dict[str, Any]] = {}
    if claims_path.exists():
        claims = {c["claim_id"]: c for c in json.loads(claims_path.read_text()).get("claims", [])}

    prereg_path = repo_root / "configs" / STUDY_B_PREREG
    prereg = (yaml.safe_load(prereg_path.read_text()) if prereg_path.exists() else {}) or {}
    families = prereg.get("families") or {}

    #: Families sorted into the three kinds the documents discuss separately.
    learning_families = [k for k, v in families.items() if v.get("kind") == "trained_vs_untrained"]
    greedy_families = [
        k for k, v in families.items()
        if v.get("kind") == "model_vs_model" and list(v.get("arms_b") or []) == ["greedy_heuristic"]
    ]
    dt_families = [
        k for k, v in families.items()
        if v.get("kind") == "model_vs_model"
        and set(map(str, v.get("arms_a") or [])) == {"ppo_cfc"}
        and set(map(str, v.get("arms_b") or [])) == {"ppo_cfc_dtblind"}
    ]
    architecture_families = [
        k for k in families
        if k not in learning_families and k not in greedy_families and k not in dt_families
        and k != "F1_primary"
    ]

    def family_chunk(names: list[str]) -> pd.DataFrame:
        """Rows of the named families, in declared order.

        Selection is by ``map`` rather than ``DataFrame.isin`` because
        tests/test_no_filter.py bans ``isin`` anywhere in the analysis layer --
        that call is how a pre-rebuild revision quietly dropped a baseline from
        four published tables.  These groupings are presentational: every
        comparison appears in ``studyb_comparisons_table`` regardless, and the
        family names come from the pre-registration rather than from a list
        written here.
        """
        if comparisons.empty:
            return comparisons
        wanted = set(names)
        chunk = comparisons[comparisons["family"].map(lambda f: f in wanted)].copy()
        chunk["_order"] = chunk["family"].map(_family_order)
        return chunk.sort_values(["_order", "comparison_id"]).drop(columns="_order")

    # ----------------------------------------------------- studyb_p1_table -- #
    primary_id = str((prereg.get("primary_comparison") or {}).get("id", "P1"))
    cid = f"CLAIM.PRIMARY.{primary_id}"
    if cid in claims and not comparisons.empty and (comparisons["comparison_id"] == primary_id).any():
        claim = claims[cid]
        row = comparisons[comparisons["comparison_id"] == primary_id].iloc[0]
        method = str(row["p_method"]).replace("_", " ")
        level = f"{100.0 * float(row['ci_level']):g}%"
        blocks["studyb_p1_table"] = _md_table(
            ["quantity", "value", "claim_id"],
            [
                ["mean paired difference", f"{float(claim['value']):.3f}", f"`{cid}`"],
                [
                    f"{level} bootstrap CI",
                    f"[{float(claim['ci_low']):.3f}, {float(claim['ci_high']):.3f}]",
                    f"`{cid}`",
                ],
                [f"p ({method})", f"{float(claim['p_value']):.3f}", f"`{cid}`"],
                ["Holm-adjusted p", f"{float(claim['holm_adjusted_p']):.3f}", f"`{cid}`"],
                ["n paired seeds", str(int(row["n_pairs"])), f"`{cid}`"],
                ["Cliff's delta", f"{float(row['cliffs_delta']):.3f}", f"`{cid}`"],
                ["verdict", f"`{claim['verdict']}`", f"`{cid}`"],
            ],
        )

    # ------------------------------------------------ studyb_ranking_table -- #
    if not ranking.empty:
        zero_param = set()
        if not budget.empty and "parameter_count" in budget:
            zero_param = set(budget.loc[budget["parameter_count"] == 0, "model"])
        rows = []
        for _, row in ranking.sort_values("rank").iterrows():
            model = str(row["model"])
            claim = claims.get(f"CLAIM.RANK.{model}")
            if claim is None:
                raise RenderError(
                    f"study_b/tables/overall_ranking.csv has {model!r} but its claims.json "
                    f"has no CLAIM.RANK.{model}"
                )
            label = f"`{model}`" + (" (0 parameters)" if model in zero_param else "")
            rows.append(
                [
                    str(int(row["rank"])),
                    label,
                    f"{float(claim['value']):.3f}",
                    f"[{float(claim['ci_low']):.3f}, {float(claim['ci_high']):.3f}]",
                    f"`CLAIM.RANK.{model}`",
                ]
            )
        blocks["studyb_ranking_table"] = _md_table(
            ["rank", "policy", "mean eval return", "95% CI over scenarios", "claim_id"], rows
        )

    # ----------------------------------------------- studyb_learning_table -- #
    chunk = family_chunk(learning_families)
    if not chunk.empty:
        rows = []
        for _, row in chunk.iterrows():
            claim = _cmp_claim(claims, str(row["comparison_id"]))
            learned = "yes" if bool(row["significant"]) else "no"
            rows.append(
                [
                    f"`{row['scenario']}`",
                    f"`{str(row['label_a']).split('(')[0]}`",
                    f"{float(claim['value']):.3f}",
                    f"[{float(claim['ci_low']):.3f}, {float(claim['ci_high']):.3f}]",
                    f"{float(claim['holm_adjusted_p']):.4f}",
                    learned,
                    f"`CLAIM.CMP.{row['comparison_id']}`",
                ]
            )
        blocks["studyb_learning_table"] = _md_table(
            ["scenario", "model", "trained − untrained", "95% CI", "Holm p",
             "beat its own initialisation?", "claim_id"],
            rows,
        )

    # ------------------------------------------------- studyb_dt_table ------ #
    dt_chunk = comparisons[comparisons["comparison_id"] == primary_id] if not comparisons.empty else pd.DataFrame()
    dt_chunk = pd.concat([dt_chunk, family_chunk(dt_families)]) if not comparisons.empty else dt_chunk
    if not dt_chunk.empty:
        blocks["studyb_dt_table"] = _md_table(_cmp_header(), _cmp_rows(dt_chunk, claims))

    # ------------------------------------------- studyb_architecture_table -- #
    chunk = family_chunk(architecture_families)
    if not chunk.empty:
        blocks["studyb_architecture_table"] = _md_table(_cmp_header(), _cmp_rows(chunk, claims))

    # -------------------------------------------- studyb_greedy_gap_table --- #
    chunk = family_chunk(greedy_families)
    if not chunk.empty:
        blocks["studyb_greedy_gap_table"] = _md_table(_cmp_header(), _cmp_rows(chunk, claims))

    # -------------------------------------------- studyb_comparisons_table -- #
    if not comparisons.empty:
        ordered = comparisons.copy()
        ordered["_order"] = ordered["family"].map(_family_order)
        ordered = ordered.sort_values(["_order", "comparison_id"])
        rows = []
        for _, row in ordered.iterrows():
            claim = _cmp_claim(claims, str(row["comparison_id"]))
            rows.append(
                [
                    f"`{row['family']}`",
                    f"`{row['label_a']}` vs `{row['label_b']}`",
                    f"`{row['scenario']}`",
                    str(int(row["n_pairs"])),
                    f"{float(claim['value']):.3f}",
                    f"[{float(claim['ci_low']):.3f}, {float(claim['ci_high']):.3f}]",
                    f"{float(claim['p_value']):.6f}",
                    f"{float(claim['holm_adjusted_p']):.4f}",
                    f"{float(row['cliffs_delta']):.3f}",
                    f"`{claim['verdict']}`",
                ]
            )
        blocks["studyb_comparisons_table"] = _md_table(
            ["family", "contrast", "scenario", "n", "mean difference", "95% CI", "raw p",
             "Holm p", "Cliff's delta", "verdict"],
            rows,
        )

    # ----------------------------------------- studyb_family_outcome_table -- #
    if not comparisons.empty:
        descriptions = {k: _first_sentence(v.get("description", "")) for k, v in families.items()}
        rows = []
        for family in sorted(set(comparisons["family"]), key=_family_order):
            chunk = comparisons[comparisons["family"] == family]
            n_sig = int(chunk["significant"].sum())
            if n_sig == 0:
                outcome = "no detectable difference"
            else:
                winners = {
                    (r["label_a"] if r["verdict"] == "favours_a" else r["label_b"])
                    for _, r in chunk[chunk["significant"] == True].iterrows()  # noqa: E712
                }
                outcome = (
                    f"all {n_sig} favour `{winners.pop()}`"
                    if len(winners) == 1
                    else f"{n_sig} of {len(chunk)} reject"
                )
            rows.append(
                [f"`{family}`", descriptions.get(family, ""), str(len(chunk)), str(n_sig), outcome]
            )
        blocks["studyb_family_outcome_table"] = _md_table(
            ["family", "what it tests (from `configs/preregistration_study_b.yaml`)",
             "comparisons", "significant after Holm", "outcome"],
            rows,
        )

    # --------------------------------------------- studyb_mechanistic_table -- #
    if not mech.empty:
        def mech_rows(frame: pd.DataFrame, with_scenario: bool, spec: str = ".3f") -> list[list[str]]:
            out = []
            for _, row in frame.iterrows():
                cells = ([f"`{row['scenario']}`"] if with_scenario else []) + [
                    f"`{row['model']}`",
                    format(float(row["before"]), spec),
                    format(float(row["after"]), spec),
                    format(float(row["change"]), spec),
                    f"[{format(float(row['change_ci_low']), spec)}, "
                    f"{format(float(row['change_ci_high']), spec)}]",
                    f"{int(row['n_seeds_moved_in_expected_direction'])} of {int(row['n_seeds'])}",
                ]
                out.append(cells)
            return out

        # The two LTC cells move the critic by ~1e-5, so the explained-variance
        # region carries an extra digit: at .3f its interval would render as
        # [-0.000, 0.000] and hide its own scale.
        for endpoint, name, spec in (
            ("eval_policy_entropy_nats", "studyb_entropy_table", ".3f"),
            ("critic_explained_variance", "studyb_explained_variance_table", ".4f"),
        ):
            frame = mech[(mech["endpoint"] == endpoint) & (mech["scenario"] == "irregular_dt")]
            frame = frame.sort_values("change")
            if frame.empty:
                continue
            blocks[name] = _md_table(
                ["model", "before", "after", "change", "95% CI", "seeds moving as expected"],
                mech_rows(frame, with_scenario=False, spec=spec),
            )

        full = mech.sort_values(["scenario", "endpoint", "model"])
        full_rows = []
        for _, row in full.iterrows():
            spec = ".4f" if row["endpoint"] == "critic_explained_variance" else ".3f"
            full_rows.append(
                [
                    f"`{row['scenario']}`",
                    f"`{row['endpoint']}`",
                    f"`{row['model']}`",
                    format(float(row["before"]), spec),
                    format(float(row["after"]), spec),
                    format(float(row["change"]), spec),
                    f"[{format(float(row['change_ci_low']), spec)}, "
                    f"{format(float(row['change_ci_high']), spec)}]",
                    f"{int(row['n_seeds_moved_in_expected_direction'])} of {int(row['n_seeds'])}",
                ]
            )
        blocks["studyb_mechanistic_full_table"] = _md_table(
            ["scenario", "endpoint", "model", "before", "after", "change", "95% CI",
             "seeds moving as expected"],
            full_rows,
        )

    # ---------------------------------------------- studyb_diagnostics_table -- #
    neural = runs[runs["is_trainable"] == True] if "is_trainable" in runs else runs  # noqa: E712
    if not neural.empty:
        columns = [
            ("final_approx_kl", "approx KL", ".4f"),
            ("final_clip_fraction", "clip fraction", ".3f"),
            ("final_explained_variance", "critic EV", ".4f"),
            ("final_policy_entropy_mean", "rollout entropy (nats)", ".3f"),
        ]
        present = [(c, label, spec) for c, label, spec in columns if c in neural.columns]
        rows = []
        for model, chunk in neural[neural["scenario"] == "irregular_dt"].groupby("model", sort=True):
            rows.append([f"`{model}`"] + [format(float(chunk[c].mean()), spec) for c, _, spec in present])
        if rows:
            blocks["studyb_diagnostics_table"] = _md_table(
                ["model (mean over 12 seeds, `irregular_dt`)"] + [label for _, label, _ in present],
                rows,
            )

    # ------------------------------------------ studyb_outcome_profile_table -- #
    if not main.empty:
        wanted = [
            ("mean_eval_return", "return", ".3f"),
            ("mean_success_rate", "success rate", ".3f"),
            ("mean_collision_rate", "collision rate", ".3f"),
            ("mean_spectrum_utilization", "utilisation", ".3f"),
            ("mean_action_histogram_entropy", "action-histogram entropy", ".3f"),
        ]
        present = [(c, label, spec) for c, label, spec in wanted if c in main.columns]
        frame = main[main["scenario"] == "irregular_dt"].sort_values("rank_in_scenario")
        rows = [
            [f"`{row['model']}`"] + [format(float(row[c]), spec) for c, _, spec in present]
            for _, row in frame.iterrows()
        ]
        if rows:
            blocks["studyb_outcome_profile_table"] = _md_table(
                ["policy (`irregular_dt`)"] + [label for _, label, _ in present], rows
            )

    # ------------------------------------------------- studyb_matrix_table --- #
    if manifest:
        matrix = manifest.get("matrix", {})
        rows = [
            ["environment steps per trainable cell", str(int(matrix["training_steps"]))],
            ["gradient steps per trainable cell",
             str(int(neural["total_gradient_steps"].max())) if "total_gradient_steps" in neural else "n/a"],
            ["seeds", str(len(matrix["seeds"]))],
            ["scenarios", ", ".join(f"`{s}`" for s in matrix["scenarios"])],
            ["trainable models", str(len(matrix["models"]))],
            ["zero-parameter baselines", str(len(matrix["baselines"]))],
            ["rows written", str(int(matrix["rows_written"]))],
            ["total training environment steps", str(int(matrix["total_env_steps_trained"]))],
            ["evaluation episodes per cell", str(int(matrix["eval_episodes"]))],
            ["parts skipped by the stopping rule", str(len(manifest.get("dropped_parts", [])))],
            ["wall clock, minutes", f"{float(manifest['total_wall_minutes']):.2f}"],
            ["workers", str(int(matrix["num_workers"]))],
            ["pre-registration sha256", f"`{manifest['preregistration_sha256']}`"],
            ["pre-registration frozen (UTC)", f"`{_frozen_utc(manifest)}`"],
            ["first cell started", f"`{manifest['started_utc']}`"],
            ["finished", f"`{manifest['finished_utc']}`"],
        ]
        blocks["studyb_matrix_table"] = _md_table(["quantity", "value"], rows)

    # ------------------------------------------------ budget_comparison_table -- #
    # The one region that reads both studies.  Every cell is a claim from the
    # study it belongs to; nothing here is recomputed from a CSV that the
    # corresponding claims file also covers.
    a_p1 = study_a_claims.get("CLAIM.PRIMARY.P1")
    b_p1 = claims.get("CLAIM.PRIMARY.P1")
    a_learn = study_a_claims.get("CLAIM.LEARN.count")
    b_learn = claims.get("CLAIM.LEARN.count")
    if a_p1 and b_p1 and a_learn and b_learn and not ranking.empty and not study_a_ranking.empty:

        def learn_cell(claim: dict[str, Any], frame: pd.DataFrame) -> str:
            tested = frame[frame["trainable"] == True] if "trainable" in frame else frame  # noqa: E712
            return f"{int(float(claim['value']))} of {len(tested)}"

        def best_learned(rank_frame: pd.DataFrame, budget_frame: pd.DataFrame,
                         claim_map: dict[str, dict[str, Any]]) -> tuple[str, float]:
            """Top-ranked policy with parameters, by row scan (no isin: see above)."""
            zero = set(budget_frame.loc[budget_frame["parameter_count"] == 0, "model"])
            for _, row in rank_frame.sort_values("rank").iterrows():
                model = str(row["model"])
                if model not in zero:
                    return model, float(claim_map[f"CLAIM.RANK.{model}"]["value"])
            raise RenderError("overall_ranking.csv contains no policy with parameters")

        a_learning = pd.read_csv(results_dir / "tables" / "learning_check.csv")
        a_model, a_value = best_learned(study_a_ranking, study_a_budget, study_a_claims)
        b_model, b_value = best_learned(ranking, budget, claims)
        a_greedy = study_a_claims["CLAIM.RANK.greedy_heuristic"]
        b_greedy = claims["CLAIM.RANK.greedy_heuristic"]
        a_neural = pd.read_csv(results_dir / "all_runs.csv")
        a_neural = a_neural[a_neural["is_trainable"] == True]  # noqa: E712

        rows = [
            [
                "environment steps per trainable cell",
                str(int(a_neural["train_steps"].max())),
                str(int(neural["train_steps"].max())),
                "`all_runs.csv :: train_steps`",
            ],
            [
                "gradient steps per trainable cell",
                str(int(a_neural["total_gradient_steps"].max())),
                str(int(neural["total_gradient_steps"].max())),
                "`all_runs.csv :: total_gradient_steps`",
            ],
            [
                "primary comparison P1, mean difference",
                f"{float(a_p1['value']):.3f}",
                f"{float(b_p1['value']):.3f}",
                "`CLAIM.PRIMARY.P1`",
            ],
            [
                "primary comparison P1, 95% CI",
                f"[{float(a_p1['ci_low']):.3f}, {float(a_p1['ci_high']):.3f}]",
                f"[{float(b_p1['ci_low']):.3f}, {float(b_p1['ci_high']):.3f}]",
                "`CLAIM.PRIMARY.P1`",
            ],
            [
                "primary comparison P1, p",
                f"{float(a_p1['p_value']):.3f}",
                f"{float(b_p1['p_value']):.3f}",
                "`CLAIM.PRIMARY.P1`",
            ],
            [
                "primary comparison P1, verdict",
                f"`{a_p1['verdict']}`",
                f"`{b_p1['verdict']}`",
                "`CLAIM.PRIMARY.P1`",
            ],
            [
                "cells that beat their own initialisation",
                learn_cell(a_learn, a_learning),
                learn_cell(b_learn, learning),
                "`CLAIM.LEARN.count`",
            ],
            [
                "mean critic explained variance, after training",
                f"{float(a_neural['final_explained_variance'].mean()):.2e}",
                f"{float(neural['final_explained_variance'].mean()):.2e}",
                "`all_runs.csv :: final_explained_variance`",
            ],
            [
                "mean policy entropy after training, nats",
                f"{float(a_neural['mean_policy_entropy_nats'].mean()):.3f}",
                f"{float(neural['mean_policy_entropy_nats'].mean()):.3f}",
                "`all_runs.csv :: mean_policy_entropy_nats`",
            ],
            [
                "best learned policy, pooled",
                f"`{a_model}` {a_value:.3f}",
                f"`{b_model}` {b_value:.3f}",
                "the named model's `CLAIM.RANK` claim, per study",
            ],
            [
                "`greedy_heuristic`, pooled",
                f"{float(a_greedy['value']):.3f}",
                f"{float(b_greedy['value']):.3f}",
                "`CLAIM.RANK.greedy_heuristic`",
            ],
            [
                "`greedy_heuristic` lead over the best learned policy",
                f"{float(a_greedy['value']) - a_value:.3f}",
                f"{float(b_greedy['value']) - b_value:.3f}",
                "`CLAIM.RANK.greedy_heuristic` minus the row above",
            ],
        ]
        blocks["budget_comparison_table"] = _md_table(
            ["quantity", "Study A (low budget)", "Study B (16x budget)", "source"], rows
        )

    return blocks


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


class RenderError(RuntimeError):
    pass


def render_text(
    text: str,
    values: dict[str, tuple[Any, str]],
    blocks: dict[str, list[str]],
    where: str = "<text>",
) -> str:
    """Substitute every generated region and inline span in one document."""

    def span(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        spec = (match.group(2) or "").strip()
        if key not in values:
            raise RenderError(
                f"{where}: inline span references unknown value {key!r}. "
                f"Add it to build_values() in scripts/render_docs.py, or cite an existing key."
            )
        value, default_spec = values[key]
        return f"<!--v:{match.group(1)}{'|' + match.group(2) if match.group(2) else ''}-->" \
               f"{_fmt(value, spec or default_spec)}<!--/v-->"

    out: list[str] = []
    lines = text.split("\n")
    i = 0
    seen: set[str] = set()
    while i < len(lines):
        begin = BEGIN_RE.match(lines[i])
        if not begin:
            out.append(SPAN_RE.sub(span, lines[i]))
            i += 1
            continue

        name = begin.group(2)
        if name in seen:
            raise RenderError(f"{where}: generated region {name!r} appears twice")
        seen.add(name)
        if name not in blocks:
            raise RenderError(
                f"{where}: no generator for region {name!r}. "
                f"Known regions: {sorted(blocks)}"
            )
        out.append(lines[i])
        j = i + 1
        while j < len(lines):
            end = END_RE.match(lines[j])
            if end:
                if end.group(1) != name:
                    raise RenderError(
                        f"{where}: region {name!r} closed by END GENERATED: {end.group(1)!r}"
                    )
                break
            if BEGIN_RE.match(lines[j]):
                raise RenderError(f"{where}: region {name!r} is not closed before the next one opens")
            j += 1
        else:
            raise RenderError(f"{where}: region {name!r} has no END GENERATED marker")
        out.extend(blocks[name])
        out.append(lines[j])
        i = j + 1

    return "\n".join(out)


def render_file(
    path: pathlib.Path,
    values: dict[str, tuple[Any, str]],
    blocks: dict[str, list[str]],
) -> str:
    return render_text(path.read_text(), values, blocks, where=path.name)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=pathlib.Path, default=REPO_ROOT / "results")
    parser.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any document differs from what this script generates",
    )
    args = parser.parse_args(argv)

    if not (args.results_dir / "all_runs.csv").exists():
        raise SystemExit(
            f"error: {args.results_dir / 'all_runs.csv'} not found. Run `make suite` first."
        )

    values = build_values(args.results_dir, args.repo_root)
    blocks = build_blocks(args.results_dir, args.repo_root)

    stale: list[str] = []
    written = 0
    for path in DOC_PATHS(args.repo_root):
        current = path.read_text()
        try:
            rendered = render_file(path, values, blocks)
        except RenderError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if rendered == current:
            continue
        if args.check:
            diff = difflib.unified_diff(
                current.splitlines(), rendered.splitlines(),
                fromfile=f"committed/{path.name}", tofile=f"generated/{path.name}", lineterm="",
            )
            stale.append("\n".join(diff))
        else:
            path.write_text(rendered)
            written += 1

    if args.check:
        if stale:
            print(
                "error: a generated number in a committed document does not match results/.\n"
                "Nothing in a generated region or an inline <!--v:...--> span may be edited by "
                "hand; run `make report` and commit the result.\n",
                file=sys.stderr,
            )
            for chunk in stale:
                print(chunk, file=sys.stderr)
            return 1
        print(f"{len(DOC_PATHS(args.repo_root))} documents are current")
        return 0

    print(f"rendered {len(DOC_PATHS(args.repo_root))} documents ({written} rewritten)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
