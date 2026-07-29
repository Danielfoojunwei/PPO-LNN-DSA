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
