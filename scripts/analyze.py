#!/usr/bin/env python3
"""Turn ``results/*.csv`` into tables, figures and a machine-readable claims file.

This script has a real CLI.  The one it replaces had its input and output paths
hardcoded at module level, so running the documented workflow silently rewrote
tracked evidence with no way to redirect it.  Here ``--results-dir`` and
``--output-dir`` are separate, ``--check`` regenerates into a temporary
directory and diffs without writing anything, and every path is explicit.

It also has no model filter.  The script it replaces defined a ``CORE_MODELS``
tuple that omitted ``random_policy`` and applied it to exactly the four tables
the README cited.  Restored, that baseline ranked 4th of 7, ahead of three PPO
variants.  ``tests/test_no_filter.py`` fails the build if such a list reappears.

Usage
-----
    python scripts/analyze.py
    python scripts/analyze.py --results-dir results --output-dir /tmp/analysis
    python scripts/analyze.py --check          # verify committed tables are current
"""

from __future__ import annotations

import argparse
import filecmp
import json
import pathlib
import shutil
import sys
import tempfile

import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dsa.analysis.plots import build_all_figures  # noqa: E402
from dsa.analysis.tables import build_all_tables, build_claims  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A PPO run whose first-epoch importance ratio is not 1.0 has a stale hidden
#: state, a dropout mismatch or a train/eval-mode bug.  Its numbers are not
#: interpretable, so the analysis refuses to build a report from them.
MAX_ACCEPTABLE_RATIO_DEVIATION = 1e-3


def _load(path: pathlib.Path, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise SystemExit(
                f"error: {path} not found. Run `make suite` (and `make federated`) first."
            )
        return pd.DataFrame()
    return pd.read_csv(path)


def _check_ratio_gate(all_runs: pd.DataFrame) -> None:
    column = "max_first_epoch_ratio_deviation"
    if column not in all_runs.columns:
        return
    values = all_runs[column].dropna()
    if values.empty:
        return
    worst = float(values.max())
    if worst > MAX_ACCEPTABLE_RATIO_DEVIATION:
        offenders = all_runs.loc[all_runs[column] > MAX_ACCEPTABLE_RATIO_DEVIATION, ["scenario", "model", "seed"]]
        raise SystemExit(
            f"error: first-epoch PPO ratio deviated by up to {worst:.3e}, above the "
            f"{MAX_ACCEPTABLE_RATIO_DEVIATION:.0e} gate, in {len(offenders)} run(s).\n"
            "That means the update did not reproduce the rollout's own log-probs -- a stale\n"
            "hidden state, live dropout, or a missing eval()/train() switch. These runs are\n"
            "not interpretable and no report will be generated from them.\n"
            f"{offenders.head(10).to_string(index=False)}"
        )
    print(f"  ratio gate ok: worst first-epoch deviation {worst:.2e} (limit {MAX_ACCEPTABLE_RATIO_DEVIATION:.0e})")


def analyze(
    results_dir: pathlib.Path,
    output_dir: pathlib.Path,
    prereg_path: pathlib.Path,
    figures: bool = True,
) -> dict[str, object]:
    all_runs = _load(results_dir / "all_runs.csv", required=True)
    fed_runs = _load(results_dir / "federated_all_runs.csv", required=False)
    prereg = yaml.safe_load(prereg_path.read_text())

    print(f"  all_runs: {len(all_runs)} rows, {all_runs['model'].nunique()} policies")
    if not fed_runs.empty:
        print(f"  federated_all_runs: {len(fed_runs)} rows, {fed_runs['arm'].nunique()} arms")
    _check_ratio_gate(all_runs)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables = build_all_tables(all_runs, fed_runs, prereg, tables_dir)
    print(f"  wrote {len(tables)} tables to {tables_dir}")

    written_figures: list[pathlib.Path] = []
    if figures:
        written_figures = build_all_figures(tables, figures_dir)
        print(f"  wrote {len(written_figures)} figures to {figures_dir}")

    claims = build_claims(tables, prereg)
    claims_path = output_dir / "claims.json"
    claims_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fields": [
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
                ],
                "note": (
                    "Every number in RESULTS.md and README.md must correspond to one of these "
                    "claim_ids. Each entry names the CSV and column it was read from so it can "
                    "be re-derived without re-running anything."
                ),
                "claims": claims,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )
    print(f"  wrote {len(claims)} claims to {claims_path}")
    return {"tables": tables, "claims": claims, "figures": written_figures}


def _diff_trees(reference: pathlib.Path, candidate: pathlib.Path, patterns: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    for pattern in patterns:
        for produced in sorted(candidate.glob(pattern)):
            expected = reference / produced.relative_to(candidate)
            if not expected.exists():
                problems.append(f"missing from committed evidence: {produced.relative_to(candidate)}")
            elif not filecmp.cmp(produced, expected, shallow=False):
                problems.append(f"differs from committed evidence: {produced.relative_to(candidate)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=pathlib.Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None, help="defaults to --results-dir")
    parser.add_argument("--prereg", type=pathlib.Path, default=REPO_ROOT / "configs" / "preregistration.yaml")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temporary directory and diff against --output-dir; write nothing",
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = pathlib.Path(tmp)
            print(f"regenerating into {candidate} for comparison against {output_dir}")
            analyze(results_dir, candidate, args.prereg, figures=False)
            problems = _diff_trees(output_dir, candidate, ("tables/*.csv", "claims.json"))
        if problems:
            print("\nCOMMITTED EVIDENCE IS STALE:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print("\nRun `make analyze` and commit the result.", file=sys.stderr)
            return 1
        print("\ncommitted tables and claims are current")
        return 0

    print(f"analysing {results_dir} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    analyze(results_dir, output_dir, args.prereg, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
