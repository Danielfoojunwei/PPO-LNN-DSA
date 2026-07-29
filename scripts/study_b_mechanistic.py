#!/usr/bin/env python3
"""Mechanistic endpoints for Study B: did the optimiser do anything at all?

These are the endpoints declared under ``mechanistic_endpoints`` in
``configs/preregistration_study_b.yaml``.  They are DESCRIPTIVE BY DECLARATION:
they carry seed-bootstrapped intervals but **no p-value and no Holm correction**,
and they are not members of any confirmatory family.  They answer a different
question from the reward comparisons -- "did entropy fall and did the critic
start explaining variance" -- and they are reported even when they disagree with
the reward numbers, which is the case they exist for.

Study A's answer to all three was flat: policy entropy stayed at the uniform
ln(8) = 2.0794 nats and critic explained variance sat at ~0.0000 after 240
gradient steps, which is why Study A's architectural contrasts compared
initialisations rather than learned policies.

Usage
-----
    python scripts/study_b_mechanistic.py
    python scripts/study_b_mechanistic.py --results-dir results/study_b
    python scripts/study_b_mechanistic.py --check   # committed table is current?
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dsa.analysis.stats import ANALYSIS_SEED, CI_LEVEL, paired_bootstrap_ci  # noqa: E402
from dsa.seeding import derive_seed  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: (endpoint name, column before, column after).  Every pair is measured on the
#: same run, so the difference is paired within a cell as well as within a seed.
ENDPOINTS = (
    (
        "eval_policy_entropy_nats",
        "untrained_mean_policy_entropy_nats",
        "mean_policy_entropy_nats",
    ),
    (
        "rollout_policy_entropy_nats",
        "first_update_policy_entropy_mean",
        "final_policy_entropy_mean",
    ),
    (
        "critic_explained_variance",
        "first_update_explained_variance",
        "final_explained_variance",
    ),
)

#: Entropy of the uniform distribution over the 8 channels, the value a policy
#: that has learned nothing sits at.  Computed, not typed.
UNIFORM_ENTROPY_NATS = float(np.log(8))


def build(all_runs: pd.DataFrame) -> pd.DataFrame:
    if "is_trainable" in all_runs.columns:
        runs = all_runs[all_runs["is_trainable"].astype(bool)]
    else:
        runs = all_runs
    rows: list[dict[str, object]] = []
    for (scenario, model), chunk in runs.groupby(["scenario", "model"], sort=True):
        chunk = chunk.sort_values("seed")
        for name, col_before, col_after in ENDPOINTS:
            if col_before not in chunk.columns or col_after not in chunk.columns:
                continue
            before = chunk[col_before].to_numpy(dtype=np.float64)
            after = chunk[col_after].to_numpy(dtype=np.float64)
            mask = np.isfinite(before) & np.isfinite(after)
            before, after = before[mask], after[mask]
            if before.size == 0:
                continue
            diff = after - before
            lo, hi = (
                paired_bootstrap_ci(
                    diff,
                    level=CI_LEVEL,
                    seed=derive_seed(ANALYSIS_SEED, "study_b_mech", scenario, model, name),
                )
                if diff.size > 1
                else (float(diff[0]), float(diff[0]))
            )
            rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "endpoint": name,
                    "n_seeds": int(diff.size),
                    "before": float(before.mean()),
                    "after": float(after.mean()),
                    "change": float(diff.mean()),
                    "change_ci_low": float(lo),
                    "change_ci_high": float(hi),
                    "n_seeds_moved_in_expected_direction": int(
                        (diff < 0).sum() if "entropy" in name else (diff > 0).sum()
                    ),
                    "expected_direction": "down" if "entropy" in name else "up",
                    "uniform_entropy_reference_nats": (
                        UNIFORM_ENTROPY_NATS if "entropy" in name else float("nan")
                    ),
                    "correction": "none -- descriptive endpoint, never tested",
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "endpoint", "model"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=pathlib.Path, default=REPO_ROOT / "results" / "study_b")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild into a temporary file and diff against the committed table; write nothing",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    source = args.results_dir / "all_runs.csv"
    if not source.exists():
        raise SystemExit(f"error: {source} not found; run scripts/run_study_b.py first")
    table = build(pd.read_csv(source))
    out = args.output or (args.results_dir / "mechanistic_endpoints.csv")

    # ``--check`` is the same gate the tables and claims get: these endpoints are
    # cited in README.md and docs/STUDY_B.md through generated regions, so a
    # stale committed CSV would silently pin a stale number into the documents.
    if args.check:
        if not out.exists():
            print(f"error: {out} does not exist; run this script without --check", file=sys.stderr)
            return 1
        with tempfile.TemporaryDirectory() as tmp:
            candidate = pathlib.Path(tmp) / "mechanistic_endpoints.csv"
            table.round(6).to_csv(candidate, index=False)
            if candidate.read_bytes() != out.read_bytes():
                print(
                    f"error: {out} is stale -- it does not match a fresh build from {source}.\n"
                    "Run `make analyze-study-b` and commit the result.",
                    file=sys.stderr,
                )
                return 1
        print(f"{out} is current ({len(table)} rows)")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(out, index=False)
    print(f"wrote {out} ({len(table)} rows) from {source}")
    print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
