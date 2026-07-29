#!/usr/bin/env python3
"""Run Study B: the confirmatory matrix at a budget where learning happens.

Study B is a SEPARATE study from Study A.  It writes only under
``results/study_b/`` and it never touches ``results/all_runs.csv``,
``results/tables/`` or ``results/claims.json``.  Study A remains committed
evidence; see ``configs/preregistration_study_b.yaml`` for why the two plans live
in two files rather than one.

Three properties this script exists to guarantee
------------------------------------------------
1.  **The pre-registration is provably frozen first.**  ``--freeze`` writes
    ``results/study_b/preregistration.sha256``.  Every subsequent run recomputes
    the sha256 of ``configs/preregistration_study_b.yaml`` and refuses to execute
    a single cell unless it matches the frozen digest.  An auditor can check the
    mtime of that file against the ``started_utc`` of every part in
    ``results/study_b/manifest.json``.

2.  **Nothing is silently truncated.**  The matrix is executed as one
    ``(scenario, model)`` part at a time.  The wall clock is checked only at part
    boundaries.  Any part that is skipped because the declared cap was reached is
    recorded by name, with the elapsed time that caused the skip, under
    ``dropped_parts``.  Seeds are never reduced.

3.  **A crash costs one part, not the study.**  Each part is a separate
    ``scripts/run_suite.py`` process writing its own CSV under
    ``results/study_b/parts/``.  Re-running resumes: a part whose CSV already has
    the expected row count is skipped as complete.

Usage
-----
    python scripts/run_study_b.py --freeze          # freeze the prereg, run nothing
    python scripts/run_study_b.py --plan            # probe, project, run nothing
    python scripts/run_study_b.py                   # run the matrix
    python scripts/run_study_b.py --merge-only      # rebuild all_runs.csv from parts
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dsa.envs.config import obs_dim_for  # noqa: E402
from dsa.envs.scenarios import get_scenario  # noqa: E402
from dsa.learner.ppo import PPOHyperParams  # noqa: E402
from dsa.models.registry import MODEL_REGISTRY, model_summary  # noqa: E402
from scripts.probe_throughput import probe_models, project_wall_minutes  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SUITE_CONFIG = REPO_ROOT / "configs" / "suite_study_b.yaml"
PREREG = REPO_ROOT / "configs" / "preregistration_study_b.yaml"
DEFAULT_OUT = REPO_ROOT / "results" / "study_b"

#: Study A's tracked evidence.  This script must not write any of it.
STUDY_A_PATHS = ("results/all_runs.csv", "results/tables", "results/claims.json")


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Pre-registration freeze
# --------------------------------------------------------------------------- #


def freeze(out_dir: pathlib.Path) -> pathlib.Path:
    """Record the pre-registration digest.  Must happen before the first cell."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preregistration.sha256"
    if path.exists():
        raise SystemExit(
            f"error: {path} already exists.\n"
            "Re-freezing a pre-registration after a run has started destroys the very\n"
            "property the file exists to prove.  If the plan genuinely must change, the\n"
            "honest move is to relabel the affected comparisons exploratory."
        )
    digest = _sha256(PREREG)
    path.write_text(
        f"{digest}  {PREREG.relative_to(REPO_ROOT)}\n"
        f"frozen_utc: {_utc()}\n"
        f"git_commit: {_git_commit()}\n"
    )
    print(f"froze {PREREG.name}\n  sha256 {digest}\n  -> {path}")
    return path


def verify_freeze(out_dir: pathlib.Path) -> str:
    path = out_dir / "preregistration.sha256"
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found.  Run `python scripts/run_study_b.py --freeze`\n"
            "BEFORE any confirmatory cell.  Study B refuses to run unpre-registered."
        )
    frozen = path.read_text().split()[0]
    current = _sha256(PREREG)
    if frozen != current:
        raise SystemExit(
            f"error: {PREREG.name} has changed since it was frozen.\n"
            f"  frozen  {frozen}\n  current {current}\n"
            "The confirmatory status of this run is void.  Do not re-freeze."
        )
    return frozen


# --------------------------------------------------------------------------- #
# Matrix -> parts
# --------------------------------------------------------------------------- #


def build_parts(config: dict, prereg: dict) -> list[dict]:
    """One part per (scenario, policy-group).  Primary block first, in order.

    The scenario order is taken from the pre-registration's `blocks`, not from the
    suite config, so the block that carries the confirmatory families always runs
    before the block that can be dropped.
    """
    blocks = prereg["blocks"]
    order = [blocks["primary_block"]["scenario"], blocks["replication_block"]["scenario"]]
    known = list(config["scenarios"])
    scenarios = [s for s in order if s in known] + [s for s in known if s not in order]

    seeds = [int(s) for s in config["seeds"]]
    models = list(config["models"])
    parts: list[dict] = []
    for scenario in scenarios:
        for position, model in enumerate(models):
            # The three zero-parameter baselines ride along with the first (and
            # cheapest) model of each block: ~0.3 core-seconds a cell, and it
            # gives the block a complete, analysable baseline row set within
            # seconds of starting.  They are NOT a part of their own, because
            # run_suite.py's --models takes nargs="+" and cannot be handed an
            # empty model list -- a baselines-only part would have quietly
            # trained the fallback model for 163,840 steps.
            baselines = list(config["baselines"]) if position == 0 else []
            parts.append(
                {
                    "part_id": f"{scenario}__{model}",
                    "scenario": scenario,
                    "models": [model],
                    "baselines": baselines,
                    "expected_rows": len(seeds) * (1 + len(baselines)),
                }
            )
    return parts


def part_csv(out_dir: pathlib.Path, part_id: str) -> pathlib.Path:
    return out_dir / "parts" / part_id / "all_runs.csv"


def part_is_complete(out_dir: pathlib.Path, part: dict) -> bool:
    path = part_csv(out_dir, part["part_id"])
    if not path.exists():
        return False
    try:
        return len(pd.read_csv(path)) == int(part["expected_rows"])
    except Exception:
        return False


def run_part(part: dict, config: dict, out_dir: pathlib.Path, workers: int) -> dict:
    """Invoke scripts/run_suite.py for one part, in its own process."""
    part_dir = out_dir / "parts" / part["part_id"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_suite.py"),
        "--config", str(SUITE_CONFIG),
        "--output-dir", str(part_dir),
        "--scenarios", part["scenario"],
        "--seeds", *[str(s) for s in config["seeds"]],
        "--training-steps", str(int(config["training_steps"])),
        "--eval-episodes", str(int(config["eval_episodes"])),
        "--workers", str(workers),
        # The per-part guard is off because run_study_b.py applies the declared
        # ceiling itself, once, against the whole matrix -- not 15 times against
        # slices of it, which would shrink training_steps per part and silently
        # make the parts non-comparable.
        "--no-guard",
        "--models", *part["models"],
    ]
    # `--baselines` with no values is an explicit empty list; omitting the flag
    # would fall back to the config's three heuristics and run them 7x per block.
    cmd += ["--baselines", *part["baselines"]] if part["baselines"] else ["--baselines"]

    started = time.perf_counter()
    started_utc = _utc()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    record = {
        "part_id": part["part_id"],
        "scenario": part["scenario"],
        "models": part["models"],
        "baselines": part["baselines"],
        "command_line": cmd,
        "started_utc": started_utc,
        "finished_utc": _utc(),
        "wall_seconds": round(elapsed, 3),
        "returncode": proc.returncode,
        "expected_rows": int(part["expected_rows"]),
    }
    if proc.returncode != 0:
        record["stderr_tail"] = proc.stderr[-4000:]
        print(f"  !! part {part['part_id']} FAILED rc={proc.returncode}", flush=True)
        print(proc.stderr[-2000:], file=sys.stderr, flush=True)
    return record


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


def merge_parts(out_dir: pathlib.Path, parts: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    frames, present = [], []
    for part in parts:
        path = part_csv(out_dir, part["part_id"])
        if path.exists():
            frame = pd.read_csv(path)
            # Verify, never filter.  A part that produced a policy it was not
            # asked for means a stale directory from an earlier matrix, and the
            # honest response is to stop -- dropping the extra rows silently is
            # exactly the curation tests/test_no_filter.py exists to forbid.
            expected = set(part["models"]) | set(part["baselines"])
            found = set(frame["model"].astype(str).unique())
            if found != expected:
                raise SystemExit(
                    f"error: part {part['part_id']} holds policies {sorted(found)} but was "
                    f"asked for {sorted(expected)}.  Delete {path.parent} and re-run that "
                    "part; do not merge a part that does not match the declared matrix."
                )
            if not frame.empty:
                frames.append(frame)
                present.append(part["part_id"])
    if not frames:
        raise SystemExit("error: no part produced any rows; nothing to merge")
    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["scenario", "model", "seed"], keep="first")
        .sort_values(["scenario", "model", "seed"])
        .reset_index(drop=True)
    )
    return merged, present


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--freeze", action="store_true", help="freeze the pre-registration digest and exit")
    parser.add_argument("--plan", action="store_true", help="probe, project and list the parts; run nothing")
    parser.add_argument("--merge-only", action="store_true", help="rebuild all_runs.csv from existing parts")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    torch.set_num_threads(1)
    out_dir = pathlib.Path(args.output_dir)

    if args.freeze:
        freeze(out_dir)
        return 0

    config = yaml.safe_load(SUITE_CONFIG.read_text())
    prereg = yaml.safe_load(PREREG.read_text())
    workers = int(args.workers or config.get("num_workers", 4))
    parts = build_parts(config, prereg)
    cap_minutes = float(prereg["stopping_rule"]["wall_clock_cap_minutes"])

    if args.merge_only:
        merged, present = merge_parts(out_dir, parts)
        (out_dir / "all_runs.csv").write_text(merged.to_csv(index=False))
        print(f"merged {len(present)} parts -> {out_dir / 'all_runs.csv'} ({len(merged)} rows)")
        return 0

    prereg_sha = verify_freeze(out_dir)
    print(f"pre-registration verified: {prereg_sha}", flush=True)

    hp = PPOHyperParams(**config["ppo"])
    print(f"probing throughput for {len(config['models'])} models...", flush=True)
    probe = probe_models(config["models"], scenario=parts[0]["scenario"], rollouts=2, hparams=hp)
    projection = project_wall_minutes(
        probe,
        int(config["training_steps"]),
        len(config["seeds"]) * len(config["scenarios"]),
        int(config["eval_episodes"]),
        hp.horizon,
        workers,
    )
    guard_limit = float((config.get("guard") or {}).get("max_projected_wall_minutes", 170.0))
    print(
        f"projection: {projection['wall_minutes']:.1f} wall-min on {workers} workers "
        f"({projection['core_seconds'] / 60:.0f} core-min), guard {guard_limit:.0f}",
        flush=True,
    )
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "timing_probe.json").write_text(
        json.dumps({"probe": probe, "projection": projection, "measured_utc": _utc()}, indent=2) + "\n"
    )

    if args.plan:
        for part in parts:
            print(f"  {part['part_id']:<40} expect {part['expected_rows']:>3} rows")
        return 0
    if projection["wall_minutes"] > guard_limit:
        raise SystemExit(
            f"error: projection {projection['wall_minutes']:.1f} min exceeds the "
            f"{guard_limit:.0f} min guard in {SUITE_CONFIG.name}.  Shrink the matrix, "
            "never the seed count."
        )

    started = time.perf_counter()
    started_utc = _utc()
    records: list[dict] = []
    dropped: list[dict] = []
    for i, part in enumerate(parts, 1):
        elapsed_min = (time.perf_counter() - started) / 60.0
        if part_is_complete(out_dir, part):
            print(f"[{i}/{len(parts)}] {part['part_id']}: already complete, skipping", flush=True)
            records.append({"part_id": part["part_id"], "status": "already_complete"})
            continue
        if elapsed_min > cap_minutes:
            print(f"[{i}/{len(parts)}] {part['part_id']}: DROPPED, cap reached at {elapsed_min:.1f} min", flush=True)
            dropped.append(
                {
                    "part_id": part["part_id"],
                    "scenario": part["scenario"],
                    "models": part["models"],
                    "baselines": part["baselines"],
                    "reason": "declared wall-clock cap reached at a part boundary",
                    "elapsed_minutes_at_skip": round(elapsed_min, 2),
                    "cap_minutes": cap_minutes,
                }
            )
            continue
        print(f"[{i}/{len(parts)}] {part['part_id']} starting at t+{elapsed_min:.1f} min", flush=True)
        record = run_part(part, config, out_dir, workers)
        records.append(record)
        print(f"    done in {record['wall_seconds'] / 60:.1f} min (rc={record['returncode']})", flush=True)

    merged, present = merge_parts(out_dir, parts)
    csv_path = out_dir / "all_runs.csv"
    merged.to_csv(csv_path, index=False)

    obs_dim = obs_dim_for(get_scenario(parts[0]["scenario"]).num_channels)
    action_dim = get_scenario(parts[0]["scenario"]).num_channels
    manifest = {
        "schema_version": 1,
        "study": "B",
        "supersedes": "none -- Study A (results/all_runs.csv) is untouched committed evidence",
        "study_a_paths_not_written": list(STUDY_A_PATHS),
        "git_commit": _git_commit(),
        "generated_by": "scripts/run_study_b.py",
        "command_line": ["python", "scripts/run_study_b.py"] + list(sys.argv[1:]),
        "started_utc": started_utc,
        "finished_utc": _utc(),
        "total_wall_minutes": round((time.perf_counter() - started) / 60.0, 2),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "preregistration_file": str(PREREG.relative_to(REPO_ROOT)),
        "preregistration_sha256": prereg_sha,
        "preregistration_frozen_at": (out_dir / "preregistration.sha256").read_text(),
        "suite_config_file": str(SUITE_CONFIG.relative_to(REPO_ROOT)),
        "suite_config_sha256": _sha256(SUITE_CONFIG),
        "suite_config_contents": config,
        "matrix": {
            "scenarios": list(config["scenarios"]),
            "models": list(config["models"]),
            "baselines": list(config["baselines"]),
            "seeds": list(config["seeds"]),
            "training_steps": int(config["training_steps"]),
            "eval_episodes": int(config["eval_episodes"]),
            "num_workers": workers,
            "rows_written": int(len(merged)),
            "total_env_steps_trained": int(
                merged.loc[merged["is_trainable"] == True, "train_steps"].sum()  # noqa: E712
            ),
        },
        "ppo_hyperparameters": config["ppo"],
        "models_solved": {k: model_summary(k, obs_dim, action_dim) for k in sorted(MODEL_REGISTRY)},
        "probe": probe,
        "projection": projection,
        "parts": records,
        "parts_merged": present,
        "dropped_parts": dropped,
        "outputs": {"all_runs.csv": _sha256(csv_path)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")

    print(f"\nwrote {csv_path} ({len(merged)} rows) in {manifest['total_wall_minutes']:.1f} min")
    if dropped:
        print(f"DROPPED {len(dropped)} part(s): {[d['part_id'] for d in dropped]}")
    failed = [r for r in records if r.get("returncode") not in (0, None)]
    if failed:
        print(f"FAILED {len(failed)} part(s): {[r['part_id'] for r in failed]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
