#!/usr/bin/env python3
"""Run the confirmatory suite: every (scenario, model, seed) cell, in parallel.

Determinism is the point of this script.  Every environment stream, every weight
initialisation and every action sample is derived from the base seed through
``dsa.seeding.derive_seed``, which is a pure blake2b function of its arguments.
Nothing depends on construction order, process identity, or which models and
scenarios happened to be requested in the same invocation.  Two invocations with
different ``--models`` ordering produce byte-identical rows for the cells they
share; ``tests/test_reproducibility.py`` proves it.

The previous version of this repository seeded environments from a mutable
class-level instance counter plus ``config.seed + instance_id``, which made the
seed a function of the position of a model in the ``--models`` list.  A fixed
policy scored between -68.5 and -28.3 across six instances at one nominal seed:
a 40-point nuisance spread against a 1.98-point headline effect.  Nothing in the
old results/ was a controlled comparison.

Usage
-----
    python scripts/run_suite.py --config configs/suite.yaml --workers 4
    python scripts/run_suite.py --smoke --output-dir /tmp/smoke
    python scripts/run_suite.py --dry-run          # project wall time, run nothing
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
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

from dsa.envs import spectrum  # noqa: E402
from dsa.envs.config import obs_dim_for  # noqa: E402
from dsa.envs.heuristics import BASELINE_KEYS, HEURISTIC_REGISTRY  # noqa: E402
from dsa.envs.scenarios import get_scenario  # noqa: E402
from dsa.envs.vector import SyncVectorEnv  # noqa: E402
from dsa.learner.evaluate import evaluate_policy  # noqa: E402
from dsa.learner.ppo import PPOHyperParams, RecurrentPPO, train  # noqa: E402
from dsa.models.registry import MODEL_REGISTRY, build_model, model_summary  # noqa: E402
from dsa.seeding import derive_seed, make_torch_generator  # noqa: E402
from scripts.probe_throughput import probe_models, project_wall_minutes  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "suite.yaml"


# --------------------------------------------------------------------------- #
# Seed plumbing -- the role table of spec 3.2, in one place
# --------------------------------------------------------------------------- #


def eval_episode_seeds(base_seed: int, scenario: str, n: int) -> list[int]:
    """Evaluation stream.  Deliberately excludes the model name, so every policy
    is scored on byte-identical episodes and the paired statistics are valid."""
    return [derive_seed(base_seed, scenario, "env", "eval", i) for i in range(n)]


def train_lane_seeds(base_seed: int, scenario: str, num_envs: int, rollout: int) -> list[int]:
    """Training stream for one rollout batch.  Also excludes the model name."""
    return [derive_seed(base_seed, scenario, "env", "train", lane, rollout) for lane in range(num_envs)]


# --------------------------------------------------------------------------- #
# One cell.  Must be a module-level function so ProcessPoolExecutor can pickle it.
# --------------------------------------------------------------------------- #


def run_cell(spec: dict) -> dict:
    """Train (if trainable) and evaluate one (scenario, policy, seed) cell."""
    torch.set_num_threads(1)  # parallelism lives across processes, not inside them

    scenario = spec["scenario"]
    policy_key = spec["model"]
    seed = int(spec["seed"])
    eval_episodes = int(spec["eval_episodes"])

    config = get_scenario(scenario)
    obs_dim = obs_dim_for(config.num_channels)
    action_dim = config.num_channels
    eval_seeds = eval_episode_seeds(seed, scenario, eval_episodes)

    started = time.perf_counter()
    row: dict[str, object] = {
        "scenario": scenario,
        "model": policy_key,
        "seed": seed,
        "eval_episodes": eval_episodes,
    }

    if policy_key in HEURISTIC_REGISTRY:
        policy = HEURISTIC_REGISTRY[policy_key](
            action_dim, derive_seed(seed, "heuristic", policy_key, scenario)
        )
        eval_start = time.perf_counter()
        summary, _ = evaluate_policy(policy, config, eval_seeds)
        eval_seconds = time.perf_counter() - eval_start

        row.update(
            {
                "policy_type": "heuristic",
                "is_trainable": False,
                "family": "heuristic",
                "dt_aware": False,
                "cell_kinds": "",
                "hidden_dim": 0,
                "num_heads": 0,
                "parameter_count": 0,
                "param_error_frac": float("nan"),
                "train_steps": 0,
                "declared_train_steps": 0,
                "num_updates": 0,
                "total_gradient_steps": 0,
                "max_first_epoch_ratio_deviation": float("nan"),
                "final_policy_loss": float("nan"),
                "final_value_loss": float("nan"),
                "final_approx_kl": float("nan"),
                "final_clip_fraction": float("nan"),
                "final_explained_variance": float("nan"),
                # Mechanistic columns.  A heuristic runs no update, so it has no
                # critic and no behaviour-policy distribution: NaN, not zero.
                "first_update_explained_variance": float("nan"),
                "first_update_policy_entropy_mean": float("nan"),
                "final_policy_entropy_mean": float("nan"),
                "first_update_mean_rollout_return": float("nan"),
                "final_mean_rollout_return": float("nan"),
                # A zero-parameter heuristic has no initialisation to improve on,
                # so its "untrained" and "trained" scores are the same number by
                # definition.  Recorded explicitly rather than left blank so the
                # baseline is never silently dropped from a table.
                "mean_eval_return_untrained": float(summary["mean_eval_return"]),
                "mean_eval_return_trained": float(summary["mean_eval_return"]),
                "wall_train_seconds": 0.0,
                "wall_eval_seconds": float(eval_seconds),
            }
        )
        for key, value in sorted(summary.items()):
            row[key] = float(value)
        row["wall_seconds"] = time.perf_counter() - started
        return row

    # ---------------------------------------------------------------- neural #
    summary_spec = model_summary(policy_key, obs_dim, action_dim)
    model = build_model(policy_key, obs_dim, action_dim, derive_seed(seed, "policy_init", policy_key))

    # Learning check, half one: score the freshly initialised weights on the
    # identical evaluation stream the trained model will see.
    eval_start = time.perf_counter()
    untrained_summary, _ = evaluate_policy(model, config, eval_seeds)
    untrained_eval_seconds = time.perf_counter() - eval_start

    hp = PPOHyperParams(**spec["ppo"])
    agent = RecurrentPPO(
        model,
        hp,
        action_generator=make_torch_generator(seed, "action", policy_key, scenario),
        shuffle_generator=make_torch_generator(seed, "shuffle", policy_key, scenario),
    )
    vec = SyncVectorEnv(config, train_lane_seeds(seed, scenario, hp.num_envs, 0))

    train_start = time.perf_counter()
    logs = train(
        agent,
        vec,
        int(spec["training_steps"]),
        lambda r: train_lane_seeds(seed, scenario, hp.num_envs, r),
    )
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    trained_summary, _ = evaluate_policy(model, config, eval_seeds)
    trained_eval_seconds = time.perf_counter() - eval_start

    ratio_devs = [float(l["first_epoch_max_ratio_deviation"]) for l in logs if "first_epoch_max_ratio_deviation" in l]
    last = logs[-1] if logs else {}
    # Mechanistic evidence that the update did something, recorded at both ends of
    # training rather than only at the end.  `explained_variance` is the critic's
    # fit to its own GAE returns on the rollout it was about to be trained on, and
    # `policy_entropy_mean` is the genuine distributional entropy of the behaviour
    # policy (Categorical(logits).entropy()), not an action histogram.  Without the
    # first-update values a flat entropy trace and a falling one are indistinguishable.
    first = logs[0] if logs else {}

    row.update(
        {
            "policy_type": "neural",
            "is_trainable": True,
            "family": str(summary_spec["family"]),
            "dt_aware": bool(summary_spec["dt_aware"]),
            "cell_kinds": "+".join(summary_spec["cell_kinds"]),
            "hidden_dim": int(summary_spec["hidden_dim"]),
            "num_heads": int(summary_spec["num_heads"] or 0),
            "parameter_count": int(summary_spec["parameter_count"]),
            "param_error_frac": float(summary_spec["param_error_frac"]),
            "train_steps": int(agent.total_env_steps),
            "declared_train_steps": int(spec["training_steps"]),
            "num_updates": len(logs),
            "total_gradient_steps": int(agent.total_gradient_steps),
            "max_first_epoch_ratio_deviation": float(max(ratio_devs)) if ratio_devs else float("nan"),
            "final_policy_loss": float(last.get("policy_loss", float("nan"))),
            "final_value_loss": float(last.get("value_loss", float("nan"))),
            "final_approx_kl": float(last.get("approx_kl", float("nan"))),
            "final_clip_fraction": float(last.get("clip_fraction", float("nan"))),
            "final_explained_variance": float(last.get("explained_variance", float("nan"))),
            "first_update_explained_variance": float(first.get("explained_variance", float("nan"))),
            "first_update_policy_entropy_mean": float(first.get("policy_entropy_mean", float("nan"))),
            "final_policy_entropy_mean": float(last.get("policy_entropy_mean", float("nan"))),
            "first_update_mean_rollout_return": float(first.get("mean_rollout_return", float("nan"))),
            "final_mean_rollout_return": float(last.get("mean_rollout_return", float("nan"))),
            "mean_eval_return_untrained": float(untrained_summary["mean_eval_return"]),
            "mean_eval_return_trained": float(trained_summary["mean_eval_return"]),
            "wall_train_seconds": float(train_seconds),
            "wall_eval_seconds": float(untrained_eval_seconds + trained_eval_seconds),
        }
    )
    for key, value in sorted(trained_summary.items()):
        row[key] = float(value)
    for key, value in sorted(untrained_summary.items()):
        row[f"untrained_{key}"] = float(value)
    row["wall_seconds"] = time.perf_counter() - started
    return row


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: pathlib.Path) -> str:
    import hashlib

    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    out_dir: pathlib.Path,
    config: dict,
    matrix: dict,
    probe: list[dict] | None,
    projection: dict | None,
    guard_note: str | None,
    argv: list[str],
    csv_paths: list[pathlib.Path],
) -> pathlib.Path:
    """The reproducibility anchor.  Weights are not committed; this is."""
    obs_dim = obs_dim_for(get_scenario(matrix["scenarios"][0]).num_channels)
    action_dim = get_scenario(matrix["scenarios"][0]).num_channels

    manifest = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "command_line": ["python"] + argv,
        "generated_by": "scripts/run_suite.py",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "config_file": str(matrix.get("config_path", "")),
        "config_contents": config,
        "preregistration_sha256": _sha256(REPO_ROOT / "configs" / "preregistration.yaml"),
        "matrix": {
            "scenarios": matrix["scenarios"],
            "models": matrix["models"],
            "baselines": matrix["baselines"],
            "seeds": matrix["seeds"],
            "training_steps": matrix["training_steps"],
            "eval_episodes": matrix["eval_episodes"],
            "num_cells": matrix["num_cells"],
        },
        "ppo_hyperparameters": matrix["ppo"],
        "models_solved": {
            key: model_summary(key, obs_dim, action_dim) for key in sorted(MODEL_REGISTRY)
        },
        # Four environment constants live at module level in dsa/envs/spectrum.py
        # rather than as ScenarioConfig fields, because the config's field list is
        # frozen by the specification.  They are recorded here so a run is
        # reproducible from the manifest alone and nobody has to read source to
        # learn what dynamics produced a number.  docs/PROTOCOL.md section 2.0
        # explains what BG_HOP_RATE buys and why the task is partially observed.
        "environment_constants": {
            "background_hop_rate": spectrum.BG_HOP_RATE,
            "occupancy_relaxation_rate": spectrum._BUSY_RATE,
            "quality_relaxation_rate": spectrum._QUALITY_RATE,
            "interference_relaxation_rate": spectrum._INTERFERENCE_RATE,
            "sensed_occupancy_reports": "primary occupant only; "
            "background devices are hidden terminals",
        },
        "probe": probe,
        "projection": projection,
        "guard": {
            "applied": guard_note is not None,
            "note": guard_note,
            "requested_training_steps": matrix.get("requested_training_steps"),
            "effective_training_steps": matrix["training_steps"],
            "seed_count_reduced": False,
        },
        "outputs": {str(p.name): _sha256(p) for p in csv_paths},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _resolve_matrix(config: dict, args: argparse.Namespace) -> dict:
    block = dict(config)
    if args.smoke:
        block.update(config.get("smoke") or {})

    models = args.models if args.models else list(block["models"])
    baselines = args.baselines if args.baselines is not None else list(block.get("baselines", BASELINE_KEYS))
    scenarios = args.scenarios if args.scenarios else list(block["scenarios"])
    seeds = [int(s) for s in (args.seeds if args.seeds else block["seeds"])]
    training_steps = int(args.training_steps or block["training_steps"])

    ppo = dict(config.get("ppo") or {})
    ppo.update(block.get("ppo") or {})

    return {
        "models": list(models),
        "baselines": list(baselines),
        "scenarios": list(scenarios),
        "seeds": seeds,
        "training_steps": training_steps,
        "requested_training_steps": training_steps,
        "eval_episodes": int(args.eval_episodes or block["eval_episodes"]),
        "num_workers": int(args.workers or block.get("num_workers", 4)),
        "ppo": ppo,
        "config_path": str(args.config),
        "num_cells": len(scenarios) * len(seeds) * (len(models) + len(baselines)),
    }


def _apply_guard(matrix: dict, config: dict, verbose: bool = True) -> tuple[dict, list[dict], dict, str | None]:
    """Re-measure throughput and shrink ``training_steps`` if the budget blows.

    Seeds are never reduced: statistical power is the one thing the budget is
    not allowed to buy back.  Any reduction is recorded in the manifest with its
    reason, so a shortened run can never be mistaken for the full one.
    """
    guard = config.get("guard") or {}
    limit = float(guard.get("max_projected_wall_minutes", 40.0))
    batch = int(guard.get("rollout_batch", 1024))

    hp = PPOHyperParams(**matrix["ppo"])
    if verbose:
        print(f"probing throughput for {len(matrix['models'])} model(s)...", flush=True)
    probe = probe_models(matrix["models"], rollouts=2, hparams=hp)
    cells_per_model = len(matrix["scenarios"]) * len(matrix["seeds"])

    projection = project_wall_minutes(
        probe,
        matrix["training_steps"],
        cells_per_model,
        matrix["eval_episodes"],
        hp.horizon,
        matrix["num_workers"],
    )
    note = None
    if projection["wall_minutes"] > limit:
        per_step = projection["core_seconds_per_env_step_all_models"]
        budget_core_seconds = limit * 60.0 * matrix["num_workers"] * 0.98
        eval_effective = matrix["eval_episodes"] * hp.horizon * 2 * 0.25
        affordable = budget_core_seconds / (cells_per_model * per_step) - eval_effective
        reduced = max(batch, int(affordable // batch) * batch)
        note = (
            f"measured throughput on this machine projects {projection['wall_minutes']:.1f} min "
            f"for training_steps={matrix['training_steps']}, above the {limit:.0f} min guard; "
            f"training_steps reduced to {reduced} (largest multiple of {batch} that fits). "
            f"Seed count held at {len(matrix['seeds'])} -- the guard never reduces seeds."
        )
        matrix["training_steps"] = reduced
        projection = project_wall_minutes(
            probe, reduced, cells_per_model, matrix["eval_episodes"], hp.horizon, matrix["num_workers"]
        )
        if verbose:
            print(f"GUARD: {note}", flush=True)
    if verbose:
        print(
            f"projection: {projection['wall_minutes']:.1f} min "
            f"({projection['core_seconds']:.0f} core-s / {matrix['num_workers']} workers)",
            flush=True,
        )
    return matrix, probe, projection, note


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=pathlib.Path, default=REPO_ROOT / "results")
    parser.add_argument("--models", nargs="+", default=None, choices=list(MODEL_REGISTRY))
    parser.add_argument("--baselines", nargs="*", default=None, choices=list(HEURISTIC_REGISTRY))
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny CI budget from the config's smoke block")
    parser.add_argument("--dry-run", action="store_true", help="probe and project, run nothing")
    parser.add_argument("--no-guard", action="store_true", help="skip the throughput guard (not for confirmatory runs)")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    config = yaml.safe_load(args.config.read_text())
    matrix = _resolve_matrix(config, args)

    probe = projection = guard_note = None
    if args.dry_run or not args.no_guard:
        matrix, probe, projection, guard_note = _apply_guard(matrix, config)

    if args.dry_run:
        print(json.dumps({"matrix": {k: v for k, v in matrix.items() if k != "ppo"}, "projection": projection}, indent=2, default=str))
        return 0

    policies = list(matrix["models"]) + list(matrix["baselines"])
    specs = [
        {
            "scenario": scenario,
            "model": policy,
            "seed": seed,
            "eval_episodes": matrix["eval_episodes"],
            "training_steps": matrix["training_steps"],
            "ppo": matrix["ppo"],
        }
        for scenario in matrix["scenarios"]
        for policy in policies
        for seed in matrix["seeds"]
    ]

    print(f"running {len(specs)} cells on {matrix['num_workers']} workers", flush=True)
    started = time.perf_counter()
    rows: list[dict] = []
    if matrix["num_workers"] <= 1:
        for i, spec in enumerate(specs, 1):
            rows.append(run_cell(spec))
            print(f"  [{i}/{len(specs)}] {spec['scenario']}/{spec['model']}/{spec['seed']}", flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=matrix["num_workers"]) as pool:
            for i, row in enumerate(pool.map(run_cell, specs), 1):
                rows.append(row)
                if i % 10 == 0 or i == len(specs):
                    print(f"  [{i}/{len(specs)}] {time.perf_counter() - started:.0f}s", flush=True)

    # Sorted before writing, so the CSV is independent of completion order.
    frame = pd.DataFrame(rows).sort_values(["scenario", "model", "seed"]).reset_index(drop=True)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_runs.csv"
    frame.to_csv(csv_path, index=False)

    manifest = write_manifest(out_dir, config, matrix, probe, projection, guard_note, argv, [csv_path])
    elapsed = time.perf_counter() - started
    print(f"\nwrote {csv_path} ({len(frame)} rows) and {manifest} in {elapsed:.0f}s")

    bad = frame["max_first_epoch_ratio_deviation"].dropna()
    if len(bad) and float(bad.max()) > 1e-3:
        print(
            f"WARNING: first-epoch PPO ratio deviated by {float(bad.max()):.2e} (> 1e-3). "
            "That indicates a stale-hidden-state or train/eval-mode bug; the analysis "
            "layer will refuse to build a report from these runs.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
