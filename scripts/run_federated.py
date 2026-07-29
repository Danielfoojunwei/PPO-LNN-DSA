#!/usr/bin/env python3
"""Run the federated topology study: three arms x N seeds, in parallel.

The study varies **topology** at fixed architecture, fixed scenario and strictly
matched compute.  Its conclusion is therefore about topology and is not evidence
about architecture; ``RESULTS.md`` says so.

Two properties this script exists to keep honest:

*   ``train_steps`` is the sum of ``agent.total_env_steps`` over every agent the
    arm actually trained, and it is written next to ``declared_train_steps`` with
    a boolean match column.  The previous version recorded 3,072 for every
    federated row while executing 18,432 -- a 6x compute advantage, unlabelled.
*   Communication is read from ``CommunicationLedger.totals()``, which counts
    real tensor bytes at the moment of each transfer.  The previous version
    hand-added a closed form, producing a flat 4/3 ratio with a standard
    deviation of exactly zero across all five seeds.

Usage
-----
    python scripts/run_federated.py --config configs/federated.yaml --workers 4
    python scripts/run_federated.py --smoke --output-dir /tmp/fed
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

from dsa.federated.runners import FEDERATED_ARMS, federated_row, run_arm  # noqa: E402
from dsa.federated.topology import FederatedConfig  # noqa: E402
from dsa.learner.ppo import PPOHyperParams  # noqa: E402
from dsa.models.registry import MODEL_REGISTRY  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "federated.yaml"


def run_fed_cell(spec: dict) -> dict:
    """One (arm, seed) cell.  Module-level so ProcessPoolExecutor can pickle it."""
    torch.set_num_threads(1)
    fed = FederatedConfig(**spec["topology"])
    hp = PPOHyperParams(**spec["ppo"]) if spec.get("ppo") else None
    result = run_arm(spec["arm"], fed, int(spec["seed"]), hp)
    # `fed` is passed so the topology snapshot (config_* columns) reaches the CSV.
    # ``RESULT_KEYS`` is frozen by the spec and has no config slot, so this
    # second argument is the ONLY route by which the arm's topology becomes
    # part of the committed evidence rather than something a reader must infer.
    return federated_row(result, fed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _resolve(config: dict, args: argparse.Namespace) -> dict:
    block = dict(config)
    if args.smoke:
        block.update(config.get("smoke") or {})

    topology = dict(config.get("topology") or {})
    topology.update(block.get("topology") or {})
    topology["model_key"] = args.model or block.get("model_key", config.get("model_key"))
    topology["scenario"] = args.scenario or block.get("scenario", config.get("scenario"))

    ppo = dict(config.get("ppo") or {})
    ppo.update(block.get("ppo") or {})
    # Federated clients run one whole rollout per local phase; the local vector
    # width is a property of the topology, not of the PPO block.
    ppo["num_envs"] = int(topology["local_num_envs"])
    ppo["horizon"] = int(topology["horizon"])

    return {
        "arms": args.arms if args.arms else list(block.get("arms", FEDERATED_ARMS)),
        "seeds": [int(s) for s in (args.seeds if args.seeds else block["seeds"])],
        "topology": topology,
        "ppo": ppo,
        "num_workers": int(args.workers or block.get("num_workers", 4)),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=pathlib.Path, default=REPO_ROOT / "results")
    parser.add_argument("--arms", nargs="+", default=None, choices=list(FEDERATED_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--model", default=None, choices=list(MODEL_REGISTRY))
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    config = yaml.safe_load(args.config.read_text())
    matrix = _resolve(config, args)

    fed_probe = FederatedConfig(**matrix["topology"])
    specs = [
        {"arm": arm, "seed": seed, "topology": matrix["topology"], "ppo": matrix["ppo"]}
        for arm in matrix["arms"]
        for seed in matrix["seeds"]
    ]
    print(
        f"running {len(specs)} federated cells on {matrix['num_workers']} workers; "
        f"every arm consumes exactly {fed_probe.total_env_steps_per_arm()} env steps",
        flush=True,
    )

    started = time.perf_counter()
    rows: list[dict] = []
    if matrix["num_workers"] <= 1:
        for i, spec in enumerate(specs, 1):
            rows.append(run_fed_cell(spec))
            print(f"  [{i}/{len(specs)}] {spec['arm']}/{spec['seed']}", flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=matrix["num_workers"]) as pool:
            for i, row in enumerate(pool.map(run_fed_cell, specs), 1):
                rows.append(row)
                print(f"  [{i}/{len(specs)}] {time.perf_counter() - started:.0f}s", flush=True)

    frame = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "federated_all_runs.csv"
    frame.to_csv(csv_path, index=False)

    manifest_path = out_dir / "federated_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "git_commit": _git_commit(),
                "command_line": ["python"] + argv,
                "generated_by": "scripts/run_federated.py",
                "versions": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
                "config_contents": config,
                "matrix": {k: v for k, v in matrix.items()},
                "topology_resolved": fed_probe.to_dict(),
                "declared_train_steps_per_arm": fed_probe.total_env_steps_per_arm(),
                "wall_seconds": time.perf_counter() - started,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    if "train_steps_match_declared" in frame.columns:
        mismatched = frame[~frame["train_steps_match_declared"].astype(bool)]
        if len(mismatched):
            print(
                f"ERROR: {len(mismatched)} federated rows executed a different number of env "
                "steps than they declared. Compute matching is broken.",
                file=sys.stderr,
            )
            return 1

    print(f"\nwrote {csv_path} ({len(frame)} rows) and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
