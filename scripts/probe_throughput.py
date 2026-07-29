#!/usr/bin/env python3
"""Measure real rollout+update throughput on THIS box, for THIS environment.

Nothing in this repository quotes a throughput number that was not produced by
this script.  ``scripts/run_suite.py --dry-run`` imports ``probe_models`` and
refuses to start a run whose projected wall time exceeds the configured guard,
so the budget is checked against measured reality rather than against a comment.

Usage
-----
    python scripts/probe_throughput.py
    python scripts/probe_throughput.py --models ppo_ltc ppo_cfc --rollouts 3
    python scripts/probe_throughput.py --json results/probe.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

import torch

# Make `python scripts/foo.py` work from a clean checkout without installation.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dsa.envs.config import obs_dim_for  # noqa: E402
from dsa.envs.scenarios import get_scenario  # noqa: E402
from dsa.envs.vector import SyncVectorEnv  # noqa: E402
from dsa.learner.ppo import PPOHyperParams, RecurrentPPO  # noqa: E402
from dsa.models.registry import MODEL_REGISTRY, PRIMARY_MODELS, build_model, model_summary  # noqa: E402
from dsa.seeding import derive_seed, make_torch_generator  # noqa: E402

DEFAULT_ROLLOUTS = 3
PROBE_SCENARIO = "irregular_dt"
PROBE_SEED = 7

#: Evaluation is rollout-only (no gradient), costed at a quarter of a training
#: step.  Each run evaluates twice -- untrained and trained.
EVAL_COST_FACTOR = 0.25
EVALS_PER_RUN = 2


def measure_model_throughput(
    model_key: str,
    scenario: str = PROBE_SCENARIO,
    rollouts: int = DEFAULT_ROLLOUTS,
    hparams: PPOHyperParams | None = None,
    seed: int = PROBE_SEED,
) -> dict[str, float | str | int]:
    """Time ``rollouts`` full collect+update cycles.  Returns steps/s and cost."""
    torch.set_num_threads(1)
    hp = hparams or PPOHyperParams()
    config = get_scenario(scenario)
    obs_dim = obs_dim_for(config.num_channels)
    action_dim = config.num_channels

    model = build_model(model_key, obs_dim, action_dim, derive_seed(seed, "policy_init", model_key))
    agent = RecurrentPPO(
        model,
        hp,
        action_generator=make_torch_generator(seed, "action", model_key, scenario),
        shuffle_generator=make_torch_generator(seed, "shuffle", model_key, scenario),
    )

    def lane_seeds(r: int) -> list[int]:
        return [derive_seed(seed, scenario, "env", "train", lane, r) for lane in range(hp.num_envs)]

    vec = SyncVectorEnv(config, lane_seeds(0))

    # One untimed cycle: the first pass pays lazy allocation and kernel warm-up
    # that the steady-state figure should not carry.
    vec.reseed(lane_seeds(0))
    agent.update(agent.collect(vec))

    start = time.perf_counter()
    for r in range(rollouts):
        vec.reseed(lane_seeds(r + 1))
        agent.update(agent.collect(vec))
    elapsed = time.perf_counter() - start

    env_steps = rollouts * hp.num_envs * hp.horizon
    steps_per_second = env_steps / elapsed if elapsed > 0 else float("inf")
    summary = model_summary(model_key, obs_dim, action_dim)
    return {
        "model": model_key,
        "hidden_dim": int(summary["hidden_dim"]),
        "parameter_count": int(summary["parameter_count"]),
        "env_steps": int(env_steps),
        "wall_seconds": float(elapsed),
        "steps_per_second": float(steps_per_second),
        "core_seconds_per_env_step": float(1.0 / steps_per_second) if steps_per_second else float("inf"),
    }


def probe_models(
    model_keys=PRIMARY_MODELS,
    scenario: str = PROBE_SCENARIO,
    rollouts: int = DEFAULT_ROLLOUTS,
    hparams: PPOHyperParams | None = None,
) -> list[dict[str, float | str | int]]:
    return [measure_model_throughput(k, scenario, rollouts, hparams) for k in model_keys]


def project_wall_minutes(
    probe: list[dict],
    training_steps: int,
    num_cells_per_model: int,
    eval_episodes: int,
    horizon: int,
    num_workers: int,
    contention: float = 0.98,
) -> dict[str, float]:
    """Project suite wall time from measured per-model cost.

    ``effective_steps_per_run`` charges evaluation at ``EVAL_COST_FACTOR`` of a
    training step and counts both the untrained and the trained evaluation.
    """
    eval_effective = eval_episodes * horizon * EVALS_PER_RUN * EVAL_COST_FACTOR
    effective_steps = float(training_steps) + eval_effective
    core_seconds_per_step = sum(float(p["core_seconds_per_env_step"]) for p in probe)
    core_seconds = num_cells_per_model * core_seconds_per_step * effective_steps
    wall_seconds = core_seconds / max(1, num_workers) / contention
    return {
        "effective_steps_per_run": effective_steps,
        "core_seconds_per_env_step_all_models": core_seconds_per_step,
        "core_seconds": core_seconds,
        "wall_seconds": wall_seconds,
        "wall_minutes": wall_seconds / 60.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODELS), choices=list(MODEL_REGISTRY))
    parser.add_argument("--scenario", default=PROBE_SCENARIO)
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--json", type=pathlib.Path, default=None, help="write the measurements here")
    args = parser.parse_args()

    torch.set_num_threads(1)
    print(f"python {platform.python_version()}  torch {torch.__version__}  threads {torch.get_num_threads()}")
    print(f"scenario={args.scenario}  rollouts={args.rollouts}\n")
    print(f"{'model':<20} {'H':>4} {'params':>8} {'steps/s':>10} {'core-s/step':>13}")
    print("-" * 60)

    probe = []
    for key in args.models:
        row = measure_model_throughput(key, args.scenario, args.rollouts)
        probe.append(row)
        print(
            f"{row['model']:<20} {row['hidden_dim']:>4} {row['parameter_count']:>8} "
            f"{row['steps_per_second']:>10.1f} {row['core_seconds_per_env_step']:>13.2e}"
        )

    total = sum(float(p["core_seconds_per_env_step"]) for p in probe)
    print("-" * 60)
    print(f"{'TOTAL':<20} {'':>4} {'':>8} {'':>10} {total:>13.2e}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(probe, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
