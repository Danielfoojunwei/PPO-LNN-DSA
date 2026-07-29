#!/usr/bin/env python3
"""EXPLORATORY diagnostic 3: does the LTC learn once its maps get gradient?

NOT PRE-REGISTERED.  Writes only under ``results/diagnostics/``.  Study A and
Study B are untouched.  Nothing here may be cited as a confirmatory result.

The chain so far
----------------
* Study B: ``ppo_ltc`` and ``ppo_ltc_cfc`` are the only two of seven
  architectures that fail to beat their own initialisation, at 163,840 steps.
* Diagnostic 1: **refuted** the obvious explanation.  Recurrent state and
  ``d x_T / d x_0`` decay steeply for *every* cell here -- the CfC's gradient
  ratio is eight orders of magnitude *smaller* than the LTC's, and the CfC
  learns.  Vanishing recurrent gradient is not the discriminator.
* Diagnostic 2: on the real registry models and a real first-update PPO loss,
  ``ppo_ltc``'s recurrent block has a median update-to-weight ratio of ~5.9e-05
  against 1.9e-03 to 3.9e-03 for every other model.  The per-parameter split
  localises it: ``A`` receives healthy gradient while ``input_map``,
  ``recurrent_map`` and ``log_tau`` receive two to three orders less.

Mechanism proposed
------------------
In the fused update

    x <- (x + h A f) / (1 + h (1/tau + f)),   f = sigmoid(W_x u + W_h x + b),
    h = dt / unfolds

the synaptic maps reach the state only through ``f``, and ``f``'s pathway into
``x`` is scaled by ``A`` (numerator) and by ``x`` (denominator), then by
``sigmoid'`` <= 1/4 and by ``h = 1/6``.  ``custom_init_`` sets
``A ~ N(0, 0.1^2)``.  So the maps are starved by roughly ``|A|`` relative to
``A`` itself, and with a learning rate shared across all seven models they move
~50x less per step than any competitor's recurrent weights.

If that is right, the Study B failure is an initialisation/tuning artifact of
this implementation, not a property of liquid time-constant networks.

Prediction, tested here
-----------------------
Two interventions should each rescue ``ppo_ltc`` on ``stationary``, where Study B
found a *negative* point estimate:

  ``a_scale``  -- multiply ``A`` at initialisation by 10 (0.1 -> ~1.0), feeding
                  the maps proportionally more gradient.  Changes the model.
  ``lr10``     -- multiply only the learning rate by 10.  Changes the optimiser.

``default`` reproduces Study B's configuration exactly and is the control;
``ppo_gru`` confirms the harness itself learns at this budget.  If neither
intervention moves ``ppo_ltc``, the mechanism above is wrong and the failure is
something else.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import importlib.util
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dsa.envs import SyncVectorEnv, get_scenario, obs_dim_for  # noqa: E402
from dsa.learner.evaluate import evaluate_policy  # noqa: E402
from dsa.learner.ppo import PPOHyperParams, RecurrentPPO, train  # noqa: E402
from dsa.models import build_model  # noqa: E402
from dsa.seeding import derive_seed, make_torch_generator  # noqa: E402

# eval_episode_seeds / train_lane_seeds live in run_suite.py, not in dsa.seeding.
# They are imported rather than copied on purpose: a divergent copy would give
# this diagnostic a different environment stream from Study B and silently
# invalidate every comparison against it.
_rs_spec = importlib.util.spec_from_file_location(
    "_run_suite", pathlib.Path(__file__).resolve().parent / "run_suite.py"
)
_run_suite = importlib.util.module_from_spec(_rs_spec)
_rs_spec.loader.exec_module(_run_suite)
eval_episode_seeds = _run_suite.eval_episode_seeds
train_lane_seeds = _run_suite.train_lane_seeds

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "diagnostics"

BASE_PPO = dict(
    num_envs=16,
    horizon=64,
    update_epochs=6,
    num_sequence_minibatches=4,
    learning_rate=3.0e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
)

ARMS = [
    ("ppo_ltc", "default", {}),
    ("ppo_ltc", "a_scale10", {"a_scale": 10.0}),
    ("ppo_ltc", "lr10", {"lr_mult": 10.0}),
    ("ppo_gru", "default", {}),
]


def run_one(spec: dict) -> dict:
    torch.set_num_threads(1)
    key, arm, seed = spec["model"], spec["arm"], int(spec["seed"])
    scenario, steps = spec["scenario"], int(spec["training_steps"])

    config = get_scenario(scenario)
    obs_dim = obs_dim_for(config.num_channels)
    action_dim = config.num_channels
    eval_seeds = eval_episode_seeds(seed, scenario, int(spec["eval_episodes"]))

    started = time.perf_counter()
    model = build_model(key, obs_dim, action_dim, derive_seed(seed, "policy_init", key))

    # The intervention, applied to the initialised weights and nowhere else.
    a_scale = float(spec["opts"].get("a_scale", 1.0))
    if a_scale != 1.0:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith(".A"):
                    p.mul_(a_scale)

    untrained, _ = evaluate_policy(model, config, eval_seeds)

    ppo = dict(BASE_PPO)
    ppo["learning_rate"] *= float(spec["opts"].get("lr_mult", 1.0))
    hp = PPOHyperParams(**ppo)
    agent = RecurrentPPO(
        model,
        hp,
        action_generator=make_torch_generator(seed, "action", key, scenario),
        shuffle_generator=make_torch_generator(seed, "shuffle", key, scenario),
    )
    vec = SyncVectorEnv(config, train_lane_seeds(seed, scenario, hp.num_envs, 0))
    logs = train(agent, vec, steps, lambda r: train_lane_seeds(seed, scenario, hp.num_envs, r))
    trained, _ = evaluate_policy(model, config, eval_seeds)

    first, last = (logs[0] if logs else {}), (logs[-1] if logs else {})
    return {
        "model": key,
        "arm": arm,
        "seed": seed,
        "scenario": scenario,
        "training_steps": steps,
        "learning_rate": ppo["learning_rate"],
        "a_scale": a_scale,
        "untrained_return": float(untrained["mean_eval_return"]),
        "trained_return": float(trained["mean_eval_return"]),
        "delta": float(trained["mean_eval_return"]) - float(untrained["mean_eval_return"]),
        "first_explained_variance": float(first.get("explained_variance", float("nan"))),
        "final_explained_variance": float(last.get("explained_variance", float("nan"))),
        "first_policy_entropy": float(first.get("policy_entropy_mean", float("nan"))),
        "final_policy_entropy": float(last.get("policy_entropy_mean", float("nan"))),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="stationary")
    ap.add_argument("--training-steps", type=int, default=163840)
    ap.add_argument("--eval-episodes", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 23, 41])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "model": key,
            "arm": arm,
            "opts": opts,
            "seed": s,
            "scenario": args.scenario,
            "training_steps": args.training_steps,
            "eval_episodes": args.eval_episodes,
        }
        for key, arm, opts in ARMS
        for s in args.seeds
    ]
    print(f"EXPLORATORY: {len(specs)} cells, {args.training_steps} steps, {args.workers} workers")

    rows: list[dict] = []
    started = time.perf_counter()
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, specs):
            rows.append(row)
            print(
                f"  {row['model']:9s} {row['arm']:10s} seed={row['seed']:3d} "
                f"delta={row['delta']:+8.3f} ev={row['final_explained_variance']:+.3f} "
                f"H={row['final_policy_entropy']:.3f} ({row['wall_seconds']/60:.1f}m)",
                flush=True,
            )

    path = args.out / "intervention.json"
    path.write_text(
        json.dumps(
            {
                "exploratory": True,
                "pre_registered": False,
                "scenario": args.scenario,
                "training_steps": args.training_steps,
                "seeds": args.seeds,
                "wall_minutes": (time.perf_counter() - started) / 60.0,
                "rows": rows,
            },
            indent=2,
        )
    )

    print(f"\n-> {path}\n")
    print(f"{'model':10s} {'arm':11s} {'mean delta':>11s} {'min':>9s} {'max':>9s} {'n>0':>5s}")
    for key, arm, _ in ARMS:
        sel = [r["delta"] for r in rows if r["model"] == key and r["arm"] == arm]
        if not sel:
            continue
        t = torch.tensor(sel)
        print(
            f"{key:10s} {arm:11s} {t.mean():+11.3f} {t.min():+9.3f} {t.max():+9.3f} "
            f"{sum(1 for v in sel if v > 0):3d}/{len(sel)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
