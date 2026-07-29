#!/usr/bin/env python3
"""EXPLORATORY diagnostic 2: parameter-gradient scale on the real models.

NOT PRE-REGISTERED.  Writes only under ``results/diagnostics/``.  Study A and
Study B are untouched.

Why this exists
---------------
Diagnostic 1 (``scripts/diagnose_ltc.py``) tested the obvious hypothesis -- that
the LTC's state and its backward gradient decay too fast to carry credit across
a 64-step episode -- and **refuted it**.  Over 64 steps the CfC retains *less*
state than the LTC and its ``d x_T / d x_0`` is eight orders of magnitude
smaller, yet the CfC learns and the LTC does not.  The GRU's gradient vanishes
too.  Recurrent-gradient decay is therefore common to every cell here and cannot
be what separates the trainable ones from the untrainable ones.

That leaves the other live confound named in the README: the models share a
learning rate of 3e-4, and nothing has ever checked that their *parameter*
gradients live on the same scale.  If the LTC's are systematically much smaller
or much larger, a shared step size is either a no-op or a divergence for it, and
the "LTC does not learn" result is a tuning artifact rather than a property of
the architecture.

Unlike diagnostic 1 this builds the **real registry models** through
``build_model``, so the initialisation, capacity matching, encoder, norms and
heads are exactly those used by both studies.

Measured, per model, on an identical batch of real environment trajectories:
  * per-parameter-group grad norm, and the global grad norm
  * the update-to-weight ratio ``lr * ||g|| / ||w||`` -- the scale-free quantity
    that says whether 3e-4 is a sensible step for that tensor
  * for the LTC specifically, the split between ``log_tau`` / ``A`` and the maps
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dsa.envs import SpectrumEnv, get_scenario  # noqa: E402
from dsa.models import build_model  # noqa: E402
from dsa.models.registry import MODEL_REGISTRY  # noqa: E402
from dsa.seeding import derive_seed  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "diagnostics"

# Derived from the registry, never typed. A literal list of policy keys is the
# shape of defect D4 -- the pre-rebuild analysis layer filtered random_policy out
# of every published table with exactly such a list -- and tests/test_no_filter.py
# rejects any list literal naming three or more registry keys. Deriving it also
# means a newly registered model is diagnosed automatically instead of silently
# skipped.
MODELS = sorted(MODEL_REGISTRY)
LR = 3.0e-4  # configs/suite_study_b.yaml


def collect_batch(scenario: str, seed: int, episodes: int, horizon: int):
    """Real observations/dt/actions/returns from the actual environment."""
    cfg = get_scenario(scenario)
    obs_b, dt_b, act_b, ret_b = [], [], [], []
    for ep in range(episodes):
        env = SpectrumEnv(cfg, episode_seed=derive_seed(seed, scenario, "diag", ep))
        obs, dts, rews, acts = [], [], [], []
        o, _ = env.reset()
        gen = torch.Generator().manual_seed(derive_seed(seed, scenario, "act", ep))
        for _ in range(horizon):
            a = int(torch.randint(0, env.action_dim, (1,), generator=gen).item())
            obs.append(torch.as_tensor(o["obs"], dtype=torch.float32))
            dts.append(float(o["dt"]))
            o, r, terminated, truncated, _ = env.step(a)
            rews.append(float(r))
            acts.append(a)
            if terminated or truncated:
                break
        n = len(obs)
        if n < horizon:
            continue
        ret, acc = [0.0] * n, 0.0
        for t in reversed(range(n)):
            acc = rews[t] + 0.99 * acc
            ret[t] = acc
        obs_b.append(torch.stack(obs))
        dt_b.append(torch.tensor(dts, dtype=torch.float32))
        act_b.append(torch.tensor(acts, dtype=torch.long))
        ret_b.append(torch.tensor(ret, dtype=torch.float32))
    return (
        torch.stack(obs_b),
        torch.stack(dt_b),
        torch.stack(act_b),
        torch.stack(ret_b),
    )


def grad_report(key: str, batch, seed: int) -> dict:
    obs, dt, act, ret = batch
    model = build_model(key, obs.shape[-1], int(act.max()) + 1, seed)
    model.train()

    state = model.initial_state(obs.shape[0])
    logits, values, _ = model.unroll(obs, dt, state)

    adv = ret - values.detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    logp = torch.log_softmax(logits, dim=-1).gather(-1, act.unsqueeze(-1)).squeeze(-1)
    entropy = -(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1).mean()
    # A first-update PPO surrogate: ratio == 1, so this is the policy-gradient
    # term PPO actually starts from, plus the value loss and entropy bonus.
    loss = -(logp * adv).mean() + 0.5 * F.mse_loss(values, ret) - 0.01 * entropy

    model.zero_grad(set_to_none=True)
    loss.backward()

    groups: dict[str, dict] = {}
    total_sq = 0.0
    for name, p in model.named_parameters():
        g = 0.0 if p.grad is None else float(p.grad.norm())
        w = float(p.detach().norm())
        total_sq += g * g
        groups[name] = {
            "grad_norm": g,
            "weight_norm": w,
            "update_over_weight": (LR * g / w) if w > 0 else None,
            "numel": p.numel(),
        }

    block = {k: v for k, v in groups.items() if k.startswith("blocks.")}
    head = {k: v for k, v in groups.items() if not k.startswith("blocks.")}
    return {
        "model": key,
        "loss": float(loss),
        "entropy_nats": float(entropy),
        "global_grad_norm": total_sq**0.5,
        "recurrent_grad_norm": sum(v["grad_norm"] ** 2 for v in block.values()) ** 0.5,
        "head_grad_norm": sum(v["grad_norm"] ** 2 for v in head.values()) ** 0.5,
        "median_update_over_weight_recurrent": float(
            torch.tensor(
                [v["update_over_weight"] for v in block.values() if v["update_over_weight"]]
            ).median()
        )
        if block
        else None,
        "median_update_over_weight_head": float(
            torch.tensor(
                [v["update_over_weight"] for v in head.values() if v["update_over_weight"]]
            ).median()
        ),
        "per_parameter": groups,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="stationary")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)

    batch = collect_batch(args.scenario, args.seed, args.episodes, args.horizon)
    reports = [
        grad_report(k, batch, derive_seed(args.seed, "diag", "init", k)) for k in MODELS
    ]

    path = args.out / "param_gradients.json"
    path.write_text(
        json.dumps(
            {"scenario": args.scenario, "lr": LR, "batch": list(batch[0].shape), "models": reports},
            indent=2,
        )
    )

    print(f"EXPLORATORY diagnostic -> {path}")
    print(f"scenario={args.scenario} batch={tuple(batch[0].shape)} lr={LR}\n")
    print(
        f"{'model':18s} {'|g| total':>10s} {'|g| recur':>10s} {'|g| head':>9s} "
        f"{'lr|g|/|w| recur':>16s} {'head':>9s}"
    )
    for r in reports:
        mr = r["median_update_over_weight_recurrent"]
        print(
            f"{r['model']:18s} {r['global_grad_norm']:10.3e} {r['recurrent_grad_norm']:10.3e} "
            f"{r['head_grad_norm']:9.3e} "
            f"{(f'{mr:.3e}' if mr is not None else 'n/a'):>16s} "
            f"{r['median_update_over_weight_head']:9.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
