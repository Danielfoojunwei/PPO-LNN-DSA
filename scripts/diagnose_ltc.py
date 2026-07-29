#!/usr/bin/env python3
"""EXPLORATORY diagnostic: why does ppo_ltc fail to train?

NOT PRE-REGISTERED.  Nothing produced by this script is confirmatory evidence,
nothing it emits may be cited as a result, and it writes only under
``results/diagnostics/``.  Study A and Study B are untouched by it.

Motivation
----------
Study B (163,840 env steps, 12 seeds, capacity-matched) found that five of
seven architectures beat their own random initialisation while ``ppo_ltc`` and
``ppo_ltc_cfc`` did not, on either scenario, with negative point estimates on
``stationary``.  A null like that is *undiagnosed*: "the LTC did not learn" is a
statement about this implementation until a mechanism is shown.  The live
confounds are gradient scale through the solver, a learning rate shared with the
other models, the ``tau`` initialisation range, and ODE stiffness.

Hypothesis under test
---------------------
The LTC update integrates ``dx/dt = -(1/tau + f) x + A f`` with

    tau_sys = tau / (1 + tau * f),      f = sigmoid(W_x u + W_h x + b)

so over one environment step of size ``dt`` the state retains a factor of
roughly ``exp(-dt / tau_sys)``.  At initialisation ``f ~ 0.5``, and with
``tau in [0.5, 8]`` that puts ``tau_sys`` in roughly ``[0.4, 1.6]`` against
``dt = 1``.  If so the hidden state — and with it the gradient flowing backwards
through the recurrence — decays by a large factor *every single timestep*, and
the cell cannot carry information across the 64-step episode no matter how it is
optimised.  That would make the failure structural rather than a tuning artifact,
and it would predict that changing ``unfolds`` does not help (the contraction is
a property of the continuous-time dynamics, not of the discretisation) while
raising ``tau_max`` does.

What is measured
----------------
1. ``retention``  -- ||x_t|| along a 64-step rollout, per cell type, and the
   fitted per-step retention factor.  Control: CfC and GRU under identical input.
2. ``gradient``   -- ||d x_T / d x_0||, i.e. how much gradient survives the full
   BPTT path, and the per-timestep profile ||d x_T / d x_t||.  This is the
   quantity that decides whether PPO can credit an early action.
3. ``tau_sys``    -- the realised distribution of tau_sys at initialisation,
   which is what the hypothesis is actually about.
4. ``sweep``      -- (learning rate x unfolds x tau_max) on the LTC alone,
   scored by whether the resulting policy beats its own initialisation.

Usage::

    python scripts/diagnose_ltc.py --stage mechanism      # fast, no training
    python scripts/diagnose_ltc.py --stage sweep          # slow, trains
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dsa.models.cells import CfCCell, GRUBlock, LTCCell  # noqa: E402
from dsa.seeding import derive_seed  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "diagnostics"

HORIZON = 64  # the episode length used by every scenario in both studies
DT = 1.0  # irregular_dt varies this; stationary holds it here


def _build(kind: str, hidden: int, seed: int, **kw):
    gen = torch.Generator().manual_seed(seed)
    cls = {"ltc": LTCCell, "cfc": CfCCell, "gru": GRUBlock}[kind]
    cell = cls(input_dim=hidden, hidden_dim=hidden, **kw)
    for p in cell.parameters():
        if p.dim() > 1:
            torch.nn.init.orthogonal_(p, generator=gen)
    if hasattr(cell, "custom_init_"):
        cell.custom_init_(gen)
    return cell.double()


def measure_retention(kind: str, hidden: int, seed: int, **kw) -> dict:
    """||x_t|| along a rollout driven by a fixed-scale random input."""
    torch.manual_seed(seed)
    cell = _build(kind, hidden, seed, **kw)
    u = torch.randn(1, HORIZON, hidden, dtype=torch.float64) * 0.5
    dt = torch.full((1,), DT, dtype=torch.float64)

    x = torch.randn(1, cell.state_size, dtype=torch.float64)
    norms = [float(x.norm())]
    with torch.no_grad():
        for t in range(HORIZON):
            _, x = cell.forward(u[:, t], x, dt)
            norms.append(float(x.norm()))

    # Retention of the *initial condition* specifically: rerun with x0 = 0 and
    # measure how the difference decays.  This separates memory of the past from
    # the driven response to u.
    x0a = torch.randn(1, cell.state_size, dtype=torch.float64)
    x0b = x0a + torch.randn(1, cell.state_size, dtype=torch.float64) * 1e-3
    xa, xb, sep = x0a.clone(), x0b.clone(), [float((x0b - x0a).norm())]
    with torch.no_grad():
        for t in range(HORIZON):
            _, xa = cell.forward(u[:, t], xa, dt)
            _, xb = cell.forward(u[:, t], xb, dt)
            sep.append(float((xb - xa).norm()))

    ratios = [sep[t + 1] / sep[t] for t in range(HORIZON) if sep[t] > 0]
    return {
        "kind": kind,
        "state_norm": norms,
        "perturbation_norm": sep,
        "median_per_step_retention": float(torch.tensor(ratios).median()) if ratios else 0.0,
        "perturbation_surviving_full_episode": sep[-1] / sep[0] if sep[0] else 0.0,
        "steps_to_1e3_decay": next(
            (t for t, s in enumerate(sep) if s < sep[0] * 1e-3), None
        ),
    }


def measure_gradient(kind: str, hidden: int, seed: int, **kw) -> dict:
    """||d x_T / d x_t|| for every t -- the BPTT credit-assignment profile."""
    cell = _build(kind, hidden, seed, **kw)
    u = torch.randn(1, HORIZON, hidden, dtype=torch.float64) * 0.5
    dt = torch.full((1,), DT, dtype=torch.float64)

    states = [torch.randn(1, cell.state_size, dtype=torch.float64, requires_grad=True)]
    x = states[0]
    for t in range(HORIZON):
        _, x = cell.forward(u[:, t], x, dt)
        x.retain_grad()
        states.append(x)

    loss = states[-1].pow(2).sum()
    loss.backward()

    grads = [float(s.grad.norm()) if s.grad is not None else 0.0 for s in states]
    ref = grads[-1] if grads[-1] else 1.0
    return {
        "kind": kind,
        "grad_norm_by_t": grads,
        "grad_at_t0_over_grad_at_T": grads[0] / ref,
        "effective_horizon_steps": sum(1 for g in grads if g > ref * 1e-6),
    }


def measure_tau_sys(hidden: int, seed: int, **kw) -> dict:
    """The realised tau_sys distribution at initialisation, vs dt."""
    cell = _build("ltc", hidden, seed, **kw)
    u = torch.randn(256, hidden, dtype=torch.float64) * 0.5
    x = torch.randn(256, cell.state_size, dtype=torch.float64)
    with torch.no_grad():
        tau_sys = cell.system_time_constant(u, x)
        f = cell.synaptic_activation(u, x)
        tau = cell.tau()
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    return {
        "tau_min_max_configured": [float(tau.min()), float(tau.max())],
        "tau_sys_quantiles": [float(v) for v in torch.quantile(tau_sys.flatten(), q)],
        "f_median": float(f.median()),
        "dt": DT,
        # exp(-dt / tau_sys): the fraction of state surviving one env step
        "per_step_retention_quantiles": [
            float(math.exp(-DT / v)) for v in torch.quantile(tau_sys.flatten(), q)
        ],
    }


def stage_mechanism(args) -> dict:
    hidden = args.hidden
    out: dict = {"stage": "mechanism", "horizon": HORIZON, "dt": DT, "hidden": hidden}

    out["tau_sys_default"] = measure_tau_sys(hidden, derive_seed(7, "diag", "tau", 0))
    out["tau_sys_wide"] = measure_tau_sys(
        hidden, derive_seed(7, "diag", "tau", 0), tau_min=0.5, tau_max=200.0
    )

    out["retention"] = []
    out["gradient"] = []
    for kind, kw in [
        ("ltc", {}),
        ("ltc", {"unfolds": 1}),
        ("ltc", {"unfolds": 24}),
        ("ltc", {"tau_min": 0.5, "tau_max": 200.0}),
        ("cfc", {}),
        ("gru", {}),
    ]:
        label = kind + ("" if not kw else " " + ",".join(f"{k}={v}" for k, v in kw.items()))
        seed = derive_seed(7, "diag", label, 0)
        r = measure_retention(kind, hidden, seed, **kw)
        g = measure_gradient(kind, hidden, seed, **kw)
        r["label"] = g["label"] = label
        out["retention"].append(r)
        out["gradient"].append(g)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["mechanism"], default="mechanism")
    ap.add_argument("--hidden", type=int, default=77)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)

    result = stage_mechanism(args)
    path = args.out / f"{args.stage}.json"
    path.write_text(json.dumps(result, indent=2))

    print(f"EXPLORATORY diagnostic -> {path}")
    print()
    print("tau_sys at init (quantiles 5/25/50/75/95), and state surviving one env step:")
    for name in ("tau_sys_default", "tau_sys_wide"):
        d = result[name]
        tq = " ".join(f"{v:7.3f}" for v in d["tau_sys_quantiles"])
        rq = " ".join(f"{v:7.4f}" for v in d["per_step_retention_quantiles"])
        print(f"  {name:18s} tau_sys {tq}")
        print(f"  {'':18s} retain  {rq}")
    print()
    print(f"{'cell':28s} {'per-step':>9s} {'survives':>10s} {'grad t0/T':>11s} {'eff.horizon':>12s}")
    for r, g in zip(result["retention"], result["gradient"]):
        print(
            f"  {r['label']:26s} {r['median_per_step_retention']:9.4f} "
            f"{r['perturbation_surviving_full_episode']:10.2e} "
            f"{g['grad_at_t0_over_grad_at_T']:11.2e} {g['effective_horizon_steps']:9d}/{HORIZON}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
