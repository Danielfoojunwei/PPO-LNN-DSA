# Why `ppo_ltc` does not train

> **This document is exploratory.** Nothing in it is pre-registered, nothing in it
> carries a permutation test or a multiplicity correction, and the intervention
> experiment uses <!--v:diag.int.seeds-->4<!--/v--> seeds on one scenario. It may
> not be cited as a confirmatory result. Its numbers are generated from
> `results/diagnostics/*.json` by `scripts/render_docs.py`, like every other
> number in this repository.

## The question Study B could not answer

[Study B](STUDY_B.md) found that five of seven capacity-matched architectures beat
their own random initialisation at <!--v:studyb.runs.train_steps-->163840<!--/v-->
environment steps, and that `ppo_ltc` and `ppo_ltc_cfc` did not — on either
scenario, with negative point estimates on `stationary`.

That is a defect report, not a finding. "The LTC did not learn" is a statement
about *this implementation* until a mechanism is shown, and the repository is
named after the architecture in question, so the temptation to leave it as an
architectural result is exactly the temptation this project exists to resist.

Three diagnostics were run. **Two hypotheses were refuted and the failure remains
unexplained.** What follows is what was ruled out, what was measured, and what a
reader should therefore conclude.

## Diagnostic 1 — recurrent gradient decay. Refuted.

`scripts/diagnose_ltc.py`

The obvious explanation: the LTC's `tau_sys = tau / (1 + tau * f)` with `f ≈ 0.5`
at initialisation and `tau ∈ [0.5, 8]` puts its memory half-life near a single
timestep against `dt = 1`, so neither state nor gradient survives the
<!--v:diag.horizon-->64<!--/v-->-step episode.

The state and gradient decay is real. It is also **not specific to the LTC**:

| cell | median per-step retention | ‖∂x_T/∂x_0‖ ÷ ‖∂x_T/∂x_T‖ | effective horizon |
|---|---|---|---|
| `ltc` | <!--v:diag.retain.ltc-->0.550<!--/v--> | <!--v:diag.gradratio.ltc-->2.2e-17<!--/v--> | <!--v:diag.effhorizon.ltc-->23<!--/v--> / <!--v:diag.horizon-->64<!--/v--> |
| `cfc` | <!--v:diag.retain.cfc-->0.457<!--/v--> | <!--v:diag.gradratio.cfc-->1.3e-25<!--/v--> | <!--v:diag.effhorizon.cfc-->16<!--/v--> / <!--v:diag.horizon-->64<!--/v--> |
| `gru` | <!--v:diag.retain.gru-->0.623<!--/v--> | <!--v:diag.gradratio.gru-->1.3e-13<!--/v--> | <!--v:diag.effhorizon.gru-->29<!--/v--> / <!--v:diag.horizon-->64<!--/v--> |

The CfC retains **less** state than the LTC and its gradient ratio is eight orders
of magnitude smaller — and the CfC is among the models that learn. The GRU's
gradient vanishes too. Varying `unfolds` from 1 to 24 barely moves the figures,
and widening `tau_max` to 200 does not rescue them.

Vanishing recurrent gradient is a property of every cell in this repository at
this horizon, so it cannot be what separates the trainable from the untrainable.
**Hypothesis rejected.**

*Caveat:* this diagnostic constructs cells directly with its own orthogonal
initialisation rather than through the registry, so its absolute numbers are not
the shipped models'. It is used only for the between-cell comparison, which is
the part that does the rejecting.

## Diagnostic 2 — parameter-gradient starvation. Measured, and real.

`scripts/diagnose_ltc_params.py`

This one uses the **real registry models** via `build_model`, on real environment
trajectories, with a real first-update PPO surrogate. Median recurrent-block
update-to-weight ratio `lr·‖g‖/‖w‖` at the shared `lr = 3e-4`, across all
<!--v:diag.uw.n_compared-->8<!--/v--> registered models:

| model | `lr·‖g‖/‖w‖` |
|---|---|
| `ppo_mlp` | <!--v:diag.uw.ppo_mlp-->3.9e-03<!--/v--> |
| `ppo_ltc_cfc` | <!--v:diag.uw.ppo_ltc_cfc-->3.5e-03<!--/v--> |
| `ppo_cfc` | <!--v:diag.uw.ppo_cfc-->2.6e-03<!--/v--> |
| `ppo_cfc_dtblind` | <!--v:diag.uw.ppo_cfc_dtblind-->2.4e-03<!--/v--> |
| `ppo_gru` | <!--v:diag.uw.ppo_gru-->1.9e-03<!--/v--> |
| `ppo_lstm` | <!--v:diag.uw.ppo_lstm-->1.8e-03<!--/v--> |
| `ppo_transformer` | <!--v:diag.uw.ppo_transformer-->9.8e-04<!--/v--> |
| **`ppo_ltc`** | **<!--v:diag.uw.ppo_ltc-->5.9e-05<!--/v-->** |

`ppo_ltc` is the sole outlier, between <!--v:diag.uw.ratio_min-->17<!--/v-->× and
<!--v:diag.uw.ratio_max-->66<!--/v-->× smaller than every other model, while its
*total* gradient norm is unremarkable. The per-parameter split localises it
precisely:

| `ppo_ltc` parameter | `lr·‖g‖/‖w‖` |
|---|---|
| `blocks.0.A` | <!--v:diag.ltc_param.A-->2.65e-03<!--/v--> |
| `blocks.0.input_map.weight` | <!--v:diag.ltc_param.input_map-->1.53e-05<!--/v--> |
| `blocks.0.log_tau` | <!--v:diag.ltc_param.log_tau-->7.58e-06<!--/v--> |
| `blocks.0.recurrent_map.weight` | <!--v:diag.ltc_param.recurrent_map-->3.20e-06<!--/v--> |

`A` receives healthy gradient; the synaptic maps and the time constants receive
two to three orders of magnitude less. The structural reason is visible in the
update. In

    x  <-  (x + h·A·f) / (1 + h·(1/tau + f)),    f = sigmoid(W_x u + W_h x + b),
    h = dt / unfolds

the maps reach the state **only** through `f`, and `f`'s pathway is scaled by `A`
in the numerator, then by `sigmoid' ≤ 1/4`, then by `h = 1/6`. `custom_init_` sets
`A ~ N(0, 0.1²)`. So the maps are starved by roughly `|A|` relative to `A` itself.

This is a real, measured, LTC-specific pathology. The obvious inference — that a
learning rate shared across all eight models is therefore ~50× too small for this
one, making the Study B result a tuning artifact — is **wrong**, as diagnostic 3
shows.

## Diagnostic 3 — do the candidate fixes rescue it? No.

`scripts/diagnose_ltc_intervention.py`

Two interventions, each targeting the starvation directly, at Study B's full
budget of <!--v:diag.int.steps-->163840<!--/v--> steps on
`<!--v:diag.int.scenario-->stationary<!--/v-->`, over
<!--v:diag.int.seeds-->4<!--/v--> seeds. `a_scale10` multiplies `A` by 10 at
initialisation, feeding the maps proportionally more gradient. `lr10` raises only
the learning rate by 10×. `ppo_gru` is the control that must learn.

| arm | mean Δ vs own init | n > 0 | max explained variance | final entropy range |
|---|---|---|---|---|
| `ppo_ltc` default | <!--v:diag.int.ppo_ltc.default.mean_delta-->-2.21<!--/v--> | <!--v:diag.int.ppo_ltc.default.n_positive-->1<!--/v-->/<!--v:diag.int.ppo_ltc.default.n-->4<!--/v--> | <!--v:diag.int.ppo_ltc.default.max_ev-->0.000<!--/v--> | <!--v:diag.int.ppo_ltc.default.min_entropy-->0.044<!--/v-->–<!--v:diag.int.ppo_ltc.default.max_entropy-->0.573<!--/v--> |
| `ppo_ltc` `a_scale10` | <!--v:diag.int.ppo_ltc.a_scale10.mean_delta-->+0.63<!--/v--> | <!--v:diag.int.ppo_ltc.a_scale10.n_positive-->2<!--/v-->/<!--v:diag.int.ppo_ltc.a_scale10.n-->4<!--/v--> | <!--v:diag.int.ppo_ltc.a_scale10.max_ev-->0.000<!--/v--> | <!--v:diag.int.ppo_ltc.a_scale10.min_entropy-->0.039<!--/v-->–<!--v:diag.int.ppo_ltc.a_scale10.max_entropy-->0.602<!--/v--> |
| `ppo_ltc` `lr10` | <!--v:diag.int.ppo_ltc.lr10.mean_delta-->-1.14<!--/v--> | <!--v:diag.int.ppo_ltc.lr10.n_positive-->1<!--/v-->/<!--v:diag.int.ppo_ltc.lr10.n-->4<!--/v--> | <!--v:diag.int.ppo_ltc.lr10.max_ev-->0.426<!--/v--> | <!--v:diag.int.ppo_ltc.lr10.min_entropy-->0.345<!--/v-->–<!--v:diag.int.ppo_ltc.lr10.max_entropy-->1.535<!--/v--> |
| `ppo_gru` control | <!--v:diag.int.ppo_gru.default.mean_delta-->+16.47<!--/v--> | <!--v:diag.int.ppo_gru.default.n_positive-->3<!--/v-->/<!--v:diag.int.ppo_gru.default.n-->4<!--/v--> | <!--v:diag.int.ppo_gru.default.max_ev-->0.377<!--/v--> | <!--v:diag.int.ppo_gru.default.min_entropy-->0.545<!--/v-->–<!--v:diag.int.ppo_gru.default.max_entropy-->1.256<!--/v--> |

Neither intervention brings `ppo_ltc` anywhere near the control. The GRU gains
about sixteen reward points on the same harness, the same seeds and the same
environment streams; the LTC's three arms are all within noise of zero and
disagree in sign across seeds.

So the starvation is real but **not sufficient** to explain the failure.
Correcting it does not restore learning. **Hypothesis rejected as the cause.**

One partial effect is worth recording because it sharpens the picture rather than
resolving it. `lr10` is the only LTC arm whose critic fits anything at all —
explained variance reaches <!--v:diag.int.ppo_ltc.lr10.max_ev-->0.426<!--/v-->
against <!--v:diag.int.ppo_ltc.default.max_ev-->0.000<!--/v--> for the default —
and it is the only arm that keeps policy entropy off the floor. The optimisation
became measurably healthier without the reward moving. A faster optimiser is not
the missing piece.

## What the failure actually looks like

Across every LTC arm, including both interventions, the same signature appears:
the critic explains approximately zero variance while policy entropy collapses
from uniform over the eight channels (`ln 8` nats) to as little as
<!--v:diag.int.ppo_ltc.a_scale10.min_entropy-->0.039<!--/v-->. The actor is moving
hard, and moving onto nothing. Several LTC cells return *byte-identical* deltas
across different arms, which is what a policy collapsed onto a single constant
action looks like under deterministic evaluation — the trained return stops
depending on the configuration because the policy has stopped depending on the
observation.

A coherent story is that the recurrent block never produces informative features,
so the critic cannot fit, so the advantages are noise, so the actor collapses on
noise. The gradient starvation of diagnostic 2 is consistent with the first link.
But diagnostic 3 shows that relieving the starvation does not break the chain, so
either the starvation is not the first link or there is something further upstream
that all three diagnostics have missed.

## What a reader should conclude

- **Do not read Study B's LTC result as a finding about liquid time-constant
  networks.** It is a robust, reproducible failure *of this implementation*, now
  characterised in more detail and still not explained.
- The two most plausible mechanisms have been tested and rejected. That raises,
  rather than lowers, the prior that there is a remaining defect in
  `dsa/models/cells.py::LTCCell` or in how it is initialised — a correct
  implementation of Hasani et al. (2021) that cannot be trained by a PPO setup
  which trains seven other architectures is more likely to be subtly wrong than
  to be a discovery.
- **Every architectural comparison in either study that involves `ppo_ltc` or
  `ppo_ltc_cfc` is a comparison against a non-learner** and must be read that
  way. That includes the hybrid decomposition, since `ppo_ltc_cfc` contains an
  LTC block.
- The primary pre-registered comparison, `ppo_cfc` vs `ppo_cfc_dtblind`, does
  **not** involve an LTC and is unaffected by any of this.

## What would settle it

In rough order of expected information per unit compute:

1. **A minimal supervised task with a known answer.** Train the LTC block alone
   to copy an input delayed by *k* steps. If it fails there, the defect is in the
   cell and the whole RL layer is a distraction. This is minutes of compute and it
   should have been step one.
2. **Compare against a reference implementation.** Hasani et al.'s `ncps` package
   provides an LTC cell. Feed both identical weights and inputs and diff the
   trajectories. The existing unit tests check the cell against the paper's
   equation as *this repository transcribed it*, which cannot catch a
   misreading of the paper.
3. **A wider hyperparameter sweep than two points.** `a_scale` and `lr` were
   tested at one alternative value each. `unfolds`, `tau_max`, gradient clipping
   and the LayerNorm placement after the block are all untested.
4. **Only then** treat any surviving failure as a claim about the architecture,
   and pre-register it before running it.
