# Model cards

One card per entry in `MODEL_REGISTRY` (`dsa/models/registry.py`).

**No measured quantity appears in this document.** Solved hidden widths, head counts,
parameter counts and state sizes are computed by `dsa.models.model_summary` and written
to `results/manifest.json` by the runner. To see them for your build:

```bash
python -c "from dsa.models import MODEL_REGISTRY, model_summary; \
           [print(model_summary(k, 40, 8)) for k in MODEL_REGISTRY]"
```

---

## Vocabulary

Liquid Foundation Models are built on the **closed-form continuous-time (CfC) cell** of
Hasani et al. (2022). In this repository the "LFM" component of the title is `ppo_cfc`.

A previous version of this repository applied the name "LFM" to a block that was
`nn.MultiheadAttention` + a GELU feed-forward + two LayerNorms + mean-pooling. That is a
transformer. It is now called `ppo_transformer`, it is a **baseline, not a liquid model**,
and `tests/test_registry.py::test_no_model_is_named_lfm` keeps the name from coming back.

`ppo_ltc_cfc` is the model the repository title abbreviates as "PPO-LTC-LFM": an LTC block
followed by a CfC (liquid-foundation) block.

---

## Shared architecture — identical for every model

Every registry entry is the same wrapper around two interchangeable blocks
(`dsa/models/policy.py`):

```
obs (B, obs_dim), dt (B,)
  u0      = tanh( Linear(obs_dim + 1, H) @ [obs ; dt] )    # encoder, identical for all models
  y1, s1' = block1(u0, s1, dt)
  u1      = LayerNorm_1(y1)
  y2, s2' = block2(u1, s2, dt)
  z       = LayerNorm_2(y2)
  logits  = Linear(H, action_dim)( tanh( Linear(H, H)(z) ) )
  value   = Linear(H, 1)( tanh( Linear(H, H)(z) ) ).squeeze(-1)
```

Three properties are load-bearing and are each pinned by a test:

* **Equal depth.** Exactly two blocks for every model, no exceptions
  (`test_equal_depth`). This is what makes `ppo_ltc_cfc` differ from `ppo_ltc` in one
  block and from `ppo_cfc` in one block (`test_ablation_single_factor`).
* **Equal information.** Every model consumes exactly `(obs, dt, state)` through an
  encoder of identical shape. There is no frame stack, and no model — the transformer
  included — receives a privileged input channel
  (`test_every_model_has_the_same_input_contract`, `test_no_model_sees_the_future`).
* **Two, and only two, routes for `dt`.**
  (a) the encoder feature, available to **all** models
  (`test_dt_reaches_every_model_through_the_encoder`);
  (b) the block's internal state update, only for `dt_aware` blocks
  (`test_dt_enters_the_state_update_only_for_dt_aware_cells`).
  The `ppo_cfc` vs `ppo_cfc_dtblind` contrast isolates (b) with (a) held fixed.

Hidden state is **carried across environment steps**. `initial_state(batch)` is both the
constructor of a fresh state and the episode-boundary reset. `unroll` recomputes the whole
hidden trajectory from the supplied initial state and is required to equal that many
successive `step` calls (`test_step_unroll_equivalence`).

`dropout` is zero for every registry entry, and `.train()` / `.eval()` are wired up
regardless. `test_eval_determinism_test_has_teeth` builds a dropout-carrying model to
prove the mode switch is real rather than vacuous.

---

## Capacity matching

The hidden width `H` is **solved**, never hardcoded:

```python
PARAM_TARGET    = 40_000
PARAM_TOLERANCE = 0.05      # +/- 5 %
solve_hidden_dim(spec, obs_dim, action_dim) -> H
```

`P(H)` is monotonically increasing for every registered cell family, so the solver
brackets the crossing by doubling, bisects, then checks the three integers around it and
returns the argmin of `|P(H) - target|`; ties resolve to the smaller `H`. It is a pure
function — no globals, no order-dependent caching (`test_solve_hidden_dim_is_pure_and_order_independent`),
and it really does return the argmin rather than merely a feasible point
(`test_solve_hidden_dim_finds_the_argmin`).

For attention specs the search is **joint** over `(H, num_heads)` with
`num_heads` in `(4, 2, 1)` and `H % num_heads == 0`, minimising `|P - target|` and
breaking ties toward the larger head count. **Head count is not a factor under study**;
it is a consequence of the divisibility constraint that `nn.MultiheadAttention` imposes,
and the solved value is recorded in `results/manifest.json`.

Why this matters: in the previous version of this repository the parameter counts spanned
a factor greater than two, parameter count correlated with reward across the PPO models at
Spearman rho above four fifths, and the two "liquid" winners were simply the two largest
networks. `test_param_budget` and `test_param_spread` make that confound impossible.

---

## `ppo_ltc` — Liquid Time-constant network

Hasani, Lechner, Amini, Rus, Grosu. *Liquid Time-constant Networks*. AAAI 2021.

`cell_kinds = ("ltc", "ltc")`, `dt_aware = True`, family `liquid`.

The ODE (paper eq. 1), elementwise over the `H` hidden units:

```
dx/dt = -[ 1/tau + f(x, I, t, theta) ] * x(t)  +  f(x, I, t, theta) * A
f(x, I, t, theta) = sigmoid( W_x @ I(t) + W_h @ x(t) + b )
```

The **defining property** is the input-dependent system time constant (paper eq. 5):

```
tau_sys = tau / ( 1 + tau * f(x, I, t, theta) )
```

`tau_sys` is a function of the input `I` **and** the state `x`. It is exposed as
`LTCCell.system_time_constant(u, x)` so that it can be measured directly.

The integrator is the fused / semi-implicit Euler solver of paper eq. 6. With
`K = unfolds` sub-steps of size `h = dt / K`:

```
for k in range(K):
    f_k = sigmoid( W_x @ u + W_h @ x + b )
    x   = ( x + h * f_k * A ) / ( 1 + h * ( 1/tau + f_k ) )
```

`u` is held constant across the sub-steps. `dt` enters **only** through `h`, multiplying
both the numerator input term and the denominator decay term; that is the entire
continuous-time content of the cell. The implementation hoists the loop-invariant `h * A`
and `1 + h/tau` out of the sub-step loop, which is an algebraic identity, not a change of
dynamics.

Parameters: `W_x` and `b` (`input_map`), `W_h` (`recurrent_map`), `log_tau`, `A`.

Initialisation: `log_tau = linspace(log(tau_min), log(tau_max), H)` with
`tau_min` a half and `tau_max` eight, giving log-spaced timescale diversity across units;
`A ~ N(0, 0.1^2)` from the model's explicit generator. Default `unfolds` is six,
configurable through `ModelSpec.cell_kwargs`; lowering it must be recorded in
`results/manifest.json`.

### What the previous version had, and how the tests catch it

The old `empirical/models.py` declared `self.log_tau = nn.Parameter(torch.zeros(hidden_dim))`
and used it as a constant. With no dependence on `x` or `I` the update reduces
algebraically to a plain sigmoid forget gate, and every unit started at the same time
constant, so there was no timescale diversity either.

* `test_ltc_time_constant_is_input_dependent` measures the relative spread of `tau_sys`
  across inputs and requires it to be non-trivial.
* `test_ltc_time_constant_check_rejects_a_constant_tau_cell` runs **the same check**
  against a deliberately reconstructed constant-`log_tau` cell and asserts it fails. A
  test that cannot fail is not evidence.
* `test_ltc_time_constant_depends_on_state` does the same across states.
* `test_ltc_timescale_diversity` requires `tau.max() / tau.min()` to be at least five.

### Continuous-time properties that are actually verified

* `test_ltc_zero_dt_is_exactly_a_no_op` — `dt = 0` leaves the state bitwise unchanged.
* `test_ltc_state_change_vanishes_linearly_as_dt_goes_to_zero` — halving `dt` halves the
  state change, i.e. the update is genuinely `O(dt)`.
* `test_ltc_is_unconditionally_stable_for_large_dt` — the semi-implicit denominator grows
  with `h`, so `|x|` stays bounded by `max|A|` even at enormous `dt`. An explicit Euler
  discretisation would blow up here.
* `test_ltc_fused_solver_converges_to_the_ltc_ode` — the strongest of the four. It
  integrates the LTC ODE with RK4 in float64 and shows the fused solver converges to that
  reference at first order as `unfolds` grows. A cell that merely resembled an ODE solver
  would not converge to the ODE's solution.

---

## `ppo_cfc` — Closed-form Continuous-time network

Hasani, Lechner, Amini, Liebenwein, Ray, Tschaikowski, Teschl, Rus. *Closed-form
Continuous-time Neural Networks*. Nature Machine Intelligence 2022.

`cell_kinds = ("cfc", "cfc")`, `dt_aware = True`, family `liquid`. **This is the
liquid-foundation primitive.**

The closed-form solution (paper eq. 10):

```
x(t) = sigmoid(-f(x, I) * t) * g(x, I)  +  [1 - sigmoid(-f(x, I) * t)] * h(x, I)
```

implemented per decision step with `t := dt` and the shared backbone of the official cell:

```
z     = tanh( W_bb @ [x_prev ; u] + b_bb )    # shared backbone, one layer
g     = tanh( W_g @ z + b_g )                 # initial-condition branch
h     = tanh( W_h @ z + b_h )                 # steady-state branch
f     =       W_f @ z + b_f                   # unconstrained time-constant logit
gate  = sigmoid( -( f * dt + b_tau ) )        # closed-form time gate
x_new = gate * g + (1 - gate) * h
```

`dt` enters **only** through `f * dt` in the gate exponent. `b_tau` is a learned `(H,)`
time offset initialised to zeros; with `b_tau` zero this is literally paper eq. 10. The
offset is the official `ncps` "explicit time-gating bias" form.

There is no state-space discretisation, no attention and no mean-pooling anywhere in this
cell.

### A limit that is honestly not satisfied

Eq. 10 is a closed-form **solution**, not an incremental integrator, so `x_new` does *not*
tend to `x_prev` as `dt` goes to zero — it tends to the `b_tau`-gated mixture of `g` and
`h`. That is a property of the published cell, not a bug in this implementation, and it is
documented rather than papered over. What is verified instead:

* `test_cfc_matches_numerical_ode_integration` — differentiating the closed form gives
  the logistic relaxation `dx/dt = -f (x - h) (1 - (x - h)/(g - h))` starting from the
  gated mixture. The test integrates *that* with RK4 in float64 and requires agreement to
  well below a nanounit. The closed form solves the ODE it is the closed-form solution of.
* `test_cfc_time_gate_limits` — as `dt` grows the gate closes and the state relaxes to the
  steady-state branch `h`; as `dt` goes to zero it returns the gated mixture.
* `test_cfc_is_bounded_for_large_dt` — both branches are `tanh` and the gate is a convex
  combination, so the output is bounded for arbitrarily large `dt`.

---

## `ppo_cfc_dtblind` — the one-factor control

`cell_kinds = ("cfc_dtblind", "cfc_dtblind")`, `dt_aware = False`, family
`liquid_control`.

`CfCDtBlindCell` subclasses `CfCCell` and overrides one method, `effective_dt`, to return
ones. Identical architecture, identical parameter count, identical initialisation
distribution — and, for a given seed, **identical initial weights**
(`test_cfc_and_dtblind_have_identical_initial_weights`).

It still receives `dt` as an encoder input feature. What is removed is only the second
route: `dt` entering the state update. This is the exactly-one-factor control that carries
the pre-registered primary comparison, and it can come out negative.

* `test_cfc_dtblind_is_bitwise_identical_when_only_dt_changes` — the cell's output is
  bitwise invariant to `dt` across four orders of magnitude.
* `test_cfc_dtblind_is_architecturally_identical_to_cfc` — same state-dict keys, same
  shapes, same weights, and the same function at unit `dt`.

---

## `ppo_ltc_cfc` — the hybrid the title names

`cell_kinds = ("ltc", "cfc")`, `dt_aware = True`, family `liquid`.

An LTC block followed by a CfC block, at the same depth and the same parameter budget as
every other model. It differs from `ppo_ltc` in exactly one block and from `ppo_cfc` in
exactly one block, which is what makes the hybrid decomposition a clean single-factor
ablation rather than a confound with depth and capacity.

---

## `ppo_transformer` — attention baseline, honestly named

`cell_kinds = ("attn", "attn")`, `dt_aware = False`, family `attention`.

Pre-norm self-attention over a **banded-causal window** of the last `K` block inputs, then
a pre-norm `Linear(H, 2H) -> GELU -> Linear(2H, H)` feed-forward, both residual. Relative
position enters as a learned additive attention bias indexed by distance-to-query, so the
block is position-aware while staying exactly reproducible in both stepping and unrolling
mode. State is the ring buffer of the last `K` inputs plus a fill counter, so
`state_size = K * H + 1`.

The window `K` is eight; it is the value the throughput budget was measured at, and
changing it must be recorded in `results/manifest.json`.

The block is banded-causal in **both** `step` and `unroll`, which is what lets the
batched unroll be exact rather than approximate. Being a transformer is not a defect —
misnaming one as a liquid model was. Tests:

* `test_attention_block_is_causal` — perturbing a future input changes no earlier output.
* `test_attention_block_horizon_is_the_declared_window` — an input older than `K` steps has
  no influence on the current output, so the declared state size is the true memory.
* `test_no_model_sees_the_future` — the same causality check applied to the full model, for
  every registry key.

---

## `ppo_mlp`, `ppo_gru`, `ppo_lstm` — memoryless and gated baselines

| key | blocks | state | `dt` in the state update? |
|---|---|---|---|
| `ppo_mlp` | `Linear -> Tanh -> Linear -> Tanh` | none, `state_size` is zero | no |
| `ppo_gru` | `nn.GRUCell` | `h` | no |
| `ppo_lstm` | `nn.LSTMCell` | `cat([h, c])`, so `state_size = 2H` | no |

All three still receive `dt` as an encoder feature — they are dt-*blind in the state
update*, not dt-unaware.

`ppo_lstm` is registered, capacity-matched and unit-tested, but it is **not** in the
primary matrix: it is a near-duplicate control of `ppo_gru` and is dropped from the
confirmatory runs to buy statistical power. It remains runnable via `--models ppo_lstm` as
an exploratory arm.

---

## Deterministic construction

`build_model(key, obs_dim, action_dim, seed)` is a pure function of its arguments:

* Initialisation draws come **only** from an explicit `torch.Generator`.
  `torch.manual_seed` is never called on the global RNG.
* Module construction is wrapped in `torch.random.fork_rng`, so building a model does not
  perturb global RNG state either — `nn.Linear.__init__` would otherwise consume global
  draws before `deterministic_init_` overwrites them.
* Traversal for re-initialisation is `nn.Module.modules()` pre-order DFS, a fixed order for
  a fixed module tree, so the sequence of draws is fully determined.

Recipe: `nn.Linear` weight and bias uniform on `+/- 1/sqrt(fan_in)`; `nn.LayerNorm` weight
one, bias zero; `nn.GRUCell` / `nn.LSTMCell` uniform on `+/- 1/sqrt(hidden_size)`;
`nn.MultiheadAttention` projection weights uniform on `+/- 1/sqrt(embed_dim)` with zero
biases; `LTCCell.log_tau` log-spaced; `LTCCell.A` normal; `CfCCell.b_tau` zero.

Pinned by `test_build_model_is_deterministic`, `test_build_model_depends_on_the_seed`,
`test_build_model_ignores_the_global_rng` and
`test_build_model_does_not_disturb_the_global_rng`.
