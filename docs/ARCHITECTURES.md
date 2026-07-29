# Architectures

The equations this repository actually implements, with citations, and — equally important —
what each block is **not**.

**Scope.** This is the research-facing description of the model family: the mathematics, the
literature it comes from, the honest naming, and the design of the comparison.
[`MODEL_CARDS.md`](MODEL_CARDS.md) is the implementation-facing companion: per-model cards
tied to the unit tests that pin each equation. [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md)
covers the experiment design. Source of truth for behaviour is `dsa/models/cells.py`,
`dsa/models/policy.py` and `dsa/models/registry.py`.

**This document states no measurements.** Results live in [`../RESULTS.md`](../RESULTS.md).

---

## 1. Naming, and why it needed fixing

A previous version of this repository shipped three names that were false:

| old name | what the code did | now called |
|---|---|---|
| `ppo_ltc` | a learned time constant that was a **constant**, with no dependence on input or state; algebraically a sigmoid forget gate | `ppo_ltc`, reimplemented as a real LTC |
| `ppo_lfm` | `nn.MultiheadAttention` + GELU feed-forward + two layer norms + mean pooling | `ppo_transformer` |
| `ppo_ltc_lfm` | the above two, at unequal depth and unequal capacity | `ppo_ltc_cfc` |

The rule now is simply that names describe implementations. A block that is attention is
called a transformer. The liquid-foundation primitive is the closed-form continuous-time
cell, which is what `ppo_cfc` is.

> Liquid Foundation Models are built on the **closed-form continuous-time (CfC) cell** of
> Hasani et al. (2022). In this repository the "LFM" component of the title is **`ppo_cfc`**.
> The attention block that a previous version called "LFM" is now **`ppo_transformer`** and
> is a baseline, not a liquid model.

> **`ppo_ltc_cfc` is the model the repository title abbreviates as "PPO-LTC-LFM":** an LTC
> block followed by a CfC (liquid-foundation) block.

Being a transformer is not a defect. Calling one a liquid model was.

---

## 2. Shared architecture — identical for every model

Every registry entry is the same network with two blocks swapped out. Nothing else varies.

```
obs (B, obs_dim), dt (B,)

u0      = tanh( Linear(obs_dim + 1, H) @ [obs ; dt] )     # encoder, identical for all models
y1, s1' = block1(u0, s1, dt)
u1      = LayerNorm_1(y1)
y2, s2' = block2(u1, s2, dt)
z       = LayerNorm_2(y2)
logits  = Linear(H, action_dim)( tanh( Linear(H, H)(z) ) )
value   = Linear(H, 1)( tanh( Linear(H, H)(z) ) )
```

Two consequences worth being explicit about:

- **Depth is exactly two blocks for every model**, with no exceptions. In the previous
  version the "LFM" model had two layers while the "LTC+LFM" model had one, and in the
  hybrid branch the layer count sized only the recurrent stack while attention and the
  feed-forward were single hardcoded modules. The ablation that claimed to remove the LTC
  also removed a transformer layer.
- **`dt` reaches every model twice over, and the difference matters.** Path (a) is the
  encoder input feature — available to *all* models, including the ones called
  `dt`-blind. Path (b) is the block's internal state update — only for `dt`-aware cells. The
  `ppo_cfc` vs `ppo_cfc_dtblind` contrast isolates path (b) with path (a) held fixed. No
  model in this repository is `dt`-unaware; the ones labelled blind are blind *in the state
  update*.

Dropout is zero in every model. Hidden state is carried across environment steps — this is
genuinely recurrent PPO, not a stateless encoder over a frame stack.

---

## 3. `ppo_ltc` — Liquid Time-constant network

Hasani, Lechner, Amini, Rus, Grosu. *Liquid Time-constant Networks.* AAAI 2021.
arXiv:2006.04439.

### 3.1 The ODE

Elementwise over the `H` hidden units (paper eq. 1):

```
dx/dt = -[ 1/tau + f(x, I, t, theta) ] * x(t)  +  f(x, I, t, theta) * A

f(x, I, t, theta) = sigmoid( W_x @ I(t) + W_h @ x(t) + b )
```

The **defining property** is the input- and state-dependent system time constant
(paper eq. 5):

```
tau_sys = tau / ( 1 + tau * f(x, I, t, theta) )
```

`tau_sys` varies with the input `I` and with the state `x`. That is the entire content of
the word "liquid": the network's characteristic timescale is itself a function of what it is
processing. An implementation whose effective decay is a learned constant independent of
`(x, I)` is a sigmoid forget gate wearing the name — which is exactly what the previous
version was, verified numerically to reduce to `h_new = f_eff * h + (1 - decay) * i * g`.

`LTCCell.system_time_constant(u, x)` exposes the quantity directly, and
`tests/test_cells.py::test_ltc_time_constant_is_input_dependent` fails if it does not move
with `(u, x)`.

### 3.2 The solver

The fused / semi-implicit Euler scheme of the paper's eq. 6, with `K` sub-steps of size
`h = dt / K`:

```
for k in range(K):
    f_k = sigmoid( W_x @ u + W_h @ x + b )
    x   = ( x + h * f_k * A ) / ( 1 + h * ( 1/tau + f_k ) )
```

`dt` enters **only** through `h`, multiplying both the numerator input term and the
denominator decay term. That is the entire continuous-time content of the cell. Two
properties follow and are tested: the state change vanishes as `dt` goes to zero, and the
update is unconditionally stable for large `dt`, because the semi-implicit denominator grows
with `h`. The encoder output `u` is held constant across sub-steps; the output is `x` after
the sub-steps, with no extra activation and no output gate — normalisation lives in the
policy wrapper, not the cell.

### 3.3 Initialisation, and timescale diversity

`log_tau = linspace(log(tau_min), log(tau_max), H)`, giving log-spaced time constants across
units, and `A ~ N(0, 0.1^2)` drawn from the model's explicit generator. A test asserts the
ratio of the largest to the smallest `tau` exceeds a floor. The previous version initialised
every unit to the same time constant, so even the parameterisation it did have carried no
timescale diversity.

---

## 4. `ppo_cfc` — Closed-form Continuous-time network

Hasani, Lechner, Amini, Liebenwein, Ray, Tschaikowski, Teschl, Rus. *Closed-form
Continuous-time Neural Networks.* Nature Machine Intelligence 4, 2022. arXiv:2106.13898.

**This is the liquid-foundation primitive** — the cell that "LFM" refers to.

### 4.1 The closed form

Paper eq. 10:

```
x(t) = sigmoid( -f(x, I) * t ) * g(x, I)  +  [ 1 - sigmoid( -f(x, I) * t ) ] * h(x, I)
```

Implemented per decision step with `t := dt`, using the shared backbone of the official
cell:

```
z     = tanh( W_bb @ [x_prev ; u] + b_bb )     # shared backbone, one layer
g     = tanh( W_g @ z + b_g )                  # initial-condition branch
h     = tanh( W_h @ z + b_h )                  # steady-state branch
f     =       W_f @ z + b_f                    # unconstrained time-constant logit
gate  = sigmoid( -( f * dt + b_tau ) )         # closed-form time gate
x_new = gate * g + (1 - gate) * h
```

`dt` enters **only** through `f * dt` in the gate exponent. `b_tau` is a learned per-unit
time offset initialised to zero; with `b_tau = 0` this is literally the paper's eq. 10, and
the offset is the official `ncps` "explicit time-gating bias" form. There is no state-space
discretisation, no attention and no pooling anywhere in this cell — the previous version's
"LFM" had all three of those and none of this.

### 4.2 A limit that is honestly not satisfied

Eq. 10 is a closed-form *solution*, not an incremental integrator, so `x_new` does **not**
tend to `x_prev` as `dt` goes to zero; it tends to the `b_tau`-gated mixture of `g` and `h`.
That is a property of the published cell rather than a defect of this implementation, and it
is documented rather than glossed. What is verified instead:

- differentiating the closed form yields the logistic relaxation it is the solution of, and
  RK4 integration of that ODE in float64 agrees with the closed form to far below a
  nanounit;
- as `dt` grows the gate closes and the state relaxes monotonically to the steady-state
  branch;
- the output is bounded for arbitrarily large `dt`, since both branches are `tanh` and the
  gate forms a convex combination.

---

## 5. `ppo_cfc_dtblind` — the one-factor control

`CfCDtBlindCell` subclasses `CfCCell` and overrides exactly one method: the interval the
time gate sees is pinned to a constant. Identical architecture, identical parameter count,
identical initialisation, identical inputs — it still receives `dt` as an encoder feature.
What is removed is path (b) of §2 and nothing else.

Tests pin both sides of the contrast: the `ppo_cfc` cell's output must change when only
`dt` changes, and the blind cell's output must be **bitwise identical** when only `dt`
changes.

This is the control that carries the pre-registered primary comparison, and it is the
reason that comparison is capable of returning a clean null: there is no confound left to
attribute a null to except the absence of an effect or the size of the budget.

---

## 6. `ppo_ltc_cfc` — the hybrid the title names

An LTC block followed by a CfC block. Depth two, like everything else. It differs from
`ppo_ltc` in exactly one position and from `ppo_cfc` in exactly one position, which is what
makes the F3 decomposition family a clean test rather than a comparison of two arbitrary
networks. A registry test asserts that single-position property directly.

---

## 7. Baselines, honestly described

| key | block | state carried | `dt` in the state update? |
|---|---|---|---|
| `ppo_mlp` | `Linear -> Tanh -> Linear -> Tanh` | none | no |
| `ppo_gru` | `nn.GRUCell` | `h` | no |
| `ppo_lstm` | `nn.LSTMCell` | `[h ; c]` | no |
| `ppo_transformer` | banded-causal windowed self-attention + GELU feed-forward, pre-norm, residual | ring buffer of the last `K` block inputs plus a fill counter | no |

`ppo_transformer` attends over a fixed causal window with a learned additive attention bias
indexed by distance-to-query, and is banded-causal in **both** stepping and unrolling, so the
batched update is exact rather than approximate. Tests assert that perturbing a future input
changes no earlier output, and that an input older than the window has no influence at all —
so the declared state size is the true memory.

`ppo_lstm` is registered, capacity-matched and unit-tested but excluded from the confirmatory
matrix as a near-duplicate of `ppo_gru`; see `BENCHMARK_PROTOCOL.md` §2.3.

All four still receive `dt` as an encoder feature. None of them is `dt`-unaware; they are
`dt`-blind in the state update.

### 7.1 Zero-parameter policies

Not neural, and first-class rows in every table.

- `random_policy` — uniform over channels, from an explicit seeded generator.
- `constant_channel` — always channel zero. A degenerate-behaviour detector: any learned
  policy that collapses to one channel should land on top of it.
- `greedy_heuristic` — a genuine competitor, not a strawman. It maintains a `dt`-decayed
  exponential memory of sensed primary occupancy and a contention estimate over hidden
  background terminals inferred from its own collision history, and scores channels by a
  weighted combination of current sensing, that memory, inferred contention and channel
  quality. It indexes the exported `OBS_LAYOUT` and reads nothing a neural policy cannot
  read.

That last point is worth dwelling on. The strongest policy in this benchmark is itself
stateful and `dt`-aware; those inductive biases are hand-coded into it. The learned models
were asked to *acquire* them from reward.

---

## 8. Capacity matching

Capacity is a confound unless it is controlled, and in the previous version it was not: the
largest model had more than twice the parameters of the smallest, and parameter count
correlated with reward at Spearman <!--v:hist.spearman_param_reward-->0.829<!--/v--> across
the model set (an audit figure for a deleted revision, registered in
`docs/historical_figures.yaml`; it is not derivable from `results/`). The two "liquid
winners" were simply the two biggest networks.

The rule now: a common parameter target with a tight tolerance, and the hidden width `H` is
**solved**, never hardcoded. `solve_hidden_dim` bisects for the crossing and checks the
neighbouring integers, returning the argmin of the absolute parameter error; for attention
it searches jointly over `(H, num_heads)` subject to divisibility, since
`nn.MultiheadAttention` constrains the embedding width. Solved widths and head counts are
written to `results/manifest.json` at run time — never into source.

Invariants, each a test in `tests/test_registry.py`:

- every model's parameter count is within tolerance of the target;
- the ratio of the largest to the smallest count across the registry is bounded (the
  measured value is published as `CLAIM.BUDGET.spread`);
- every model has exactly two blocks;
- `ppo_ltc_cfc` differs from `ppo_ltc` and from `ppo_cfc` in exactly one block position, and
  `ppo_cfc_dtblind` differs from `ppo_cfc` in no architectural position at all, with equal
  parameter counts;
- encoder, actor and critic parameter shapes are identical across all keys at a fixed `H`,
  so only the two blocks ever differ.

Head count for the attention baseline is a solved consequence of the budget, not a factor
under study, and no claim in this repository depends on it.

---

## 9. Deterministic construction

`build_model(key, obs_dim, action_dim, seed)` produces an identical `state_dict` for
identical arguments. After the module tree is built, every parameter is re-initialised from
an explicit `torch.Generator`: linear weights and biases uniform on `+/- 1/sqrt(fan_in)`,
layer norms to unit weight and zero bias, recurrent cells uniform on `+/- 1/sqrt(H)`,
attention projections uniform on `+/- 1/sqrt(embed_dim)` with zero biases, `log_tau` to the
log-spaced schedule, `A` normal, `b_tau` zero. `torch.manual_seed` is never used and the
global RNG is never load-bearing.

The initialisation seed includes the model key, because initialisation is a property of the
arm. Environment seeds do not — see `BENCHMARK_PROTOCOL.md` §4.2.

---

## 10. What the architecture comparison can and cannot show

The family is constructed so that a difference between two arms has few places to hide:
equal depth, matched capacity, shared encoder and heads, shared environment streams, and a
control that differs in exactly one factor. If the comparison returns a null, the honest
readings are (a) the effect is not large at this task and budget, or (b) the learner never
reached a regime where an architectural difference could express itself.

Distinguishing (a) from (b) is not an architectural question, and the architecture layer
does not answer it. The learner diagnostics published per run in `results/all_runs.csv` —
critic explained variance, policy entropy in nats, gradient-step count, and the first-epoch
importance-ratio deviation — are what a reader should use to decide which reading applies.
The README's limitations section states which one this repository's evidence supports.

---

## References

- Hasani, Lechner, Amini, Rus, Grosu. *Liquid Time-constant Networks.* AAAI 2021.
  arXiv:2006.04439.
- Hasani, Lechner, Amini, Liebenwein, Ray, Tschaikowski, Teschl, Rus. *Closed-form
  Continuous-time Neural Networks.* Nature Machine Intelligence 4, 2022. arXiv:2106.13898.
- Lechner, Hasani, Amini, Henzinger, Rus, Grosu. *Neural Circuit Policies Enabling Auditable
  Autonomy.* Nature Machine Intelligence 2, 2020. (The `ncps` reference implementation whose
  explicit time-gating bias form the CfC cell here follows.)
- Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. *Attention Is All
  You Need.* NeurIPS 2017. arXiv:1706.03762.
- Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk, Bengio. *Learning Phrase
  Representations using RNN Encoder–Decoder for Statistical Machine Translation.* EMNLP 2014.
  arXiv:1406.1078.
- Hochreiter, Schmidhuber. *Long Short-Term Memory.* Neural Computation 9, 1997.
- Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization Algorithms.*
  2017. arXiv:1707.06347.
