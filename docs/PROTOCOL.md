# Experiment protocol

This document specifies the seeding contract, the experiment matrix, the PPO learner
contract and the statistical analysis. It contains **no measurements**; every number the
project reports lives in `RESULTS.md`, which is generated from `results/`.

---

## 1. The seeding contract

This is the highest-value guarantee in the repository. Everything downstream — the paired
statistics, the reproducibility gate, the claim that any comparison is *controlled* —
rests on it.

### 1.1 Derivation

```python
def derive_seed(base_seed: int, *parts: object) -> int:
    d = hashlib.blake2b(digest_size=8)
    d.update(str(int(base_seed)).encode("utf-8"))
    for p in parts:
        d.update(b"\x1f")                  # unambiguous separator
        d.update(str(p).encode("utf-8"))
    return int.from_bytes(d.digest(), "big") & SEED_MASK
```

`blake2b`, not Python's built-in `hash()`. `hash()` is salted per process for `str` and
`bytes`, so a seed derived from it would differ between a parent process and its worker —
silently, and only under multiprocessing.

### 1.2 Role table

| consumer | derivation | contains model name? |
|---|---|---|
| training env, lane `l`, rollout `r` | `derive_seed(base, scenario, "env", "train", l, r)` | **no** |
| evaluation env, episode `i` | `derive_seed(base, scenario, "env", "eval", i)` | **no** |
| env internal substream `s` | `derive_seed(episode_seed, "sub", s)` | n/a |
| policy weight init | `derive_seed(base, "policy_init", model_key)` | yes |
| action sampling | `derive_seed(base, "action", model_key, scenario)` | yes |
| minibatch shuffling | `derive_seed(base, "shuffle", model_key, scenario)` | yes |
| heuristic policy RNG | `derive_seed(base, "heuristic", policy_key, scenario)` | yes |
| federated client `c`, round `r`, edge round `k` | `derive_seed(base, "fed", arm, r, k, c, "env")` | no |
| bootstrap / permutation | `derive_seed(ANALYSIS_SEED, "boot", comparison_id)` | n/a |

**Environment seeds deliberately exclude the model name.** All policies therefore observe
byte-identical environment streams for a given `(base_seed, scenario)`. This is what makes
the paired statistics tight, and it is what makes a comparison a comparison.

### 1.3 Prohibitions

1. **No mutable class-level counters.** `SpectrumEnv.__init__(config, episode_seed)` takes
   the seed as an explicit argument. There is no instance counter and no module-level
   mutable state.
2. **No load-bearing global RNG.** `np.random.seed`, `random.seed` and `torch.manual_seed`
   may be called for library hygiene but must not affect any recorded metric. Every
   stochastic operation uses an explicit `np.random.Generator` or `torch.Generator`.
3. **No `hash()` on strings** anywhere in seed derivation.

### 1.4 The naming rule that makes determinism checkable

**Every wall-clock column in every emitted CSV is named with the prefix `wall_`, and no
other column may be non-deterministic.** The reproducibility tests drop exactly `^wall_`
and require bitwise equality on everything else. A non-deterministic column without that
prefix is a bug either way: either the value is wrong, or the name is.

### 1.5 Why this matters

The previous implementation used a class-level `_instance_counter` and
`base_seed = config.seed + instance_id`, while the runner built a fresh environment per
episode in a single process. The seed offset was therefore a function of a model's
position in the `--models` list. Measured with a **fixed** policy at one nominal seed, six
consecutive instances spanned roughly forty reward points — against a headline effect of
about two. No result produced that way was a controlled comparison, and rerunning the
documented command moved the flagship model from first to sixth of seven.

---

## 2. Observation and environment

`obs_dim = 4 * num_channels + 8`. The layout is exported as `OBS_LAYOUT` and is what the
heuristic policies index — no magic numbers.

| slice | contents |
|---|---|
| `[0:C]` | sensed occupancy, each entry bit-flipped with probability `sensing_noise` |
| `[C:2C]` | sensed interference, plus Gaussian noise, clipped |
| `[2C:3C]` | sensed channel quality, plus Gaussian noise, clipped |
| `[3C:4C]` | one-hot of the previous action |
| `[4C+0..7]` | scaled last reward, last success, last collision, running success rate, running collision rate, step fraction, scaled `dt`, log `dt` |

`num_background_users` is **not** an observation feature. The agent must infer load from
contention.

### 2.0 What "sensed occupancy" contains, and why the task is partially observed

`obs[0:C]` reports the **primary occupant of each channel only**. Background IoT devices
are *hidden terminals*: they contend for the same channels, they cause the agent's
collisions, and they are invisible to its sensing. They hop and camp at a fixed rate
(`BG_HOP_RATE` in `dsa/envs/spectrum.py`), so the history of the agent's own collisions is
genuinely predictive of where they are now.

This is a deliberate design decision and it is the single most load-bearing one in the
environment, so it is recorded here rather than left in a docstring. An earlier
implementation reported both primary and background occupancy. Measured, that made the
task a **fully observed MDP with a myopic optimum**: a memoryless sensing heuristic came
within a few percent of the per-step oracle in all five primary scenarios. Under those
dynamics every architecture in the registry would tie at the same ceiling and the entire
recurrence-and-continuous-time thesis would be unfalsifiable — not confirmed, not refuted,
just untestable. Hidden background terminals are also the canonical partial-observability
structure of the dynamic-spectrum-access literature, and they are what gives the sentence
above ("the agent must infer load from contention") any force at all.

The consequence is that `greedy_heuristic` is a *competitive* floor, not a strawman. The
previous repository's greedy baseline lost to uniform random in six of seven scenarios;
this one wins in all seven. If the learned policies do not beat it, `RESULTS.md` says so —
that is the reporting rule, and weakening the baseline to avoid the finding is not
available.

Four environment constants live as module-level constants rather than `ScenarioConfig`
fields, because the config's field list is frozen by the specification. They are recorded
in `results/manifest.json` under `environment_constants` so a run is reproducible from the
manifest alone: `BG_HOP_RATE` (background hop/camp rate) and the three relaxation rates
for occupancy, quality and interference.

### 2.1 The `dt` single source of truth

The interval is sampled **once**, at the top of the transition, and the same scalar
advances the clock, drives the channel relaxation, drives the load process, and reaches
the recurrent cell:

```python
def step(self, action):
    reward, info = self._resolve(action)      # resolve against the CURRENT latent state
    dt = self._sample_dt()                    # sampled exactly once, here
    self.sim_time += dt
    self._advance_background_load(dt)         # one dt, three consumers
    self._relax_channels(dt)
    obs = self._build_features(dt_elapsed=dt)
    ...
```

The previous implementation advanced the clock with the *old* `dt` while relaxing channel
dynamics with the *newly sampled* one, and it was the new value that reached the cell — an
off-by-one that made the continuous-time signal incoherent with the dynamics it was
supposed to track.

All time-dependent processes are driven by `sim_time`, never by `step_count`, so they stay
consistent under irregular sampling.

### 2.2 Scenarios

Five primary dynamics classes: `stationary`, `non_stationary`, `interference_heavy`,
`irregular_dt`, `bursty_irregular`. `dt` genuinely varies in two of them.

`irregular_dt` differs from `stationary` in **exactly one process** — how the decision
interval is drawn — which makes the primary comparison an interaction test against a clean
control. Literally, the two configs differ in the fields `{name, dt_mode, dt_min, dt_max}`;
`dt_min` and `dt_max` only parameterise `dt_mode` and are inert while it is `constant`.
`tests/test_envs.py::test_irregular_dt_differs_from_stationary_only_in_the_dt_process`
pins that difference set exactly, so a fifth differing field cannot be added silently.

Two exploratory scenarios (`noisy_partial`, `bursty_load`) are registered and runnable but
excluded from confirmatory analysis.

`ScenarioConfig` is a frozen dataclass; in-place mutation of a config is forbidden and
every scenario is fully specified in the registry. The previous repository shipped two
"scenarios" that both set `scenario_name="stationary"` and differed only in a constant
background-user count that was never mutated — a static load sweep presented as population
dynamics.

---

## 3. The PPO-RNN learner contract

1. **Rollout.** `model.eval()`, `torch.no_grad()`, 16 lanes stepped synchronously for a
   64-step horizon. The buffer stores `init_state` explicitly so truncated rollouts stay
   correct if the horizon changes.
2. **Storage.** `(N, T, ...)` tensors plus `init_state` and `last_value`. **Per-step hidden
   states are never stored and never used as update inputs.**
3. **Update.** Each minibatch calls `model.unroll(obs, dt, init_state)` and **recomputes
   the entire hidden trajectory from the stored initial state.** Stored per-step hidden
   states are never fed back in; nothing is detached inside the unroll. This is the single
   most common fatal PPO-RNN bug and it is forbidden here.
4. **Minibatching is over the sequence axis** and is reachable by construction. The
   previous implementation could never reach its minibatch path at all: episodes were
   always exactly the buffer length and the buffer flushed at every episode end, so the
   guard condition could never fire.
5. **GAE** over the `(N, T)` grid, with `dones` masking the bootstrap. Advantage
   normalisation is over the whole batch, computed once before the epoch loop.
6. **The ratio-sanity invariant.** On epoch 0, minibatch 0, the recomputed
   log-probabilities must reproduce the rollout's own, so the importance ratio must be
   exactly 1. `first_epoch_max_ratio_deviation` is returned from every update and logged
   for every run; `scripts/analyze.py` **refuses to build a report** if any run exceeded
   `1e-3`. This one check catches stale hidden state, dropout mismatch and eval/train-mode
   bugs simultaneously.
7. **Dropout is zero for every model, and `.train()` / `.eval()` are used correctly.** The
   previous implementation had neither: `.eval()` and `.train()` appeared nowhere in the
   entire live path, so dropout was active during rollout, update and evaluation. Two
   successive greedy actions on identical input returned different actions, and simulating
   the real rollout-then-update path produced spurious importance ratios consuming about a
   third of the trust region.
8. **Action sampling** uses an explicit `torch.Generator`, never the global RNG.
9. **`total_env_steps`** increments by `num_envs` per vector step and is the **only** source
   for the `train_steps` column. It cannot be a config value copied into a field.
10. **`torch.set_num_threads(1)`** at every worker entry point. Parallelism lives across
    processes rather than inside the ops, because scaling across processes was measured to be
    near-linear at four workers during design. That figure comes from the design-time probe
    and is **not** recorded in `results/`: the committed run was launched with `--no-guard`,
    so `results/manifest.json` has `probe: null`. Run `make probe` to measure it on your box.

---

## 4. Statistical protocol

`scipy` is not installed and is not a dependency. Everything is numpy.

**Unit of analysis.** One `(scenario, model, seed)` cell yields one scalar:
`mean_eval_return`, the mean undiscounted episode return over the shared evaluation
episodes under the greedy policy.

**Pairing.** Comparisons are paired by seed. Because environment seeds exclude the model
name, both arms of every comparison saw byte-identical environment streams.

**Confidence intervals.** Percentile bootstrap over the paired differences, with the
generator seeded from `(ANALYSIS_SEED, "boot", comparison_id)` so every comparison has its
own reproducible stream and no interval depends on how many other comparisons ran first.

> One consequence worth stating, because it looks like an inconsistency and is not.
> The primary comparison P1 (family F1) and the `irregular_dt` row of family F4 are the
> same underlying contrast, deliberately reported under both — F1 is the pre-registered
> singleton, F4 places it in the across-scenario control set. Because the bootstrap
> generator is keyed on `comparison_id`, the two rows carry independently resampled
> intervals that can differ in the last reported decimal. Their **p-values are identical**,
> since the sign-flip test in the exact regime consults no RNG at all.

**Hypothesis test.** Two-sided paired sign-flip permutation test. For `n <= 20` **all
`2**n` sign vectors are enumerated exactly** — no RNG, no sampling error, a deterministic
p-value. Enumeration uses a subset-sum recursion rather than a `2**n × n` matrix: flipping
a subset maps the total `T` to `T - 2·sum(subset)`, so the cost is `O(2**n)` rather than
`O(2**n · n)`. Beyond that limit the test falls back to Monte-Carlo with the
Phipson–Smyth `+1` correction, so a p-value can never be zero. `p_method` is recorded in
every result row.

**Effect size.** Cliff's delta, a rank-based dominance measure in `[-1, +1]`, reported
alongside the mean difference — a difference in reward units is not comparable across
scenarios with different reward scales, but delta is. Magnitude thresholds follow Romano
et al. (2006).

**Multiplicity.** Holm–Bonferroni within each declared family. Holm controls the
family-wise error rate without assuming independence, which matters because comparisons
inside a family share arms and are correlated. Adjusted p-values and a `significant`
boolean are columns in `results/tables/comparisons.csv`.

**Pre-registration.** `configs/preregistration.yaml` is written before any confirmatory run
and its sha256 is recorded in `results/manifest.json`. It declares the primary comparison
and seven families. Comparison expansion is a pure function of that file, so the set of
tested comparisons is fixed before any result is read — which is what makes the Holm
correction honest rather than a post-hoc count.

**Reporting rules.**

- Every reported difference carries `n`, mean, interval, p, `p_method`, family and adjusted p.
- Numbers are formatted to three decimals in prose. The previous ablation table reported
  "gains" to seventeen significant digits with no uncertainty column at all, and in most
  rows the magnitude of the reported gain was smaller than the baseline's own across-seed
  standard deviation.
- **`derive_verdict` is the only source of a verdict, and a directional one needs both an
  interval that excludes zero and survival of Holm correction.** `compare_paired` calls it
  before any family is known, so only the interval speaks there; `holm_bonferroni` calls it
  again with `significant` supplied and demotes any row the correction did not reject to
  `no_detectable_difference`.
- **Any difference whose verdict is not directional is described as "no detectable
  difference", never as a gain.** `describe_comparison` reads the verdict rather than the
  raw interval, so wording and verdict cannot disagree, and a unit test asserts the words
  "gain", "improvement", "wins" and "better" cannot appear in such a sentence.

**Mandatory baselines.** `random_policy`, `constant_channel` and `greedy_heuristic` appear
as rows in **every** generated table. Any model filter in the analysis layer is forbidden
and `tests/test_no_filter.py` enforces it both structurally (no hardcoded policy list, no
`isin` filtering) and behaviourally (every table retains every policy that ran).

### 4.1 The learning check

Every training run evaluates the model **twice on the identical evaluation stream**: once
with freshly initialised weights and once after training. Both go into `all_runs.csv`.

If a model does not beat its own initialisation, the generated report says exactly that. If
PPO does not beat `random_policy`, the report says exactly that. That is the result.

**This check is a precondition, not a footnote.** Two studies at budgets a factor of sixteen
apart disagree on it — Study A's cells did not clear it and Study B's mostly did — and the
disagreement propagates to nearly every architectural family. Any family whose arms did not
clear the learning check is comparing initialisations, and the documents say so at the point
of use. Study B additionally records, per run, the **first-update** values of the critic's
explained variance and the rollout policy entropy alongside the final ones
(`first_update_explained_variance`, `first_update_policy_entropy_mean`,
`final_policy_entropy_mean`, `first_update_mean_rollout_return`,
`final_mean_rollout_return` — added to `scripts/run_suite.py` as strictly additive columns,
`NaN` for heuristics). Those support the descriptive mechanistic endpoints in
[`STUDY_B.md`](STUDY_B.md) §6, whose one transferable warning is that **falling policy
entropy alone does not establish learning**: it fell hardest in the two models that learned
nothing at all.

---

## 5. Federated design

**Genuine hierarchy.** Clients initialise from **their edge**, never from the cloud, and
the edge parameters **persist across the inner round loop and are the aggregation target**.
The previous implementation seeded every client from the global model and computed an edge
state that it then discarded, making the hierarchy a numerical no-op: hierarchical and flat
FedAvg agreed to within <!--v:hist.fedavg_agreement_max_abs-->1.19e-07<!--/v--> over every
tensor (an audit figure for a deleted revision, registered in
`docs/historical_figures.yaml`; it is not derivable from `results/`). A regression test now
asserts the two are meaningfully different.

**Strict compute matching.** Every arm consumes exactly the same number of environment
steps, and each runner reports `train_steps` as the **sum of `agent.total_env_steps` over
every agent it actually trained**, next to `declared_train_steps` with a boolean match
column. The previous implementation recorded 3,072 steps for every federated row while
executing 18,432 — a sixfold compute advantage, unlabelled.

**Measured communication.** Every transfer is recorded at the moment it happens and
`num_bytes` is computed from the actual tensors. No closed-form arithmetic anywhere. The
previous implementation hand-added a formula, producing a ratio of exactly 4/3 and a
standard deviation of exactly zero across all seeds.

Design intent the implementation must make true: hierarchical places only edge↔cloud
payloads on the wide-area link, so it moves **fewer cloud bytes and more total bytes** than
flat. All three columns — `cloud_megabytes`, `edge_local_megabytes`, `total_megabytes` — are
reported in every federated table, so the hierarchy's cost is visible next to its benefit.

`centralized` transmits no model parameters; its ledger totals are zero and it is labelled
a **performance reference, not a communication baseline**.

The federated study varies topology at fixed architecture. **Its conclusion is about
topology and is not evidence about architecture**, and the report says so.

---

## 6. What the benchmark is allowed to conclude

Nothing, until the runs finish. If the pre-registered primary comparison comes out with an
interval containing zero, the report states that the repository found **no detectable
benefit from continuous-time state updates under irregular decision intervals at that
budget**, and the README says it on the first screen. If `random_policy` outranks a PPO
model, the table shows it in the same font as everything else. If a model fails the learning
check, the report says the model did not learn.

**And a conclusion is scoped to its budget.** Both studies are reported, neither is hidden,
and every claim in the documentation names the budget that produced it. Where they disagree,
the disagreement is reported as the finding rather than resolved in favour of the more
flattering one. A result that moves when the step count moves is a fact about this
benchmark, and burying either half of it would be the same failure as publishing a fabricated
number — a document that does not match its artifacts.

The repository's value is a correct, seeded, capacity-matched, pre-registered, statistically
honest benchmark — not a win.
