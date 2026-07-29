# PPO-LTC-CfC for Dynamic Spectrum Access in Hierarchical Federated IoT Networks

A seeded, capacity-matched, pre-registered benchmark of **continuous-time recurrent
policies** (LTC, CfC) against gated, attention and memoryless baselines on a partially
observed dynamic-spectrum-access task, plus a hierarchical federated topology study.

**The headline result is negative.** The pre-registered primary comparison found no
detectable effect, no architectural family separated from any other, and the only
comparisons that survive multiplicity correction are the ones in which a zero-parameter
hand-written heuristic beats every learned policy. That is what the artifacts say, so
that is what this document says.

---

## Abstract

This is a **low-budget study**, and the budget is the first thing you should know about
it. Each training cell runs <!--v:matrix.training_steps-->10240<!--/v--> environment steps
and <!--v:runs.gradient_steps-->240<!--/v--> gradient updates on 4 CPU cores; there is no
pretraining, no hyperparameter search and no curriculum. The confirmatory matrix is
<!--v:matrix.n_scenarios-->5<!--/v--> scenarios × <!--v:matrix.n_seeds-->12<!--/v--> seeds ×
(<!--v:matrix.n_models-->7<!--/v--> PPO models + <!--v:matrix.n_baselines-->3<!--/v-->
zero-parameter heuristics) = <!--v:matrix.num_cells-->600<!--/v--> cells, each evaluated over
<!--v:matrix.eval_episodes-->32<!--/v--> held-out episodes, with a separate federated study
of 3 topology arms × <!--v:matrix.n_seeds-->12<!--/v--> seeds at
<!--v:fed.hierarchical_federated.train_steps-->36864<!--/v--> environment steps per arm. The
configuration is recorded in `results/manifest.json` and `configs/suite.yaml`.

Seven recurrent architectures are compared at a **matched parameter budget** (an eighth,
`ppo_lstm`, is registered and capacity-matched but is exploratory and was not run) — every
model is exactly two blocks deep and its hidden width is solved, not chosen, so that across
the seven that ran the largest and smallest trainable parameter counts differ by under two
percent (`CLAIM.BUDGET.spread`). Environment seeds are derived by `blake2b` from
`(base_seed, scenario, role, index)` and deliberately **exclude the model name**, so every
policy in a cell sees a byte-identical environment stream and every comparison is paired by
seed. The statistical protocol — pre-registered in `configs/preregistration.yaml`, whose
sha256 is recorded in the manifest before any confirmatory run — is a two-sided paired
**exact** sign-flip permutation test (all `2^12` sign vectors enumerated), a 20,000-draw
percentile bootstrap interval, and Holm–Bonferroni correction within each declared family
at α = <!--v:prereg.alpha-->0.05<!--/v-->.

The pre-registered primary comparison asks whether letting the elapsed decision interval
enter the recurrent state update helps under irregular decision intervals, using two arms
that are architecturally identical, parameter-identical and input-identical, and differ
only in whether `dt` reaches the cell's state update. **It returns no detectable
difference: the interval spans zero and the exact test does not come close to rejecting**
(`CLAIM.PRIMARY.P1`). Across <!--v:cmp.total-->104<!--/v--> pre-registered comparisons,
<!--v:cmp.significant-->35<!--/v--> are significant after correction, and all of them are
cases where the `greedy_heuristic` baseline beats a learned policy — in every scenario, at
every seed (Cliff's delta <!--v:f6.cliffs_delta_max-->-1.000<!--/v--> throughout). No learned
model beat its own random initialisation after correction (`CLAIM.LEARN.count`). The
federated study reproduces the intended topology trade-off in measured bytes — hierarchical
aggregation moves far less wide-area traffic than flat federated
(`CLAIM.FED.cloud_mb.hierarchical_federated` vs `CLAIM.FED.cloud_mb.flat_federated`) —
while the three arms' returns are statistically indistinguishable.

The contribution here is therefore **methodological, not architectural**: a correct,
seeded, capacity-matched, pre-registered and statistically honest benchmark that reports a
null. See [Limitations](#limitations-and-threats-to-validity) for why the compute budget is
the leading candidate explanation, and what evidence in the run logs supports that.

---

## Headline numbers

**Every table and every number in this section is written by `scripts/render_docs.py` from
`results/claims.json` and `results/tables/*.csv`.** The blocks below sit between
`<!-- BEGIN GENERATED -->` markers and are rewritten in place by `make report`; the inline
values sit in `<!--v:key-->` spans. `make check` fails the build if any of them differs from
what the committed results produce, so a number in this file cannot drift from the evidence
and cannot be edited by hand. [`RESULTS.md`](RESULTS.md) is generated the same way by
`scripts/make_report.py` and carries the full tables.

**Pre-registered primary comparison P1** — `ppo_cfc` vs `ppo_cfc_dtblind` on
`irregular_dt`, seed-matched pairs:

<!-- BEGIN GENERATED: p1_table -->
| quantity | value | claim_id |
|---|---|---|
| mean paired difference | -0.007 | `CLAIM.PRIMARY.P1` |
| 95% bootstrap CI | [-3.869, 3.980] | `CLAIM.PRIMARY.P1` |
| p (exact sign flip) | 0.998 | `CLAIM.PRIMARY.P1` |
| Holm-adjusted p | 0.998 | `CLAIM.PRIMARY.P1` |
| n paired seeds | 12 | `CLAIM.PRIMARY.P1` |
| verdict | `no_detectable_difference` | `CLAIM.PRIMARY.P1` |
<!-- END GENERATED: p1_table -->

The interval contains zero. **At this budget, this repository finds no detectable benefit
from letting the elapsed decision interval enter the recurrent state update under irregular
decision intervals.**

**Overall ranking**, pooled over the confirmatory scenarios, every policy that ran, no filter:

<!-- BEGIN GENERATED: ranking_table -->
| rank | policy | mean eval return | 95% CI over scenarios | claim_id |
|---|---|---|---|---|
| 1 | `greedy_heuristic` (0 parameters) | 8.291 | [-7.753, 19.881] | `CLAIM.RANK.greedy_heuristic` |
| 2 | `ppo_transformer` | -63.830 | [-77.352, -51.902] | `CLAIM.RANK.ppo_transformer` |
| 3 | `ppo_cfc` | -64.177 | [-76.721, -53.891] | `CLAIM.RANK.ppo_cfc` |
| 4 | `ppo_mlp` | -64.260 | [-77.338, -53.606] | `CLAIM.RANK.ppo_mlp` |
| 5 | `ppo_ltc` | -64.293 | [-76.882, -52.883] | `CLAIM.RANK.ppo_ltc` |
| 6 | `ppo_gru` | -64.351 | [-76.740, -54.201] | `CLAIM.RANK.ppo_gru` |
| 7 | `ppo_cfc_dtblind` | -64.533 | [-77.492, -52.890] | `CLAIM.RANK.ppo_cfc_dtblind` |
| 8 | `ppo_ltc_cfc` | -64.622 | [-77.732, -53.419] | `CLAIM.RANK.ppo_ltc_cfc` |
| 9 | `random_policy` (0 parameters) | -64.882 | [-77.760, -53.366] | `CLAIM.RANK.random_policy` |
| 10 | `constant_channel` (0 parameters) | -65.039 | [-77.544, -53.715] | `CLAIM.RANK.constant_channel` |
<!-- END GENERATED: ranking_table -->

The seven learned policies occupy ranks 2 through 8, spanning under one reward point, with
`random_policy` immediately below them and a zero-parameter heuristic far above. **None of
the gaps between ranks 2 and 8 is statistically detectable** — see
[Negative and null results](#negative-and-null-results). Treat that block as a tie.

---

## Findings

### 1. The pre-registered primary question returns a null

`ppo_cfc` and `ppo_cfc_dtblind` share an architecture, a parameter count, an
initialisation distribution, an input vector and an environment stream. They differ in
exactly one factor: whether `dt` enters the CfC time gate or is pinned to a constant there.
Both still receive `dt` as an encoder input feature. This is the sharpest available test of
the repository's thesis, and it was declared before the run.

It came out null (`CLAIM.PRIMARY.P1`, above). The same contrast repeated across every
scenario (family F4) is null in all <!--v:f4.n-->5<!--/v-->, and its two largest magnitudes
point in *opposite* directions: <!--v:claim.CLAIM.CMP.F4_dt_awareness:ppo_cfc_vs_ppo_cfc_dtblind:bursty_irregular.value-->2.577<!--/v--> on `bursty_irregular`
(`CLAIM.CMP.F4_dt_awareness:ppo_cfc_vs_ppo_cfc_dtblind:bursty_irregular`) and
<!--v:claim.CLAIM.CMP.F4_dt_awareness:ppo_cfc_vs_ppo_cfc_dtblind:stationary.value-->-3.112<!--/v--> on `stationary`
(`CLAIM.CMP.F4_dt_awareness:ppo_cfc_vs_ppo_cfc_dtblind:stationary`). The
largest apparent effect is on `stationary`, a scenario in which `dt` never varies and the
two arms are identical in expectation. That is a direct signature of noise, not of an
effect, and it is the reason the F4 family is reported as five nulls rather than as one
promising scenario.

### 2. A zero-parameter heuristic wins everything

Family F6 tests each of the <!--v:matrix.n_models-->7<!--/v--> PPO models against
`greedy_heuristic` in each of the <!--v:matrix.n_scenarios-->5<!--/v--> scenarios:
<!--v:f6.n-->35<!--/v--> pre-registered comparisons. **All of them favour the heuristic and
all of them remain significant after Holm correction** at the exact test's floor (worst raw
p across the family = <!--v:f6.p_value_max-->0.000488<!--/v-->, worst Holm-adjusted p =
<!--v:f6.holm_adjusted_p_max-->0.0171<!--/v-->), with Cliff's delta
<!--v:f6.cliffs_delta_max-->-1.000<!--/v--> — the heuristic wins at every single seed of
every single comparison. Margins run from <!--v:claim.CLAIM.CMP.F6_vs_greedy_baseline:ppo_gru_vs_greedy_heuristic:interference_heavy.value-->-66.265<!--/v-->
(`CLAIM.CMP.F6_vs_greedy_baseline:ppo_gru_vs_greedy_heuristic:interference_heavy`) to
<!--v:claim.CLAIM.CMP.F6_vs_greedy_baseline:ppo_ltc_vs_greedy_heuristic:non_stationary.value-->-81.172<!--/v-->
(`CLAIM.CMP.F6_vs_greedy_baseline:ppo_ltc_vs_greedy_heuristic:non_stationary`).

The baseline is not a strawman. `GreedyOccupancyPolicy` carries a `dt`-decayed exponential
memory of sensed primary occupancy plus a contention estimate inferred from its own
collision history, and it reads nothing a neural policy cannot read
(`dsa/envs/heuristics.py`, and `docs/PROTOCOL.md` §2.0 for why the task was deliberately
kept partially observed). Note what that means: the strongest policy in this benchmark is
*already* stateful and `dt`-aware — those inductive biases are hand-coded into it. What
this study failed to show is that PPO can **learn** them at this budget, not that they are
worthless.

### 3. Nothing learned

Every run is evaluated twice on the identical evaluation stream: once with freshly
initialised weights and once after training. Family F5 tests that paired difference for all
<!--v:learn.tested-->35<!--/v--> (model, scenario) cells. **<!--v:learn.learned-->0<!--/v-->
of <!--v:learn.tested-->35<!--/v--> improved significantly over their own random
initialisation after correction** (`CLAIM.LEARN.count`).
<!--v:f5.interval_excludes_zero-->4<!--/v--> cells had raw bootstrap intervals excluding zero
— in both directions, so some models trained to a *worse* policy than their own
initialisation — and <!--v:f5.excludes_zero_not_significant-->4<!--/v--> of those
<!--v:f5.interval_excludes_zero-->4<!--/v--> fail Holm correction. Every one of them is
therefore published with the verdict `no_detectable_difference`: under a pre-registered
protocol, a raw interval that excludes zero is not a result if the correction the protocol
declared does not reject. All <!--v:learn.tested-->35<!--/v--> are reported as null.

The diagnostics in the committed run log say why, and they are more informative than the
returns themselves. Read from `results/all_runs.csv` over the
<!--v:runs.neural_rows-->420<!--/v--> neural runs:

<!-- BEGIN GENERATED: diagnostics_table -->
| diagnostic | column | value |
|---|---|---|
| critic explained variance (max over all runs) | `final_explained_variance` | `1.28e-04` |
| mean policy entropy after training | `mean_policy_entropy_nats` | `1.971` |
| mean policy entropy before training | `untrained_mean_policy_entropy_nats` | `2.046` |
| entropy of a uniform policy over 8 channels | `ln(num_channels)` | `2.079` |
| gradient steps per run | `total_gradient_steps` | `240` |
<!-- END GENERATED: diagnostics_table -->

The critic explains essentially none of the return variance, so the GAE advantages the
policy gradient consumes are close to noise, and the policy stays close to uniform — the
last two rows of that table are the measured entropy and the entropy of a uniform policy
over the channel set. These are direct reads of columns in a committed CSV, not
pre-registered claims; they are diagnostics of *why* the confirmatory result is null.

### 4. No architectural family separated from any other

- **F2, continuous-time vs gated** (`ppo_ltc`, `ppo_cfc`, `ppo_ltc_cfc` vs `ppo_gru`,
  <!--v:f2.n-->15<!--/v--> comparisons): <!--v:f2.significant-->0<!--/v--> significant,
  every interval spans zero.
- **F3, hybrid decomposition** (`ppo_ltc_cfc` vs each of its two components,
  <!--v:f3.n-->10<!--/v--> comparisons): <!--v:f3.significant-->0<!--/v--> significant. The
  hybrid the repository title names is neither better nor worse than either half, and it
  finishes last of the seven learned models in the pooled ranking
  (`CLAIM.RANK.ppo_ltc_cfc`).
- **F4, `dt`-awareness** (<!--v:f4.n-->5<!--/v--> comparisons):
  <!--v:f4.significant-->0<!--/v--> significant, see finding 1.

An observation, offered as a ranking and explicitly *not* as a claim, because none of the
underlying differences is detectable: the best-placed learned model is `ppo_transformer`
(`CLAIM.RANK.ppo_transformer`) — the block a previous version of this repository
mislabelled "LFM" — and the two genuinely liquid cells and the hybrid sit below it.

### 5. The federated hierarchy does what it is supposed to do, in measured bytes

All three arms are strictly compute-matched: `train_steps == declared_train_steps ==`
<!--v:fed.hierarchical_federated.train_steps-->36864<!--/v--> for every row of
`results/federated_all_runs.csv`. Communication is counted from the actual tensors at the
moment of transfer by `CommunicationLedger`; no script, table or document computes a byte
count from a formula.

<!-- BEGIN GENERATED: federated_bytes_table -->
| arm | wide-area (cloud) MB | claim_id | edge-local MB | total MB |
|---|---|---|---|---|
| `centralized` | 0.0000 | `CLAIM.FED.cloud_mb.centralized` | 0.0000 | 0.0000 |
| `flat_federated` | 16.5672 | `CLAIM.FED.cloud_mb.flat_federated` | 0.0000 | 16.5672 |
| `hierarchical_federated` | 2.2302 | `CLAIM.FED.cloud_mb.hierarchical_federated` | 15.2928 | 17.5230 |
<!-- END GENERATED: federated_bytes_table -->

(Edge-local and total columns are read from `results/tables/federated_table.csv`;
`centralized` transmits no parameters and is a performance reference, not a communication
baseline.) Hierarchical aggregation moves
<!--v:fed.cloud_ratio_flat_over_hier-->7.43<!--/v-->× less wide-area traffic than flat
federation while moving *more* bytes in total — exactly the predicted trade-off, and the
direction is a property of the topology rather than a tuned outcome.

The returns, however, are indistinguishable. All <!--v:f7.n-->3<!--/v--> F7 comparisons are
non-significant after correction (`CLAIM.CMP.F7:hier_vs_flat`,
`CLAIM.CMP.F7:hier_vs_centralized`, `CLAIM.CMP.F7:flat_vs_centralized`); the largest is
<!--v:claim.CLAIM.CMP.F7:hier_vs_centralized.value-->-2.744<!--/v--> between hierarchical and
centralized with Holm-adjusted p =
<!--v:claim.CLAIM.CMP.F7:hier_vs_centralized.holm_adjusted_p-->0.245<!--/v-->.
<!--v:f7.excludes_zero_not_significant-->2<!--/v--> of the <!--v:f7.n-->3<!--/v--> bootstrap
intervals marginally exclude zero while the pre-registered exact test does not reject, which
is exactly the situation the pre-registration exists to adjudicate: **the test governs, we
do not claim a difference, and the published verdict on all three rows is
`no_detectable_difference`.** The federated study is evidence about *topology and
communication cost*, not about architecture — a single model was used for all arms.

### 6. The seeding defect that invalidated the previous results is fixed and gated

The four cells `irregular_dt × {ppo_cfc, ppo_gru} × {seed 7, 19}` were run three ways: in a
2-model invocation, in a differently-ordered 4-model invocation, and inside the full
<!--v:matrix.num_cells-->600<!--/v-->-cell run. Every non-`wall_` column is byte-identical
across all three, maximum absolute difference zero. The equivalent nuisance spread in the
previous implementation was roughly forty reward points, against a headline effect of about
two. CI re-runs an order-invariance check on every push.

The learner is also verifiably correct on the check that catches stale hidden states,
dropout mismatch and train/eval-mode bugs simultaneously: the maximum first-epoch PPO ratio
deviation over all <!--v:runs.neural_rows-->420<!--/v--> neural runs is
<!--v:runs.ratio_deviation_max-->7.15e-07<!--/v-->
(`results/all_runs.csv :: max_first_epoch_ratio_deviation`) — three orders of magnitude
inside the unit test's `1e-4` bound, and inside the `1e-3` gate that `scripts/analyze.py`
applies to the whole suite before it will emit a report. **The null is not a stale-hidden-state,
dropout or train/eval-mode bug in the update.**

---

## Negative and null results

Mandatory section, listing every comparison family that came out null. Source:
`results/tables/comparisons.csv`, <!--v:cmp.total-->104<!--/v--> pre-registered comparisons
in <!--v:cmp.families-->7<!--/v--> families, <!--v:cmp.significant-->35<!--/v--> significant
after Holm correction — all of them in F6. The "what it tests" column is the first sentence
of each family's declaration in `configs/preregistration.yaml`; the counts and the outcome
are computed from the comparisons table.

<!-- BEGIN GENERATED: family_outcome_table -->
| family | what it tests (from `configs/preregistration.yaml`) | comparisons | significant after Holm | outcome |
|---|---|---|---|---|
| `F1_primary` | The pre-registered primary comparison, reported unadjusted. | 1 | 0 | no detectable difference |
| `F2_continuous_vs_gated` | Do continuous-time cells beat a conventional gated RNN of equal parameter budget and equal depth? | 15 | 0 | no detectable difference |
| `F3_hybrid_decomposition` | Does the LTC+CfC hybrid the repository title names beat each of its two single-cell components? | 10 | 0 | no detectable difference |
| `F4_dt_awareness` | The one-factor dt-awareness contrast, run across all five primary scenarios. | 5 | 0 | no detectable difference |
| `F5_learning_check` | Did each model beat its own random initialisation on the identical evaluation stream? | 35 | 0 | no detectable difference |
| `F6_vs_greedy_baseline` | Each learned policy against a fixed, zero-parameter greedy occupancy heuristic. | 35 | 35 | all 35 favour `greedy_heuristic` |
| `F7_federated` | Topology comparison at fixed architecture and strictly matched compute. | 3 | 0 | no detectable difference |
<!-- END GENERATED: family_outcome_table -->

In plain terms, the following claims are **not** supported by this repository's evidence:

- that liquid (LTC / CfC) cells outperform gated recurrence on this task;
- that continuous-time state updates help under irregular decision intervals;
- that the LTC + CfC hybrid is better than either of its parts;
- that any of the seven learned policies is better than any other;
- that PPO at this budget learns anything at all on this task;
- that hierarchical federation changes achieved return.

Supported: that hierarchical federation reduces measured wide-area traffic while increasing
total traffic; that a hand-written heuristic beats every learned policy here.

Descriptive, untested, and offered only as an ordering: the seven learned policies all sit
above `random_policy` in the pooled ranking — the gap between rank 2 and rank 9 is the
difference of `CLAIM.RANK.ppo_transformer` and `CLAIM.RANK.random_policy`, about one reward
point against a between-scenario spread of tens of points. **No test was run on this
contrast and none should be read into it.** It is noted only because in the previous
version of this repository random beat three of six trained models, and that is no longer
the case.

---

## Limitations and threats to validity

**1. Compute budget — the leading explanation for the null.**
<!--v:matrix.training_steps-->10240<!--/v--> environment steps and
<!--v:runs.gradient_steps-->240<!--/v--> gradient updates per cell is small for PPO on a
partially observed task. The
diagnostic evidence in finding 3 (critic explained variance indistinguishable from zero,
policy entropy barely below uniform) says the learner never reached the regime where its
advantage estimates carry signal. A null under those conditions is a statement about *this
budget*, not about liquid networks. An exploratory budget ladder was run locally on two
models, one scenario and two seeds, at multiples of the confirmatory budget; it indicated
that learning does emerge — entropy falls, explained variance rises, trained beats untrained
consistently — at a budget roughly an order of magnitude larger than the one used here. That
diagnostic is **not** pre-registered, **not** part of the evidence base, and **not**
committed (it lands in the gitignored `results/raw/`), so no number from it appears
anywhere in this repository's documentation. It is recorded here as the most likely
direction for future work, and it should be treated as a hypothesis, not a result. It also
does not rescue F6: the ceiling that ladder approached remains far below `greedy_heuristic`.

**2. A single environment family.** All five confirmatory scenarios are variants of one
simulator with one observation layout, 8 channels, 64-step episodes and one reward
function. `irregular_dt` differs from `stationary` in exactly one field, which is good for
causal attribution and bad for external validity. Conclusions do not transfer to other
spectrum models, to continuous action spaces, or to longer horizons.

**3. Simulator fidelity.** The environment is a stochastic abstraction, not a radio. Channel
occupancy is a relaxing Bernoulli process, interference and quality are smoothed scalars,
background devices are a birth–death count of hidden terminals, and propagation, protocol
overhead, retransmission and hardware effects are absent. It is a controlled testbed for a
learning question, and results should be read as such.

**4. `dt` varies in only 2 of the 5 confirmatory scenarios.** The continuous-time thesis is
only genuinely under test on `irregular_dt` and `bursty_irregular`; the other three are
controls. That is deliberate, but it means the primary question rests on
<!--v:matrix.n_seeds-->12<!--/v--> paired seeds in one scenario. With `n = 12` the exact
sign-flip test's p-value floor is `2/2^12`, which is enough power to detect a large
consistent effect and not enough to detect a small one. This study can distinguish "no large
effect" from "a large effect"; it cannot rule out a small one.

**5. Multiplicity, and what correction costs.** <!--v:cmp.total-->104<!--/v--> comparisons
were run. Holm–Bonferroni within family is the pre-registered correction and it is
conservative: <!--v:cmp.excludes_zero_not_significant-->6<!--/v--> comparisons have a raw
bootstrap interval that excludes zero and an adjusted p-value that does not reject. All
<!--v:cmp.excludes_zero_not_significant-->6<!--/v--> are published with the verdict
`no_detectable_difference`, which is the right call under a pre-registered protocol and is
also the call that loses information. Anyone re-analysing should read
`results/tables/comparisons.csv`, which carries raw p, adjusted p, interval and Cliff's
delta for every comparison, so the demotion is fully reversible from the committed data.

**6. Statistical scope.** Bootstrap intervals are percentile intervals over
<!--v:matrix.n_seeds-->12<!--/v--> paired differences — small-sample intervals with real
coverage error. The unit of analysis is the per-cell mean over
<!--v:matrix.eval_episodes-->32<!--/v--> evaluation episodes, so within-cell episode variance
is not propagated into the family-level tests.

**7. Evaluation protocol.** Reported returns are under a greedy (argmax) policy on a fixed
shared evaluation stream. A stochastic-policy evaluation would give different numbers, and
the two entropy metrics are deliberately reported separately
(`mean_action_histogram_entropy` is an action histogram under argmax evaluation;
`mean_policy_entropy_nats` is the real policy entropy) because conflating them is how the
previous version produced a spurious "entropy collapse" narrative.

**8. One model in the federated study.** All three topology arms use `ppo_ltc_cfc`. The
federated conclusion is about topology and communication accounting only; it is not
evidence about architecture, and the arms' returns were indistinguishable anyway.

**9. Manifest provenance.** `results/manifest.json` records the git commit that was HEAD
when the run started, which necessarily precedes the commit that contains the results. Use
the recorded config contents, pre-registration sha256 and per-CSV checksums as the
provenance anchor rather than that commit id.

**10. `ppo_lstm` and two scenarios are exploratory.** `ppo_lstm`, `noisy_partial` and
`bursty_load` are registered, capacity-matched, unit-tested and runnable, but are excluded
from the confirmatory matrix — `ppo_lstm` as a near-duplicate of `ppo_gru`, the two
scenarios to buy statistical power at 12 seeds. They are not part of any family and no
claim rests on them.

---

## Relationship to previous versions of this repository

Earlier revisions of this repository published results that do not hold up. This is stated
here because the credibility of the current numbers depends on it being said plainly.

The figures in this section come from an adversarial audit of those earlier revisions, each
defect independently reproduced against the code and artifacts as they then stood. They are
**not** derived from the current `results/` tree, they carry no `claim_id`, and none of the
old artifacts survives in this repository — the entire previous `results/` tree, the
fabricated report generator and its rendered output were deleted rather than archived. What
remains verifiable today is the regression test attached to each defect.

They are the *only* numbers in this repository's documentation that no expression over
`results/` can produce, so they are the only ones exempt from the generator. Each is
registered in [`docs/historical_figures.yaml`](docs/historical_figures.yaml) with what was
measured and against what, and each is rendered from that registry — an entry cannot be
added without recording its provenance.

- **Fabricated artifacts shipped.** A committed benchmark report and its generator produced
  numbers that were string literals in the source — no file was ever read — while the
  generated document asserted that the results were empirical and unbiased. A committed
  results JSON recorded four different algorithms with a byte-identical final reward, and a
  demo output file contained a curve that was another curve plus a constant offset. Those
  files have been deleted rather than archived.
- **The flagship result did not reproduce.** Re-running the documented command moved the
  headline model from first to sixth of seven. The cause was an environment-seeding bug: a
  mutable class-level instance counter added an offset to the base seed, so a model's
  environment stream depended on its position in the `--models` list. Measured with a fixed
  policy, that nuisance was worth up to roughly forty reward points against a headline
  effect of about two. Nothing in the old `results/` tree was a controlled comparison, so
  the entire tree was deleted.
- **The published tables were filtered.** The old analysis script defined a core-model list
  that omitted the uniform-random baseline and filtered it out of exactly the four tables
  the README cited. Restored, random ranked fourth of seven, ahead of three trained models.
  There is now no model filter anywhere in the analysis layer, and a test
  (`tests/test_no_filter.py`) fails the build if one appears.
- **The models were not what their names said.** The "LTC" had a constant learned time
  constant with no dependence on input or state, which reduces algebraically to a sigmoid
  forget gate; the "LFM" was `nn.MultiheadAttention` plus a GELU feed-forward and two layer
  norms; the ablation that removed "LTC" also removed a transformer layer, and parameter
  count correlated with reward at Spearman
  <!--v:hist.spearman_param_reward-->0.829<!--/v--> across the model set. All three of the
  load-bearing words in the old title were wrong.
- **Nothing was trained, and the minibatching was unreachable.** The old budget amounted to
  72 full-batch gradient steps, and dropout was live during rollout, update and evaluation
  because `.train()` and `.eval()` appeared nowhere in the live path.
- **The federated hierarchy was a no-op that was reported as a saving.** Clients were seeded
  from the global model rather than their edge, the edge state was computed and discarded,
  and communication was hand-computed arithmetic rather than measured bytes. The arms were
  also a six-fold compute mismatch recorded as if matched.

What replaced each of those defects is described in
[`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) and
[`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md), and each has a regression test. The
current repository publishes a null. That is a worse headline and a better artifact.

---

## What "LTC", "CfC" and "LFM" mean here

Names describe implementations. Full equations and citations are in
[`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md).

> Liquid Foundation Models are built on the **closed-form continuous-time (CfC) cell** of
> Hasani et al. (2022). In this repository the "LFM" component of the title is
> **`ppo_cfc`**. The attention block that a previous version of this repository called
> "LFM" is now **`ppo_transformer`** and is a baseline, not a liquid model.

> **`ppo_ltc_cfc` is the model the repository title abbreviates as "PPO-LTC-LFM":** an LTC
> block followed by a CfC (liquid-foundation) block.

- **`ppo_ltc`** implements the Liquid Time-constant network of Hasani et al. (AAAI 2021):
  the input- and state-dependent system time constant `tau_sys = tau / (1 + tau * f(x, I))`
  and the fused semi-implicit Euler solver of the paper's eq. 6, with `dt` entering only
  through the sub-step size. A unit test measures `tau_sys` directly and fails if it is
  constant in `(x, I)`.
- **`ppo_cfc`** implements the Closed-form Continuous-time cell of Hasani et al. (Nature
  Machine Intelligence 2022): the closed-form solution of the paper's eq. 10, with `dt`
  entering only through the time-gate exponent.
- **`ppo_transformer`** is banded-causal windowed self-attention with a GELU feed-forward
  and layer norms. It is called a transformer because that is what it is.
- **`ppo_cfc_dtblind`** is `ppo_cfc` with `dt` pinned to a constant inside the cell only.
  Same architecture, same parameter count, same initialisation, same inputs. It is the
  one-factor control that carries the primary comparison.

Every model is exactly two blocks deep, every hidden width is solved to hit a common
parameter target, and the seven that ran land within a couple of percent of each other
(`CLAIM.BUDGET.spread`) — so capacity is not a confound. (`ppo_lstm`, the eighth registered
model, is inside the same tolerance; its solved width and parameter count are in
`results/manifest.json :: models_solved`, but it was not run and no claim rests on it.) The
previous version's spread was more than two-fold.

---

## Reproduction

Requires Python `3.11` or newer, 4 CPU cores, no GPU. Every command below was run against
this tree; every path and flag resolves.

```bash
git clone <this repository>
cd PPO-LTC-LFM-for-Dynamic-Spectrum-Access-in-Hierarchical-Federated-IoT-Networks

make install      # pip install -r requirements.txt
make test         # pytest -q  -- the full unit suite
make check        # tables, RESULTS.md, README.md and docs/ all match a fresh analysis
```

`make check` is the honesty gate, and it has three parts, all of which write nothing:

1. `scripts/analyze.py --check` re-derives every table in `results/tables/` from
   `results/all_runs.csv` and fails on any difference;
2. `scripts/make_report.py --check` regenerates `RESULTS.md` and fails on any difference;
3. `scripts/render_docs.py --check` regenerates every generated region and every
   `<!--v:...-->` span in this README and in `docs/*.md`, and fails on any difference.

Part 3 is new, and it exists because parts 1 and 2 were not enough. An adversarial auditor
falsified two headline figures in this file — the primary comparison's mean difference and
the greedy baseline's pooled return — and the whole test suite and the CI report gate stayed
green, because the scan that was supposed to catch hand-typed metrics skipped markdown table
rows and stripped inline-code spans, which is exactly where those two numbers lived. Both
holes are now closed from both sides: the numbers are generated, and
`tests/test_no_hardcoded_metrics.py` scans table cells and inline code instead of skipping
them.

To reproduce the evidence from scratch:

```bash
make probe        # measure real steps/s on your box
make suite        # the confirmatory matrix
make federated    # the 3-arm topology study
make analyze      # tables, figures, claims.json
make report       # regenerate RESULTS.md and every generated block in the docs
# or all five in order:
make all
```

Budget note, so that `make all` does not surprise you. The committed evidence took
<!--v:runs.wall_minutes-->56.8<!--/v--> minutes of wall clock for the suite and
<!--v:fed.wall_minutes-->33.4<!--/v--> minutes for the federated study at
<!--v:config.num_workers-->4<!--/v--> workers (`results/all_runs.csv :: wall_seconds` and
`results/federated_all_runs.csv :: wall_seconds`, summed and divided by the worker count).
The throughput guard in `configs/suite.yaml` would have intervened above
<!--v:config.max_projected_wall_minutes-->40<!--/v--> projected minutes; it was explicitly
disabled with `--no-guard`, and `results/manifest.json` records `guard.applied = false` with
the requested and effective step counts equal. The matrix and the seed count were **not**
reduced to hit any projection. No probe throughput is recorded in the manifest for this run,
because the guard was skipped, so these figures come from the committed wall-clock columns
and nothing else.

Every script has a real CLI; these are the flags, not a sketch:

```bash
python scripts/run_suite.py --models ppo_cfc ppo_cfc_dtblind --scenarios irregular_dt \
                            --seeds 7 19 --training-steps 10240 --workers 4
python scripts/run_suite.py --smoke --output-dir results_smoke      # tiny CI budget
python scripts/run_suite.py --dry-run                               # probe and project only
python scripts/run_federated.py --arms hierarchical_federated flat_federated --seeds 7 19
python scripts/analyze.py --results-dir results --output-dir /tmp/scratch
python scripts/analyze.py --check                                   # tables are current?
python scripts/make_report.py --check                               # RESULTS.md is current?
python scripts/render_docs.py --check                               # README/docs are current?
```

To re-derive the primary claim yourself without trusting the analysis layer, take
`results/all_runs.csv`, filter `scenario == "irregular_dt"`, pivot
`mean_eval_return_trained` by seed for `ppo_cfc` and `ppo_cfc_dtblind`, difference the two
columns and enumerate all `2^12` sign vectors. That reproduces `CLAIM.PRIMARY.P1` exactly.

### What is committed, and what is not

Committed: `results/all_runs.csv`, `results/federated_all_runs.csv`,
`results/manifest.json`, `results/claims.json`, `results/tables/*.csv`,
`results/figures/*.png`, and the generated `RESULTS.md`. Per-run logs, exploratory
diagnostics and checkpoints go to `results/raw/` and `results/checkpoints/` and are
gitignored. `results/manifest.json` — not committed weights — is the reproducibility anchor:
git commit, exact command line, full config contents, pre-registration sha256, solved hidden
dimensions and parameter counts, library versions, real step counts, and a sha256 checksum
of `results/all_runs.csv`. (Only that one CSV is checksummed; the tables and
`federated_all_runs.csv` are instead verified by regenerating them — `make check`.)

`results/claims.json` gives all <!--v:claims.total-->120<!--/v--> citable numbers a
`claim_id` with source file, source column, interval, p-value and verdict. Prose that cites
a number cites a `claim_id`, and `scripts/render_docs.py` writes the number itself.

---

## Repository layout

```
dsa/seeding.py       pure, order-independent seed derivation
dsa/envs/            spectrum environment, scenarios, heuristic policies
dsa/models/          recurrent cells, shared actor-critic, capacity-matched registry
dsa/learner/         sequence-based recurrent PPO, evaluation harness
dsa/federated/       aggregation, topology, the three federated arms
dsa/analysis/        statistics, tables, figures
scripts/             probe, run_suite, run_federated, analyze, make_report, render_docs
configs/             suite.yaml, federated.yaml, preregistration.yaml
docs/                prose, plus historical_figures.yaml (the one non-results/ number source)
tests/               one test file per owned module, plus the honesty gates
```

## Documentation

- [`RESULTS.md`](RESULTS.md) — generated results, authoritative, regenerated by CI.
- [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) — the pre-registered design,
  the statistical protocol, the seeding contract, and how to verify any published number.
- [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md) — the equations actually implemented,
  with citations, and what each one is not.
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — the implementation contract the code is written
  against (seeding, environment, learner, analysis internals).
- [`docs/MODEL_CARDS.md`](docs/MODEL_CARDS.md) — per-model implementation cards and the
  tests that pin each equation.

## References

- Hasani, Lechner, Amini, Rus, Grosu. *Liquid Time-constant Networks.* AAAI 2021.
  arXiv:2006.04439.
- Hasani, Lechner, Amini, Liebenwein, Ray, Tschaikowski, Teschl, Rus. *Closed-form
  Continuous-time Neural Networks.* Nature Machine Intelligence 4, 2022. arXiv:2106.13898.
- Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization Algorithms.*
  2017. arXiv:1707.06347.
- Schulman, Moritz, Levine, Jordan, Abbeel. *High-Dimensional Continuous Control Using
  Generalized Advantage Estimation.* ICLR 2016. arXiv:1506.02438.
- Holm. *A simple sequentially rejective multiple test procedure.* Scandinavian Journal of
  Statistics 6, 1979.
- Phipson, Smyth. *Permutation p-values should never be zero.* Statistical Applications in
  Genetics and Molecular Biology 9, 2010.
- Efron, Tibshirani. *An Introduction to the Bootstrap.* Chapman & Hall, 1993.
- Cliff. *Dominance statistics: ordinal analyses to answer ordinal questions.* Psychological
  Bulletin 114, 1993.
- Romano, Kromrey, Coraggio, Skowronek. *Appropriate statistics for ordinal level data.*
  FAIR, 2006 (Cliff's delta magnitude thresholds).
- McMahan, Moore, Ramage, Hampson, Agüera y Arcas. *Communication-Efficient Learning of Deep
  Networks from Decentralized Data.* AISTATS 2017. arXiv:1602.05629.
- Liu, Zhang, Song, Letaief. *Client-Edge-Cloud Hierarchical Federated Learning.* IEEE ICC
  2020. arXiv:1905.06641.

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Daniel Foo Jun Wei.
