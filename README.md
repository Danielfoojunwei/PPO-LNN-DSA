# PPO-LTC-CfC for Dynamic Spectrum Access in Hierarchical Federated IoT Networks

A seeded, capacity-matched, pre-registered benchmark of **continuous-time recurrent
policies** (LTC, CfC) against gated, attention and memoryless baselines on a partially
observed dynamic-spectrum-access task, plus a hierarchical federated topology study.

**This repository reports two pre-registered studies at two compute budgets, and they do
not agree.** Study A trained every cell for <!--v:matrix.training_steps-->10240<!--/v-->
environment steps; Study B repeated the core of the matrix at exactly
<!--v:studyb.budget.env_step_ratio-->16<!--/v-->× that budget,
<!--v:studyb.matrix.training_steps-->163840<!--/v--> steps per cell. Study A concluded that
nothing learned, that no architecture separated from any other, and that a zero-parameter
heuristic beat everything. Study B, on the two scenarios it re-ran, reverses the first two
of those three conclusions and leaves the third exactly where it was.

**That disagreement is the headline result, and it is a result about benchmarks rather than
about liquid networks.** Study A's architectural comparisons were comparisons between random
initialisations, and they looked like clean, well-powered, correctly-corrected nulls. Nothing
in Study A's statistics was wrong. Only the budget was. Any RL comparison that does not show
its learning check is one budget away from meaning the opposite of what it says.

What did **not** move with the budget: the pre-registered primary comparison is a null at
both budgets, and the zero-parameter `greedy_heuristic` beats every learned policy at both
budgets, at every seed, in every scenario tested.

---

## Abstract

Two studies, pre-registered separately, are reported side by side and neither supersedes the
other.

**Study A (low budget).** <!--v:matrix.n_scenarios-->5<!--/v--> scenarios ×
<!--v:matrix.n_seeds-->12<!--/v--> seeds × (<!--v:matrix.n_models-->7<!--/v--> PPO models +
<!--v:matrix.n_baselines-->3<!--/v--> zero-parameter heuristics) =
<!--v:matrix.num_cells-->600<!--/v--> cells at <!--v:matrix.training_steps-->10240<!--/v-->
environment steps and <!--v:runs.gradient_steps-->240<!--/v--> gradient updates per cell, plus
a federated study of 3 topology arms × <!--v:matrix.n_seeds-->12<!--/v--> seeds. Its plan is
`configs/preregistration.yaml`, its evidence is `results/`, and its headline is a null: the
primary comparison did not reject, <!--v:cmp.significant-->35<!--/v--> of
<!--v:cmp.total-->104<!--/v--> comparisons were significant after Holm correction and all of
them favoured the heuristic, and <!--v:learn.learned-->0<!--/v--> of
<!--v:learn.tested-->35<!--/v--> trainable cells beat their own random initialisation.

**Study B (<!--v:studyb.budget.env_step_ratio-->16<!--/v-->× the budget).**
<!--v:studyb.matrix.n_scenarios-->2<!--/v--> scenarios ×
<!--v:studyb.matrix.n_seeds-->12<!--/v--> seeds ×
(<!--v:studyb.matrix.n_models-->7<!--/v--> PPO models +
<!--v:studyb.matrix.n_baselines-->3<!--/v--> heuristics) =
<!--v:studyb.matrix.num_cells-->240<!--/v--> cells at
<!--v:studyb.matrix.training_steps-->163840<!--/v--> environment steps and
<!--v:studyb.runs.gradient_steps-->3840<!--/v--> gradient updates per cell —
<!--v:studyb.matrix.total_env_steps_trained-->27525120<!--/v--> training environment steps in
total, <!--v:studyb.runs.wall_minutes-->143.3<!--/v--> wall-clock minutes on
<!--v:studyb.config.num_workers-->4<!--/v--> workers. Its plan is the separate file
`configs/preregistration_study_b.yaml`, frozen and sha256-recorded before its first cell ran,
and its evidence is `results/study_b/`. Three of Study A's five scenarios
(<!--v:studyb.descoped.scenarios-->`non_stationary`, `interference_heavy`, `bursty_irregular`<!--/v-->) and the federated sub-study were descoped to
pay for the budget increase; the seed count, the primary comparison, all
<!--v:studyb.matrix.n_models-->7<!--/v--> models and all
<!--v:studyb.matrix.n_baselines-->3<!--/v--> baselines were kept.

**What the budget changed.** At Study A's budget the critic explained no return variance and
the policy never left uniform, so the advantages driving the policy gradient were noise. At
Study B's budget <!--v:studyb.learn.learned-->10<!--/v--> of
<!--v:studyb.learn.tested-->14<!--/v--> trainable cells beat their own initialisation after
Holm correction (`CLAIM.LEARN.count`), gaining up to
<!--v:studyb.f8.margin_max-->24.517<!--/v--> reward points, and
<!--v:studyb.cmp.significant-->42<!--/v--> of <!--v:studyb.cmp.total-->52<!--/v-->
pre-registered comparisons are significant. Five of the seven architectures moved clear of
`random_policy`; `ppo_ltc` and `ppo_ltc_cfc` did not, and remain level with it. That contrast
is untested by design — the pre-registration puts `random_policy` in every table with an
interval but in no tested family — so it is reported here as a ranking observation, not as a
result.

**What the budget did not change.** The primary comparison — `ppo_cfc` vs
`ppo_cfc_dtblind` on `irregular_dt`, two arms that are architecturally identical,
parameter-identical and input-identical and differ only in whether the elapsed decision
interval `dt` reaches the recurrent state update — returns **no detectable difference at both
budgets** (`CLAIM.PRIMARY.P1` in each study). Study B's null is the stronger one, because
both of its arms demonstrably learned first. And `greedy_heuristic`, a hand-written
zero-parameter policy, still beats every learned policy in every one of the
<!--v:studyb.f7.n-->7<!--/v--> + <!--v:studyb.f10.n-->7<!--/v--> comparisons that test it,
with Cliff's delta <!--v:studyb.f7.cliffs_delta_max-->-1.000<!--/v--> — it wins at every seed —
and a pooled lead of <!--v:studyb.rank.heuristic_lead-->46.535<!--/v--> reward points over the
best learned policy.

**Three results that were not anticipated and are reported because they are real.** (i) The
pre-registered *control* for the `dt` contrast — the same two arms on a constant-`dt`
scenario, where they are informationally identical and the expected result is nothing — is
the only place in either study where the `dt`-awareness contrast reaches significance
(family F4, `CLAIM.CMP.F4_dt_awareness_control:ppo_cfc_vs_ppo_cfc_dtblind:stationary`). That
is a nuisance effect, and it caps how tightly the primary null can be read. (ii)
<!--v:studyb.learn.not_learned_model_count-->2<!--/v--> of the
<!--v:studyb.matrix.n_models-->7<!--/v--> architectures —
<!--v:studyb.learn.not_learned_models-->`ppo_ltc`, `ppo_ltc_cfc`<!--/v-->, the two containing an LTC block — do not
train at all even at this budget, on either scenario; every comparison involving them is a
comparison against a non-learner. (iii) The memoryless MLP beats both memory mechanisms
(family F6), so no memory mechanism in the registry buys anything on this task and two of
them cost.

The statistical protocol is identical in both studies: a two-sided paired **exact** sign-flip
permutation test (all `2^12` sign vectors enumerated), a 20,000-draw percentile bootstrap
interval, and Holm–Bonferroni within each declared family at α =
<!--v:prereg.alpha-->0.05<!--/v-->. Environment seeds are derived by `blake2b` from
`(base_seed, scenario, role, index)` and deliberately **exclude the model name**, so every
policy in a cell sees a byte-identical environment stream and every comparison is paired by
seed. Every model is exactly two blocks deep and its hidden width is solved, not chosen, so
the largest and smallest trainable parameter counts differ by under two percent
(`CLAIM.BUDGET.spread`).

The contribution is **methodological, not architectural**: a benchmark that reports its own
budget dependence, with both budgets' artifacts committed, and whose central architectural
thesis is unsupported at both. See [Limitations](#limitations-and-threats-to-validity) for
what Study B settled, what it did not, and what remains confounded.

---

## The two studies side by side

**Every table and every number in this README is written by `scripts/render_docs.py` from
`results/claims.json`, `results/tables/*.csv`, `results/study_b/claims.json` and
`results/study_b/tables/*.csv`.** The blocks below sit between `<!-- BEGIN GENERATED -->`
markers and are rewritten in place by `make report`; the inline values sit in `<!--v:key-->`
spans. `make check` fails the build if any of them differs from what the committed results
produce, so a generated number cannot drift from the evidence, and a hand-edit to one fails
the build. The scan also covers plain prose, headings, table cells, inline code, figure
captions, link titles and HTML comment bodies — an audit attacked all of those and each went
red. Fenced code blocks are exempt from the metric scan because they hold transcripts, so
they are checked separately for result vocabulary; that hole was found by the same audit and
is the reason this paragraph no longer claims a number "cannot be edited by hand" without
qualification. [`RESULTS.md`](RESULTS.md) is generated the same way by `scripts/make_report.py` and
carries Study A's full tables; [`docs/STUDY_B.md`](docs/STUDY_B.md) carries Study B's.

<!-- BEGIN GENERATED: budget_comparison_table -->
| quantity | Study A (low budget) | Study B (16x budget) | source |
|---|---|---|---|
| environment steps per trainable cell | 10240 | 163840 | `all_runs.csv :: train_steps` |
| gradient steps per trainable cell | 240 | 3840 | `all_runs.csv :: total_gradient_steps` |
| primary comparison P1, mean difference | -0.007 | 0.474 | `CLAIM.PRIMARY.P1` |
| primary comparison P1, 95% CI | [-3.869, 3.980] | [-5.184, 5.898] | `CLAIM.PRIMARY.P1` |
| primary comparison P1, p | 0.998 | 0.859 | `CLAIM.PRIMARY.P1` |
| primary comparison P1, verdict | `no_detectable_difference` | `no_detectable_difference` | `CLAIM.PRIMARY.P1` |
| cells that beat their own initialisation | 0 of 35 | 10 of 14 | `CLAIM.LEARN.count` |
| mean critic explained variance, after training | -4.18e-06 | 1.05e-01 | `all_runs.csv :: final_explained_variance` |
| mean policy entropy after training, nats | 1.971 | 0.762 | `all_runs.csv :: mean_policy_entropy_nats` |
| best learned policy, pooled | `ppo_transformer` -63.830 | `ppo_cfc` -26.106 | the named model's `CLAIM.RANK` claim, per study |
| `greedy_heuristic`, pooled | 8.291 | 20.429 | `CLAIM.RANK.greedy_heuristic` |
| `greedy_heuristic` lead over the best learned policy | 72.122 | 46.535 | `CLAIM.RANK.greedy_heuristic` minus the row above |
<!-- END GENERATED: budget_comparison_table -->

Read that table as one sentence: **sixteen times the compute changed every diagnostic, most
of the architectural verdicts, and neither of the two conclusions this repository is named
after.**

---

## Headline numbers

### Pre-registered primary comparison P1

`ppo_cfc` vs `ppo_cfc_dtblind` on `irregular_dt`, seed-matched pairs. Study A, at
<!--v:matrix.training_steps-->10240<!--/v--> steps per cell:

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

Study B, at <!--v:studyb.matrix.training_steps-->163840<!--/v--> steps per cell:

<!-- BEGIN GENERATED: studyb_p1_table -->
| quantity | value | claim_id |
|---|---|---|
| mean paired difference | 0.474 | `CLAIM.PRIMARY.P1` |
| 95% bootstrap CI | [-5.184, 5.898] | `CLAIM.PRIMARY.P1` |
| p (exact sign flip) | 0.859 | `CLAIM.PRIMARY.P1` |
| Holm-adjusted p | 0.859 | `CLAIM.PRIMARY.P1` |
| n paired seeds | 12 | `CLAIM.PRIMARY.P1` |
| Cliff's delta | 0.028 | `CLAIM.PRIMARY.P1` |
| verdict | `no_detectable_difference` | `CLAIM.PRIMARY.P1` |
<!-- END GENERATED: studyb_p1_table -->

Both intervals contain zero. **At neither budget does this repository find a detectable
benefit from letting the elapsed decision interval enter the recurrent state update under
irregular decision intervals.** The two nulls are not equally informative, and the difference
is the whole reason Study B exists: in Study A neither arm had learned anything, so the
comparison was between two random initialisations; in Study B both arms beat their own
initialisation first (`ppo_cfc` by <!--v:studyb.claim.CLAIM.CMP.F2_learning_check:ppo_cfc:trained_vs_untrained:irregular_dt.value-->19.430<!--/v-->,
`ppo_cfc_dtblind` by <!--v:studyb.claim.CLAIM.CMP.F2_learning_check:ppo_cfc_dtblind:trained_vs_untrained:irregular_dt.value-->24.416<!--/v-->,
both at Holm p = <!--v:studyb.claim.CLAIM.CMP.F2_learning_check:ppo_cfc:trained_vs_untrained:irregular_dt.holm_adjusted_p|.4f-->0.0034<!--/v-->),
so Study B's null compares two policies that had something to compare.

### Overall ranking

Pooled over each study's confirmatory scenarios, every policy that ran, no filter. Study A,
over <!--v:rank.scenarios-->5<!--/v--> scenarios:

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

Study B, over <!--v:studyb.rank.scenarios-->2<!--/v--> scenarios — a different and smaller
scenario set, so the two tables are not directly comparable row for row:

<!-- BEGIN GENERATED: studyb_ranking_table -->
| rank | policy | mean eval return | 95% CI over scenarios | claim_id |
|---|---|---|---|---|
| 1 | `greedy_heuristic` (0 parameters) | 20.429 | [19.781, 21.077] | `CLAIM.RANK.greedy_heuristic` |
| 2 | `ppo_cfc` | -26.106 | [-27.234, -24.977] | `CLAIM.RANK.ppo_cfc` |
| 3 | `ppo_mlp` | -26.471 | [-26.667, -26.275] | `CLAIM.RANK.ppo_mlp` |
| 4 | `ppo_cfc_dtblind` | -29.022 | [-30.336, -27.708] | `CLAIM.RANK.ppo_cfc_dtblind` |
| 5 | `ppo_transformer` | -33.269 | [-34.924, -31.614] | `CLAIM.RANK.ppo_transformer` |
| 6 | `ppo_gru` | -34.248 | [-34.783, -33.712] | `CLAIM.RANK.ppo_gru` |
| 7 | `ppo_ltc` | -50.852 | [-52.047, -49.658] | `CLAIM.RANK.ppo_ltc` |
| 8 | `random_policy` (0 parameters) | -50.859 | [-50.902, -50.816] | `CLAIM.RANK.random_policy` |
| 9 | `ppo_ltc_cfc` | -50.885 | [-51.451, -50.319] | `CLAIM.RANK.ppo_ltc_cfc` |
| 10 | `constant_channel` (0 parameters) | -51.455 | [-51.461, -51.450] | `CLAIM.RANK.constant_channel` |
<!-- END GENERATED: studyb_ranking_table -->

In Study A the seven learned policies occupied ranks 2 through 8 within one reward point of
each other and of `random_policy`, and none of the gaps was detectable. In Study B five of
them have separated upward by tens of points, `ppo_cfc` leads them, and the two LTC-containing
models sit among the zero-parameter baselines. `greedy_heuristic` is first in both.

**No test was run on the pooled ranking in either study**; it is a descriptive ordering, its
intervals are over <!--v:studyb.rank.scenarios-->2<!--/v--> scenario means in Study B, and
the tested contrasts are in the families below.

---

## Findings

Each finding names the study and the budget that produced it. Where the two studies disagree,
both answers are given.

### 1. The primary question returns a null at both budgets — and its control fired the wrong way

`ppo_cfc` and `ppo_cfc_dtblind` share an architecture, a parameter count, an initialisation
*distribution*, an input vector and an environment stream. They differ in exactly one factor:
whether `dt` enters the CfC time gate or is pinned to a constant there. Both still receive
`dt` as an encoder input feature. This is the sharpest available test of the repository's
thesis, and it was declared before each run.

Study A: null (`CLAIM.PRIMARY.P1`, above), with neither arm having learned anything.
Study B: null again, with both arms having learned. Study B also ran the contrast on
`stationary`, a constant-`dt` scenario where the two arms are informationally identical and
the pre-registered expectation was **no difference** — the control that tells you whether the
contrast measures `dt` at all:

<!-- BEGIN GENERATED: studyb_dt_table -->
| contrast | scenario | mean difference | 95% CI | Holm p | Cliff's delta | verdict | claim_id |
|---|---|---|---|---|---|---|---|
| `ppo_cfc` vs `ppo_cfc_dtblind` | `irregular_dt` | 0.474 | [-5.184, 5.898] | 0.8589 | 0.028 | `no_detectable_difference` | `CLAIM.CMP.P1` |
| `ppo_cfc` vs `ppo_cfc_dtblind` | `stationary` | 5.359 | [1.822, 8.794] | 0.0166 | 0.472 | `favours_a` | `CLAIM.CMP.F4_dt_awareness_control:ppo_cfc_vs_ppo_cfc_dtblind:stationary` |
<!-- END GENERATED: studyb_dt_table -->

**The control is the significant one.** On `irregular_dt`, where `dt` is drawn loguniformly
and actually varies, the difference is
<!--v:studyb.claim.CLAIM.PRIMARY.P1.value-->0.474<!--/v--> with an interval spanning zero. On
`stationary`, where `dt` cannot carry information, `ppo_cfc` beats `ppo_cfc_dtblind` by
<!--v:studyb.claim.CLAIM.CMP.F4_dt_awareness_control:ppo_cfc_vs_ppo_cfc_dtblind:stationary.value-->5.359<!--/v-->
with interval
[<!--v:studyb.claim.CLAIM.CMP.F4_dt_awareness_control:ppo_cfc_vs_ppo_cfc_dtblind:stationary.ci_low-->1.822<!--/v-->,
<!--v:studyb.claim.CLAIM.CMP.F4_dt_awareness_control:ppo_cfc_vs_ppo_cfc_dtblind:stationary.ci_high-->8.794<!--/v-->]
and p = <!--v:studyb.claim.CLAIM.CMP.F4_dt_awareness_control:ppo_cfc_vs_ppo_cfc_dtblind:stationary.p_value-->0.017<!--/v-->.

The honest reading is that this is a **nuisance effect**, not a `dt` effect. The two arms are
separately parameterised models with independently drawn initialisations: their shapes and
parameter counts match, but their weight distributions are not identical draws, and the
`cfc_dtblind` cell is a distinct module rather than a runtime switch inside `ppo_cfc`. A
difference of this size can therefore appear where `dt` is constant. Two consequences, both
uncomfortable and both stated rather than buried:

1. No `dt`-awareness contrast in this repository — **including the primary comparison** — can
   currently be read as a clean measurement of `dt`-awareness.
2. The primary null must not be read as a tight null. A nuisance effect the size of the F4
   estimate sits comfortably inside the primary comparison's own interval.

This is a defect in the experimental design, not in the analysis, and more compute does not
fix it. The fix is a within-model switch that toggles `dt` on one set of weights; that is
future work and is named as such in [Limitations](#limitations-and-threats-to-validity).

### 2. Learning: nothing at the low budget, most things at <!--v:studyb.budget.env_step_ratio-->16<!--/v-->×

Every run is evaluated twice on the identical evaluation stream: once with freshly
initialised weights and once after training. That paired difference is a **pre-registered
family in both studies** (Study A's F5; Study B's F2 on `irregular_dt` and F8 on
`stationary`), and it is the precondition every architectural comparison depends on.

- **Study A**, <!--v:matrix.training_steps-->10240<!--/v--> steps / <!--v:runs.gradient_steps-->240<!--/v-->
  gradient updates: <!--v:learn.learned-->0<!--/v--> of <!--v:learn.tested-->35<!--/v--> cells
  improved significantly over their own initialisation (`CLAIM.LEARN.count`).
- **Study B**, <!--v:studyb.matrix.training_steps-->163840<!--/v--> steps /
  <!--v:studyb.runs.gradient_steps-->3840<!--/v--> gradient updates:
  <!--v:studyb.learn.learned-->10<!--/v--> of <!--v:studyb.learn.tested-->14<!--/v--> cells did
  (`CLAIM.LEARN.count` in `results/study_b/claims.json`),
  <!--v:studyb.learn.learned.irregular_dt-->5<!--/v--> of
  <!--v:studyb.learn.tested.irregular_dt-->7<!--/v--> on `irregular_dt` and
  <!--v:studyb.learn.learned.stationary-->5<!--/v--> of
  <!--v:studyb.learn.tested.stationary-->7<!--/v--> on `stationary`.

<!-- BEGIN GENERATED: studyb_learning_table -->
| scenario | model | trained − untrained | 95% CI | Holm p | beat its own initialisation? | claim_id |
|---|---|---|---|---|---|---|
| `irregular_dt` | `ppo_cfc` | 19.430 | [14.923, 24.057] | 0.0034 | yes | `CLAIM.CMP.F2_learning_check:ppo_cfc:trained_vs_untrained:irregular_dt` |
| `irregular_dt` | `ppo_cfc_dtblind` | 24.416 | [20.892, 27.675] | 0.0034 | yes | `CLAIM.CMP.F2_learning_check:ppo_cfc_dtblind:trained_vs_untrained:irregular_dt` |
| `irregular_dt` | `ppo_gru` | 17.078 | [10.444, 22.699] | 0.0044 | yes | `CLAIM.CMP.F2_learning_check:ppo_gru:trained_vs_untrained:irregular_dt` |
| `irregular_dt` | `ppo_ltc` | 1.549 | [-2.226, 5.234] | 0.9141 | no | `CLAIM.CMP.F2_learning_check:ppo_ltc:trained_vs_untrained:irregular_dt` |
| `irregular_dt` | `ppo_ltc_cfc` | 0.600 | [-4.506, 5.675] | 0.9141 | no | `CLAIM.CMP.F2_learning_check:ppo_ltc_cfc:trained_vs_untrained:irregular_dt` |
| `irregular_dt` | `ppo_mlp` | 24.134 | [20.509, 27.714] | 0.0034 | yes | `CLAIM.CMP.F2_learning_check:ppo_mlp:trained_vs_untrained:irregular_dt` |
| `irregular_dt` | `ppo_transformer` | 18.824 | [12.273, 24.249] | 0.0039 | yes | `CLAIM.CMP.F2_learning_check:ppo_transformer:trained_vs_untrained:irregular_dt` |
| `stationary` | `ppo_cfc` | 24.517 | [20.146, 29.832] | 0.0034 | yes | `CLAIM.CMP.F8_replication_learning_check:ppo_cfc:trained_vs_untrained:stationary` |
| `stationary` | `ppo_cfc_dtblind` | 20.098 | [16.143, 23.842] | 0.0034 | yes | `CLAIM.CMP.F8_replication_learning_check:ppo_cfc_dtblind:trained_vs_untrained:stationary` |
| `stationary` | `ppo_gru` | 15.396 | [10.703, 20.138] | 0.0034 | yes | `CLAIM.CMP.F8_replication_learning_check:ppo_gru:trained_vs_untrained:stationary` |
| `stationary` | `ppo_ltc` | -1.470 | [-3.840, 0.715] | 0.4180 | no | `CLAIM.CMP.F8_replication_learning_check:ppo_ltc:trained_vs_untrained:stationary` |
| `stationary` | `ppo_ltc_cfc` | -1.644 | [-3.897, 0.741] | 0.4180 | no | `CLAIM.CMP.F8_replication_learning_check:ppo_ltc_cfc:trained_vs_untrained:stationary` |
| `stationary` | `ppo_mlp` | 23.749 | [20.414, 27.054] | 0.0034 | yes | `CLAIM.CMP.F8_replication_learning_check:ppo_mlp:trained_vs_untrained:stationary` |
| `stationary` | `ppo_transformer` | 20.616 | [15.349, 25.473] | 0.0034 | yes | `CLAIM.CMP.F8_replication_learning_check:ppo_transformer:trained_vs_untrained:stationary` |
<!-- END GENERATED: studyb_learning_table -->

**Study A's null was substantially a compute artifact, and this is the evidence that says
so.** It is not a claim that Study A was analysed wrongly — the same analysis code, the same
seeds and the same tests produce both tables.

### 3. Two of seven architectures do not train at all, at either budget

<!--v:studyb.learn.not_learned_models-->`ppo_ltc`, `ppo_ltc_cfc`<!--/v--> — the two models containing an LTC block —
are the <!--v:studyb.learn.not_learned_model_count-->2<!--/v--> that fail the precondition in
Study B, on **both** scenarios, and on `stationary` their point estimates are negative. Their
mechanistic diagnostics agree with their returns and rule out "it learned but the reward
metric missed it":

<!-- BEGIN GENERATED: studyb_diagnostics_table -->
| model (mean over 12 seeds, `irregular_dt`) | approx KL | clip fraction | critic EV | rollout entropy (nats) |
|---|---|---|---|---|
| `ppo_cfc` | 0.0193 | 0.104 | 0.1312 | 0.724 |
| `ppo_cfc_dtblind` | 0.0556 | 0.145 | 0.1356 | 0.762 |
| `ppo_gru` | 0.0064 | 0.065 | 0.2244 | 1.017 |
| `ppo_ltc` | 0.0022 | 0.025 | 0.0000 | 0.546 |
| `ppo_ltc_cfc` | 0.0011 | 0.017 | 0.0000 | 0.569 |
| `ppo_mlp` | 0.0124 | 0.097 | 0.0818 | 0.597 |
| `ppo_transformer` | 0.0179 | 0.136 | 0.0720 | 0.969 |
<!-- END GENERATED: studyb_diagnostics_table -->

Their critic explained variance is zero to four decimal places, their approximate KL and clip
fractions are an order of magnitude below every other model's — the update is barely moving
the policy — and their evaluation-time action histogram is a point mass
(<!--v:studyb.main.irregular_dt.ppo_ltc.mean_action_histogram_entropy-->0.000<!--/v--> nats,
identical to `constant_channel`), so they collapse onto a single fixed channel and score like
it.

**Every architectural comparison involving either model is therefore a comparison against a
non-learner and must be read that way**, including all six `F5`/`F9` contrasts that appear to
show "liquid loses to conventional". Exploratory, and not pre-registered: this looks like an
optimisation pathology in the LTC cell under this PPO configuration rather than a property of
continuous-time models, because `ppo_cfc` is also liquid, also `dt`-aware, also
capacity-matched, and ranks <!--v:studyb.rank.ppo_cfc.rank-->2<!--/v--> of
<!--v:studyb.rank.policies-->10<!--/v-->. The LTC cell's unit tests pass and it is the verified
Hasani et al. (2021) formulation, so this is a learner/architecture interaction the existing
test suite cannot see. It deserves a dedicated investigation and has not had one.

### 4. A zero-parameter heuristic wins everything, at both budgets

Study A: <!--v:f6.n-->35<!--/v--> comparisons against `greedy_heuristic`,
<!--v:f6.significant-->35<!--/v--> significant, all favouring the heuristic. Study B:
<!--v:studyb.f7.n-->7<!--/v--> on `irregular_dt` and <!--v:studyb.f10.n-->7<!--/v--> on
`stationary`, all <!--v:studyb.f7.significant-->7<!--/v--> and
<!--v:studyb.f10.significant-->7<!--/v--> significant, all favouring the heuristic, Cliff's
delta <!--v:studyb.f7.cliffs_delta_max-->-1.000<!--/v--> throughout — the heuristic wins at every
one of the <!--v:studyb.matrix.n_seeds-->12<!--/v--> seeds in every comparison.

<!-- BEGIN GENERATED: studyb_greedy_gap_table -->
| contrast | scenario | mean difference | 95% CI | Holm p | Cliff's delta | verdict | claim_id |
|---|---|---|---|---|---|---|---|
| `ppo_cfc_dtblind` vs `greedy_heuristic` | `irregular_dt` | -48.786 | [-51.661, -45.263] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F7_vs_greedy_baseline:ppo_cfc_dtblind_vs_greedy_heuristic:irregular_dt` |
| `ppo_cfc` vs `greedy_heuristic` | `irregular_dt` | -48.311 | [-51.279, -45.254] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F7_vs_greedy_baseline:ppo_cfc_vs_greedy_heuristic:irregular_dt` |
| `ppo_gru` vs `greedy_heuristic` | `irregular_dt` | -55.861 | [-60.419, -51.709] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F7_vs_greedy_baseline:ppo_gru_vs_greedy_heuristic:irregular_dt` |
| `ppo_ltc_cfc` vs `greedy_heuristic` | `irregular_dt` | -71.396 | [-75.138, -68.008] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F7_vs_greedy_baseline:ppo_ltc_cfc_vs_greedy_heuristic:irregular_dt` |
| `ppo_ltc` vs `greedy_heuristic` | `irregular_dt` | -70.736 | [-73.507, -67.580] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F7_vs_greedy_baseline:ppo_ltc_vs_greedy_heuristic:irregular_dt` |
| `ppo_mlp` vs `greedy_heuristic` | `irregular_dt` | -47.745 | [-51.658, -43.898] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F7_vs_greedy_baseline:ppo_mlp_vs_greedy_heuristic:irregular_dt` |
| `ppo_transformer` vs `greedy_heuristic` | `irregular_dt` | -56.001 | [-60.865, -52.366] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F7_vs_greedy_baseline:ppo_transformer_vs_greedy_heuristic:irregular_dt` |
| `ppo_cfc_dtblind` vs `greedy_heuristic` | `stationary` | -50.118 | [-54.020, -46.437] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F10_replication_vs_greedy:ppo_cfc_dtblind_vs_greedy_heuristic:stationary` |
| `ppo_cfc` vs `greedy_heuristic` | `stationary` | -44.758 | [-47.583, -41.517] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F10_replication_vs_greedy:ppo_cfc_vs_greedy_heuristic:stationary` |
| `ppo_gru` vs `greedy_heuristic` | `stationary` | -53.494 | [-59.053, -48.823] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F10_replication_vs_greedy:ppo_gru_vs_greedy_heuristic:stationary` |
| `ppo_ltc_cfc` vs `greedy_heuristic` | `stationary` | -71.232 | [-73.299, -69.155] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F10_replication_vs_greedy:ppo_ltc_cfc_vs_greedy_heuristic:stationary` |
| `ppo_ltc` vs `greedy_heuristic` | `stationary` | -71.828 | [-74.267, -69.415] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F10_replication_vs_greedy:ppo_ltc_vs_greedy_heuristic:stationary` |
| `ppo_mlp` vs `greedy_heuristic` | `stationary` | -46.056 | [-49.977, -41.988] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F10_replication_vs_greedy:ppo_mlp_vs_greedy_heuristic:stationary` |
| `ppo_transformer` vs `greedy_heuristic` | `stationary` | -51.395 | [-56.380, -46.905] | 0.0034 | -1.000 | `favours_b` | `CLAIM.CMP.F10_replication_vs_greedy:ppo_transformer_vs_greedy_heuristic:stationary` |
<!-- END GENERATED: studyb_greedy_gap_table -->

Sixteen times the compute closed a large part of the gap and none of it. What the extra
budget bought is visible in the components the return is made of:

<!-- BEGIN GENERATED: studyb_outcome_profile_table -->
| policy (`irregular_dt`) | return | success rate | collision rate | utilisation | action-histogram entropy |
|---|---|---|---|---|---|
| `greedy_heuristic` | 21.077 | 0.728 | 0.272 | 0.379 | 1.785 |
| `ppo_mlp` | -26.667 | 0.500 | 0.500 | 0.212 | 0.712 |
| `ppo_cfc` | -27.234 | 0.498 | 0.502 | 0.210 | 0.761 |
| `ppo_cfc_dtblind` | -27.708 | 0.495 | 0.505 | 0.209 | 0.756 |
| `ppo_gru` | -34.783 | 0.449 | 0.551 | 0.189 | 0.606 |
| `ppo_transformer` | -34.924 | 0.449 | 0.551 | 0.189 | 0.636 |
| `ppo_ltc` | -49.658 | 0.359 | 0.641 | 0.152 | 0.000 |
| `ppo_ltc_cfc` | -50.319 | 0.355 | 0.645 | 0.151 | 0.000 |
| `random_policy` | -50.816 | 0.355 | 0.645 | 0.149 | 2.022 |
| `constant_channel` | -51.450 | 0.351 | 0.649 | 0.148 | 0.000 |
<!-- END GENERATED: studyb_outcome_profile_table -->

The best learned policy on `irregular_dt` (`ppo_mlp`) transmits successfully on about
<!--v:studyb.main.irregular_dt.ppo_mlp.mean_success_rate-->0.500<!--/v--> of its attempts against
the heuristic's <!--v:studyb.main.irregular_dt.greedy_heuristic.mean_success_rate-->0.728<!--/v-->,
and achieves <!--v:studyb.main.irregular_dt.ppo_mlp.mean_spectrum_utilization-->0.212<!--/v-->
spectrum utilisation against <!--v:studyb.main.irregular_dt.greedy_heuristic.mean_spectrum_utilization-->0.379<!--/v-->.
The baseline is not a strawman: `GreedyOccupancyPolicy` carries a `dt`-decayed exponential
memory of sensed primary occupancy plus a contention estimate inferred from its own collision
history, and it reads nothing a neural policy cannot read (`dsa/envs/heuristics.py`, and
`docs/PROTOCOL.md` §2.0 for why the task was deliberately kept partially observed). The
strongest policy in this benchmark is *already* stateful and `dt`-aware — those inductive
biases are hand-coded into it. What neither study showed is that PPO can **learn** them here.
[Limitations](#limitations-and-threats-to-validity) §7 argues concretely about what that says
about the environment.

### 5. Architecture: CfC ties a memoryless MLP, memory costs, and the hybrid loses to its own half

All Study B contrasts between architectures, Holm-corrected within their declared families:

<!-- BEGIN GENERATED: studyb_architecture_table -->
| contrast | scenario | mean difference | 95% CI | Holm p | Cliff's delta | verdict | claim_id |
|---|---|---|---|---|---|---|---|
| `ppo_ltc_cfc` vs `ppo_cfc` | `irregular_dt` | -23.085 | [-28.825, -17.763] | 0.0010 | -1.000 | `favours_b` | `CLAIM.CMP.F3_hybrid_decomposition:ppo_ltc_cfc_vs_ppo_cfc:irregular_dt` |
| `ppo_ltc_cfc` vs `ppo_ltc` | `irregular_dt` | -0.661 | [-5.188, 3.629] | 0.7832 | 0.049 | `no_detectable_difference` | `CLAIM.CMP.F3_hybrid_decomposition:ppo_ltc_cfc_vs_ppo_ltc:irregular_dt` |
| `ppo_cfc` vs `ppo_gru` | `irregular_dt` | 7.550 | [1.608, 14.141] | 0.0752 | 0.583 | `no_detectable_difference` | `CLAIM.CMP.F5_liquid_vs_conventional:ppo_cfc_vs_ppo_gru:irregular_dt` |
| `ppo_cfc` vs `ppo_mlp` | `irregular_dt` | -0.567 | [-5.454, 4.390] | 0.8325 | -0.069 | `no_detectable_difference` | `CLAIM.CMP.F5_liquid_vs_conventional:ppo_cfc_vs_ppo_mlp:irregular_dt` |
| `ppo_cfc` vs `ppo_transformer` | `irregular_dt` | 7.690 | [2.398, 13.584] | 0.0527 | 0.569 | `no_detectable_difference` | `CLAIM.CMP.F5_liquid_vs_conventional:ppo_cfc_vs_ppo_transformer:irregular_dt` |
| `ppo_ltc_cfc` vs `ppo_gru` | `irregular_dt` | -15.536 | [-20.776, -10.205] | 0.0068 | -0.833 | `favours_b` | `CLAIM.CMP.F5_liquid_vs_conventional:ppo_ltc_cfc_vs_ppo_gru:irregular_dt` |
| `ppo_ltc_cfc` vs `ppo_mlp` | `irregular_dt` | -23.652 | [-28.214, -18.982] | 0.0044 | -1.000 | `favours_b` | `CLAIM.CMP.F5_liquid_vs_conventional:ppo_ltc_cfc_vs_ppo_mlp:irregular_dt` |
| `ppo_ltc_cfc` vs `ppo_transformer` | `irregular_dt` | -15.395 | [-21.039, -8.939] | 0.0078 | -0.875 | `favours_b` | `CLAIM.CMP.F5_liquid_vs_conventional:ppo_ltc_cfc_vs_ppo_transformer:irregular_dt` |
| `ppo_ltc` vs `ppo_gru` | `irregular_dt` | -14.875 | [-20.201, -9.502] | 0.0068 | -0.875 | `favours_b` | `CLAIM.CMP.F5_liquid_vs_conventional:ppo_ltc_vs_ppo_gru:irregular_dt` |
| `ppo_ltc` vs `ppo_mlp` | `irregular_dt` | -22.991 | [-26.110, -20.033] | 0.0044 | -1.000 | `favours_b` | `CLAIM.CMP.F5_liquid_vs_conventional:ppo_ltc_vs_ppo_mlp:irregular_dt` |
| `ppo_ltc` vs `ppo_transformer` | `irregular_dt` | -14.734 | [-19.488, -10.071] | 0.0068 | -0.833 | `favours_b` | `CLAIM.CMP.F5_liquid_vs_conventional:ppo_ltc_vs_ppo_transformer:irregular_dt` |
| `ppo_gru` vs `ppo_mlp` | `irregular_dt` | -8.116 | [-12.833, -2.987] | 0.0273 | -0.597 | `favours_b` | `CLAIM.CMP.F6_within_conventional:ppo_gru_vs_ppo_mlp:irregular_dt` |
| `ppo_transformer` vs `ppo_mlp` | `irregular_dt` | -8.257 | [-14.718, -1.651] | 0.0361 | -0.625 | `favours_b` | `CLAIM.CMP.F6_within_conventional:ppo_transformer_vs_ppo_mlp:irregular_dt` |
| `ppo_cfc` vs `ppo_gru` | `stationary` | 8.735 | [3.981, 14.299] | 0.0059 | 0.667 | `favours_a` | `CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_cfc_vs_ppo_gru:stationary` |
| `ppo_cfc` vs `ppo_mlp` | `stationary` | 1.298 | [-2.715, 5.191] | 0.5405 | 0.111 | `no_detectable_difference` | `CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_cfc_vs_ppo_mlp:stationary` |
| `ppo_cfc` vs `ppo_transformer` | `stationary` | 6.637 | [2.082, 11.323] | 0.0469 | 0.444 | `favours_a` | `CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_cfc_vs_ppo_transformer:stationary` |
| `ppo_ltc_cfc` vs `ppo_gru` | `stationary` | -17.738 | [-23.234, -10.965] | 0.0059 | -0.833 | `favours_b` | `CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_ltc_cfc_vs_ppo_gru:stationary` |
| `ppo_ltc_cfc` vs `ppo_mlp` | `stationary` | -25.175 | [-29.114, -21.485] | 0.0044 | -1.000 | `favours_b` | `CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_ltc_cfc_vs_ppo_mlp:stationary` |
| `ppo_ltc_cfc` vs `ppo_transformer` | `stationary` | -19.837 | [-25.152, -14.017] | 0.0044 | -1.000 | `favours_b` | `CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_ltc_cfc_vs_ppo_transformer:stationary` |
| `ppo_ltc` vs `ppo_gru` | `stationary` | -18.334 | [-22.669, -13.291] | 0.0049 | -0.840 | `favours_b` | `CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_ltc_vs_ppo_gru:stationary` |
| `ppo_ltc` vs `ppo_mlp` | `stationary` | -25.771 | [-30.070, -21.620] | 0.0044 | -1.000 | `favours_b` | `CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_ltc_vs_ppo_mlp:stationary` |
| `ppo_ltc` vs `ppo_transformer` | `stationary` | -20.433 | [-24.993, -15.411] | 0.0044 | -1.000 | `favours_b` | `CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_ltc_vs_ppo_transformer:stationary` |
<!-- END GENERATED: studyb_architecture_table -->

Reading only the rows whose verdict is directional, and only those whose two arms both
cleared the learning check:

- **Liquid vs conventional is not a coherent question here.** `ppo_cfc` is
  rank <!--v:studyb.rank.ppo_cfc.rank-->2<!--/v-->; `ppo_ltc` is rank
  <!--v:studyb.rank.ppo_ltc.rank-->7<!--/v-->. Both are "liquid". The class does not predict
  the outcome; the specific cell does.
- **CfC vs the memoryless MLP: no detectable difference on either scenario.** The best liquid
  model cannot separate itself from a feedforward policy of the same depth and parameter
  budget.
- **CfC vs GRU and CfC vs attention:** directional on `stationary`
  (<!--v:studyb.claim.CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_cfc_vs_ppo_gru:stationary.value-->8.735<!--/v-->
  and <!--v:studyb.claim.CLAIM.CMP.F9_replication_liquid_vs_conventional:ppo_cfc_vs_ppo_transformer:stationary.value-->6.637<!--/v-->,
  favouring CfC), **no detectable difference** on `irregular_dt`, where the raw intervals point
  the same way but the pre-registered Holm correction declines to reject
  (<!--v:studyb.f5.excludes_zero_not_significant-->2<!--/v--> comparisons in F5 have an
  interval excluding zero and an adjusted p that does not reject; all of them are published as
  `no_detectable_difference`).
- **Memory is worse than no memory** (family F6, `irregular_dt`): `ppo_gru` loses to
  `ppo_mlp` by <!--v:studyb.claim.CLAIM.CMP.F6_within_conventional:ppo_gru_vs_ppo_mlp:irregular_dt.value-->-8.116<!--/v-->
  and `ppo_transformer` by
  <!--v:studyb.claim.CLAIM.CMP.F6_within_conventional:ppo_transformer_vs_ppo_mlp:irregular_dt.value-->-8.257<!--/v-->,
  both significant after Holm. This is the family that makes the rest interpretable: recurrence
  and attention are not merely unnecessary on this task, they cost. Combined with CfC ≈ MLP,
  **no memory mechanism in the registry buys anything here and two of them are actively
  worse.**
- **The hybrid the repository is named after loses to one of its own components** (family F3):
  `ppo_ltc_cfc` vs `ppo_cfc` is
  <!--v:studyb.claim.CLAIM.CMP.F3_hybrid_decomposition:ppo_ltc_cfc_vs_ppo_cfc:irregular_dt.value-->-23.085<!--/v-->
  favouring `ppo_cfc`, while `ppo_ltc_cfc` vs `ppo_ltc` shows no detectable difference.
  Adding an LTC block to a CfC costs about as much as the CfC had gained, and the hybrid
  behaves like its LTC half. Caveat from finding 3: the hybrid and `ppo_ltc` are both
  non-learners, so this pair of rows is "a non-learner loses to a learner, and two
  non-learners tie".

In Study A none of these contrasts was detectable, because none of the arms had learned. The
change in verdict between the studies is a change in what the arms *were*, not in how they
were tested.

### 6. Mechanism: entropy fell everywhere, explained variance only where learning happened

Three mechanistic endpoints were declared in Study B's pre-registration as **descriptive**:
they carry seed-bootstrapped intervals and **no p-value and no correction**, and they are not
members of any family. They were declared that way so that they could not later be presented
as tested, and they are reported here even though they disagree with the reward result for two
models — which is the case they exist for. Source:
`results/study_b/mechanistic_endpoints.csv`.

Policy entropy at evaluation time, before and after training, on `irregular_dt`. Uniform over
<!--v:env.num_channels-->8<!--/v--> channels is
<!--v:env.uniform_policy_entropy_nats-->2.079<!--/v--> nats:

<!-- BEGIN GENERATED: studyb_entropy_table -->
| model | before | after | change | 95% CI | seeds moving as expected |
|---|---|---|---|---|---|
| `ppo_ltc` | 2.043 | 0.536 | -1.507 | [-1.693, -1.297] | 12 of 12 |
| `ppo_ltc_cfc` | 2.051 | 0.575 | -1.476 | [-1.672, -1.276] | 12 of 12 |
| `ppo_mlp` | 2.050 | 0.591 | -1.459 | [-1.537, -1.370] | 12 of 12 |
| `ppo_cfc` | 2.048 | 0.718 | -1.329 | [-1.412, -1.250] | 12 of 12 |
| `ppo_cfc_dtblind` | 2.047 | 0.778 | -1.269 | [-1.355, -1.186] | 12 of 12 |
| `ppo_transformer` | 2.040 | 0.975 | -1.064 | [-1.189, -0.938] | 12 of 12 |
| `ppo_gru` | 2.046 | 1.078 | -0.967 | [-1.084, -0.843] | 12 of 12 |
<!-- END GENERATED: studyb_entropy_table -->

Critic explained variance, first update to last update, on `irregular_dt`:

<!-- BEGIN GENERATED: studyb_explained_variance_table -->
| model | before | after | change | 95% CI | seeds moving as expected |
|---|---|---|---|---|---|
| `ppo_ltc` | -0.0000 | 0.0000 | 0.0000 | [-0.0002, 0.0002] | 7 of 12 |
| `ppo_ltc_cfc` | -0.0001 | 0.0000 | 0.0001 | [-0.0002, 0.0003] | 7 of 12 |
| `ppo_transformer` | -0.0021 | 0.0720 | 0.0741 | [0.0393, 0.1147] | 11 of 12 |
| `ppo_mlp` | -0.0012 | 0.0818 | 0.0830 | [0.0545, 0.1133] | 12 of 12 |
| `ppo_cfc` | -0.0008 | 0.1312 | 0.1320 | [0.0847, 0.1915] | 12 of 12 |
| `ppo_cfc_dtblind` | -0.0002 | 0.1356 | 0.1358 | [0.0956, 0.1807] | 12 of 12 |
| `ppo_gru` | -0.0035 | 0.2244 | 0.2280 | [0.1514, 0.3179] | 12 of 12 |
<!-- END GENERATED: studyb_explained_variance_table -->

**The two endpoints disagree, and the disagreement is the finding.** `ppo_ltc` and
`ppo_ltc_cfc` show the *largest* entropy collapse of any model
(<!--v:studyb.mech.irregular_dt.ppo_ltc.eval_policy_entropy_nats.change-->-1.507<!--/v--> and
<!--v:studyb.mech.irregular_dt.ppo_ltc_cfc.eval_policy_entropy_nats.change-->-1.476<!--/v--> nats,
every seed moving down) while their critics explain zero variance and their returns do not
move. **Falling entropy is not evidence of learning.** Explained variance is the endpoint that
tracks the reward result here: every model whose critic moved improved, and both models whose
critic did not move did not. Study A reported flat entropy and flat explained variance
together, so the two could not be separated; Study B separates them and shows entropy is the
weaker signal.

Mechanistically the two are consistent with finding 3: for the LTC-containing models the
recurrent body is barely updating while the policy head saturates onto one action.

### 7. The federated hierarchy does what it is supposed to do, in measured bytes (Study A only)

Study B did not re-run the federated sub-study — it is orthogonal to the budget question and
was descoped in the pre-registration. **The evidence below is Study A's, at Study A's budget,
and inherits Study A's caveat that its policies had not learned.**

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
<!--v:fed.cloud_ratio_flat_over_hier-->7.43<!--/v-->× less wide-area traffic than flat federation
while moving *more* bytes in total — exactly the predicted trade-off, and the direction is a
property of the topology rather than a tuned outcome.

The returns, however, are indistinguishable. All <!--v:f7.n-->3<!--/v--> F7 comparisons are
non-significant after correction (`CLAIM.CMP.F7:hier_vs_flat`,
`CLAIM.CMP.F7:hier_vs_centralized`, `CLAIM.CMP.F7:flat_vs_centralized`); the largest is
<!--v:claim.CLAIM.CMP.F7:hier_vs_centralized.value-->-2.744<!--/v--> between hierarchical and
centralized with Holm-adjusted p =
<!--v:claim.CLAIM.CMP.F7:hier_vs_centralized.holm_adjusted_p-->0.245<!--/v-->.
<!--v:f7.excludes_zero_not_significant-->2<!--/v--> of the <!--v:f7.n-->3<!--/v--> bootstrap
intervals marginally exclude zero while the pre-registered exact test does not reject, which
is exactly the situation the pre-registration exists to adjudicate: **the test governs, we do
not claim a difference, and the published verdict on all three rows is
`no_detectable_difference`.** The federated study is evidence about *topology and
communication cost*, not about architecture — a single model was used for all arms — and a
communication result does not depend on the policies having learned, while the returns
comparison does.

### 8. The seeding defect that invalidated the pre-rebuild results is fixed and gated

The four cells `irregular_dt × {ppo_cfc, ppo_gru} × {seed 7, 19}` were run three ways: in a
2-model invocation, in a differently-ordered 4-model invocation, and inside the full
<!--v:matrix.num_cells-->600<!--/v-->-cell Study A run. Every non-`wall_` column is
byte-identical across all three, maximum absolute difference zero. The equivalent nuisance
spread in the pre-rebuild implementation was roughly forty reward points, against a headline
effect of about two. CI re-runs an order-invariance check on every push.

The learner is also verifiably correct on the check that catches stale hidden states, dropout
mismatch and train/eval-mode bugs simultaneously: the maximum first-epoch PPO ratio deviation
is <!--v:runs.ratio_deviation_max-->7.15e-07<!--/v--> over Study A's
<!--v:runs.neural_rows-->420<!--/v--> neural runs and
<!--v:studyb.runs.ratio_deviation_max-->4.29e-06<!--/v--> over Study B's
<!--v:studyb.runs.neural_rows-->168<!--/v--> — three orders of magnitude inside the unit test's
`1e-4` bound, and inside the `1e-3` gate that `scripts/analyze.py` applies to a whole suite
before it will emit a report. **Neither null is a stale-hidden-state, dropout or
train/eval-mode bug in the update.**

An incidental cross-check that costs nothing and is worth stating: the three zero-parameter
baselines are deterministic given a seed and a scenario, and their rows in Study B are
byte-identical to their rows in Study A on the two shared scenarios. The two studies
therefore demonstrably evaluated on the same environment streams, and the change in the
learned policies' returns cannot be an artifact of a changed evaluation.

---

## Negative and null results

Mandatory section, listing every comparison family that came out null in each study. The
"what it tests" column is the first sentence of each family's declaration in that study's
pre-registration; the counts and the outcome are computed from that study's comparisons
table.

**Study A** — `results/tables/comparisons.csv`, <!--v:cmp.total-->104<!--/v--> comparisons in
<!--v:cmp.families-->7<!--/v--> families, <!--v:cmp.significant-->35<!--/v--> significant after
Holm correction, all of them in F6:

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

**Study B** — `results/study_b/tables/comparisons.csv`, <!--v:studyb.cmp.total-->52<!--/v-->
comparisons in <!--v:studyb.cmp.families-->10<!--/v--> families,
<!--v:studyb.cmp.significant-->42<!--/v--> significant after Holm correction:

<!-- BEGIN GENERATED: studyb_family_outcome_table -->
| family | what it tests (from `configs/preregistration_study_b.yaml`) | comparisons | significant after Holm | outcome |
|---|---|---|---|---|
| `F1_primary` | The pre-registered primary comparison, reported unadjusted. | 1 | 0 | no detectable difference |
| `F2_learning_check` | THE PRECONDITION. | 7 | 5 | 5 of 7 reject |
| `F3_hybrid_decomposition` | Does the LTC+CfC hybrid the repository title names beat each of its two single-cell components? | 2 | 1 | all 1 favour `ppo_cfc` |
| `F4_dt_awareness_control` | The dt-awareness contrast on a CONSTANT-dt scenario. | 1 | 1 | all 1 favour `ppo_cfc` |
| `F5_liquid_vs_conventional` | Liquid / continuous-time cells against conventional cells at equal depth (2 blocks) and equal capacity (all within 5% of 40,000 parameters): {LTC, CfC, LTC+CfC} x {gated GRU, windowed self-attention, memoryless MLP}. | 9 | 6 | 6 of 9 reject |
| `F6_within_conventional` | Does either conventional memory mechanism beat having no memory at all? | 2 | 2 | all 2 favour `ppo_mlp` |
| `F7_vs_greedy_baseline` | Each learned policy against a FIXED, zero-parameter greedy occupancy heuristic chosen in advance -- deliberately not "the best baseline per scenario", which would be a selection bias. | 7 | 7 | all 7 favour `greedy_heuristic` |
| `F8_replication_learning_check` | The F2 precondition, replicated on the constant-dt scenario. | 7 | 5 | 5 of 7 reject |
| `F9_replication_liquid_vs_conventional` | F5, replicated on the constant-dt scenario. | 9 | 8 | 8 of 9 reject |
| `F10_replication_vs_greedy` | F7, replicated on the constant-dt scenario. | 7 | 7 | all 7 favour `greedy_heuristic` |
<!-- END GENERATED: studyb_family_outcome_table -->

Not supported by either study's evidence:

- that letting the elapsed decision interval enter the recurrent state update helps under
  irregular decision intervals — the repository's central thesis, tested at two budgets
  <!--v:studyb.budget.env_step_ratio-->16<!--/v-->× apart and unsupported at both;
- that liquid cells as a class outperform conventional cells — `ppo_cfc` and `ppo_ltc` sit at
  opposite ends of the same ranking;
- that the LTC + CfC hybrid is better than either of its parts;
- that either memory mechanism beats having no memory;
- that hierarchical federation changes achieved return.

Not supported by Study A, supported by Study B: that PPO learns anything at all on this task.
That is a budget-dependent conclusion, and it is the reason both studies are published.

Supported by both: that a hand-written zero-parameter heuristic beats every learned policy
here. Supported by Study A: that hierarchical federation reduces measured wide-area traffic
while increasing total traffic.

Descriptive, untested, and offered only as an ordering: in Study B's pooled ranking
`ppo_ltc` (`CLAIM.RANK.ppo_ltc`) and `ppo_ltc_cfc` (`CLAIM.RANK.ppo_ltc_cfc`) sit either side
of `random_policy` (`CLAIM.RANK.random_policy`) within a reward point. **No test was run on
that contrast and none should be read into it** — but it is consistent with finding 3, where
the pre-registered learning check does test those two models and does not reject.

---

## Limitations and threats to validity

### What Study B resolved

**1. The compute budget was the leading explanation for Study A's null, and it was the right
explanation — for five of seven architectures.** Study A's diagnostics (critic explained
variance indistinguishable from zero, policy entropy at uniform) said the learner never
reached the regime where its advantage estimates carry signal. At
<!--v:studyb.budget.env_step_ratio-->16<!--/v-->× the budget, entropy falls in every cell,
explained variance rises in every cell that learns, and
<!--v:studyb.learn.learned-->10<!--/v--> of <!--v:studyb.learn.tested-->14<!--/v--> cells beat
their own initialisation. Study A's architectural comparisons were comparisons between
initialisations. **They should not be cited as evidence about architectures, and this
document does not cite them that way.**

**2. The exploratory budget ladder that motivated Study B understated the problem and hid its
heterogeneity.** That ladder (`results/raw/budget_sensitivity_diagnostic.jsonl`, gitignored,
not pre-registered, no number from it appears anywhere in this documentation) used two seeds
and two models, and both of the models it happened to test are models that learn. Had it
included `ppo_ltc` it would have shown a <!--v:studyb.budget.env_step_ratio-->16<!--/v-->×
budget increase buying nothing. **Two seeds and two models is not enough to size a study.**

### What Study B did not resolve

**3. The primary comparison is still a null, and its control is now broken.** The
repository's central thesis — that letting the elapsed decision interval into the recurrent
state update helps under irregular decision intervals — has now been tested at
<!--v:matrix.training_steps-->10240<!--/v--> and at
<!--v:studyb.matrix.training_steps-->163840<!--/v--> steps per cell and is unsupported both times,
the second time with both arms demonstrably learning. Worse, the pre-registered control
(family F4: the same contrast on constant-`dt` `stationary`, where the two arms are
informationally identical) is the only place the contrast reaches significance. Until that
nuisance effect is explained, **no `dt`-awareness contrast in this repository can be read as
a clean measurement**, and the primary null cannot be read as a tight one: an effect the size
of the F4 estimate sits inside the primary comparison's own interval. The cause is a design
choice — `ppo_cfc` and `ppo_cfc_dtblind` are separately instantiated models with
independently drawn weights rather than one model with a runtime switch — so more compute
cannot fix it. The fix is a single parameterisation with a `dt`-gate toggle, evaluated from
one set of weights. That has not been built.

**4. Two of seven architectures do not train under this repository's PPO configuration, at
any budget tested.** <!--v:studyb.learn.not_learned_models-->`ppo_ltc`, `ppo_ltc_cfc`<!--/v--> fail the
pre-registered learning check on both scenarios. Their unit tests pass and the LTC cell is
the verified Hasani et al. (2021) formulation with input-dependent `tau_sys`, so this is a
learner/architecture interaction the existing test suite cannot see. **Every comparison
involving either model — six of the nine rows in each of F5 and F9, and both rows of F3 —
measures a non-learner.** Read literally they say "liquid loses to conventional"; read
honestly they say "a model that did not train loses to models that did". No claim in this
document reads them the first way.

**5. Three of Study A's five scenarios, and the whole federated sub-study, were not re-run at
the higher budget.** Study B kept <!--v:studyb.matrix.n_scenarios-->2<!--/v--> scenarios and
dropped <!--v:studyb.descoped.n_scenarios-->3<!--/v-->
(<!--v:studyb.descoped.scenarios-->`non_stationary`, `interference_heavy`, `bursty_irregular`<!--/v-->) to pay for the budget increase; the drops are
declared in `configs/preregistration_study_b.yaml` under `descoped`, with reasons, and no
result was moved into or out of that list after seeing anything. Study B demonstrates that
the compute artifact was real on the two scenarios it tested, so **Study A's nulls on the
other three should be assumed contaminated by the same artifact rather than treated as
settled.** The same applies to Study A's federated returns comparison; its *communication*
result does not depend on the policies having learned, and stands.

**6. Study B is a two-scenario study, so "pooled" means less in it than in Study A.** Study
B's pooled ranking is over <!--v:studyb.rank.scenarios-->2<!--/v--> scenario means and Study
A's over <!--v:rank.scenarios-->5<!--/v-->; the two rankings are not directly comparable and
neither is a tested contrast. Study B's per-scenario intervals are narrower simply because
they average two numbers, not because the estimate is better.

### What the heuristic result says about the environment

**7. A zero-parameter greedy occupancy heuristic beats capacity-matched deep RL here by a
margin that <!--v:studyb.budget.env_step_ratio-->16<!--/v-->× more compute does not dent, and
the interesting question is which of three explanations that supports.** Taking them
concretely, and in order:

*Is the task too easy for a hand-written policy?* **No, and this is checked mechanically.**
Background IoT devices are hidden terminals: the sensed-occupancy block of the observation
reports the licensed primary occupant only, so contention must be inferred from the agent's
own collision history, and because devices camp on a channel and hop slowly that history is
genuinely predictive. `tests/test_envs.py::test_greedy_heuristic_leaves_headroom_for_a_learner`
fails the build if the greedy baseline comes close to the per-step oracle in any primary
scenario, and an earlier fully observed draft of this environment was rejected for exactly
that reason. The heuristic wins this benchmark while remaining far from what the environment
allows.

*Is the task too hard for PPO at any reachable budget?* **Not demonstrated, and the trend
argues against it.** Between the two budgets the learned policies moved from
`random_policy`'s success rate to about
<!--v:studyb.main.irregular_dt.ppo_mlp.mean_success_rate-->0.500<!--/v-->, roughly half the
distance to the heuristic's
<!--v:studyb.main.irregular_dt.greedy_heuristic.mean_success_rate-->0.728<!--/v-->, and the
critic went from explaining nothing to explaining a real fraction of return variance. Nothing
in that trajectory says the remaining gap is unreachable. What this repository can say is
that it is not reachable at <!--v:studyb.matrix.training_steps-->163840<!--/v--> steps per cell on
four CPU cores, which is a statement about this budget and not about PPO.

*Is it badly posed as an RL problem?* **Partly, and this is the most concrete thing to take
away.** The reward is dense and steeply asymmetric — a collision costs up to twice
`collision_penalty` while the best possible success pays at most
`success_reward_scale + idle_bias`, and a uniform policy collides on
<!--v:studyb.main.irregular_dt.random_policy.mean_collision_rate-->0.645<!--/v--> of its steps —
so early training is dominated by the penalty term and every action looks bad. The policies respond by collapsing: their evaluation-time action histograms sit near
<!--v:studyb.main.irregular_dt.ppo_mlp.mean_action_histogram_entropy-->0.712<!--/v--> nats, i.e.
they concentrate on one or two preferred channels, whereas the heuristic's histogram sits at
<!--v:studyb.main.irregular_dt.greedy_heuristic.mean_action_histogram_entropy-->1.785<!--/v-->
nats — it spreads, state-dependently, without ever being uniform
(<!--v:env.uniform_policy_entropy_nats-->2.079<!--/v--> nats). **Camping is exactly the failure
mode this environment punishes**, because hidden background devices camp too. So the shape of
the solution PPO converges to under a fixed entropy coefficient is structurally mismatched to
the shape of the solution the task rewards. That is a falsifiable statement rather than a
hedge: if it is right, raising the entropy coefficient or annealing it should show up as
action-histogram entropy rising toward the heuristic's *while* success rate rises, and if
those two move apart the explanation is wrong. No such run has been done, and no
hyperparameter was tuned in either study — Study B changed the step count and nothing else,
deliberately, so that the budget manipulation is not confounded with tuning.

One gap that blocks a sharper answer: the environment computes a per-step oracle reward in
`info["oracle_reward"]` and the test suite asserts a margin against it, but no committed CSV
aggregates it. **This repository therefore cannot quote how far either the heuristic or the
learned policies are from the achievable ceiling**, and adding that column is the single
cheapest improvement available to the next study.

### Threats that apply to both studies

**8. A single environment family.** Every scenario in both studies is a variant of one
simulator with one observation layout, <!--v:env.num_channels-->8<!--/v--> channels, 64-step
episodes and one reward function. `irregular_dt` differs from `stationary` in exactly one
field, which is good for causal attribution and bad for external validity. Conclusions do not
transfer to other spectrum models, to continuous action spaces, or to longer horizons.

**9. Simulator fidelity.** The environment is a stochastic abstraction, not a radio. Channel
occupancy is a relaxing Bernoulli process, interference and quality are smoothed scalars,
background devices are a birth–death count of hidden terminals, and propagation, protocol
overhead, retransmission and hardware effects are absent.

**10. Statistical scope and multiplicity.** Bootstrap intervals are percentile intervals over
<!--v:matrix.n_seeds-->12<!--/v--> paired differences — small-sample intervals with real
coverage error — and the unit of analysis is the per-cell mean over
<!--v:matrix.eval_episodes-->32<!--/v--> evaluation episodes, so within-cell episode variance
is not propagated. With `n = 12` the exact sign-flip test's p-value floor is `2/2^12`: enough
power to detect a large consistent effect, not enough for a small one. Holm–Bonferroni within
family is conservative and it costs: <!--v:cmp.excludes_zero_not_significant-->6<!--/v-->
comparisons in Study A and <!--v:studyb.cmp.excludes_zero_not_significant-->2<!--/v--> in
Study B have a raw bootstrap interval that excludes zero and an adjusted p-value that does
not reject. **All of them are published with the verdict `no_detectable_difference`**, which
is the right call under a pre-registered protocol and is also the call that loses
information. Both `comparisons.csv` files carry raw p, adjusted p, interval and Cliff's delta
for every comparison, so the demotion is fully reversible from the committed data.

**11. Evaluation protocol.** Reported returns are under a greedy (argmax) policy on a fixed
shared evaluation stream. A stochastic-policy evaluation would give different numbers, and
the two entropy metrics are deliberately reported separately
(`mean_action_histogram_entropy` is an action histogram under argmax evaluation;
`mean_policy_entropy_nats` is the real distributional policy entropy) because conflating them
is how a pre-rebuild version of this repository produced a spurious "entropy collapse"
narrative. Finding 6 depends on that distinction: the LTC models' *distributional* entropy
falls furthest of any model while they learn nothing at all.

**12. One model in the federated study, at one budget.** All three topology arms use
`ppo_ltc_cfc` — which Study B later showed does not train — at Study A's budget. The
federated conclusion is about topology and communication accounting only. Its returns
comparison should be read as uninformative rather than as a null.

**13. Provenance wrinkles, both minor and both stated.** `results/manifest.json` and
`results/study_b/manifest.json` record the git commit that was HEAD when each run started,
which necessarily precedes the commit containing its results; use the recorded config
contents, pre-registration sha256 and per-CSV checksums as the anchor instead. And the
`source_file` field inside `results/study_b/claims.json` reads `results/tables/...`: the
claims builder in `dsa/analysis/tables.py` labels paths relative to the study directory it
was pointed at and does not know which study that is. For Study B, read those paths as
`results/study_b/tables/...`. The values themselves are re-derivable —
`python scripts/analyze.py --results-dir results/study_b --output-dir results/study_b --prereg configs/preregistration_study_b.yaml --check`
regenerates every Study B table and claim and diffs them.

Two further wrinkles, both found by audit and both corrections of things this project
asserted too strongly. First, **the commit that added the pre-registration is dated after the
runs it pre-registers.** Its message says it "deliberately lands before any Study B result
exists"; that is wrong. The commit was authored at 10:40:39 UTC, whereas the first
confirmatory cell started at 09:38:23 and eight of fourteen parts were already complete. What
genuinely precedes the runs is the *freeze*, not the commit:
`results/study_b/preregistration.sha256` was written at 09:37:03, eighty seconds before the
first cell, and the ordering is machine-enforced rather than asserted —
`scripts/run_study_b.py` calls `verify_freeze()` and refuses to execute a single cell unless
the recomputed digest matches, and `freeze()` refuses to overwrite an existing digest file.
Audit the freeze and the enforcement, not the commit date.

Second, **the frozen pre-registration's budget rationale misquotes the ladder that motivated
it.** `configs/preregistration_study_b.yaml` justifies the 163,840-step rung with a pair of
figures per model, phrased "depending on seed" in a way that reads as two per-seed values
when it is not that. The numbers are not reproduced here — the honesty gate correctly
rejected an earlier draft of this paragraph that quoted them, since they are derivable from
nothing in the repository. Read them in the frozen file if you want them, and treat them as
unverifiable. The file is hash-locked, so it is left exactly
as frozen rather than quietly corrected — editing it would break the freeze that makes the
pre-registration meaningful. The ladder itself is gitignored and not pre-registered, so
nothing in the repository can check those numbers; **no figure from the ladder appears
anywhere in this documentation, and none should be cited.** The rung choice stands on its
own: at that budget, 10 of 14 cells beat their initialisation, which is checkable from
`results/study_b/tables/learning_check.csv`.

**14. `ppo_lstm` and two scenarios remain exploratory.** `ppo_lstm`, `noisy_partial` and
`bursty_load` are registered, capacity-matched, unit-tested and runnable, but are in neither
confirmatory matrix — `ppo_lstm` as a near-duplicate of `ppo_gru`, the two scenarios to buy
statistical power at <!--v:matrix.n_seeds-->12<!--/v--> seeds. No claim rests on them.

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
current repository publishes a null on the question it is named after, at two budgets. That
is a worse headline and a better artifact.

### The defect the rebuild did not catch, and what Study B added

The rebuild fixed correctness: seeding, the cells, the learner, the federated aggregation,
the analysis, and the honesty gates that keep every published number tied to a file. It then
ran Study A and published a clean, well-corrected null.

**The null was substantially an artifact of the compute budget, and none of those gates could
see it.** Every test passed. Every number was generated from `results/`. `make check` was
green. The pre-registration was frozen before the run and its digest recorded. And the
conclusion — "no architecture separates from any other" — was still wrong, in the sense that
running the same code with a larger step count reverses it on the two scenarios Study B
tested. A benchmark can be fully reproducible, fully pre-registered, statistically
conservative, and still report the wrong thing, because reproducibility is a property of the
pipeline and adequacy is a property of the budget.

Three concrete changes came out of that, and they are the transferable part of this
repository:

- **The learning check is a pre-registered family, not a diagnostic.** Both studies test
  trained-versus-untrained on the identical evaluation stream, per model and per scenario,
  with the same correction as every other family (Study A's F5; Study B's F2 and F8). Study
  A's headline should have been "<!--v:learn.learned-->0<!--/v--> of
  <!--v:learn.tested-->35<!--/v--> cells learned anything", not its architectural nulls, and
  it is written that way now.
- **Critic explained variance is the endpoint that tracks learning; policy entropy is not.**
  Study B's mechanistic endpoints show entropy collapsing hardest in the two models that
  learn nothing. An entropy plot alone would have certified them as trained.
- **A budget ladder needs breadth, not depth.** The `n = 2` exploratory ladder that sized
  Study B tested two models, both of which learn, and predicted a uniform benefit that did
  not materialise for <!--v:studyb.learn.not_learned_model_count-->2<!--/v--> of
  <!--v:studyb.matrix.n_models-->7<!--/v--> architectures.

The pre-rebuild repository published results that were not real. Study A published results
that were real and under-powered. Study B publishes results that are real, better powered on
two scenarios, and still do not support the repository's title. Each step is an improvement
and none of them is a vindication.

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

**What the evidence says about the title.** At the budget where models learn, the two cells
the title names behave completely differently: `ppo_cfc` — the "LFM" component — is the best
learned policy in Study B's pooled ranking (rank <!--v:studyb.rank.ppo_cfc.rank-->2<!--/v-->
of <!--v:studyb.rank.policies-->10<!--/v-->), while `ppo_ltc` does not train at all and ranks
<!--v:studyb.rank.ppo_ltc.rank-->7<!--/v-->, and `ppo_ltc_cfc`, the hybrid the title
abbreviates, ranks <!--v:studyb.rank.ppo_ltc_cfc.rank-->9<!--/v--> and is beaten by its own
CfC half. "Liquid" is not a coherent performance category in this benchmark; the specific
cell is what matters. The title is retained because it names what the repository implements,
not because the evidence supports it.

---

## Reproduction

Requires Python `3.11` or newer, 4 CPU cores, no GPU. Every command below was run against
this tree; every path and flag resolves.

```bash
git clone <this repository>
cd PPO-LTC-LFM-for-Dynamic-Spectrum-Access-in-Hierarchical-Federated-IoT-Networks

make install      # pip install -r requirements.txt
make test         # pytest -q  -- the full unit suite
make check        # both studies' tables, RESULTS.md, README.md and docs/ all match
```

`make check` is the honesty gate. It has four parts, all of which write nothing:

1. `scripts/analyze.py --results-dir results/study_b --prereg configs/preregistration_study_b.yaml --check`
   re-derives every Study B table and claim from `results/study_b/all_runs.csv`, and
   `scripts/study_b_mechanistic.py --check` re-derives the mechanistic endpoints; both fail
   on any difference (`make check-study-b` runs just these);
2. `scripts/analyze.py --check` does the same for Study A, from `results/all_runs.csv`;
3. `scripts/make_report.py --check` regenerates `RESULTS.md` and fails on any difference;
4. `scripts/render_docs.py --check` regenerates every generated region and every
   `<!--v:...-->` span in this README and in `docs/*.md` — from **both** studies — and fails
   on any difference.

Part 4 exists because parts 2 and 3 were not enough. An adversarial auditor falsified two
headline figures in this file — the primary comparison's mean difference and the greedy
baseline's pooled return — and the whole test suite and the CI report gate stayed green,
because the scan that was supposed to catch hand-typed metrics skipped markdown table rows
and stripped inline-code spans, which is exactly where those two numbers lived. Both holes
are now closed from both sides: the numbers are generated, and
`tests/test_no_hardcoded_metrics.py` scans table cells and inline code instead of skipping
them. CI additionally falsifies a claim in a scratch copy of `results/` on every run and
asserts that part 4 goes red, for each study separately.

To reproduce Study A from scratch:

```bash
make probe        # measure real steps/s on your box
make suite        # the confirmatory matrix
make federated    # the 3-arm topology study
make analyze      # tables, figures, claims.json
make report       # regenerate RESULTS.md and every generated block in the docs
# or all five in order:
make all
```

To reproduce Study B from scratch — a separate study with a separate pre-registration and a
separate output tree, which `make all` deliberately does not include:

```bash
make study-b           # scripts/run_study_b.py --output-dir results/study_b --workers 4
make analyze-study-b   # Study B tables, figures, claims.json, mechanistic endpoints
make report            # rewrites the generated blocks in README.md and docs/ from both
```

Budget note, so that neither target surprises you. Study A took
<!--v:runs.wall_minutes-->56.8<!--/v--> minutes of wall clock for the suite and
<!--v:fed.wall_minutes-->33.4<!--/v--> minutes for the federated study at
<!--v:config.num_workers-->4<!--/v--> workers (`results/all_runs.csv :: wall_seconds` and
`results/federated_all_runs.csv :: wall_seconds`, summed and divided by the worker count).
The throughput guard in `configs/suite.yaml` would have intervened above
<!--v:config.max_projected_wall_minutes-->40<!--/v--> projected minutes; it was explicitly
disabled with `--no-guard`, and `results/manifest.json` records `guard.applied = false` with
the requested and effective step counts equal. Study B took
<!--v:studyb.runs.wall_minutes-->143.3<!--/v--> minutes at the same worker count against a
projection of <!--v:studyb.projection.wall_minutes-->146.7<!--/v--> minutes measured by a
throughput probe plus a four-worker calibration run before the study started
(`results/study_b/timing_probe.json`), under a declared wall-clock cap of
<!--v:studyb.prereg.wall_cap_minutes-->150<!--/v--> minutes. **In neither study was the matrix
or the seed count reduced to hit a projection**; Study B's stopping rule can only skip whole
`(scenario, model)` parts, it records any it skips by name in
`results/study_b/manifest.json :: dropped_parts`, and it skipped
<!--v:studyb.matrix.dropped_parts-->0<!--/v-->.

Every script has a real CLI; these are the flags, not a sketch:

```bash
python scripts/run_suite.py --models ppo_cfc ppo_cfc_dtblind --scenarios irregular_dt \
                            --seeds 7 19 --training-steps 10240 --workers 4
python scripts/run_suite.py --smoke --output-dir results_smoke      # tiny CI budget
python scripts/run_suite.py --dry-run                               # probe and project only
python scripts/run_federated.py --arms hierarchical_federated flat_federated --seeds 7 19
python scripts/run_study_b.py --freeze          # freeze the pre-registration digest, run nothing
python scripts/run_study_b.py --plan            # probe, project and list the parts, run nothing
python scripts/run_study_b.py --workers 4       # run the matrix, one (scenario, model) part at a time
python scripts/run_study_b.py --merge-only      # rebuild all_runs.csv from the finished parts
python scripts/study_b_mechanistic.py --results-dir results/study_b
python scripts/analyze.py --results-dir results --output-dir /tmp/scratch
python scripts/analyze.py --check                                   # Study A tables current?
python scripts/analyze.py --results-dir results/study_b --output-dir results/study_b \
                          --prereg configs/preregistration_study_b.yaml --check
python scripts/make_report.py --check                               # RESULTS.md is current?
python scripts/render_docs.py --check                               # README/docs are current?
```

To re-derive either study's primary claim yourself without trusting the analysis layer, take
that study's `all_runs.csv` (`results/` or `results/study_b/`), filter
`scenario == "irregular_dt"`, pivot `mean_eval_return_trained` by seed for `ppo_cfc` and
`ppo_cfc_dtblind`, difference the two columns and enumerate all `2^12` sign vectors. That
reproduces `CLAIM.PRIMARY.P1` in the corresponding `claims.json` exactly.

### Two pre-registrations, in two files, on purpose

Study B's plan is `configs/preregistration_study_b.yaml`, **not** a section appended to
`configs/preregistration.yaml`. `dsa.analysis.tables.expand_families()` expands every family
in whichever pre-registration file it is handed against whichever `all_runs.csv` it is
handed, and Study A's matrix is a superset of Study B's — so a Study B family living in
Study A's plan would silently add rows to `results/tables/comparisons.csv` the next time
Study A was analysed, rewriting committed evidence and turning the docs gate red for a
reason unrelated to Study A. Two files is what makes "Study A is untouched" mechanically
true rather than merely promised. `configs/preregistration.yaml` is byte-identical to the
version frozen for Study A and its sha256 in `results/manifest.json` still matches.

The freeze ordering is machine-enforced rather than asserted:
`scripts/run_study_b.py --freeze` writes `results/study_b/preregistration.sha256` and
refuses to overwrite an existing digest; every subsequent invocation recomputes the sha256 of
the plan and refuses to execute a single cell unless it matches. The digest recorded in
`results/study_b/manifest.json` is
`<!--v:studyb.prereg.sha256-->f4b76c9a0b38f813f8c222702822ea663bb2c652398a40ec4b62640769614f45<!--/v-->`, and the manifest carries the freeze timestamp and
the start time of the first cell so the ordering can be checked by an auditor rather than
believed.

### What is committed, and what is not

Committed, Study A: `results/all_runs.csv`, `results/federated_all_runs.csv`,
`results/manifest.json`, `results/claims.json`, `results/tables/*.csv`,
`results/figures/*.png`, and the generated `RESULTS.md`.

Committed, Study B: `results/study_b/all_runs.csv`, `results/study_b/manifest.json`,
`results/study_b/claims.json`, `results/study_b/tables/*.csv`,
`results/study_b/mechanistic_endpoints.csv`, `results/study_b/figures/*.png`,
`results/study_b/preregistration.sha256`, `results/study_b/timing_probe.json`, and the
generated [`docs/STUDY_B.md`](docs/STUDY_B.md).

Not committed: per-run logs, exploratory diagnostics and checkpoints (`results/raw/`,
`results/checkpoints/`), and Study B's per-part intermediate CSVs
(`results/study_b/parts/`), all gitignored. Each `manifest.json` — not committed weights —
is the reproducibility anchor: git commit, exact command line, full config contents,
pre-registration sha256, solved hidden dimensions and parameter counts, library versions,
real step counts, and a sha256 checksum of that study's `all_runs.csv`. (Only that one CSV
is checksummed per study; the tables are instead verified by regenerating them — `make
check`.)

The two `claims.json` files give all <!--v:claims.total-->120<!--/v--> Study A and
<!--v:studyb.claims.total-->65<!--/v--> Study B citable numbers a `claim_id` with source file,
source column, interval, p-value and verdict. Prose that cites a number cites a `claim_id`,
and `scripts/render_docs.py` writes the number itself. Claim ids are per study and collide by
design — `CLAIM.PRIMARY.P1` exists in both — so every citation in this document says which
study it belongs to, and the generator reads Study B's under a separate `studyb.` namespace.

---

## Repository layout

```
dsa/seeding.py       pure, order-independent seed derivation
dsa/envs/            spectrum environment, scenarios, heuristic policies
dsa/models/          recurrent cells, shared actor-critic, capacity-matched registry
dsa/learner/         sequence-based recurrent PPO, evaluation harness
dsa/federated/       aggregation, topology, the three federated arms
dsa/analysis/        statistics, tables, figures
scripts/             probe, run_suite, run_federated, run_study_b, study_b_mechanistic,
                     analyze, make_report, render_docs
configs/             suite.yaml, federated.yaml, preregistration.yaml   (Study A)
                     suite_study_b.yaml, preregistration_study_b.yaml   (Study B)
results/             Study A evidence; results/study_b/ is Study B's, never written by A
docs/                prose, plus historical_figures.yaml (the one non-results/ number source)
tests/               one test file per owned module, plus the honesty gates
```

## Documentation

- [`RESULTS.md`](RESULTS.md) — Study A's generated results, authoritative, regenerated by CI.
- [`docs/STUDY_B.md`](docs/STUDY_B.md) — Study B in full: the matrix, all
  <!--v:studyb.cmp.total-->52<!--/v--> comparisons, the mechanistic endpoints, and what
  changed against Study A.
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
