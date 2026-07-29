# Study B — the same benchmark at <!--v:studyb.budget.env_step_ratio-->16<!--/v-->× the compute budget

Study B is a **separate, separately pre-registered study**. It does not supersede, replace or
retro-fit Study A: Study A's plan is `configs/preregistration.yaml`, its evidence is
`results/`, and Study B writes only under `results/study_b/`. Both are reported, and where
they disagree the disagreement is the finding. [`RESULTS.md`](../RESULTS.md) is Study A's
generated report; this document is Study B's.

**Why it was run.** Study A trained each cell for
<!--v:matrix.training_steps-->10240<!--/v--> environment steps
(<!--v:runs.gradient_steps-->240<!--/v--> gradient updates) and returned a null on every
architectural family. Its own diagnostics said why: critic explained variance
indistinguishable from zero and policy entropy at the uniform
<!--v:env.uniform_policy_entropy_nats-->2.079<!--/v--> nats, so the advantages driving the
policy gradient were noise and every architectural contrast compared random initialisations.
Study B repeats the core of that matrix at exactly
<!--v:studyb.budget.env_step_ratio-->16<!--/v-->× the budget —
<!--v:studyb.matrix.training_steps-->163840<!--/v--> environment steps and
<!--v:studyb.runs.gradient_steps-->3840<!--/v--> gradient updates per trainable cell — and
asks the same questions of policies that have had a chance to learn.

**Everything below is generated.** Every number in this file is written by
`scripts/render_docs.py` from `results/study_b/claims.json`,
`results/study_b/tables/*.csv`, `results/study_b/mechanistic_endpoints.csv`,
`results/study_b/all_runs.csv` and `results/study_b/manifest.json`. `make check` fails the
build if any of them differs from what those files produce.

```bash
make study-b           # run the matrix (about 2.5 hours on 4 CPU cores)
make analyze-study-b   # tables, figures, claims.json, mechanistic endpoints
make check-study-b     # verify the committed ones are current; writes nothing
```

---

## 1. The matrix, the budget and the provenance

<!-- BEGIN GENERATED: studyb_matrix_table -->
| quantity | value |
|---|---|
| environment steps per trainable cell | 163840 |
| gradient steps per trainable cell | 3840 |
| seeds | 12 |
| scenarios | `irregular_dt`, `stationary` |
| trainable models | 7 |
| zero-parameter baselines | 3 |
| rows written | 240 |
| total training environment steps | 27525120 |
| evaluation episodes per cell | 32 |
| parts skipped by the stopping rule | 0 |
| wall clock, minutes | 143.34 |
| workers | 4 |
| pre-registration sha256 | `f4b76c9a0b38f813f8c222702822ea663bb2c652398a40ec4b62640769614f45` |
| pre-registration frozen (UTC) | `2026-07-29T09:37:03+00:00` |
| first cell started | `2026-07-29T09:38:23+00:00` |
| finished | `2026-07-29T12:01:43+00:00` |
<!-- END GENERATED: studyb_matrix_table -->

The pre-registration was frozen before the run and the ordering is machine-enforced rather
than asserted: `scripts/run_study_b.py --freeze` writes
`results/study_b/preregistration.sha256` and refuses to overwrite an existing digest, and
every subsequent invocation recomputes the sha256 of
`configs/preregistration_study_b.yaml` and **refuses to execute a single cell** unless it
matches. The run log's first line is the verified digest.

### What was descoped, and why

A <!--v:studyb.budget.env_step_ratio-->16<!--/v-->× budget increase at Study A's breadth is
roughly sixteen hours of compute; the allowance was about three. Breadth was cut in a
declared priority order — seeds first, then the primary comparison, then all three
zero-parameter baselines in every cell, then everything else — and every cut is recorded in
the pre-registration's `descoped` block:

- **<!--v:studyb.descoped.n_scenarios-->3<!--/v--> of Study A's five scenarios**
  (<!--v:studyb.descoped.scenarios-->`non_stationary`, `interference_heavy`, `bursty_irregular`<!--/v-->).
  Study A's evidence on these stands and is not revisited — but see §8: it should be assumed
  to carry the same compute artifact Study B demonstrated on the two scenarios it did run.
- **`ppo_lstm`**, which was excluded from Study A's confirmatory matrix for the same reason:
  it is a near-duplicate control of `ppo_gru`.
- **The federated topology sub-study**, which is orthogonal to the budget question. Study A's
  federated evidence stands unre-run.

Nothing else was cut. The seed count, the primary comparison, all
<!--v:studyb.matrix.n_models-->7<!--/v--> models and all
<!--v:studyb.matrix.n_baselines-->3<!--/v--> baselines were kept.

### The stopping rule, and the fact that it never fired

The pre-registration declared a wall-clock cap of
<!--v:studyb.prereg.wall_cap_minutes-->150<!--/v--> minutes, checked **only at
`(scenario, model)` part boundaries**, with any skipped part recorded by name under
`dropped_parts` and with seeds never reduced. A family whose arms are not all present
produces no rows at all rather than being tested on a subset. The run completed all
<!--v:studyb.matrix.parts-->14<!--/v--> parts in
<!--v:studyb.runs.wall_minutes-->143.3<!--/v--> minutes against a pre-run projection of
<!--v:studyb.projection.wall_minutes-->146.7<!--/v-->, and `dropped_parts` contains
<!--v:studyb.matrix.dropped_parts-->0<!--/v--> entries.

---

## 2. The pre-registered primary comparison

`ppo_cfc` vs `ppo_cfc_dtblind` on `irregular_dt` — two arms with identical architecture,
identical parameter count, identical initialisation distribution, identical inputs and
identical environment streams, differing only in whether the elapsed decision interval `dt`
reaches the CfC time gate. Reported unadjusted as the singleton family `F1_primary`.

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

The interval contains zero: **no detectable difference**, for the second study running. This
null is worth more than Study A's identical-looking one because both arms demonstrably
learned first (§3), so it compares two trained policies rather than two initialisations.

It should not, however, be read as a *tight* null. The pre-registered control for this
contrast fired the wrong way:

<!-- BEGIN GENERATED: studyb_dt_table -->
| contrast | scenario | mean difference | 95% CI | Holm p | Cliff's delta | verdict | claim_id |
|---|---|---|---|---|---|---|---|
| `ppo_cfc` vs `ppo_cfc_dtblind` | `irregular_dt` | 0.474 | [-5.184, 5.898] | 0.8589 | 0.028 | `no_detectable_difference` | `CLAIM.CMP.P1` |
| `ppo_cfc` vs `ppo_cfc_dtblind` | `stationary` | 5.359 | [1.822, 8.794] | 0.0166 | 0.472 | `favours_a` | `CLAIM.CMP.F4_dt_awareness_control:ppo_cfc_vs_ppo_cfc_dtblind:stationary` |
<!-- END GENERATED: studyb_dt_table -->

`F4_dt_awareness_control` runs the same two arms on `stationary`, a constant-`dt` scenario
where they are informationally identical and the expected result is nothing. It is the only
place in either study where the `dt`-awareness contrast reaches significance. The honest
reading is a nuisance effect — the two arms are separately instantiated models with
independently drawn weights, not one model with a runtime switch — and an effect of that size
sits inside the primary comparison's own interval. **No `dt`-awareness contrast in this
repository, including the primary one, is currently a clean measurement of `dt`-awareness.**

---

## 3. The learning check — the precondition everything else rests on

Every run is evaluated twice on the identical evaluation stream: once with freshly
initialised weights, once after training. Families `F2_learning_check` (`irregular_dt`) and
`F8_replication_learning_check` (`stationary`), Holm-corrected within family.

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

<!--v:studyb.learn.learned-->10<!--/v--> of <!--v:studyb.learn.tested-->14<!--/v--> cells
beat their own initialisation, against <!--v:learn.learned-->0<!--/v--> of
<!--v:learn.tested-->35<!--/v--> in Study A. The
<!--v:studyb.learn.not_learned-->4<!--/v--> that did not are the two LTC-containing models on
both scenarios: <!--v:studyb.learn.not_learned_models-->`ppo_ltc`, `ppo_ltc_cfc`<!--/v-->.
**Every comparison in §4 and §5 involving either of those two is a comparison against a
non-learner**, and none of the prose in this repository reads them otherwise.

---

## 4. All <!--v:studyb.cmp.total-->52<!--/v--> pre-registered comparisons

<!--v:studyb.cmp.significant-->42<!--/v--> are significant after Holm correction within
family. <!--v:studyb.cmp.excludes_zero_not_significant-->2<!--/v--> have a raw bootstrap
interval that excludes zero and an adjusted p-value that does not reject; all of those are
published with the verdict `no_detectable_difference`, which is what the pre-registered
protocol requires and is also the call that loses information. Both raw and adjusted p are in
the table so the demotion is reversible.

<!-- BEGIN GENERATED: studyb_comparisons_table -->
| family | contrast | scenario | n | mean difference | 95% CI | raw p | Holm p | Cliff's delta | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `F1_primary` | `ppo_cfc` vs `ppo_cfc_dtblind` | `irregular_dt` | 12 | 0.474 | [-5.184, 5.898] | 0.858887 | 0.8589 | 0.028 | `no_detectable_difference` |
| `F2_learning_check` | `ppo_cfc(trained)` vs `ppo_cfc(untrained)` | `irregular_dt` | 12 | 19.430 | [14.923, 24.057] | 0.000488 | 0.0034 | 1.000 | `favours_a` |
| `F2_learning_check` | `ppo_cfc_dtblind(trained)` vs `ppo_cfc_dtblind(untrained)` | `irregular_dt` | 12 | 24.416 | [20.892, 27.675] | 0.000488 | 0.0034 | 1.000 | `favours_a` |
| `F2_learning_check` | `ppo_gru(trained)` vs `ppo_gru(untrained)` | `irregular_dt` | 12 | 17.078 | [10.444, 22.699] | 0.001465 | 0.0044 | 0.903 | `favours_a` |
| `F2_learning_check` | `ppo_ltc(trained)` vs `ppo_ltc(untrained)` | `irregular_dt` | 12 | 1.549 | [-2.226, 5.234] | 0.457031 | 0.9141 | 0.236 | `no_detectable_difference` |
| `F2_learning_check` | `ppo_ltc_cfc(trained)` vs `ppo_ltc_cfc(untrained)` | `irregular_dt` | 12 | 0.600 | [-4.506, 5.675] | 0.835938 | 0.9141 | 0.146 | `no_detectable_difference` |
| `F2_learning_check` | `ppo_mlp(trained)` vs `ppo_mlp(untrained)` | `irregular_dt` | 12 | 24.134 | [20.509, 27.714] | 0.000488 | 0.0034 | 0.986 | `favours_a` |
| `F2_learning_check` | `ppo_transformer(trained)` vs `ppo_transformer(untrained)` | `irregular_dt` | 12 | 18.824 | [12.273, 24.249] | 0.000977 | 0.0039 | 0.903 | `favours_a` |
| `F3_hybrid_decomposition` | `ppo_ltc_cfc` vs `ppo_cfc` | `irregular_dt` | 12 | -23.085 | [-28.825, -17.763] | 0.000488 | 0.0010 | -1.000 | `favours_b` |
| `F3_hybrid_decomposition` | `ppo_ltc_cfc` vs `ppo_ltc` | `irregular_dt` | 12 | -0.661 | [-5.188, 3.629] | 0.783203 | 0.7832 | 0.049 | `no_detectable_difference` |
| `F4_dt_awareness_control` | `ppo_cfc` vs `ppo_cfc_dtblind` | `stationary` | 12 | 5.359 | [1.822, 8.794] | 0.016602 | 0.0166 | 0.472 | `favours_a` |
| `F5_liquid_vs_conventional` | `ppo_cfc` vs `ppo_gru` | `irregular_dt` | 12 | 7.550 | [1.608, 14.141] | 0.037598 | 0.0752 | 0.583 | `no_detectable_difference` |
| `F5_liquid_vs_conventional` | `ppo_cfc` vs `ppo_mlp` | `irregular_dt` | 12 | -0.567 | [-5.454, 4.390] | 0.832520 | 0.8325 | -0.069 | `no_detectable_difference` |
| `F5_liquid_vs_conventional` | `ppo_cfc` vs `ppo_transformer` | `irregular_dt` | 12 | 7.690 | [2.398, 13.584] | 0.017578 | 0.0527 | 0.569 | `no_detectable_difference` |
| `F5_liquid_vs_conventional` | `ppo_ltc_cfc` vs `ppo_gru` | `irregular_dt` | 12 | -15.536 | [-20.776, -10.205] | 0.000977 | 0.0068 | -0.833 | `favours_b` |
| `F5_liquid_vs_conventional` | `ppo_ltc_cfc` vs `ppo_mlp` | `irregular_dt` | 12 | -23.652 | [-28.214, -18.982] | 0.000488 | 0.0044 | -1.000 | `favours_b` |
| `F5_liquid_vs_conventional` | `ppo_ltc_cfc` vs `ppo_transformer` | `irregular_dt` | 12 | -15.395 | [-21.039, -8.939] | 0.001953 | 0.0078 | -0.875 | `favours_b` |
| `F5_liquid_vs_conventional` | `ppo_ltc` vs `ppo_gru` | `irregular_dt` | 12 | -14.875 | [-20.201, -9.502] | 0.000977 | 0.0068 | -0.875 | `favours_b` |
| `F5_liquid_vs_conventional` | `ppo_ltc` vs `ppo_mlp` | `irregular_dt` | 12 | -22.991 | [-26.110, -20.033] | 0.000488 | 0.0044 | -1.000 | `favours_b` |
| `F5_liquid_vs_conventional` | `ppo_ltc` vs `ppo_transformer` | `irregular_dt` | 12 | -14.734 | [-19.488, -10.071] | 0.000977 | 0.0068 | -0.833 | `favours_b` |
| `F6_within_conventional` | `ppo_gru` vs `ppo_mlp` | `irregular_dt` | 12 | -8.116 | [-12.833, -2.987] | 0.013672 | 0.0273 | -0.597 | `favours_b` |
| `F6_within_conventional` | `ppo_transformer` vs `ppo_mlp` | `irregular_dt` | 12 | -8.257 | [-14.718, -1.651] | 0.036133 | 0.0361 | -0.625 | `favours_b` |
| `F7_vs_greedy_baseline` | `ppo_cfc_dtblind` vs `greedy_heuristic` | `irregular_dt` | 12 | -48.786 | [-51.661, -45.263] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F7_vs_greedy_baseline` | `ppo_cfc` vs `greedy_heuristic` | `irregular_dt` | 12 | -48.311 | [-51.279, -45.254] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F7_vs_greedy_baseline` | `ppo_gru` vs `greedy_heuristic` | `irregular_dt` | 12 | -55.861 | [-60.419, -51.709] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F7_vs_greedy_baseline` | `ppo_ltc_cfc` vs `greedy_heuristic` | `irregular_dt` | 12 | -71.396 | [-75.138, -68.008] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F7_vs_greedy_baseline` | `ppo_ltc` vs `greedy_heuristic` | `irregular_dt` | 12 | -70.736 | [-73.507, -67.580] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F7_vs_greedy_baseline` | `ppo_mlp` vs `greedy_heuristic` | `irregular_dt` | 12 | -47.745 | [-51.658, -43.898] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F7_vs_greedy_baseline` | `ppo_transformer` vs `greedy_heuristic` | `irregular_dt` | 12 | -56.001 | [-60.865, -52.366] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F8_replication_learning_check` | `ppo_cfc(trained)` vs `ppo_cfc(untrained)` | `stationary` | 12 | 24.517 | [20.146, 29.832] | 0.000488 | 0.0034 | 1.000 | `favours_a` |
| `F8_replication_learning_check` | `ppo_cfc_dtblind(trained)` vs `ppo_cfc_dtblind(untrained)` | `stationary` | 12 | 20.098 | [16.143, 23.842] | 0.000488 | 0.0034 | 0.986 | `favours_a` |
| `F8_replication_learning_check` | `ppo_gru(trained)` vs `ppo_gru(untrained)` | `stationary` | 12 | 15.396 | [10.703, 20.138] | 0.000977 | 0.0034 | 0.806 | `favours_a` |
| `F8_replication_learning_check` | `ppo_ltc(trained)` vs `ppo_ltc(untrained)` | `stationary` | 12 | -1.470 | [-3.840, 0.715] | 0.259766 | 0.4180 | -0.181 | `no_detectable_difference` |
| `F8_replication_learning_check` | `ppo_ltc_cfc(trained)` vs `ppo_ltc_cfc(untrained)` | `stationary` | 12 | -1.644 | [-3.897, 0.741] | 0.208984 | 0.4180 | -0.333 | `no_detectable_difference` |
| `F8_replication_learning_check` | `ppo_mlp(trained)` vs `ppo_mlp(untrained)` | `stationary` | 12 | 23.749 | [20.414, 27.054] | 0.000488 | 0.0034 | 1.000 | `favours_a` |
| `F8_replication_learning_check` | `ppo_transformer(trained)` vs `ppo_transformer(untrained)` | `stationary` | 12 | 20.616 | [15.349, 25.473] | 0.000488 | 0.0034 | 1.000 | `favours_a` |
| `F9_replication_liquid_vs_conventional` | `ppo_cfc` vs `ppo_gru` | `stationary` | 12 | 8.735 | [3.981, 14.299] | 0.001465 | 0.0059 | 0.667 | `favours_a` |
| `F9_replication_liquid_vs_conventional` | `ppo_cfc` vs `ppo_mlp` | `stationary` | 12 | 1.298 | [-2.715, 5.191] | 0.540527 | 0.5405 | 0.111 | `no_detectable_difference` |
| `F9_replication_liquid_vs_conventional` | `ppo_cfc` vs `ppo_transformer` | `stationary` | 12 | 6.637 | [2.082, 11.323] | 0.023438 | 0.0469 | 0.444 | `favours_a` |
| `F9_replication_liquid_vs_conventional` | `ppo_ltc_cfc` vs `ppo_gru` | `stationary` | 12 | -17.738 | [-23.234, -10.965] | 0.001465 | 0.0059 | -0.833 | `favours_b` |
| `F9_replication_liquid_vs_conventional` | `ppo_ltc_cfc` vs `ppo_mlp` | `stationary` | 12 | -25.175 | [-29.114, -21.485] | 0.000488 | 0.0044 | -1.000 | `favours_b` |
| `F9_replication_liquid_vs_conventional` | `ppo_ltc_cfc` vs `ppo_transformer` | `stationary` | 12 | -19.837 | [-25.152, -14.017] | 0.000488 | 0.0044 | -1.000 | `favours_b` |
| `F9_replication_liquid_vs_conventional` | `ppo_ltc` vs `ppo_gru` | `stationary` | 12 | -18.334 | [-22.669, -13.291] | 0.000977 | 0.0049 | -0.840 | `favours_b` |
| `F9_replication_liquid_vs_conventional` | `ppo_ltc` vs `ppo_mlp` | `stationary` | 12 | -25.771 | [-30.070, -21.620] | 0.000488 | 0.0044 | -1.000 | `favours_b` |
| `F9_replication_liquid_vs_conventional` | `ppo_ltc` vs `ppo_transformer` | `stationary` | 12 | -20.433 | [-24.993, -15.411] | 0.000488 | 0.0044 | -1.000 | `favours_b` |
| `F10_replication_vs_greedy` | `ppo_cfc_dtblind` vs `greedy_heuristic` | `stationary` | 12 | -50.118 | [-54.020, -46.437] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F10_replication_vs_greedy` | `ppo_cfc` vs `greedy_heuristic` | `stationary` | 12 | -44.758 | [-47.583, -41.517] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F10_replication_vs_greedy` | `ppo_gru` vs `greedy_heuristic` | `stationary` | 12 | -53.494 | [-59.053, -48.823] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F10_replication_vs_greedy` | `ppo_ltc_cfc` vs `greedy_heuristic` | `stationary` | 12 | -71.232 | [-73.299, -69.155] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F10_replication_vs_greedy` | `ppo_ltc` vs `greedy_heuristic` | `stationary` | 12 | -71.828 | [-74.267, -69.415] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F10_replication_vs_greedy` | `ppo_mlp` vs `greedy_heuristic` | `stationary` | 12 | -46.056 | [-49.977, -41.988] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
| `F10_replication_vs_greedy` | `ppo_transformer` vs `greedy_heuristic` | `stationary` | 12 | -51.395 | [-56.380, -46.905] | 0.000488 | 0.0034 | -1.000 | `favours_b` |
<!-- END GENERATED: studyb_comparisons_table -->

### Family outcomes

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

---

## 5. Ranking, and the gap to the zero-parameter heuristic

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

The pooled ranking is **descriptive and untested**, and its intervals are over
<!--v:studyb.rank.scenarios-->2<!--/v--> scenario means — narrower than Study A's for
arithmetic reasons, not better ones.

Every learned policy against `greedy_heuristic`, families `F7_vs_greedy_baseline` and
`F10_replication_vs_greedy`:

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

All <!--v:studyb.f7.n-->7<!--/v--> + <!--v:studyb.f10.n-->7<!--/v--> favour the heuristic
with Cliff's delta <!--v:studyb.f7.cliffs_delta_max-->-1.000<!--/v--> — it wins at every one
of the <!--v:studyb.matrix.n_seeds-->12<!--/v--> seeds in every comparison, exactly as in
Study A. What sixteen times the compute bought is visible in the components of the return:

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

Note the action-histogram column. The heuristic spreads across channels
(<!--v:studyb.main.irregular_dt.greedy_heuristic.mean_action_histogram_entropy-->1.785<!--/v-->
nats) without ever being uniform
(<!--v:env.uniform_policy_entropy_nats-->2.079<!--/v--> nats); the learned policies
concentrate on one or two channels; the two non-learners collapse to a point mass and score
like `constant_channel`. Camping is what this environment punishes, because the hidden
background devices camp too.

### Architecture contrasts

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

---

## 6. Mechanistic endpoints — descriptive by declaration

These three endpoints were declared in the pre-registration as **descriptive**: intervals
where an interval is meaningful, **no p-value, no correction**, and membership in no family.
They were declared that way so they could not later be presented as if they had been tested,
and they are reported here even where they disagree with the reward result — which is the
case they exist for.

<!-- BEGIN GENERATED: studyb_mechanistic_full_table -->
| scenario | endpoint | model | before | after | change | 95% CI | seeds moving as expected |
|---|---|---|---|---|---|---|---|
| `irregular_dt` | `critic_explained_variance` | `ppo_cfc` | -0.0008 | 0.1312 | 0.1320 | [0.0847, 0.1915] | 12 of 12 |
| `irregular_dt` | `critic_explained_variance` | `ppo_cfc_dtblind` | -0.0002 | 0.1356 | 0.1358 | [0.0956, 0.1807] | 12 of 12 |
| `irregular_dt` | `critic_explained_variance` | `ppo_gru` | -0.0035 | 0.2244 | 0.2280 | [0.1514, 0.3179] | 12 of 12 |
| `irregular_dt` | `critic_explained_variance` | `ppo_ltc` | -0.0000 | 0.0000 | 0.0000 | [-0.0002, 0.0002] | 7 of 12 |
| `irregular_dt` | `critic_explained_variance` | `ppo_ltc_cfc` | -0.0001 | 0.0000 | 0.0001 | [-0.0002, 0.0003] | 7 of 12 |
| `irregular_dt` | `critic_explained_variance` | `ppo_mlp` | -0.0012 | 0.0818 | 0.0830 | [0.0545, 0.1133] | 12 of 12 |
| `irregular_dt` | `critic_explained_variance` | `ppo_transformer` | -0.0021 | 0.0720 | 0.0741 | [0.0393, 0.1147] | 11 of 12 |
| `irregular_dt` | `eval_policy_entropy_nats` | `ppo_cfc` | 2.048 | 0.718 | -1.329 | [-1.412, -1.250] | 12 of 12 |
| `irregular_dt` | `eval_policy_entropy_nats` | `ppo_cfc_dtblind` | 2.047 | 0.778 | -1.269 | [-1.355, -1.186] | 12 of 12 |
| `irregular_dt` | `eval_policy_entropy_nats` | `ppo_gru` | 2.046 | 1.078 | -0.967 | [-1.084, -0.843] | 12 of 12 |
| `irregular_dt` | `eval_policy_entropy_nats` | `ppo_ltc` | 2.043 | 0.536 | -1.507 | [-1.693, -1.297] | 12 of 12 |
| `irregular_dt` | `eval_policy_entropy_nats` | `ppo_ltc_cfc` | 2.051 | 0.575 | -1.476 | [-1.672, -1.276] | 12 of 12 |
| `irregular_dt` | `eval_policy_entropy_nats` | `ppo_mlp` | 2.050 | 0.591 | -1.459 | [-1.537, -1.370] | 12 of 12 |
| `irregular_dt` | `eval_policy_entropy_nats` | `ppo_transformer` | 2.040 | 0.975 | -1.064 | [-1.189, -0.938] | 12 of 12 |
| `irregular_dt` | `rollout_policy_entropy_nats` | `ppo_cfc` | 2.048 | 0.724 | -1.324 | [-1.401, -1.243] | 12 of 12 |
| `irregular_dt` | `rollout_policy_entropy_nats` | `ppo_cfc_dtblind` | 2.047 | 0.762 | -1.285 | [-1.375, -1.188] | 12 of 12 |
| `irregular_dt` | `rollout_policy_entropy_nats` | `ppo_gru` | 2.046 | 1.017 | -1.029 | [-1.157, -0.899] | 12 of 12 |
| `irregular_dt` | `rollout_policy_entropy_nats` | `ppo_ltc` | 2.043 | 0.546 | -1.498 | [-1.681, -1.288] | 12 of 12 |
| `irregular_dt` | `rollout_policy_entropy_nats` | `ppo_ltc_cfc` | 2.051 | 0.569 | -1.483 | [-1.676, -1.286] | 12 of 12 |
| `irregular_dt` | `rollout_policy_entropy_nats` | `ppo_mlp` | 2.049 | 0.597 | -1.452 | [-1.541, -1.354] | 12 of 12 |
| `irregular_dt` | `rollout_policy_entropy_nats` | `ppo_transformer` | 2.040 | 0.969 | -1.071 | [-1.186, -0.953] | 12 of 12 |
| `stationary` | `critic_explained_variance` | `ppo_cfc` | -0.0005 | 0.1109 | 0.1113 | [0.0674, 0.1579] | 11 of 12 |
| `stationary` | `critic_explained_variance` | `ppo_cfc_dtblind` | -0.0003 | 0.1270 | 0.1273 | [0.0809, 0.1719] | 11 of 12 |
| `stationary` | `critic_explained_variance` | `ppo_gru` | -0.0044 | 0.3072 | 0.3117 | [0.2273, 0.3921] | 12 of 12 |
| `stationary` | `critic_explained_variance` | `ppo_ltc` | -0.0000 | 0.0000 | 0.0000 | [-0.0002, 0.0003] | 7 of 12 |
| `stationary` | `critic_explained_variance` | `ppo_ltc_cfc` | -0.0000 | 0.0000 | 0.0000 | [-0.0003, 0.0003] | 7 of 12 |
| `stationary` | `critic_explained_variance` | `ppo_mlp` | -0.0026 | 0.1761 | 0.1788 | [0.1341, 0.2266] | 12 of 12 |
| `stationary` | `critic_explained_variance` | `ppo_transformer` | -0.0030 | 0.1002 | 0.1032 | [0.0604, 0.1492] | 12 of 12 |
| `stationary` | `eval_policy_entropy_nats` | `ppo_cfc` | 2.047 | 0.746 | -1.301 | [-1.398, -1.211] | 12 of 12 |
| `stationary` | `eval_policy_entropy_nats` | `ppo_cfc_dtblind` | 2.047 | 0.835 | -1.212 | [-1.291, -1.132] | 12 of 12 |
| `stationary` | `eval_policy_entropy_nats` | `ppo_gru` | 2.048 | 1.019 | -1.028 | [-1.137, -0.924] | 12 of 12 |
| `stationary` | `eval_policy_entropy_nats` | `ppo_ltc` | 2.043 | 0.351 | -1.692 | [-1.833, -1.552] | 12 of 12 |
| `stationary` | `eval_policy_entropy_nats` | `ppo_ltc_cfc` | 2.051 | 0.712 | -1.340 | [-1.541, -1.127] | 12 of 12 |
| `stationary` | `eval_policy_entropy_nats` | `ppo_mlp` | 2.049 | 0.674 | -1.375 | [-1.468, -1.281] | 12 of 12 |
| `stationary` | `eval_policy_entropy_nats` | `ppo_transformer` | 2.040 | 1.078 | -0.963 | [-1.079, -0.843] | 12 of 12 |
| `stationary` | `rollout_policy_entropy_nats` | `ppo_cfc` | 2.048 | 0.707 | -1.341 | [-1.436, -1.242] | 12 of 12 |
| `stationary` | `rollout_policy_entropy_nats` | `ppo_cfc_dtblind` | 2.048 | 0.871 | -1.177 | [-1.266, -1.083] | 12 of 12 |
| `stationary` | `rollout_policy_entropy_nats` | `ppo_gru` | 2.048 | 0.988 | -1.060 | [-1.189, -0.943] | 12 of 12 |
| `stationary` | `rollout_policy_entropy_nats` | `ppo_ltc` | 2.043 | 0.333 | -1.710 | [-1.840, -1.583] | 12 of 12 |
| `stationary` | `rollout_policy_entropy_nats` | `ppo_ltc_cfc` | 2.051 | 0.713 | -1.338 | [-1.540, -1.127] | 12 of 12 |
| `stationary` | `rollout_policy_entropy_nats` | `ppo_mlp` | 2.048 | 0.705 | -1.343 | [-1.443, -1.238] | 12 of 12 |
| `stationary` | `rollout_policy_entropy_nats` | `ppo_transformer` | 2.040 | 1.028 | -1.013 | [-1.126, -0.893] | 12 of 12 |
<!-- END GENERATED: studyb_mechanistic_full_table -->

The disagreement: `ppo_ltc` and `ppo_ltc_cfc` show the **largest** entropy collapse of any
model while their critics explain zero variance and their returns do not move. **Falling
entropy is not evidence of learning.** Explained variance is the endpoint that tracks the
reward result here.

Supporting run diagnostics, read directly from `results/study_b/all_runs.csv` and offered as
diagnostics rather than as claims:

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

The two non-learners have approximate KL and clip fractions an order of magnitude below every
other model's: the update is barely moving them. The entropy collapse is the policy head
saturating onto one action while the recurrent body stays inert.

---

## 7. What changed against Study A, and what did not

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

Changed: the learning check, every mechanistic diagnostic, and most architectural verdicts.
Unchanged: the primary comparison's verdict, and the fact that a zero-parameter hand-written
heuristic beats every learned policy at every seed.

An incidental cross-check: the three zero-parameter baselines are deterministic given a seed
and a scenario, and their rows in Study B are byte-identical to their rows in Study A on the
two shared scenarios. The two studies demonstrably evaluated on the same environment streams.

---

## 8. Limitations specific to Study B

1. **Two scenarios, not five.** "Pooled" means less here than in Study A, and the two
   studies' rankings are not directly comparable.
2. **The three descoped scenarios should be assumed contaminated, not settled.** Study B
   showed the compute artifact was real where it looked; there is no reason to expect
   `non_stationary`, `interference_heavy` and `bursty_irregular` to be exempt.
3. **The `dt` control fired the wrong way** (§2), which caps how tightly the primary null can
   be read and is a design defect that compute cannot fix.
4. **Two architectures never trained** (§3), so a third of the model-vs-model comparisons
   measure a non-learner.
5. **No hyperparameter was tuned.** Study B changed the step count and nothing else, by
   declaration, so the budget manipulation is not confounded with tuning — and equally, no
   claim here is about a tuned configuration.
6. **`source_file` inside `results/study_b/claims.json` reads `results/tables/...`.** The
   claims builder labels paths relative to the study directory it was given and does not know
   which study that is; read them as `results/study_b/tables/...`. The values are
   re-derivable with `make check-study-b`.

### Exploratory artifacts, labelled as such

Not part of Study B's evidence and read no reward: the two throughput probes and the
four-worker calibration run used to size the matrix
(`results/study_b/timing_probe.json` holds the committed measurement), and the `n = 2` budget
ladder in `results/raw/budget_sensitivity_diagnostic.jsonl` that motivated the study. The
ladder is gitignored and **no number from it appears anywhere in this repository's
documentation**. It predicted the direction correctly and both understated the effect and
hid its heterogeneity: it tested two models, both of which learn.

---

## See also

- [`../README.md`](../README.md) — both studies, side by side, with the findings.
- [`../RESULTS.md`](../RESULTS.md) — Study A's generated report.
- [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) — the design and statistical protocol both
  studies share, and how to verify any published number.
- `configs/preregistration_study_b.yaml` — Study B's plan, frozen at
  `<!--v:studyb.prereg.sha256-->f4b76c9a0b38f813f8c222702822ea663bb2c652398a40ec4b62640769614f45<!--/v-->`.
