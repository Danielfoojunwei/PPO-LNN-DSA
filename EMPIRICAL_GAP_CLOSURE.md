# EMPIRICAL_GAP_CLOSURE

**Author:** Manus AI  
**Date:** 2026-03-27

## Executive statement

This document records how the repository's original gaps were closed to meet the execution brief. The core standard was simple: **only executable training, evaluation, and saved artifacts count as evidence**. Under that standard, the repository now qualifies as a controlled empirical reinforcement-learning benchmark package for dynamic spectrum access, with real multi-seed results, explicit ablations, and a narrow but implemented hierarchical federated comparison.[1] [2]

The remaining limitations are scientific rather than evidentiary. The benchmark is still a controlled RL environment rather than a physical wireless deployment, but its outputs are produced by real environment interaction and real optimization rather than by scripted or expected-result tables.[1] [2]

## Gap-closure summary

| Requirement from execution brief | Initial gap | Closure status | Evidence |
|---|---|---|---|
| No demo outputs, fakes, stubs, or hardcoded benchmark wins | The repository contained expectation-oriented benchmark material and claim-heavy documentation | Closed for the primary evidence path; benchmark conclusions now derive from executed runs under `results/` and `results/federated/` | [1] [2] |
| Real executable RL environments | Prior evidence path was not grounded in a clean empirical suite | Closed through the new executable environment family in `empirical/envs.py` and completed benchmark runs | [1] [3] |
| Required baselines: random, greedy, PPO-MLP, PPO-LSTM, PPO-GRU, PPO-LTC, PPO-LFM, PPO-LTC-LFM | Baseline coverage was incomplete and inconsistently evidenced | Closed through the completed single-agent suite covering all requested baselines across five seeds | [1] [3] |
| Required scenarios: stationary, non-stationary, interference-heavy, varying users, noisy/partial observability | Coverage was incomplete and not tied to a uniform result pipeline | Closed through the seven-scenario matrix saved in the aggregate tables | [2] [3] |
| At least five seeds per major result | Prior material did not provide a dependable multi-seed empirical matrix | Closed with five-seed aggregation for the main single-agent and federated evidence paths | [1] [3] [6] |
| Ablation evidence for LTC and LFM | No trustworthy ablation evidence path | Closed through the LTC/LFM ablation table comparing PPO-MLP, PPO-LTC, PPO-LFM, and PPO-LTC-LFM | [5] |
| Hierarchical federated claims only if actually implemented and benchmarked | The title and prose risked overclaiming hierarchy | Closed narrowly through implemented centralized, flat federated, and hierarchical federated benchmark modes with communication metrics | [1] [2] [6] |
| Required outputs under `results/` plus runbook/spec/gap-closure docs | The required artifact contract was not satisfied | Closed through saved logs, episode metrics, aggregate tables, plots, config snapshots, and the required Markdown documents | [1] [2] |

## Environment and benchmark closure

The benchmark now covers all required controlled DSA regimes through a single executable environment family and a reproducible experiment matrix. This matters because the execution brief explicitly allowed controlled simulation-style research **only when it is a real executable RL environment with non-hardcoded outcomes**. The completed suite meets that threshold.[1] [2]

| Environment requirement | Implemented benchmark scenario(s) | Closure note |
|---|---|---|
| Stationary channel occupancy | `stationary` | Completed and aggregated across all baselines |
| Non-stationary channel occupancy | `non_stationary` | Completed and reused for federated comparison |
| Interference-heavy regime | `interference_heavy` | Completed with full baseline matrix |
| Varying user/device count | `varying_users_small`, `varying_users_large` | Completed for both lighter and denser contention |
| Partial observability or noisy sensing | `noisy_partial` | Completed with degraded sensing information |
| Irregular decision timing | `variable_dt` | Added as a stronger dt-aware stress test |

The scenario coverage is visible directly in the main paper-style summary table and winner table.[3] [7]

## Baseline closure

The benchmark now includes the exact policy family requested by the execution brief. The important point is not merely that these classes exist in code, but that they were actually run in the completed multi-seed suite.[1] [3]

| Baseline requirement | Closure status | Evidence |
|---|---|---|
| Random policy | Closed | `random_policy` rows in aggregate tables [3] |
| Greedy or occupancy-aware heuristic | Closed | `greedy_heuristic` rows in aggregate tables [3] |
| PPO-MLP | Closed | `ppo_mlp` rows in aggregate tables [3] |
| PPO-LSTM | Closed | `ppo_lstm` rows in aggregate tables [3] |
| PPO-GRU | Closed | `ppo_gru` rows in aggregate tables [3] |
| PPO-LTC | Closed | `ppo_ltc` rows in aggregate tables and ablations [3] [5] |
| PPO-LFM | Closed | `ppo_lfm` rows in aggregate tables and ablations [3] [5] |
| PPO-LTC-LFM | Closed | `ppo_ltc_lfm` rows in aggregate tables and ablations [3] [5] |

## Experiment-matrix closure

The requested experiment matrix is now complete for the main evidence path.

| Experiment requirement | Closure status | Evidence |
|---|---|---|
| Single-agent benchmark suite for all baselines | Closed | Main aggregate table [3] |
| Non-stationary stress suite | Closed | `non_stationary` rows in single-agent and federated tables [3] [6] |
| Five seeds for each major result | Closed | `num_seeds = 5` in aggregate summaries [3] [6] |
| LTC/LFM ablations | Closed | Ablation table [5] |
| Multi-client federated benchmark with communication rounds | Closed | Federated tradeoff table with communication volume [6] |

## Metrics closure

The execution brief required a specific metric set. The completed repository now records those metrics or direct operational equivalents in its aggregate result files.[1] [2]

| Required metric | Closure status | Where it is evidenced |
|---|---|---|
| Mean cumulative reward | Closed | Main aggregate tables [3] |
| Standard deviation across seeds | Closed | Main aggregate tables [3] [6] |
| Spectrum utilization | Closed | Main aggregate tables [3] |
| Collision or interference rate | Closed | Main aggregate tables [3] [6] |
| Fairness across agents if multi-agent | Closed as a benchmark fairness proxy | Background fairness columns in aggregate outputs [3] [6] |
| Convergence speed in environment steps | Closed through logged training-step summaries and evaluation intervals | Aggregate outputs and raw training logs [1] [3] |
| Communication cost and aggregation frequency if federated | Closed | Federated tradeoff table and runbook command specification [1] [6] |
| Wall-clock training time | Closed | Main and federated aggregate outputs [3] [6] |

## Hierarchical federated closure

The execution brief stated that hierarchical federated language must be removed unless the system is actually implemented and benchmarked. That gap was closed by implementing an executable three-mode comparison for the non-stationary scenario. The current repository therefore supports a **narrow and specific** federated claim: the code now includes centralized, flat federated, and hierarchical federated training procedures with communication accounting under the controlled DSA benchmark.[1] [2] [6]

The evidence does **not** support broader claims such as real device deployment, hardware radio validation, or universal hierarchy superiority. The current results instead show a communication-performance tradeoff. In the completed five-seed non-stationary federated benchmark, flat and hierarchical federated modes achieved slightly better average reward than the centralized comparator, but at much higher runtime and nonzero communication overhead.[6]

| Federated mode | Mean reward | Mean utilization | Mean collision rate | Communication volume (MB) |
|---|---:|---:|---:|---:|
| Centralized | -72.486 | 0.079 | 0.784 | 0.000 |
| Flat federated | -69.686 | 0.089 | 0.772 | 21.501 |
| Hierarchical federated | -69.456 | 0.087 | 0.767 | 28.667 |

## Outcome closure relative to the research objective

The execution brief asked the repository to demonstrate that PPO-LTC-LFM improves sample efficiency, convergence, utilization, and interference reduction relative to strong baselines, or else present the truth honestly. The benchmark now satisfies the **honesty** requirement fully, but it satisfies the **universal superiority** target only partially.

The empirical evidence is mixed. PPO-LTC-LFM is the best-reward model in the `stationary` scenario and remains competitive elsewhere, but it does not dominate every regime. PPO-LFM attains the best average reward across scenarios, while PPO-LTC is strongest in the `varying_users_large` regime. Greedy heuristics often maximize raw utilization, but typically at much worse reward.[3] [4] [5] [7]

| Empirical question | Closure judgment | Evidence |
|---|---|---|
| Is PPO-LTC-LFM competitive with strong PPO baselines? | Yes | It wins `stationary` reward and remains close to top performance elsewhere [3] [7] |
| Does PPO-LTC-LFM universally dominate every benchmark? | No | Cross-scenario ranking shows PPO-LFM has the best average reward [4] |
| Is there clear LTC/LFM ablation evidence? | Yes | Scenario-by-scenario reward gains and collision changes are reported [5] |
| Are federated claims now truthful and benchmarked? | Yes, narrowly | Implemented and measured under the non-stationary suite [6] |

## Remaining limitations

The main unresolved issues are no longer about fake evidence. They are ordinary research limitations that should be stated plainly.

| Remaining limitation | Implication |
|---|---|
| Controlled benchmark rather than physical radio deployment | Results support algorithmic comparison, not field validation |
| Federated evidence is currently concentrated on the non-stationary scenario | The repository supports a narrow federated claim, not a universal multi-scenario federated conclusion |
| PPO-LTC-LFM has higher wall-clock cost than lighter baselines | Final writeups must present performance-cost tradeoffs honestly |
| Background fairness is a benchmark proxy rather than a full network utility study | Fairness discussion should remain modest and clearly scoped |

## Final judgment

The repository now passes the gap-closure standard set by the execution brief. It contains real baseline comparisons, five-seed RL evaluation, explicit LTC/LFM ablations, truthful hierarchical federated benchmarking, and the required artifact set. The final scientific message is **credible and nuanced** rather than promotional: the liquid variants are competitive and sometimes best, but their benefits are scenario-dependent and must be weighed against training cost.[1] [3] [4] [5] [6] [7]

## References

[1]: ./EMPIRICAL_RUNBOOK.md "EMPIRICAL_RUNBOOK"
[2]: ./BENCHMARK_SPEC.md "BENCHMARK_SPEC"
[3]: ./results/aggregate_tables/paper_table_main.csv "Paper-style main benchmark table"
[4]: ./results/aggregate_tables/overall_model_ranking.csv "Overall model ranking"
[5]: ./results/aggregate_tables/ablation_ltc_lfm.csv "LTC and LFM ablation summary"
[6]: ./results/aggregate_tables/federated_tradeoff_table.csv "Federated tradeoff table"
[7]: ./results/aggregate_tables/scenario_winners.csv "Scenario winner summary"
