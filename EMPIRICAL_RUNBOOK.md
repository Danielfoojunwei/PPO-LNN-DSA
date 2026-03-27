# EMPIRICAL_RUNBOOK

**Author:** Manus AI  
**Date:** 2026-03-27

## Purpose

This repository was converted from a claim-heavy research prototype into an **executable empirical benchmark suite** for dynamic spectrum access (DSA). The benchmark now relies on real training loops, real environment interactions, multi-seed evaluation, and saved result artifacts. It no longer depends on placeholder benchmark wins, expected-result tables, or hardcoded evidence paths.

The runbook below explains exactly how the benchmark was implemented, executed, and interpreted. It also defines the boundaries of what the results do and do not prove.

## What was executed

The empirical rewrite centered on two top-level entry points.

| Script | Purpose | Main outputs |
|---|---|---|
| `run_empirical_suite.py` | Runs the single-agent benchmark matrix across all required scenarios and baselines | `results/raw_training_logs/`, `results/episode_metrics/`, `results/aggregate_tables/`, `results/plots/`, `results/config_snapshots/` |
| `run_federated_empirical.py` | Runs centralized, flat federated, and hierarchical federated comparisons for the non-stationary benchmark | `results/federated/raw_training_logs/`, `results/federated/episode_metrics/`, `results/federated/aggregate_tables/`, `results/federated/config_snapshots/` |

The supporting implementation lives under `empirical/` and contains the executable environment, model definitions, PPO learner, and federated training utilities.

## Empirical components added or stabilized

The rewrite introduced a controlled but executable DSA benchmark environment, together with a real PPO pipeline and truthful federated comparisons.

| Component | File(s) | Notes |
|---|---|---|
| Controlled DSA environment | `empirical/envs.py` | Real environment transitions, stochastic occupancy dynamics, noisy sensing, variable `dt`, interference, and changing user load |
| PPO baselines and liquid variants | `empirical/models.py` | PPO-MLP, PPO-LSTM, PPO-GRU, PPO-LTC, PPO-LFM, PPO-LTC-LFM |
| PPO training loop | `empirical/ppo.py` | Rollout collection, GAE, clipped PPO updates, evaluation, checkpoint saving |
| Federated execution | `empirical/federated.py` | Centralized-adjacent evaluation, flat FedAvg, and two-level hierarchical aggregation with communication accounting |
| Result post-processing | `analyze_empirical_results.py` | Paper-ready tables, ablations, rankings, and additional plots |

## Dependency state used for the completed runs

The repository now uses pinned dependencies in `requirements.txt` for reproducibility.

| Package | Version |
|---|---:|
| `torch` | `2.5.1` |
| `numpy` | `2.2.6` |
| `pandas` | `2.3.3` |
| `scipy` | `1.14.1` |
| `matplotlib` | `3.10.7` |
| `seaborn` | `0.13.2` |
| `gymnasium` | `1.0.0` |
| `pyyaml` | `6.0.3` |
| `tensorboard` | `2.18.0` |
| `tqdm` | `4.67.1` |

## Smoke validation before full runs

Before launching the full suite, small smoke experiments were executed to confirm that the code compiled, the PPO stack ran end to end, heuristics produced metrics, and federated modes completed without crashing.

Those checks produced:

| Path | Role |
|---|---|
| `results_smoke/` | Single-agent smoke outputs |
| `results_smoke_recheck/` | Post-fix single-agent smoke outputs |
| `results_fed_smoke/` | Federated smoke outputs |

These smoke runs were used for pipeline validation only. The main evidence path is `results/` and `results/federated/`.

## Full single-agent command that was executed

```bash
python3.11 run_empirical_suite.py \
  --results-dir results \
  --device cpu \
  --training-steps 1536 \
  --eval-every 512 \
  --eval-episodes 6 \
  --seeds 7 19 23 41 89 \
  --models random_policy greedy_heuristic ppo_mlp ppo_lstm ppo_gru ppo_ltc ppo_lfm ppo_ltc_lfm \
  --scenarios stationary non_stationary interference_heavy varying_users_small varying_users_large noisy_partial variable_dt \
  --rollout-steps 128 \
  --update-epochs 3 \
  --minibatch-size 64
```

This completed successfully and generated the aggregate tables and plots under `results/`.

## Full federated command that was executed

```bash
python3.11 run_federated_empirical.py \
  --results-dir results/federated \
  --device cpu \
  --scenario non_stationary \
  --model ppo_ltc_lfm \
  --seeds 7 19 23 41 89 \
  --rounds 6 \
  --local-timesteps 512 \
  --centralized-timesteps 3072 \
  --eval-every 512 \
  --eval-episodes 6 \
  --num-clients 6 \
  --num-edges 2 \
  --rollout-steps 128 \
  --update-epochs 3 \
  --minibatch-size 64
```

This also completed successfully and produced centralized, flat federated, and hierarchical federated result artifacts.

## Benchmark coverage actually achieved

The completed benchmark suite covers all required single-agent scenario classes and the requested multi-seed treatment.

| Requirement | Status | Evidence |
|---|---|---|
| Stationary occupancy | Completed | `results/aggregate_tables/paper_table_main.csv` |
| Non-stationary occupancy | Completed | `results/aggregate_tables/paper_table_main.csv` |
| Interference-heavy scenario | Completed | `results/aggregate_tables/paper_table_main.csv` |
| Varying number of users/devices | Completed | `varying_users_small`, `varying_users_large` rows |
| Partial observability / noisy sensing | Completed | `noisy_partial` rows |
| Variable inter-decision interval | Completed | `variable_dt` rows |
| Five seeds per major result | Completed | `num_seeds = 5` in aggregate tables |
| Heuristic baselines | Completed | `random_policy`, `greedy_heuristic` runs |
| PPO baselines | Completed | `ppo_mlp`, `ppo_lstm`, `ppo_gru`, `ppo_ltc`, `ppo_lfm`, `ppo_ltc_lfm` |
| Ablation evidence | Completed | `results/aggregate_tables/ablation_ltc_lfm.csv` |
| Federated comparison | Completed | `results/federated/aggregate_tables/federated_aggregate_results.csv` |

## Result artifact map

The repository now saves the required result types in the expected locations.

| Required output class | Path |
|---|---|
| Raw training logs | `results/raw_training_logs/*.csv` |
| Episode metrics | `results/episode_metrics/*.json` |
| Aggregate tables | `results/aggregate_tables/*.csv` |
| Plots | `results/plots/*.png` |
| Config snapshots | `results/config_snapshots/*.yaml` |
| Federated raw logs | `results/federated/raw_training_logs/*.csv` |
| Federated metrics | `results/federated/episode_metrics/*.json` |
| Federated aggregate tables | `results/federated/aggregate_tables/*.csv` |
| Federated config snapshots | `results/federated/config_snapshots/*.yaml` |

## High-level empirical findings

The new evidence supports a **credible but mixed** story rather than a universal win claim. PPO-LTC-LFM achieved the best reward in the **stationary** scenario, whereas PPO-LFM achieved the strongest **average reward across scenarios** in the aggregated ranking. PPO-LTC improved reward in the `varying_users_large` regime, while greedy heuristics often maximized raw utilization but at much worse cumulative reward.

This matters because the repository should now be read as a **benchmark suite that measures tradeoffs**, not as a proof that one architecture dominates every regime.

| Finding | Evidence |
|---|---|
| PPO-LTC-LFM wins the stationary benchmark on reward | `results/aggregate_tables/paper_table_main.csv` |
| PPO-LFM has the best average reward across all scenarios | `results/aggregate_tables/overall_model_ranking.csv` |
| PPO-LTC improves reward over PPO-MLP in `varying_users_large` | `results/aggregate_tables/ablation_ltc_lfm.csv` |
| Greedy heuristic often gives high utilization but poor reward | `results/aggregate_tables/paper_table_main.csv` |
| Federated hierarchy is now implemented as a controlled empirical benchmark with communication accounting | `results/federated/aggregate_tables/federated_aggregate_results.csv` |

## Honest interpretation boundaries

The benchmark is **real and executable**, but it is still a **controlled RL benchmark**, not a hardware deployment study. That is scientifically acceptable for DSA research as long as the repository is explicit about the setting and does not present synthetic or scripted outputs as field results.

The completed repository therefore supports the following truthful claims:

| Safe claim | Reason |
|---|---|
| The repository contains executable RL environments and real PPO training loops | Verified by completed multi-seed runs and saved artifacts |
| The repository compares liquid and recurrent PPO variants under several controlled DSA regimes | Implemented and benchmarked under `results/` |
| The repository now includes centralized, flat federated, and hierarchical federated comparisons with communication metrics | Implemented and benchmarked under `results/federated/` |
| The empirical evidence is mixed and scenario-dependent | Confirmed by aggregate and ablation tables |

The repository should **not** claim that PPO-LTC-LFM is universally state of the art, universally better than all baselines, or validated on real radios, because the current evidence does not justify those statements.

## Reproducing the completed benchmark

To reproduce the exact evidence path, install the pinned dependencies and rerun the two main commands shown above. For a quick sanity check, start with a smoke configuration and only then launch the full suite.

A practical workflow is:

1. install dependencies from `requirements.txt`;
2. run a short smoke test with fewer steps and one seed;
3. run the full single-agent suite;
4. run the federated suite;
5. regenerate analysis tables and plots with `python3.11 analyze_empirical_results.py`.

## Final repository posture

After this rewrite, the repository is suitable as an **honest empirical benchmark package** for controlled DSA reinforcement learning. It is no longer dependent on expected outputs or conceptual-only claims in the evidence path. The remaining scientific limitation is not fake evidence, but rather the normal limitation of controlled-environment evaluation.
