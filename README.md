# PPO-LTC-LFM for Dynamic Spectrum Access in Hierarchical Federated IoT Networks

> **Repository status:** this codebase now serves as an **empirical benchmark suite** for controlled dynamic spectrum access reinforcement learning. Only executed runs saved under `results/` and `results/federated/` count as evidence.[1] [2]

## Overview

This repository implements and benchmarks a family of PPO agents for dynamic spectrum access (DSA), including feed-forward, recurrent, liquid, and hybrid liquid-attention variants. The project was rewritten so that its primary evidence path is based on **real executable training and evaluation runs**, rather than demo outputs, expected-result tables, or conceptual-only claims.[1] [2]

The benchmark covers stationary and non-stationary channel dynamics, interference-heavy conditions, changing user populations, noisy partial observations, and variable inter-decision intervals. It also includes a controlled federated comparison between centralized, flat federated, and hierarchical federated training for the non-stationary case.[1] [2]

## What is implemented

The main executable components are organized as follows.

| Component | File(s) | Purpose |
|---|---|---|
| Controlled DSA environments | `empirical/envs.py` | Generates real benchmark episodes with stochastic occupancy, interference, noisy sensing, and variable `dt` |
| PPO model family | `empirical/models.py` | Implements `ppo_mlp`, `ppo_lstm`, `ppo_gru`, `ppo_ltc`, `ppo_lfm`, and `ppo_ltc_lfm` |
| PPO learner | `empirical/ppo.py` | Collects rollouts, computes advantages, updates policies, evaluates checkpoints, and logs metrics |
| Federated training utilities | `empirical/federated.py` | Implements centralized, flat federated, and hierarchical federated benchmark modes |
| Single-agent runner | `run_empirical_suite.py` | Executes the full multi-seed benchmark matrix |
| Federated runner | `run_federated_empirical.py` | Executes the federated comparison suite |
| Result analysis | `analyze_empirical_results.py` | Regenerates summary tables and plots from completed artifacts |

## Benchmark scope

The completed single-agent benchmark includes all required major baselines and scenario classes.[2] [3]

| Category | Included items |
|---|---|
| Heuristic baselines | `random_policy`, `greedy_heuristic` |
| PPO baselines | `ppo_mlp`, `ppo_lstm`, `ppo_gru` |
| Liquid variants | `ppo_ltc`, `ppo_lfm`, `ppo_ltc_lfm` |
| Scenario classes | `stationary`, `non_stationary`, `interference_heavy`, `varying_users_small`, `varying_users_large`, `noisy_partial`, `variable_dt` |
| Seed protocol | `7, 19, 23, 41, 89` |

The federated benchmark retains the repository's hierarchical framing in a narrow and truthful way: it now measures **centralized**, **flat federated**, and **hierarchical federated** PPO-LTC-LFM on the `non_stationary` scenario, together with communication volume and runtime cost.[1] [2] [6]

## Key empirical findings

The current results support a **mixed and scenario-dependent** conclusion rather than a universal superiority claim. PPO-LTC-LFM achieves the best reward in the `stationary` scenario, PPO-LFM achieves the strongest average reward across scenarios, and PPO-LTC performs best in the `varying_users_large` regime. Greedy heuristics often maximize raw utilization, but they usually do so with substantially worse reward.[3] [4] [5] [7]

| Finding | Evidence |
|---|---|
| PPO-LTC-LFM is the best-reward model in `stationary` | [3] [7] |
| PPO-LFM has the best cross-scenario average reward | [4] |
| PPO-LTC is strongest in `varying_users_large` | [3] [7] |
| LTC and LFM effects are scenario dependent | [5] |
| Federated modes trade communication cost for performance | [6] |

## Main result artifacts

The repository now saves empirical artifacts in a structured evidence path.

| Artifact type | Path |
|---|---|
| Raw training logs | `results/raw_training_logs/*.csv` |
| Episode metrics | `results/episode_metrics/*.json` |
| Aggregate tables | `results/aggregate_tables/*.csv` |
| Plots | `results/plots/*.png` |
| Config snapshots | `results/config_snapshots/*.yaml` |
| Federated aggregate tables | `results/federated/aggregate_tables/*.csv` |
| Empirical runbook | `EMPIRICAL_RUNBOOK.md` |
| Benchmark specification | `BENCHMARK_SPEC.md` |
| Gap-closure audit | `EMPIRICAL_GAP_CLOSURE.md` |

The most useful files for first inspection are:

| File | Why it matters |
|---|---|
| `results/aggregate_tables/paper_table_main.csv` | Main scenario-by-scenario benchmark summary |
| `results/aggregate_tables/overall_model_ranking.csv` | Cross-scenario ranking |
| `results/aggregate_tables/ablation_ltc_lfm.csv` | LTC and LFM ablation evidence |
| `results/aggregate_tables/federated_tradeoff_table.csv` | Centralized versus federated tradeoff summary |
| `results/plots/reward_heatmap.png` | Reward comparison across models and scenarios |

## Quick start

Install the pinned dependencies first.

```bash
pip install -r requirements.txt
```

Then run a small smoke test or the full benchmark suite.

### Full single-agent suite

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

### Full federated suite

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

### Regenerate analysis outputs

```bash
python3.11 analyze_empirical_results.py
```

## Interpretation guidance

This repository should be read as a **controlled empirical RL benchmark**, not as a field deployment study on real wireless hardware. That distinction is important. The current implementation provides real environment interaction, real policy optimization, and real saved outputs, but it does not by itself establish universal state-of-the-art performance or deployment readiness.[1] [2] [5]

Accordingly, the strongest defensible claim is that the codebase now supports **truthful, reproducible, scenario-aware comparison** among standard PPO baselines, liquid variants, and a limited hierarchical federated setup.[1] [2]

## References

[1]: ./EMPIRICAL_RUNBOOK.md "EMPIRICAL_RUNBOOK"
[2]: ./BENCHMARK_SPEC.md "BENCHMARK_SPEC"
[3]: ./results/aggregate_tables/paper_table_main.csv "Paper-style main benchmark table"
[4]: ./results/aggregate_tables/overall_model_ranking.csv "Overall model ranking"
[5]: ./results/aggregate_tables/ablation_ltc_lfm.csv "LTC and LFM ablation summary"
[6]: ./results/aggregate_tables/federated_tradeoff_table.csv "Federated tradeoff table"
[7]: ./results/aggregate_tables/scenario_winners.csv "Scenario winner summary"
