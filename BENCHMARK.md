# BENCHMARK

This repository now uses a **real empirical benchmark workflow** for dynamic spectrum access reinforcement learning. The legacy expectation-oriented benchmark narrative has been retired from the evidence path. Only executed runs saved under `results/` and `results/federated/` should be treated as benchmark evidence.

## Current benchmark entry points

| Script | Role | Main output location |
|---|---|---|
| `run_empirical_suite.py` | Runs the single-agent multi-seed benchmark matrix | `results/` |
| `run_federated_empirical.py` | Runs centralized, flat federated, and hierarchical federated comparisons | `results/federated/` |
| `analyze_empirical_results.py` | Regenerates summary tables and plots from completed runs | `results/aggregate_tables/`, `results/plots/` |

## Benchmark coverage

The executed benchmark suite includes the following scenario families and policy baselines.

| Category | Coverage |
|---|---|
| Scenarios | `stationary`, `non_stationary`, `interference_heavy`, `varying_users_small`, `varying_users_large`, `noisy_partial`, `variable_dt` |
| Heuristics | `random_policy`, `greedy_heuristic` |
| PPO baselines | `ppo_mlp`, `ppo_lstm`, `ppo_gru` |
| Liquid variants | `ppo_ltc`, `ppo_lfm`, `ppo_ltc_lfm` |
| Seed set | `7, 19, 23, 41, 89` |

## Primary evidence files

Use the following files as the main benchmark evidence path.

| File | Description |
|---|---|
| `results/aggregate_tables/paper_table_main.csv` | Main scenario-by-scenario benchmark summary |
| `results/aggregate_tables/overall_model_ranking.csv` | Cross-scenario average ranking |
| `results/aggregate_tables/ablation_ltc_lfm.csv` | LTC and LFM ablation evidence |
| `results/aggregate_tables/federated_tradeoff_table.csv` | Centralized versus federated tradeoff summary |
| `results/plots/reward_heatmap.png` | Model-scenario reward visualization |

## Reproduction

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Run the single-agent suite:

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

Run the federated suite:

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

Regenerate analysis outputs:

```bash
python3.11 analyze_empirical_results.py
```

## Interpretation standard

This benchmark is a **controlled empirical RL benchmark**. It supports truthful comparison among heuristic, recurrent, liquid, and hybrid liquid PPO policies under executable simulated environments with real optimization and saved artifacts. It should not be used to claim universal superiority, physical-radio deployment validation, or any result that does not appear in the saved artifacts.

## Further documentation

For the full execution record and requirement mapping, see:

| Document | Purpose |
|---|---|
| `EMPIRICAL_RUNBOOK.md` | Exact execution workflow and artifact map |
| `BENCHMARK_SPEC.md` | Formal benchmark definition |
| `EMPIRICAL_GAP_CLOSURE.md` | Requirement-by-requirement closure audit |
