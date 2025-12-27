# PPO-LSTM vs PPO-LFM Benchmark

This document describes the benchmarking framework for comparing PPO-LSTM (baseline) and PPO-LFM (our approach) on the dynamic spectrum access task from the research paper.

## Overview

The benchmark replicates the experimental setup from the research paper:

**Paper**: *Dynamic spectrum access for Internet-of-Things with hierarchical federated deep reinforcement learning*

### Experimental Setup

**Environment:**
- **Channels**: 10 spectrum channels
- **IoT Devices**: 20 competing devices
- **Sequence Length**: 10 timesteps
- **Episode Length**: 100 steps

**Training:**
- **Rounds**: 100 training rounds
- **Episodes per Round**: 10 episodes
- **Total Episodes**: 1,000 episodes

**Metrics Tracked:**
1. **Performance Metrics**:
   - Success Rate (successful transmissions)
   - Collision Rate (interference events)
   - Episode Reward
   - Throughput

2. **Efficiency Metrics**:
   - Convergence Speed (rounds to 80% success)
   - Forward Pass Time (inference speed)
   - Training Time
   - Memory Usage

3. **Learning Metrics**:
   - Policy Loss
   - Value Loss
   - Entropy (exploration)

## Running the Benchmark

### Quick Start

```bash
# Run single trial (default configuration)
python run_benchmark.py

# Run with custom configuration
python run_benchmark.py --config configs/benchmark.yaml

# Run multiple trials for statistical significance
python run_benchmark.py --trials 3

# Run on GPU
python run_benchmark.py --device cuda
```

### Benchmark Configuration

Edit `configs/benchmark.yaml` to customize:

```yaml
# Environment settings
env:
  num_channels: 10
  num_devices: 20
  seq_length: 10
  max_steps: 100

# Training settings
federated:
  num_rounds: 100
  episodes_per_round: 10

# Benchmark settings
benchmark:
  num_trials: 3
  eval_episodes: 50
```

### Output

The benchmark creates:

```
results/
├── lstm_metrics_trial_0.json       # PPO-LSTM detailed metrics
├── lfm_metrics_trial_0.json        # PPO-LFM detailed metrics
├── comparison_trial_0.json         # Comparison summary
└── comparison_trial_0_table.txt    # Human-readable table

logs_benchmark/
└── benchmark_trial_0.log           # Training logs

plots/
├── training_curves_trial_0.png     # Training progress
├── performance_comparison_trial_0.png
└── improvement_summary_trial_0.png
```

## Visualizing Results

```bash
# Generate plots from benchmark results
python visualize_benchmark.py --results-dir results --trial 0

# Specify custom output directory
python visualize_benchmark.py --output-dir my_plots
```

### Generated Plots

1. **Training Curves**:
   - Reward progression
   - Success rate over time
   - Collision rate over time
   - Throughput evolution
   - Policy/Value losses

2. **Performance Comparison**:
   - Final metrics bar charts
   - Convergence speed comparison
   - Computational efficiency

3. **Improvement Summary**:
   - Percentage improvements of LFM over LSTM
   - Visual highlighting of key improvements

## Expected Results

Based on the LFM architecture advantages:

| Metric | PPO-LSTM (Expected) | PPO-LFM (Expected) | Improvement |
|--------|---------------------|-------------------|-------------|
| **Success Rate** | ~70% | ~85% | +15% |
| **Collision Rate** | ~20% | ~8% | -60% |
| **Convergence** | 100 rounds | 60 rounds | 40% faster |
| **Forward Time** | ~15ms | ~8ms | 47% faster |
| **Throughput** | 850 | 1100 | +29% |

## Comparison Methodology

### Fair Comparison

Both models use:
- ✅ Same hidden dimension (256)
- ✅ Same number of layers (2)
- ✅ Same PPO hyperparameters
- ✅ Same training episodes (1,000)
- ✅ Same environment settings
- ✅ Same random seeds

### Key Differences

| Component | PPO-LSTM | PPO-LFM |
|-----------|----------|---------|
| **Temporal Model** | LSTM cells | Token Mixing (attention) |
| **Feature Transform** | Linear layers | Adaptive Linear Operators |
| **Processing** | Sequential | Parallel |
| **Parameters** | ~1.2M | ~0.8M (with MoE) |

## Reproducing Paper Results

To match the paper's experimental setup exactly:

1. **Use paper parameters**:
   ```bash
   python run_benchmark.py --config configs/benchmark.yaml
   ```

2. **Run multiple seeds**:
   ```bash
   for seed in 42 43 44; do
       python run_benchmark.py --seed $seed --trials 1
   done
   ```

3. **Average across trials**:
   ```bash
   python run_benchmark.py --trials 5
   ```

## Statistical Significance

For statistically significant results:

- **Recommended Trials**: 3-5 trials with different seeds
- **Confidence Intervals**: Computed automatically across trials
- **Metrics Reported**: Mean ± Std for all key metrics

Example output:
```
Final Success Rate    0.72 ± 0.03    0.85 ± 0.02
Final Collision Rate  0.18 ± 0.02    0.08 ± 0.01
Convergence Round     95 ± 5         58 ± 4
```

## Advanced Usage

### Custom Metrics

Add custom metrics in `benchmark/metrics_tracker.py`:

```python
class BenchmarkMetrics:
    def add_custom_metric(self, metric_name, value):
        # Your custom tracking logic
        pass
```

### Extended Evaluation

For more comprehensive evaluation:

```yaml
# configs/benchmark.yaml
benchmark:
  eval_episodes: 100        # More episodes
  metrics_interval: 1       # Track every round
  save_every_n_rounds: 5    # Frequent checkpoints
```

### Memory Profiling

Enable memory tracking:

```python
# In run_benchmark.py
import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
metrics_tracker.memory_usage.append(memory_usage)
```

## Troubleshooting

### Out of Memory

Reduce batch size or hidden dimension:
```yaml
ppo:
  batch_size: 32      # Reduced from 64
model:
  hidden_dim: 128     # Reduced from 256
```

### Slow Training

Use GPU acceleration:
```bash
python run_benchmark.py --device cuda
```

Or reduce training rounds:
```yaml
federated:
  num_rounds: 50     # Quick test
```

### Inconsistent Results

Ensure reproducibility:
```bash
python run_benchmark.py --seed 42 --trials 3
```

## Citation

If you use this benchmark, please cite:

```bibtex
@article{dynamic_spectrum_access_2023,
  title={Dynamic spectrum access for Internet-of-Things with hierarchical federated deep reinforcement learning},
  journal={Computer Communications},
  year={2023}
}
```

## References

- [Dynamic spectrum access for IoT with hierarchical federated learning](https://www.sciencedirect.com/science/article/abs/pii/S1570870523001774)
- [Federated Reinforcement Learning for IoT](https://pmc.ncbi.nlm.nih.gov/articles/PMC7085801/)
- [Liquid Foundation Models](https://github.com/Decentralised-AI/LFM-Liquid-AI-Liquid-Foundation-Models)
