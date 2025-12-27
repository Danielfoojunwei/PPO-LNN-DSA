#!/bin/bash

# Quick Benchmark Script
# Runs a fast benchmark with reduced parameters for testing

echo "========================================"
echo "Quick Benchmark: PPO-LSTM vs PPO-LFM"
echo "========================================"
echo ""
echo "This runs a quick test with reduced parameters:"
echo "  - 20 rounds (instead of 100)"
echo "  - 5 episodes per round"
echo "  - 10 eval episodes"
echo ""

# Create quick config
cat > configs/quick_benchmark.yaml << EOF
# Quick Benchmark Configuration (for testing)

model:
  hidden_dim: 128          # Reduced for speed
  num_lfm_layers: 1
  num_lstm_layers: 1
  num_heads: 4
  use_moe: false

ppo:
  lr: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  n_epochs: 5              # Reduced
  batch_size: 32           # Reduced
  update_interval: 1024    # Reduced

env:
  num_channels: 10
  num_devices: 20
  seq_length: 10
  max_steps: 100
  interference_threshold: 0.5

federated:
  num_clients: 10          # Reduced
  num_edge_servers: 2      # Reduced
  clients_per_edge: 5
  num_rounds: 20           # Reduced
  episodes_per_round: 5    # Reduced
  aggregation_method: 'fedavg'

benchmark:
  num_trials: 1
  eval_episodes: 10        # Reduced
  save_every_n_rounds: 5
  metrics_interval: 1

device: 'auto'
seed: 42
save_interval: 10
eval_interval: 5
log_interval: 1
checkpoint_dir: 'checkpoints_quick'
log_dir: 'logs_quick'
results_dir: 'results_quick'
EOF

echo "Running quick benchmark..."
python run_benchmark.py --config configs/quick_benchmark.yaml --trials 1

echo ""
echo "========================================"
echo "Quick benchmark complete!"
echo "========================================"
echo ""
echo "Results saved to: results_quick/"
echo ""
echo "To visualize results, run:"
echo "  python visualize_benchmark.py --results-dir results_quick --output-dir plots_quick"
echo ""
echo "For full benchmark, run:"
echo "  python run_benchmark.py --config configs/benchmark.yaml --trials 3"
echo ""
