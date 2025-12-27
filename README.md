# PPO-LFM for Dynamic Spectrum Access in Hierarchical Federated Learning

This repository implements a novel approach to dynamic spectrum access for Internet-of-Things (IoT) using **Proximal Policy Optimization with Liquid Foundation Models (PPO-LFM)** in a hierarchical federated learning framework.

## Overview

This implementation **replaces the traditional PPO-LSTM architecture** with PPO-LFM, leveraging the advantages of Liquid Foundation Models:

- **Adaptive Linear Operators**: Context-aware computation that adapts to input patterns
- **Token & Channel Mixing**: Enhanced feature extraction compared to LSTM
- **Mixture of Experts (MoE)**: Efficient parameter utilization
- **Better Temporal Modeling**: Superior to LSTM for sequential decision-making

## Architecture

### Hierarchical Federated Learning Structure

```
Cloud Server (Global Model)
    ↓
Edge Servers (Regional Aggregation)
    ↓
IoT Clients (Local Training)
```

### Key Components

1. **LFM-Based Actor-Critic Network**: Replaces LSTM with Liquid Foundation Model layers
2. **PPO Algorithm**: Policy optimization for spectrum access decisions
3. **Federated Learning**: Distributed training across client-edge-cloud hierarchy
4. **Dynamic Spectrum Access Environment**: IoT spectrum allocation simulation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with hierarchical federated learning
python train_federated.py --config configs/default.yaml

# Train single agent (baseline)
python train_single.py --config configs/baseline.yaml
```

### Evaluation

```bash
python evaluate.py --model checkpoints/best_model.pt
```

## Project Structure

```
FTVault/
├── models/
│   ├── lfm_layers.py          # Liquid Foundation Model layers
│   ├── ppo_lfm_agent.py       # PPO-LFM agent implementation
│   └── actor_critic.py         # LFM-based actor-critic network
├── federated/
│   ├── client.py               # IoT client implementation
│   ├── edge_server.py          # Edge server aggregation
│   └── cloud_server.py         # Cloud server coordination
├── environment/
│   └── spectrum_env.py         # Dynamic spectrum access environment
├── training/
│   ├── ppo_trainer.py          # PPO training logic
│   └── federated_trainer.py    # Federated learning orchestration
├── utils/
│   ├── config.py               # Configuration management
│   └── metrics.py              # Performance metrics
├── train_federated.py          # Main training script
├── train_single.py             # Baseline training
└── evaluate.py                 # Evaluation script
```

## Key Differences from PPO-LSTM

| Feature | PPO-LSTM | PPO-LFM (This Implementation) |
|---------|----------|-------------------------------|
| Temporal Modeling | LSTM cells | Adaptive Linear Operators + Token Mixing |
| Memory Complexity | O(4 × hidden_size²) | O(hidden_size²) with MoE |
| Adaptability | Fixed gates | Context-aware adaptive computation |
| Multi-modal Support | Text/Sequential only | Native multi-modal (future extension) |
| Inference Speed | Slower (sequential) | Faster (parallel-friendly) |

## Performance Benefits

- **Faster Convergence**: LFM's adaptive computation accelerates learning
- **Better Generalization**: Token mixing improves feature extraction
- **Lower Memory**: MoE reduces active parameters per inference
- **Scalability**: More efficient federated learning with reduced communication

## Citation

If you use this code, please cite the original PPO-LSTM paper and acknowledge the LFM integration:

```
Dynamic spectrum access for Internet-of-Things with hierarchical federated deep reinforcement learning
```

## License

MIT License
