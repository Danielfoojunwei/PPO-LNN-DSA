# PPO-LFM Architecture Documentation

## Overview

This document describes the architecture of the PPO-LFM (Proximal Policy Optimization with Liquid Foundation Models) system for dynamic spectrum access in IoT networks with hierarchical federated learning.

## System Architecture

### Hierarchical Structure

```
┌─────────────────────────────────────────────────┐
│           Cloud Server (Global Model)           │
│  • Aggregates edge server models                │
│  • Maintains global policy                      │
│  • Distributes updates to edges                 │
└────────────┬────────────────────────┬───────────┘
             │                        │
     ┌───────▼───────┐        ┌──────▼────────┐
     │ Edge Server 1 │  ....  │ Edge Server N │
     │ (Regional)    │        │ (Regional)    │
     └───┬─────┬─────┘        └───┬─────┬─────┘
         │     │                  │     │
    ┌────▼┐ ┌─▼────┐        ┌────▼┐ ┌─▼────┐
    │IoT  │ │IoT   │        │IoT  │ │IoT   │
    │Dev 1│ │Dev 2 │  ....  │Dev N│ │Dev N+│
    └─────┘ └──────┘        └─────┘ └──────┘
```

## Key Components

### 1. Liquid Foundation Model (LFM) Layers

#### Adaptive Linear Operator
- **Purpose**: Context-aware linear transformations
- **Key Features**:
  - Static component (base transformation)
  - Adaptive component (context-dependent modulation)
  - Gating mechanism to blend static and adaptive
- **Replaces**: Fixed weight matrices in LSTM

#### Token Mixing
- **Purpose**: Model dependencies across sequence positions
- **Key Features**:
  - Multi-head attention mechanism
  - Adaptive query, key, value projections
  - Parallel processing (vs LSTM's sequential)
- **Replaces**: LSTM's temporal gates

#### Channel Mixing
- **Purpose**: Feature transformation across channels
- **Key Features**:
  - Two-layer MLP with adaptive operators
  - Context-aware transformations
  - Residual connections
- **Replaces**: LSTM's hidden state transformations

#### LFM Block
- **Purpose**: Complete replacement for LSTM layer
- **Components**:
  - Token Mixing (temporal dependencies)
  - Channel Mixing (feature transformation)
  - Layer normalization
  - Residual connections

### 2. Actor-Critic Network

```
Input State [batch, seq_len, state_dim]
           ↓
    LFM Encoder (multi-layer)
    • Input projection
    • Stack of LFM blocks
    • Output normalization
           ↓
    Pooled Representation [batch, hidden_dim]
           ↓
      ┌────┴────┐
      ↓         ↓
   Actor     Critic
  (Policy)  (Value)
      ↓         ↓
  Actions   Values
```

### 3. PPO Training Algorithm

#### Training Loop
1. **Rollout Collection**: Agents interact with environment
2. **Advantage Computation**: GAE (Generalized Advantage Estimation)
3. **Policy Update**: PPO with clipped objective
4. **Value Update**: MSE loss for value function

#### Key Features
- Clipped surrogate objective prevents large policy updates
- Multiple epochs of minibatch updates
- Gradient clipping for stability
- Entropy bonus for exploration

### 4. Federated Learning Framework

#### Client (IoT Device)
- **Responsibilities**:
  - Local training on spectrum access task
  - Maintain local PPO-LFM agent
  - Send model updates to edge server
- **Data**: Local spectrum observations

#### Edge Server (Regional Aggregator)
- **Responsibilities**:
  - Aggregate client models (FedAvg)
  - Maintain regional model
  - Send aggregated update to cloud
- **Aggregation**: Weighted or simple average

#### Cloud Server (Global Coordinator)
- **Responsibilities**:
  - Aggregate edge server models
  - Maintain global policy
  - Distribute global model to edges
- **Coordination**: Synchronous rounds

### 5. Dynamic Spectrum Access Environment

#### State Space
- **Channel Quality**: [num_channels]
- **Interference Levels**: [num_channels]
- **Historical Usage**: [seq_length, num_channels]
- **Total**: [seq_length, num_channels * 3]

#### Action Space
- Discrete: Select one channel from num_channels options

#### Reward Function
- **Success**: +quality * 10.0
- **Collision**: -5.0
- **Failure**: -1.0
- **Interference Penalty**: -(interference - threshold) * 2.0

#### Dynamics
- Channel quality varies over time (fading)
- Interference changes dynamically
- Multiple devices compete for channels
- Collisions when multiple devices select same channel

## Comparison: PPO-LSTM vs PPO-LFM

| Aspect | PPO-LSTM | PPO-LFM |
|--------|----------|---------|
| **Temporal Modeling** | Sequential LSTM cells | Parallel token mixing |
| **Memory Complexity** | O(4 × hidden_size²) | O(hidden_size²) |
| **Adaptability** | Fixed gates | Context-aware operators |
| **Parallelization** | Sequential (slow) | Parallel-friendly (fast) |
| **Feature Extraction** | LSTM hidden state | Token + channel mixing |
| **Scalability** | Limited | Better (MoE optional) |

## Training Pipeline

### Federated Training Round

```
1. Cloud → Edge: Distribute global model
2. Edge → Clients: Distribute regional model
3. Clients: Local training (N episodes)
4. Clients → Edge: Send model updates
5. Edge: Aggregate client models
6. Edge → Cloud: Send regional model
7. Cloud: Aggregate edge models
8. Repeat
```

### Single-Agent Training (Baseline)

```
1. Initialize agent and environment
2. Collect rollout (interact with environment)
3. Compute advantages (GAE)
4. Update policy (PPO)
5. Repeat
```

## File Structure

```
FTVault/
├── models/
│   ├── lfm_layers.py          # LFM core components
│   ├── actor_critic.py         # Actor-critic networks
│   └── ppo_lfm_agent.py       # PPO algorithm
├── federated/
│   ├── client.py               # Client implementation
│   ├── edge_server.py          # Edge aggregation
│   └── cloud_server.py         # Cloud coordination
├── environment/
│   └── spectrum_env.py         # DSA environment
├── utils/
│   ├── config.py               # Configuration
│   └── metrics.py              # Metrics tracking
├── train_federated.py          # Federated training
├── train_single.py             # Baseline training
├── evaluate.py                 # Evaluation script
└── configs/
    ├── default.yaml            # Default config
    └── baseline.yaml           # Baseline config
```

## Key Innovations

### 1. LFM Replaces LSTM
- **Adaptive operators** instead of fixed gates
- **Token mixing** for temporal dependencies
- **Channel mixing** for feature transformation
- **Better parallelization** and efficiency

### 2. Hierarchical Federated Learning
- **Three-tier architecture**: Client-Edge-Cloud
- **Regional aggregation** reduces communication overhead
- **Scalable** to many devices

### 3. Dynamic Spectrum Access
- **Realistic environment** with interference and collisions
- **Multi-agent setting** (IoT devices compete)
- **Time-varying channels** (fading, mobility)

## Performance Metrics

### Training Metrics
- Mean episode reward
- Policy loss, value loss
- Entropy (exploration)

### Evaluation Metrics
- Success rate (successful transmissions)
- Collision rate (interference)
- Throughput (total reward)
- Spectrum efficiency

## Future Extensions

1. **Mixture of Experts (MoE)**: Already implemented, can be enabled
2. **Multi-modal support**: LFM natively supports it
3. **Asynchronous federated learning**: For better efficiency
4. **Advanced aggregation**: FedProx, FedAdam, etc.
5. **Fairness constraints**: Ensure equitable spectrum access
