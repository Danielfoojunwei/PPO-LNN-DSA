# PPO-LTC/LFM for Dynamic Spectrum Access in Hierarchical Federated IoT Networks

> **A Liquid Time-Constant Approach to Proximal Policy Optimization for Dynamic Spectrum Access with Hierarchical Federated Deep Reinforcement Learning**

---

## Abstract

Dynamic spectrum access (DSA) in Internet-of-Things (IoT) networks demands reinforcement learning agents that can adapt to non-stationary channel conditions, variable inter-decision intervals, and decentralized multi-device coordination. Prior work has relied on Proximal Policy Optimization with Long Short-Term Memory networks (PPO-LSTM), which assume fixed discrete time steps and suffer from sequential processing bottlenecks. We present **PPO-LTC/LFM**, a novel architecture that replaces LSTM temporal backbones with **Liquid Time-Constant (LTC) networks** and **Liquid Foundation Model (LFM) layers** for policy and value function approximation. Our approach introduces three key innovations: (1) continuous-time hidden state dynamics governed by learnable per-neuron time constants τ, enabling dt-aware adaptation to variable sampling intervals; (2) Adaptive Linear Operators with Token and Channel Mixing, providing context-dependent feature transformations that replace static LSTM gating; and (3) a three-tier hierarchical federated learning framework (Cloud–Edge–Client) that scales to 20+ IoT devices across multiple regional edge aggregators. We implement the full system using the open-source **ncps** (Neural Circuit Policies) library for canonical LTC cells and **Closed-form Continuous-depth (CfC)** models. The codebase provides complete reproducibility with configurable benchmarks, multiple random seeds, and comprehensive metric tracking across success rate, collision rate, convergence speed, inference latency, and spectrum efficiency on a 10-channel, 20-device DSA environment.

---

## 1. Introduction

### 1.1 Problem Statement

The proliferation of IoT devices has created severe spectrum scarcity in shared radio environments. Dynamic spectrum access allows devices to opportunistically select transmission channels, but this requires real-time decision-making under uncertainty: channel quality fluctuates due to fading and mobility, interference arises from concurrent device transmissions, and the environment is inherently non-stationary (Haykin, 2005; Zhao & Sadler, 2007).

Deep reinforcement learning (DRL) has emerged as a promising approach for DSA, with PPO-LSTM architectures achieving strong results in prior work (Wang et al., 2023; Yu et al., 2019). However, LSTM-based approaches have fundamental limitations:

- **Fixed discrete time steps**: LSTMs assume uniform inter-decision intervals, but IoT devices may observe the spectrum at irregular rates due to hardware constraints, duty cycling, or event-driven sensing.
- **Sequential processing**: LSTM's recurrent structure precludes parallelism, creating inference latency bottlenecks on resource-constrained edge devices.
- **Static gating**: LSTM's sigmoid/tanh gates are input-independent in their parameterization, limiting adaptability to heterogeneous channel dynamics.

### 1.2 Approach

We propose replacing the LSTM backbone with architectures from the **Liquid Neural Network** family (Hasani et al., 2021; Hasani et al., 2022), which model hidden state evolution as ordinary differential equations (ODEs) with learnable time constants:

```
τ_i · dh_i/dt = −h_i + f(x, h)_i
```

where τ_i is a per-neuron learnable time constant controlling temporal response speed. This formulation provides:

1. **dt-awareness**: The time interval Δt between decisions explicitly modulates hidden state updates via exponential decay `exp(−Δt/τ)`, naturally handling irregular sampling.
2. **Multi-timescale dynamics**: Different neurons learn different τ values, capturing both fast channel fluctuations and slow interference trends.
3. **Continuous-time guarantees**: The exponential Euler integration scheme preserves stability and monotonic decay properties regardless of step size.

We further extend this with Liquid Foundation Model (LFM) layers that combine LTC/CfC dynamics with self-attention (Token Mixing) and Mixture-of-Experts (MoE) feed-forward networks (Channel Mixing) for enhanced representational capacity.

### 1.3 Contributions

1. **PPO-LTC Architecture**: A complete PPO agent using canonical Liquid Time-Constant cells (from the open-source `ncps` library) as the temporal backbone, with dt-aware policy and value function approximation.
2. **PPO-LFM Architecture**: An enhanced variant combining LTC/CfC dynamics with Adaptive Linear Operators, Token Mixing (multi-head attention), and Channel Mixing with optional Mixture of Experts.
3. **Hierarchical Federated Learning**: A three-tier Cloud–Edge–Client framework using FedAvg aggregation, enabling privacy-preserving distributed training across 20 IoT devices and 4 edge servers.
4. **Comprehensive Benchmarking Framework**: Configurable evaluation comparing PPO-LSTM, PPO-LTC, and PPO-LFM with multiple trials, seeds, and metrics (success rate, collision rate, convergence speed, inference latency, parameter count, spectrum efficiency).
5. **Full Open-Source Reproducibility**: All code, configurations, and training scripts are provided for complete experimental reproduction.

---

## 2. Related Work

### 2.1 Deep Reinforcement Learning for Dynamic Spectrum Access

Wang et al. (2023) proposed hierarchical federated deep reinforcement learning for DSA in IoT, using PPO with LSTM networks to model temporal channel dynamics in a multi-device shared spectrum environment. Their three-tier architecture (Cloud–Edge–Device) demonstrated convergence advantages over centralized training. Yu et al. (2019) applied deep Q-networks to multi-channel DSA, while Naparstek & Cohen (2019) used multi-agent actor-critic methods. All prior work uses discrete-time recurrent networks (LSTM/GRU), which cannot adapt to variable sampling intervals inherent in IoT sensing.

### 2.2 Liquid Time-Constant Networks

Hasani et al. (2021) introduced Liquid Time-Constant (LTC) networks, which model neural dynamics as continuous-time ODEs with learnable time constants. The key innovation is that the time constant τ of each neuron is learned, allowing the network to adapt its temporal response to the task. The `ncps` library (Lechner et al., 2020) provides the canonical open-source implementation with Neural Circuit Policy (NCP) wiring patterns inspired by the *C. elegans* nervous system. LTC networks have demonstrated state-of-the-art performance on time-series tasks including human activity recognition, traffic forecasting, and medical signal processing.

### 2.3 Closed-form Continuous-depth Models

Hasani et al. (2022) proposed Closed-form Continuous-depth (CfC) models that provide analytical solutions to the LTC ODE, avoiding the computational cost of numerical ODE solvers. The closed-form solution uses a sigmoid-based interpolation:

```
h_new = σ(−Δt/τ) · f(x, h) + (1 − σ(−Δt/τ)) · g(x, h)
```

CfC models achieve 100x speedup over neural ODE solvers while preserving continuous-time properties, making them practical for real-time edge deployment in IoT DSA.

### 2.4 Liquid Foundation Models

Liquid AI (2024) extended the liquid neural network paradigm to foundation-scale models, combining continuous-time dynamics with transformer-style attention mechanisms and Mixture of Experts routing. These Liquid Foundation Models (LFMs) achieve competitive performance with significantly fewer parameters through adaptive computation. Our LFM implementation draws on these architectural principles while targeting the RL policy optimization setting.

### 2.5 Federated Reinforcement Learning

McMahan et al. (2017) introduced Federated Averaging (FedAvg) for distributed model training without sharing raw data. Zhuo et al. (2019) and Lim et al. (2020) extended federated learning to reinforcement learning settings. Hierarchical federated learning (Liu et al., 2020) adds regional aggregation through edge servers, reducing communication overhead and enabling heterogeneous device participation. Our work combines hierarchical FL with liquid neural network backbones, enabling dt-aware temporal modeling in the federated DSA setting.

---

## 3. Method

### 3.1 Problem Formulation

We formulate DSA as a Markov Decision Process (MDP) `⟨S, A, P, R, γ⟩`:

- **State** `s_t ∈ ℝ^{L × 3N}`: A sequence of L timesteps, each containing channel quality `q ∈ ℝ^N`, interference levels `ι ∈ ℝ^N`, and historical usage `u ∈ ℝ^N` for N spectrum channels.
- **Action** `a_t ∈ {0, 1, …, N−1}`: Channel selection for transmission.
- **Reward** `r_t`: Shaped reward combining transmission success, collision penalty, and interference cost:

```
          ┌ q(a_t) · 10.0     if successful transmission
r_t =     │ −5.0               if collision (multi-device on same channel)
          └ −1.0               if transmission failure
      − max(0, ι(a_t) − θ) · 2.0
```

where `θ = 0.5` is the interference penalty threshold.

- **Transition P**: Channel quality evolves as an AR(1) process with Gaussian noise: `q_{t+1} = 0.8·q_t + 0.2·U(0.5, 1.0) + N(0, 0.05)`. Interference evolves similarly. Other devices follow mixed greedy/random strategies (70%/30%).
- **Discount** `γ = 0.99`.

### 3.2 PPO with Liquid Time-Constant Networks

#### 3.2.1 LTC Cell Dynamics

The core building block is the Liquid Time-Constant cell. For hidden neuron i with input x_t and previous state h_{t−1}:

```
τ_i = clamp(exp(log_τ_i^raw), τ_min, τ_max)

candidate_i = σ_gate · tanh(W_x · x_t + W_h · h_{t−1}) + (1 − σ_gate) · h_{t−1,i}

h_{t,i} = exp(−Δt/τ_i) · h_{t−1,i} + (1 − exp(−Δt/τ_i)) · candidate_i
```

where `τ_i ∈ [τ_min, τ_max]` (default [0.1, 10.0]) is the learnable time constant, and `σ_gate = σ(W_g · [x_t; h_{t−1}])` is an input-dependent gate. When Δt is large relative to τ_i, the decay `exp(−Δt/τ_i) → 0` and the cell rapidly adopts the new candidate; when Δt is small, the cell retains its previous state. This provides natural handling of irregular sampling intervals.

#### 3.2.2 CfC Alternative

For latency-critical deployment, we provide a Closed-form Continuous-depth (CfC) alternative:

```
h_new = σ(−Δt/τ) · tanh(f₁(x, h)) + (1 − σ(−Δt/τ)) · tanh(f₂(x, h))
```

where f₁, f₂ are learned neural networks. This avoids ODE integration while preserving dt-awareness.

#### 3.2.3 LFM Block Architecture

Each LFM block combines three components:

1. **Liquid Cell** (LTC or CfC): dt-aware recurrent dynamics
2. **Token Mixing**: Multi-head self-attention across the sequence dimension

```
Attn(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

where Q, K, V are produced by Adaptive Linear Operators instead of static projections.

3. **Channel Mixing**: Context-dependent feature transformation via Adaptive Linear Operators:

```
ALO(x, c) = W_static · x · (1 + σ(W_g · c) · tanh(W_c · c))
```

where c is the context (mean-pooled input) and `σ(W_g · c)` gates the adaptive modulation.

Residual connections and LayerNorm are applied after each sub-layer:

```
x = x + TokenMixing(LN(x))
x = x + ChannelMixing(LN(x))
```

#### 3.2.4 Actor-Critic Architecture

```
State s_t ∈ ℝ^{L×3N}
       │
       ▼
┌─────────────────────┐
│   Input Projection   │  Linear(3N → d_hidden)
│    W_proj · s + b    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   LFM Encoder        │  K stacked LFM blocks
│   (K=2 layers)       │  with LTC/CfC + Attn + MoE
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Mean Pooling       │  ℝ^{L×d} → ℝ^d
└──────────┬──────────┘
           ▼
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│  Actor  │ │ Critic  │
│ π(a|s)  │ │  V(s)   │
│ ALO+Lin │ │ ALO+Lin │
└────┬────┘ └────┬────┘
     ▼           ▼
  Categorical   Scalar
  over N ch.    value
```

### 3.3 PPO Training

We use the clipped surrogate objective (Schulman et al., 2017):

```
L_CLIP = E_t [ min( r_t(θ)·Â_t,  clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]
```

where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` and Â_t is computed via Generalized Advantage Estimation (GAE; Schulman et al., 2016):

```
Â_t = Σ_{l=0}^{T−t} (γλ)^l · δ_{t+l},    δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
```

The total loss combines policy, value, and entropy terms:

```
L = L_CLIP + c₁ · L_VF − c₂ · H[π_θ]
```

| Hyperparameter | Symbol | Value |
|---|---|---|
| Learning rate | α | 3 × 10⁻⁴ |
| Discount factor | γ | 0.99 |
| GAE lambda | λ | 0.95 |
| Clip epsilon | ε | 0.2 |
| Value coefficient | c₁ | 0.5 |
| Entropy coefficient | c₂ | 0.01 |
| Max gradient norm | — | 0.5 |
| PPO epochs | K | 10 |
| Minibatch size | B | 64 |

### 3.4 Hierarchical Federated Learning

We employ a three-tier hierarchical architecture following Wang et al. (2023):

```
┌─────────────────────────────────────────────────────┐
│             Cloud Server (Global Model)              │
│  • Aggregates edge server models via FedAvg          │
│  • Maintains global policy π_global                  │
│  • Distributes updates to all edges                  │
└────────────┬───────────────────────┬────────────────┘
             │                       │
     ┌───────▼───────┐       ┌──────▼────────┐
     │ Edge Server 1 │  ...  │ Edge Server 4 │
     │  (Regional)   │       │  (Regional)   │
     │  FedAvg over  │       │  FedAvg over  │
     │  5 clients    │       │  5 clients    │
     └───┬─────┬─────┘       └───┬─────┬─────┘
         │     │                 │     │
    ┌────▼┐ ┌─▼────┐       ┌────▼┐ ┌─▼────┐
    │ IoT │ │ IoT  │  ...  │ IoT │ │ IoT  │
    │ D_1 │ │ D_2  │       │D_16 │ │ D_20 │
    └─────┘ └──────┘       └─────┘ └──────┘
```

**FedAvg aggregation** (McMahan et al., 2017): For K participants with model parameters θ_k trained on n_k samples:

```
θ_global = Σ_{k=1}^{K} (n_k / Σ_j n_j) · θ_k
```

**Training protocol per round r:**

1. Cloud distributes `θ_global^{(r−1)}` to all edge servers
2. Edge servers distribute to registered clients
3. Each client k performs local PPO training for E episodes, yielding `θ_k^{(r)}`
4. Edge server e aggregates its clients: `θ_e^{(r)} = FedAvg({θ_k^{(r)}}_{k ∈ C_e})`
5. Cloud aggregates edge models: `θ_global^{(r)} = FedAvg({θ_e^{(r)}}_{e=1}^{E})`

| FL Parameter | Value |
|---|---|
| Total clients | 20 |
| Edge servers | 4 |
| Clients per edge | 5 |
| Training rounds | 100 |
| Episodes per client per round | 10 |
| Aggregation method | FedAvg |

---

## 4. Experimental Setup

### 4.1 Environment: Dynamic Spectrum Access

The DSA environment simulates N = 20 IoT devices competing for C = 10 spectrum channels:

| Parameter | Value | Description |
|---|---|---|
| Channels C | 10 | Available spectrum channels |
| Devices N | 20 | Competing IoT devices |
| Sequence length L | 10 | Temporal observation window |
| Episode length T | 100 | Maximum steps per episode |
| Interference threshold θ | 0.5 | Penalty activation threshold |
| Channel quality init | U(0.5, 1.0) | Initial quality distribution |
| Interference init | U(0.0, 0.3) | Initial interference distribution |
| Quality AR(1) coeff. | 0.8 | Temporal correlation |
| Other device policy | 70% greedy / 30% random | Competitor strategy |

**State space**: `s_t ∈ ℝ^{10 × 30}` — 10 timesteps of [quality, interference, usage] for 10 channels.

**Action space**: Discrete, `|A| = 10` — select one channel.

**Collision model**: When multiple devices select the same channel, all experience collision (reward −5.0). Success probability is `p = q_c · (1 − ι_c)` for channel c with quality q_c and interference ι_c.

### 4.2 Model Configurations

All models share identical hidden dimensions, training hyperparameters, and evaluation protocols for fair comparison:

| Component | PPO-LSTM | PPO-LTC | PPO-LFM |
|---|---|---|---|
| Temporal backbone | LSTM (2 layers) | LTC cells (2 layers) | LFM blocks (2 layers) |
| Hidden dimension | 256 | 256 | 256 |
| Attention heads | — | — | 4 |
| MoE experts | — | — | Optional (8, top-2) |
| Time-awareness | No (fixed Δt = 1) | Yes (learnable τ) | Yes (learnable τ) |
| Processing | Sequential | Sequential | Parallel-friendly |
| Actor head | Linear→Tanh→Linear | Linear→Tanh→Linear | ALO→Tanh→Linear |
| Critic head | Linear→Tanh→Linear | Linear→Tanh→Linear | ALO→Tanh→Linear |

### 4.3 Metrics

We evaluate across five categories following standard RL benchmarking practices:

**Performance Metrics:**
- **Success rate**: Fraction of steps with successful transmission
- **Collision rate**: Fraction of steps with multi-device collision
- **Episode reward**: Cumulative shaped reward per episode
- **Throughput**: Total reward across episode
- **Spectrum efficiency**: Success rate × mean reward

**Convergence Metrics:**
- **Convergence round**: First round achieving ≥ 80% success rate
- **Final performance**: Metrics at training completion

**Computational Metrics:**
- **Forward pass time**: Wall-clock inference latency (ms)
- **Update time**: Wall-clock PPO update time (ms)
- **Parameter count**: Total learnable parameters

**Learning Metrics:**
- **Policy loss**: PPO clipped surrogate objective
- **Value loss**: Critic MSE
- **Entropy**: Policy entropy (exploration measure)

**Statistical Rigor:**
- Multiple independent trials (≥ 3) with different random seeds
- Mean ± standard deviation reported for all metrics
- Same seed sequence used for LSTM, LTC, and LFM runs

### 4.4 Open-Source Dependencies

| Library | Version | Purpose | Reference |
|---|---|---|---|
| `ncps` | ≥ 0.0.7 | Official LTC/CfC cells, NCP wiring | Lechner et al. (2020) |
| `torch` | ≥ 2.0.0 | Deep learning framework | Paszke et al. (2019) |
| `numpy` | ≥ 1.24.0 | Numerical computing | Harris et al. (2020) |
| `gymnasium` | ≥ 0.29.0 | RL environment API | Towers et al. (2023) |
| `scipy` | ≥ 1.10.0 | Scientific computing | Virtanen et al. (2020) |
| `matplotlib` | ≥ 3.7.0 | Visualization | Hunter (2007) |
| `tensorboard` | ≥ 2.13.0 | Training monitoring | — |
| `pyyaml` | ≥ 6.0 | Configuration management | — |
| `pandas` | ≥ 2.0.0 | Data analysis | — |
| `tabulate` | ≥ 0.9.0 | Table formatting | — |

The `ncps` library provides the **canonical** open-source implementation of:
- **LTCCell**: Liquid Time-Constant recurrent cell
- **CfC**: Closed-form Continuous-depth model
- **AutoNCP**: Automatic Neural Circuit Policy wiring
- **NCP**: Neural Circuit Policy with sensory, inter, command, and motor neuron groups

---

## 5. Architecture Details

### 5.1 LTC Cell Implementation

Our LTC cell implements the exponential Euler integration scheme for the continuous-time ODE:

```python
# Core dynamics (from preceptual_libiao/src/preceptual/rl/lnn_ltc.py)
tau = torch.clamp(torch.exp(self.log_tau), tau_min, tau_max)
decay = torch.exp(-dt / tau)                        # Exponential decay factor
h_new = decay * h + (1 - decay) * candidate         # Interpolation
```

**Key properties:**
- **Stability**: Exponential Euler is unconditionally stable for τ > 0, unlike explicit Euler which requires Δt < 2τ.
- **Monotonic decay**: `‖h_new − candidate‖ = exp(−Δt/τ) · ‖h_old − candidate‖ ≤ ‖h_old − candidate‖`.
- **dt-invariance**: Applying N steps of size Δt/N produces the same result as one step of size Δt (up to nonlinear candidate recomputation).

**Learnable time constants** are parameterized in log-space for unconstrained optimization:

```
τ = clamp(exp(log_τ_raw), 0.1, 10.0)
```

Initialization: `log_τ_raw ~ U(log(0.1), log(10.0))`.

### 5.2 Adaptive Linear Operator

The ALO replaces standard `Wx + b` with context-modulated transformations:

```python
# Static path
static_out = F.linear(x, W_static, b_static)

# Context-dependent modulation
context_pooled = context.mean(dim=1)              # Mean pooling over sequence
adaptive_scale = tanh(W_context(context_pooled))
gate_value = sigmoid(W_gate(context_pooled))

# Blended output
output = static_out * (1 + gate_value * adaptive_scale)
```

This provides a spectrum between purely static (gate → 0) and fully adaptive (gate → 1) behavior, with the network learning the optimal blend.

### 5.3 Mixture of Experts

The optional MoE layer routes tokens to top-k experts:

```
p = softmax(W_router · x)
top-k = argtopk(p, k=2)
output = Σ_{i ∈ top-k} (p_i / Σ_{j ∈ top-k} p_j) · E_i(x)
```

where each expert E_i is a two-layer MLP with GELU activation. This allows the network to specialize different experts for different channel conditions (e.g., high-interference vs. low-interference regimes).

---

## 6. Reproducibility

### 6.1 Installation

```bash
# Clone repository
git clone https://github.com/Danielfoojunwei/PPO-LTC-DSA.git
cd PPO-LTC-DSA

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Running Benchmarks

```bash
# Quick validation (20 rounds, ~5-10 minutes)
./quick_benchmark.sh

# Full benchmark (100 rounds, single trial)
python run_benchmark.py --config configs/benchmark.yaml

# Statistically rigorous benchmark (3 trials, different seeds)
python run_benchmark.py --config configs/benchmark.yaml --trials 3

# GPU acceleration
python run_benchmark.py --config configs/benchmark.yaml --device cuda

# Custom seed for exact reproducibility
python run_benchmark.py --seed 42 --trials 5
```

### 6.3 Training

```bash
# Hierarchical federated learning (PPO-LFM, 20 clients, 4 edges)
python train_federated.py --config configs/default.yaml

# Single-agent baseline training
python train_single.py --config configs/baseline.yaml
```

### 6.4 Evaluation and Visualization

```bash
# Evaluate trained model
python evaluate.py --model checkpoints/best_model.pt --plot

# Generate benchmark visualization
python visualize_benchmark.py --results-dir results --trial 0
```

### 6.5 Output Structure

```
results/
├── lstm_metrics_trial_{i}.json       # PPO-LSTM per-round metrics
├── lfm_metrics_trial_{i}.json        # PPO-LFM per-round metrics
├── comparison_trial_{i}.json         # Head-to-head comparison
└── comparison_trial_{i}_table.txt    # Human-readable results

logs_benchmark/
└── benchmark_trial_{i}.log           # Full training log

plots/
├── training_curves_trial_{i}.png     # Learning curves
├── performance_comparison_{i}.png    # Bar chart comparison
└── improvement_summary_{i}.png       # Improvement percentages
```

---

## 7. Project Structure

```
PPO-LTC-DSA/
│
├── models/                          # Core PPO-LFM implementation
│   ├── lfm_layers.py                #   Adaptive Linear Operators, Token/Channel
│   │                                #   Mixing, LFM Blocks, MoE (297 lines)
│   ├── actor_critic.py              #   LFM Actor-Critic networks (282 lines)
│   └── ppo_lfm_agent.py             #   PPO algorithm with LFM backbone (304 lines)
│
├── baseline/                        # PPO-LSTM baseline
│   └── ppo_lstm_agent.py            #   LSTM Actor-Critic + PPO (408 lines)
│
├── environment/                     # DSA environment
│   └── spectrum_env.py              #   DynamicSpectrumAccessEnv (348 lines)
│
├── federated/                       # Hierarchical FL framework
│   ├── client.py                    #   IoT client with local PPO training
│   ├── edge_server.py               #   Regional FedAvg aggregation
│   └── cloud_server.py              #   Global coordination
│
├── benchmark/                       # Metrics and comparison
│   └── metrics_tracker.py           #   BenchmarkMetrics, ComparisonMetrics
│
├── utils/                           # Utilities
│   ├── config.py                    #   YAML configuration loader
│   └── metrics.py                   #   MetricsTracker, Logger
│
├── configs/                         # Experiment configurations
│   ├── default.yaml                 #   Default PPO-LFM training
│   ├── baseline.yaml                #   Single-agent baseline
│   ├── benchmark.yaml               #   Full benchmark parameters
│   ├── quick_benchmark.yaml         #   Quick validation
│   └── stable_benchmark.yaml        #   Extended stable benchmark
│
├── preceptual_libiao/               # Extended implementations
│   ├── src/preceptual/
│   │   ├── rl/
│   │   │   ├── lnn_ltc.py           #   LTC Cell + LNN Backbone (417 lines)
│   │   │   └── ppo_lnn.py           #   PPO with LNN (610 lines)
│   │   ├── baselines/
│   │   │   ├── ppo_lfm.py           #   LFM PPO with CfC + MoE (728 lines)
│   │   │   ├── ppo_lnn.py           #   LNN PPO baseline (571 lines)
│   │   │   ├── ppo_lstm.py          #   LSTM PPO baseline (496 lines)
│   │   │   └── double_dqn.py        #   Double DQN alternative
│   │   ├── sim/                     #   DSA environment variants
│   │   ├── fl/                      #   Advanced FL components
│   │   └── slicing/                 #   Spectrum slicing
│   ├── benchmark_*.py               #   6 benchmark script variants
│   └── tests/                       #   Unit tests
│
├── run_benchmark.py                 # Main benchmark: LSTM vs LFM
├── train_federated.py               # Hierarchical federated training
├── train_single.py                  # Single-agent training
├── evaluate.py                      # Model evaluation
├── visualize_benchmark.py           # Result visualization
├── quick_benchmark.sh               # Quick benchmark script
├── requirements.txt                 # Python dependencies
├── ARCHITECTURE.md                  # Architecture documentation
└── BENCHMARK.md                     # Benchmarking documentation
```

---

## 8. Architectural Comparison: PPO-LSTM vs PPO-LTC vs PPO-LFM

### 8.1 Structural Differences

| Aspect | PPO-LSTM | PPO-LTC | PPO-LFM |
|---|---|---|---|
| **Temporal Model** | LSTM gates (discrete) | LTC ODE (continuous) | LTC/CfC + Attention |
| **Time-Awareness** | None (Δt = 1 assumed) | Explicit Δt/τ decay | Explicit Δt/τ decay |
| **Gating** | Static σ(Wx + b) | Learned τ + input gate | Adaptive Linear Operators |
| **Sequence Processing** | Sequential (no parallelism) | Sequential per step | Parallel (attention) |
| **Feature Extraction** | LSTM hidden state | LNN hidden state | Token + Channel Mixing |
| **Scalability** | Fixed computation | Fixed computation | MoE (adaptive compute) |
| **Memory** | O(4d²) per LSTM layer | O(4d²) per LTC layer | O(d²) + attention |

### 8.2 Theoretical Advantages of LTC/LFM

1. **Variable Δt handling**: LTC naturally adapts to irregular sampling via exponential decay, while LSTM treats all steps as Δt = 1.
2. **Multi-timescale dynamics**: Learned τ distribution captures both fast (small τ → channel quality) and slow (large τ → interference trends) dynamics.
3. **Stability guarantees**: Exponential Euler integration is unconditionally stable, unlike explicit methods that require small step sizes.
4. **Continuous-time generalization**: CfC models can extrapolate to unseen Δt values at test time without retraining.

### 8.3 Fair Comparison Protocol

Both models are evaluated under identical conditions:

- Same hidden dimension (256)
- Same number of layers (2)
- Same PPO hyperparameters (α, γ, λ, ε, c₁, c₂)
- Same training episodes (100 rounds × 10 episodes = 1,000 total)
- Same environment settings (10 channels, 20 devices)
- Same random seeds across trials
- Same evaluation protocol (50 deterministic episodes)

---

## 9. Discussion

### 9.1 When to Prefer LTC/LFM over LSTM

- **Irregular sampling**: IoT devices with duty cycling, event-driven sensing, or heterogeneous clock rates benefit from dt-aware dynamics.
- **Non-stationary environments**: Learnable time constants adapt to changing dynamics without manual tuning.
- **Multi-timescale tasks**: DSA requires tracking both fast channel fluctuations and slow interference patterns.
- **Edge deployment**: CfC provides faster inference than LSTM for latency-constrained devices.

### 9.2 Limitations

- **LTC sequential bottleneck**: Pure LTC cells still process sequences step-by-step (mitigated by CfC and attention in LFM).
- **Training complexity**: LFM with MoE has more hyperparameters (number of experts, top-k routing).
- **Environment fidelity**: Our DSA environment uses simplified channel models; real RF environments have more complex propagation effects.
- **Federated overhead**: Hierarchical FL adds communication rounds; optimal edge-client assignment is not addressed.

### 9.3 Ablation Components

The codebase supports ablating individual components:

| Configuration | Liquid Cell | Attention | MoE | dt-awareness |
|---|---|---|---|---|
| PPO-LSTM (baseline) | LSTM | No | No | No |
| PPO-LTC | LTC | No | No | Yes |
| PPO-CfC | CfC | No | No | Yes |
| PPO-LFM (no MoE) | CfC | Yes | No | Yes |
| PPO-LFM (full) | CfC | Yes | Yes | Yes |

---

## 10. Conclusion

We present PPO-LTC/LFM, a family of reinforcement learning architectures for dynamic spectrum access that replaces traditional LSTM backbones with Liquid Time-Constant networks and Liquid Foundation Model layers. The key innovation is the integration of continuous-time ODE dynamics with learnable per-neuron time constants into the PPO actor-critic framework, enabling dt-aware temporal modeling that naturally handles the irregular sampling intervals inherent in IoT spectrum sensing. Combined with hierarchical federated learning across a Cloud–Edge–Client architecture, the system provides a scalable, privacy-preserving solution for multi-device spectrum coordination. All code, configurations, and evaluation scripts are open-sourced for full reproducibility using the canonical `ncps` library for LTC/CfC implementations.

---

## References

Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). **Liquid Time-constant Networks.** *Proceedings of the AAAI Conference on Artificial Intelligence, 35*(9), 7657-7666. [arXiv:2006.04439](https://arxiv.org/abs/2006.04439)

Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G., & Rus, D. (2022). **Closed-form Continuous-time Neural Networks.** *Nature Machine Intelligence, 4*, 992-1003. [arXiv:2106.13898](https://arxiv.org/abs/2106.13898)

Lechner, M., Hasani, R., Amini, A., Henzinger, T. A., Rus, D., & Grosu, R. (2020). **Neural Circuit Policies Enabling Auditable Autonomy.** *Nature Machine Intelligence, 2*(10), 642-652.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). **Proximal Policy Optimization Algorithms.** [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). **High-Dimensional Continuous Control Using Generalized Advantage Estimation.** *Proceedings of ICLR 2016.* [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)

Wang, C., Wang, S., Li, Y., & Gao, F. (2023). **Dynamic spectrum access for Internet-of-Things with hierarchical federated deep reinforcement learning.** *Ad Hoc Networks, 149*, 103247.

McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data.** *Proceedings of AISTATS 2017.* [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)

Haykin, S. (2005). **Cognitive Radio: Brain-Empowered Wireless Communications.** *IEEE Journal on Selected Areas in Communications, 23*(2), 201-220.

Zhao, Q., & Sadler, B. M. (2007). **A Survey of Dynamic Spectrum Access.** *IEEE Signal Processing Magazine, 24*(3), 79-89.

Yu, Y., Wang, T., & Liew, S. C. (2019). **Deep-Reinforcement Learning Multiple Access for Heterogeneous Wireless Networks.** *IEEE Journal on Selected Areas in Communications, 37*(6), 1277-1290.

Naparstek, O., & Cohen, K. (2019). **Deep Multi-User Reinforcement Learning for Distributed Dynamic Spectrum Access.** *IEEE Transactions on Wireless Communications, 18*(1), 310-323.

Zhuo, H. H., Feng, W., Xu, Q., Yang, Q., & Lin, Y. (2019). **Federated Reinforcement Learning.** [arXiv:1901.08277](https://arxiv.org/abs/1901.08277)

Lim, W. Y. B., et al. (2020). **Federated Learning in Mobile Edge Networks: A Comprehensive Survey.** *IEEE Communications Surveys & Tutorials, 22*(3), 2031-2063.

Liu, L., Zhang, J., Song, S. H., & Letaief, K. B. (2020). **Client-Edge-Cloud Hierarchical Federated Learning.** *Proceedings of ICC 2020.* [arXiv:1905.06641](https://arxiv.org/abs/1905.06641)

Paszke, A., et al. (2019). **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** *Advances in NeurIPS 32.*

Harris, C. R., et al. (2020). **Array Programming with NumPy.** *Nature, 585*, 357-362.

Virtanen, P., et al. (2020). **SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.** *Nature Methods, 17*, 261-272.

Towers, M., et al. (2023). **Gymnasium.** [github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

Liquid AI. (2024). **Liquid Foundation Models.** [liquid.ai/research](https://www.liquid.ai/research)

---

## License

MIT License

## Citation

```bibtex
@software{ppo_ltc_dsa_2024,
  title={{PPO-LTC/LFM} for Dynamic Spectrum Access in Hierarchical
         Federated {IoT} Networks},
  author={Daniel Foo Jun Wei},
  year={2024},
  url={https://github.com/Danielfoojunwei/PPO-LTC-DSA},
  note={Open-source implementation using ncps library for canonical
        LTC/CfC cells}
}

@inproceedings{hasani2021liquid,
  title={Liquid Time-constant Networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander
          and Rus, Daniela and Grosu, Radu},
  booktitle={Proceedings of the AAAI Conference on Artificial
             Intelligence},
  volume={35},
  number={9},
  pages={7657--7666},
  year={2021}
}

@article{hasani2022closed,
  title={Closed-form Continuous-time Neural Networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander
          and Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max
          and Teschl, Gerald and Rus, Daniela},
  journal={Nature Machine Intelligence},
  volume={4},
  pages={992--1003},
  year={2022}
}

@article{wang2023dynamic,
  title={Dynamic spectrum access for {Internet-of-Things} with
         hierarchical federated deep reinforcement learning},
  author={Wang, C. and Wang, S. and Li, Y. and Gao, F.},
  journal={Ad Hoc Networks},
  volume={149},
  pages={103247},
  year={2023}
}
```
