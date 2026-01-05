# PPO-LNN Design Document

## Overview

The PPO-LNN (Proximal Policy Optimization with Liquid Neural Network) is the learning component of Preceptual.ai. It uses a continuous-time neural dynamics backbone to handle irregular sampling and non-stationary RF/traffic dynamics.

## Why LNN/LTC?

Traditional RNNs (LSTM, GRU) assume fixed time steps between observations. In real-world fleet control:
- Control ticks may vary (200-1000ms)
- Network latency causes irregular sampling
- Robot movements create non-stationary dynamics

Liquid Time-Constant (LTC) cells explicitly model time:
```
dh/dt = (-h + f(x, h)) / tau
```

Where `tau` is a learned time constant per neuron, allowing the network to:
- Adapt to varying observation intervals
- Learn appropriate time scales for different dynamics
- Handle missing or delayed observations gracefully

## Architecture

### LTC Cell

```python
class LTCCell(nn.Module):
    def forward(self, x, h, dt):
        # Compute candidate state
        candidate = i * g + f * h

        # Apply continuous-time dynamics
        decay = exp(-dt / tau)
        h_new = decay * h + (1 - decay) * candidate

        return h_new
```

Key features:
- Learnable per-neuron time constants (tau)
- dt-aware update ensures consistent behavior across different sampling rates
- Layer normalization for stability

### LNN Backbone

Stacks multiple LTC cells with:
- Input projection layer
- Skip connections between layers
- Layer normalization
- Dropout for regularization

### Hierarchical Policy

```
                    ┌─────────────────┐
                    │  Fleet State    │
                    │  Observation    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   LNN Encoder   │
                    │  (DT-Aware)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
       │ Global Head │ │Robot Heads│ │ Value Head  │
       │  - Slices   │ │(Top-K)    │ │             │
       │  - Mode     │ │ - HOLD    │ │ V(s)        │
       │  - Scans    │ │ - ARM     │ │             │
       └─────────────┘ │ - EXECUTE │ └─────────────┘
                       └───────────┘
```

## Observations

### Global Observation (42 dims)
- Fleet metrics: n_robots, n_devices, n_channels, dt
- Device health: avg_rtt, avg_loss, avg_timeout, online_ratio
- Slice QoS: drops, latency
- Load balance: Jain index

### Per-Robot Observation (14 dims)
- Position: x, y (normalized)
- Signal: rssi, rssi_slope
- Channel assignment
- Switching: time_since_last, recent_count
- Density: same_ap, same_channel, same_zone
- QoS: latency_proxy, loss_proxy
- Risk scores: switch_risk, contention_risk

## Actions

### Global Actions
- **Slice Budgets** [6 continuous]: Budget allocation for slices A-F
- **Congestion Mode** [4 categorical]: NORMAL, PROTECT_CONTROL, ROAM_RECOVERY, INCIDENT_CONTAINMENT
- **Scan Quota** [1 continuous]: Fraction of robots to scan this tick

### Per-Robot Actions (Top-K)
- **Action Type** [3 categorical]: HOLD, ARM_SWITCH, EXECUTE_SWITCH
- **Target Candidate** [K categorical]: Which candidate AP/channel

## Training

### PPO Algorithm
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Separate value and policy networks (shared encoder)

### Reward Function

```python
reward = (
    + stability_bonus           # A/B/C slice health
    + balance_bonus * jain      # Load distribution
    - switch_penalty * n_switches
    - scan_penalty * n_scans
    - safety_violation_penalty  # If hotspots occur
    + fairness_bonus            # E/F utilization
)
```

### Hyperparameters
- Learning rate: 3e-4
- Gamma (discount): 0.99
- GAE lambda: 0.95
- Clip epsilon: 0.2
- Value coefficient: 0.5
- Entropy coefficient: 0.01
- PPO epochs: 4
- Minibatch size: 64

## DT Robustness

The LNN backbone is tested for dt robustness:
- Varying dt from 0.01s to 5.0s produces valid outputs
- No NaN or Inf values
- Consistent behavior across sampling rates

## Integration with Safety Envelope

PPO outputs are ALWAYS filtered through deterministic safety constraints:
1. Check minimum dwell time
2. Check switch rate limits
3. Check slice budget minima
4. Force PROTECT_CONTROL if needed
5. Throttle as required

The safety envelope acts as a hard constraint layer that PPO learns to respect.
