# Preceptual.ai (Libiao Edition) - Airtime OS + Handover Orchestrator

## Overview

Preceptual.ai (Libiao Edition) is an **Airtime OS + Handover Orchestrator** designed for Libiao robot fleets (50→300+ robots) operating in large warehouses with severe airtime contention.

This is NOT "roaming optimization." The product delivers step-function reliability at 300 robots by controlling:

1. **Airtime budgets** (per robot + per traffic slice)
2. **Scan/measurement scheduling** (avoid scan storms)
3. **Congestion modes** (deterministic degradation policies)
4. **Handover orchestration** (predictive + verified switching)
5. **LB-AP / LBAP-102LU-900 aware load balancing** (hotspot prevention)

## Architecture

### Three Control Planes

#### PLANE 1 — Airtime OS (Real-time Allocator)
Controls scarce airtime via:
- Per-slice budgets (traffic classes A-F)
- Per-robot weights (fairness/priority)
- Admission control + deterministic congestion modes

#### PLANE 2 — Handover Orchestrator (Predictive Switching)
Controls:
- Per-robot link assignment (AP IP + channel) via RCS
- Verified switching with cooldown and ping-pong prevention
- Hotspot avoidance by moving robots before collapse

#### PLANE 3 — Learning (PPO-LNN + FL)
Learns site-specific contention and RF dynamics to optimize:
- Slice budgets
- Scan scheduling
- Switching policy
- Congestion mode triggers and recovery

### Multi-Link Support

The platform treats "network control" as multi-link:
- **LinkType A: LBAP** (900MHz, UDP gateway devices, centrally managed)
- **LinkType B: Wi-Fi** (2.4/5GHz, if present in site)

Unified objective: allocate scarce airtime + manage link switching reliably.

## Traffic Slices

| Slice | Priority | Purpose | Target Latency | Max Drop Rate |
|-------|----------|---------|----------------|---------------|
| A | 6 (highest) | Critical control (emergency, safety) | 5ms | 0.1% |
| B | 5 | Real-time control (motion commands) | 10ms | 1% |
| C | 4 | Telemetry/status (periodic reporting) | 50ms | 2% |
| D | 3 | Task coordination (job assignments) | 100ms | 5% |
| E | 2 | Bulk data (logs, diagnostics) | 500ms | 10% |
| F | 1 (lowest) | Best effort (updates, optional) | 1000ms | 20% |

## Congestion Modes

### NORMAL
- All features enabled
- Standard slice budgets
- Full switching capability

### PROTECT_CONTROL
- Freeze switching
- Throttle best-effort (E/F) slices
- Boost critical (A/B) slice budgets
- Triggered by: control plane degradation, critical slice drops

### ROAM_RECOVERY
- Allow only top-K high-risk robot moves
- Moderate throttling of non-critical slices
- Triggered by: high switch failure rate

### INCIDENT_CONTAINMENT
- Cap misbehaving robots
- Aggressive throttling
- Isolate problem areas
- Triggered by: hotspot collapse

## PPO-LNN Policy

### Architecture
- **Backbone**: Liquid Neural Network (LNN) with dt-aware LTC cells
- **Global Head**: Outputs slice budgets, congestion mode, scan quotas
- **Robot Heads**: Per-robot switching decisions for top-K risk robots

### Control Outputs
Every tick (200-1000ms), PPO outputs:
- Airtime budgets: slice budgets b = [bA..bF] and robot weights
- Scan scheduling: subset of robots to scan, aggressiveness
- Congestion mode: NORMAL / PROTECT_CONTROL / ROAM_RECOVERY / INCIDENT_CONTAINMENT
- Handover actions: HOLD / ARM_SWITCH / EXECUTE_SWITCH

### Reward Function
- **Positive**: Stable control (low A/B/C drops), balanced load
- **Negative**: Switch failures, ping-pong, excessive switching, scan overhead
- **Large negative**: Safety envelope triggers
- **Fairness term**: Jain index for best-effort slices

## Safety Envelope

Hard constraints that override PPO outputs:
- Minimum dwell time between switches
- Maximum switch rate per robot
- Never reduce slice A/B below reserved minima
- Automatic PROTECT_CONTROL on control-plane degradation
- Automatic throttle E/F on congestion detection
- Freeze switching except emergency ROAM_RECOVERY for high-risk robots

## Deployment

### Topology
```
RCS Server (or RCS cluster)
       |
       | (wired)
       v
SWITCH (PoE capable for LBAP devices)
       |---> LBAP-102LU-900 devices (x1..n) [PoE + Ethernet]
       |           |
       |           | (900MHz wireless to robots)
       |           v
       |        Robots (AGV/Sorting robots)
       |
       |---> (optional) Wi-Fi APs
       |
Preceptual.ai Edge Server
```

### Integration
- Polls RCS robot state via `/com-api/query-robots`
- Monitors LBAP devices via UDP
- Executes decisions via `/com-api/set-robot`
- Runs PPO-LNN inference at 2-5 Hz

## Acceptance Criteria

At 300 robots:
- Critical slices remain stable (A/B/C protected; E/F degrade first)
- Hotspot collapse is prevented or recovered quickly
- Scan storms are actively prevented
- Switching is verified with reduced ping-pong
- System runs end-to-end using rcs_mock + lbap simulator
