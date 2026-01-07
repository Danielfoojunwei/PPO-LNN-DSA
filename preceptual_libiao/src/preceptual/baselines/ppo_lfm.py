"""
Liquid Foundation Model (LFM) Implementation for PPO

Based on Liquid.ai's architecture and the ncps library:
- https://github.com/mlech26l/ncps (official LTC/CfC implementation)
- https://github.com/kyegomez/LFM (open source LFM)
- Paper: "Liquid Time-constant Networks" (https://arxiv.org/abs/2006.04439)
- Paper: "Closed-form Continuous-time Neural Networks" (https://arxiv.org/abs/2106.13898)

Key features:
1. Liquid Time-Constant (LTC) cells with adaptive time constants
2. dt-aware dynamics - time step affects hidden state evolution
3. Closed-form Continuous-time (CfC) for faster inference
4. Neural Circuit Policy (NCP) inspired wiring

The key advantage over LSTM:
- LSTM: h[t+1] = f(h[t], x[t]) - discrete, fixed time steps
- LTC: dh/dt = f(h, x, t) - continuous, adapts to variable dt
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class LTCCell(nn.Module):
    """
    Liquid Time-Constant Cell.

    Implements the ODE: τ * dh/dt = -h + f(x, h)

    Where τ (tau) is a learnable time-constant that adapts based on input.
    This is the core building block of Liquid Neural Networks.

    The key insight: the time-constant τ controls how fast the neuron
    responds to changes. Different τ values = different temporal scales.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input transformation
        self.W_in = nn.Linear(input_size, hidden_size, bias=use_bias)

        # Recurrent transformation (hidden-to-hidden)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Learnable time constants (τ) - one per hidden unit
        # Initialized to reasonable range for stability
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

        # Learnable activation mixing (for expressiveness)
        self.W_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=use_bias)

        # Layer normalization for stability
        self.ln = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with dt-aware dynamics.

        The ODE is solved using exponential Euler integration:
        h_new = h * exp(-dt/τ) + (1 - exp(-dt/τ)) * candidate

        This is more stable than explicit Euler and respects the
        continuous-time nature of the system.

        Args:
            x: Input tensor (batch, input_size)
            h: Hidden state (batch, hidden_size)
            dt: Time delta (batch,) or scalar

        Returns:
            New hidden state (batch, hidden_size)
        """
        batch_size = x.size(0)

        # Ensure dt has right shape
        if dt.dim() == 0:
            dt = dt.expand(batch_size)
        dt = dt.view(batch_size, 1)  # (batch, 1) for broadcasting

        # Get time constants (ensure positive via exp)
        tau = torch.exp(self.log_tau).unsqueeze(0)  # (1, hidden_size)
        tau = tau.clamp(min=0.1, max=100.0)  # Stability bounds

        # Compute decay factor: exp(-dt/τ)
        # Larger dt or smaller τ = more decay = faster response
        decay = torch.exp(-dt / tau)  # (batch, hidden_size)

        # Compute candidate activation
        x_proj = self.W_in(x)  # (batch, hidden_size)
        h_proj = self.W_h(h)  # (batch, hidden_size)
        candidate = torch.tanh(x_proj + h_proj)

        # Gating mechanism (like GRU but continuous)
        gate_input = torch.cat([x, h], dim=-1)
        gate = torch.sigmoid(self.W_gate(gate_input))

        # Apply gated candidate
        gated_candidate = gate * candidate + (1 - gate) * h

        # Exponential Euler integration
        # h_new = h * decay + (1 - decay) * gated_candidate
        h_new = decay * h + (1 - decay) * gated_candidate

        # Layer norm for stability
        h_new = self.ln(h_new)

        return h_new


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time Cell.

    A faster alternative to LTC that provides a closed-form solution
    without requiring ODE solvers. Based on the paper:
    "Closed-form Continuous-time Neural Networks" (Hasani et al., 2022)

    Key advantage: 100x faster than neural ODEs while preserving
    the continuous-time dynamics and dt-awareness.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: str = "default",  # "default", "pure", "no_gate"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode

        # Backbone network
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.SiLU(),  # Smooth activation
        )

        # Time-dependent components
        self.ff1 = nn.Linear(hidden_size, hidden_size)
        self.ff2 = nn.Linear(hidden_size, hidden_size)

        # Time constants
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

        # Layer norm
        self.ln = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Closed-form forward pass.

        Uses the closed-form solution:
        h_new = sigmoid(-t/τ) * f(x,h) + (1 - sigmoid(-t/τ)) * g(x,h)

        This approximates the ODE solution without numerical integration.
        """
        batch_size = x.size(0)

        if dt.dim() == 0:
            dt = dt.expand(batch_size)
        dt = dt.view(batch_size, 1)

        # Get time constants
        tau = torch.exp(self.log_tau).unsqueeze(0).clamp(min=0.1, max=100.0)

        # Time-dependent interpolation factor
        t_interp = torch.sigmoid(-dt / tau)

        # Compute backbone features
        concat = torch.cat([x, h], dim=-1)
        features = self.backbone(concat)

        # Two branches for interpolation
        f_out = self.ff1(features)
        g_out = self.ff2(features)

        # Closed-form interpolation
        h_new = t_interp * torch.tanh(f_out) + (1 - t_interp) * torch.tanh(g_out)

        return self.ln(h_new)


class LFMBlock(nn.Module):
    """
    Liquid Foundation Model Block.

    Combines:
    1. LTC/CfC recurrent dynamics (dt-aware)
    2. Self-attention for global context
    3. Mixture of Experts for adaptive computation

    This is inspired by Liquid.ai's architecture which combines
    liquid neural networks with transformer-style attention.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 4,
        num_experts: int = 4,
        use_cfc: bool = True,  # CfC is faster than LTC
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Liquid cell (dt-aware)
        if use_cfc:
            self.liquid_cell = CfCCell(input_size, hidden_size)
        else:
            self.liquid_cell = LTCCell(input_size, hidden_size)

        # Self-attention for global context
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Mixture of Experts FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )
            for _ in range(num_experts)
        ])

        # Router for MoE
        self.router = nn.Linear(hidden_size, num_experts)

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LFM block.

        Args:
            x: Input (batch, input_size)
            h: Hidden state (batch, hidden_size)
            dt: Time delta (batch,)

        Returns:
            (output, new_hidden)
        """
        # 1. Liquid dynamics (dt-aware)
        h_new = self.liquid_cell(x, h, dt)
        h_new = self.ln1(h_new)

        # 2. Self-attention (for sequence context if needed)
        h_seq = h_new.unsqueeze(1)  # Add sequence dim
        attn_out, _ = self.attention(h_seq, h_seq, h_seq)
        attn_out = attn_out.squeeze(1)
        h_new = h_new + self.dropout(attn_out)
        h_new = self.ln2(h_new)

        # 3. Mixture of Experts
        router_logits = self.router(h_new)  # (batch, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (batch, num_experts)

        # Compute weighted sum of expert outputs
        expert_outputs = torch.stack([
            expert(h_new) for expert in self.experts
        ], dim=1)  # (batch, num_experts, hidden)

        # Weighted combination: (batch, num_experts) @ (batch, num_experts, hidden) -> (batch, hidden)
        router_probs_expanded = router_probs.unsqueeze(-1)  # (batch, num_experts, 1)
        moe_out = (expert_outputs * router_probs_expanded).sum(dim=1)  # (batch, hidden)

        h_new = h_new + self.dropout(moe_out)
        h_new = self.ln3(h_new)

        return h_new, h_new


class LFMEncoder(nn.Module):
    """
    Liquid Foundation Model Encoder.

    Stacks multiple LFM blocks with dt-aware dynamics.
    This is the backbone for PPO-LNN.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_experts: int = 4,
        use_cfc: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # LFM blocks
        self.blocks = nn.ModuleList([
            LFMBlock(
                input_size=hidden_size if i > 0 else hidden_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_experts=num_experts,
                use_cfc=use_cfc,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize hidden states for all layers."""
        return [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with dt-aware dynamics.

        Args:
            x: Input (batch, input_size)
            hidden: List of hidden states per layer
            dt: Time delta (batch,) - THIS IS THE KEY DIFFERENCE FROM LSTM

        Returns:
            (output, new_hidden_states)
        """
        batch_size = x.size(0)
        device = x.device

        # Default dt if not provided
        if dt is None:
            dt = torch.ones(batch_size, device=device)

        # Initialize hidden if needed
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # Project input
        out = self.input_proj(x)

        # Pass through blocks
        new_hidden = []
        for i, block in enumerate(self.blocks):
            out, h_new = block(out, hidden[i], dt)
            new_hidden.append(h_new)

        # Output projection
        out = self.output_proj(out)

        return out, new_hidden


class PPOLFMAgent:
    """
    PPO Agent with Liquid Foundation Model backbone.

    Key advantages over PPO-LSTM:
    1. dt-aware: Adapts to variable time steps between decisions
    2. Continuous-time: Models dynamics as ODEs, not discrete steps
    3. Adaptive time constants: Different neurons respond at different speeds
    4. MoE: Adaptive computation based on input complexity
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.95,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.gamma = gamma

        # LFM encoder (dt-aware)
        self.encoder = LFMEncoder(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            use_cfc=True,  # Faster than LTC
        ).to(self.device)

        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        ).to(self.device)

        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(self.device)

        # Optimizer
        params = (
            list(self.encoder.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Training params
        self.clip_epsilon = 0.2
        self.ppo_epochs = 4

        # State
        self.hidden = None
        self.last_dt = 1.0

        # Buffer
        self.buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "dts": [],  # Store dt for each step
        }

        self.total_steps = 0

        # Metrics for dt-awareness analysis
        self.tau_history = []  # Track time constants over time

    def reset_hidden(self):
        """Reset hidden state for new episode."""
        self.hidden = None

    def set_dt(self, dt: float):
        """Set time delta for next forward pass."""
        self.last_dt = dt

    def select_action(
        self,
        state: np.ndarray,
        dt: Optional[float] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select action with dt-aware dynamics.

        Args:
            state: Observation
            dt: Time since last decision (seconds)
            deterministic: If True, take argmax action

        Returns:
            (action, log_prob, value)
        """
        if dt is not None:
            self.last_dt = dt

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dt_tensor = torch.tensor([self.last_dt], device=self.device)

        with torch.no_grad():
            # Forward through LFM encoder with dt
            encoded, self.hidden = self.encoder(state_tensor, self.hidden, dt_tensor)

            # Get action distribution
            logits = self.actor(encoded)
            value = self.critic(encoded).squeeze(-1)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        dt: Optional[float] = None,
    ):
        """Store transition with dt information."""
        self.buffer["states"].append(torch.FloatTensor(state))
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["values"].append(value)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.buffer["dts"].append(dt if dt is not None else self.last_dt)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        """Update policy with dt-aware rollouts."""
        if len(self.buffer["states"]) < 32:
            return {"loss": 0.0}

        # Convert to tensors
        states = torch.stack(self.buffer["states"]).to(self.device)
        actions = torch.tensor(self.buffer["actions"]).to(self.device)
        old_log_probs = torch.tensor(self.buffer["log_probs"]).to(self.device)
        values = torch.tensor(self.buffer["values"]).to(self.device)
        rewards = torch.tensor(self.buffer["rewards"], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.buffer["dones"], dtype=torch.float32).to(self.device)
        dts = torch.tensor(self.buffer["dts"], dtype=torch.float32).to(self.device)

        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

            next_val = values[t + 1] if t + 1 < len(values) else 0
            td_error = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            running_advantage = td_error + self.gamma * 0.95 * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update with dt
        total_loss = 0

        for _ in range(self.ppo_epochs):
            # Batch processing with per-sample dt values
            # Key insight: We don't need sequential hidden state for PPO training
            # Each transition is treated independently but with its own dt
            hidden = self.encoder.init_hidden(len(states), self.device)

            # Forward pass with batch of states and batch of dts
            # Each sample in the batch gets its own dt value
            encoded, _ = self.encoder(states, hidden, dts)

            logits = self.actor(encoded)
            new_values = self.critic(encoded).squeeze(-1)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values, returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.actor.parameters()) +
                list(self.critic.parameters()),
                0.5
            )
            self.optimizer.step()

            total_loss += loss.item()

        # Record tau statistics for analysis
        with torch.no_grad():
            for block in self.encoder.blocks:
                tau = torch.exp(block.liquid_cell.log_tau).cpu().numpy()
                self.tau_history.append({
                    "mean": float(tau.mean()),
                    "std": float(tau.std()),
                    "min": float(tau.min()),
                    "max": float(tau.max()),
                })

        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []

        return {"loss": total_loss / self.ppo_epochs}

    def get_tau_statistics(self) -> Dict[str, float]:
        """Get learned time constant statistics."""
        if not self.tau_history:
            return {}

        recent = self.tau_history[-10:]  # Last 10 updates
        return {
            "tau_mean": np.mean([t["mean"] for t in recent]),
            "tau_std": np.mean([t["std"] for t in recent]),
            "tau_min": np.mean([t["min"] for t in recent]),
            "tau_max": np.mean([t["max"] for t in recent]),
        }
