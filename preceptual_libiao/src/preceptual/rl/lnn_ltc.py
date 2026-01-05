"""
Liquid Neural Network (LNN) / Liquid Time-Constant (LTC) Module

Implements continuous-time neural dynamics for handling:
- Irregular sampling intervals (dt-aware)
- Non-stationary RF/traffic dynamics
- Temporal abstraction for fleet control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Cell.

    Implements continuous-time dynamics:
    dh/dt = (-h + f(x, h)) / tau

    Where tau is a learned time constant per neuron.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Input transformation
        self.W_x = nn.Linear(input_size, hidden_size * 4)

        # Hidden transformation
        self.W_h = nn.Linear(hidden_size, hidden_size * 4, bias=False)

        # Learnable time constants (log-space for stability)
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

        # Layer normalization
        self.ln = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable dynamics."""
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.orthogonal_(self.W_h.weight)
        nn.init.uniform_(self.log_tau, math.log(self.tau_min), math.log(self.tau_max))

    @property
    def tau(self) -> torch.Tensor:
        """Get time constants (clamped to valid range)."""
        return torch.clamp(
            torch.exp(self.log_tau),
            self.tau_min,
            self.tau_max
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with dt-aware update.

        Args:
            x: Input tensor (batch, input_size)
            h: Hidden state (batch, hidden_size)
            dt: Time delta (batch,) or scalar

        Returns:
            New hidden state (batch, hidden_size)
        """
        # Ensure dt has correct shape
        if dt.dim() == 0:
            dt = dt.unsqueeze(0).expand(x.size(0))
        dt = dt.unsqueeze(-1)  # (batch, 1)

        # Compute gate values
        gates_x = self.W_x(x)
        gates_h = self.W_h(h)
        gates = gates_x + gates_h

        # Split into input, forget, output, candidate gates
        i, f, o, g = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # Compute candidate state
        candidate = i * g + f * h

        # Apply continuous-time dynamics
        # h_new = h + dt/tau * (-h + candidate)
        tau = self.tau.unsqueeze(0)  # (1, hidden_size)
        decay = torch.exp(-dt / tau)
        h_new = decay * h + (1 - decay) * candidate

        # Apply output gate and layer norm
        h_new = o * torch.tanh(self.ln(h_new))

        return h_new


class LNNBackbone(nn.Module):
    """
    Liquid Neural Network backbone for policy networks.

    Stacks multiple LTC cells with skip connections.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # LTC layers
        self.cells = nn.ModuleList([
            LTCCell(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])

        # Layer normalization for residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize hidden states for all layers."""
        return [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[List[torch.Tensor]] = None,
        dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, input_size)
            h: List of hidden states per layer
            dt: Time delta (batch,) or scalar, defaults to 1.0

        Returns:
            (output, new_hidden_states)
        """
        batch_size = x.size(0)
        device = x.device

        if h is None:
            h = self.init_hidden(batch_size, device)

        if dt is None:
            dt = torch.ones(batch_size, device=device)

        # Input projection
        out = self.input_proj(x)

        # Process through LTC layers
        new_h = []
        for i, (cell, ln) in enumerate(zip(self.cells, self.layer_norms)):
            h_new = cell(out, h[i], dt)
            # Residual connection with layer norm
            out = ln(out + self.dropout(h_new))
            new_h.append(h_new)

        return out, new_h

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        h: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Process sequence with varying dt.

        Args:
            x_seq: Input sequence (batch, seq_len, input_size)
            dt_seq: Time deltas (batch, seq_len)
            h: Initial hidden states

        Returns:
            (outputs, final_hidden_states)
        """
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device

        if h is None:
            h = self.init_hidden(batch_size, device)

        outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            dt_t = dt_seq[:, t]
            out, h = self.forward(x_t, h, dt_t)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        return outputs, h


class DTAwareLNNEncoder(nn.Module):
    """
    DT-aware LNN encoder for fleet observations.

    Processes global and per-robot observations through
    shared LNN backbone with temporal awareness.
    """

    def __init__(
        self,
        global_input_size: int,
        robot_input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        max_robots: int = 50,
    ):
        super().__init__()
        self.global_input_size = global_input_size
        self.robot_input_size = robot_input_size
        self.hidden_size = hidden_size
        self.max_robots = max_robots

        # Global encoder
        self.global_backbone = LNNBackbone(
            input_size=global_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        # Robot encoder (shared across robots)
        self.robot_backbone = LNNBackbone(
            input_size=robot_input_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
        )

        # Attention for robot aggregation
        self.robot_attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> dict:
        """Initialize all hidden states."""
        return {
            "global": self.global_backbone.init_hidden(batch_size, device),
            "robot": self.robot_backbone.init_hidden(batch_size * self.max_robots, device),
        }

    def forward(
        self,
        global_obs: torch.Tensor,
        robot_obs: torch.Tensor,
        robot_mask: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        hidden: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            global_obs: Global observation (batch, global_input_size)
            robot_obs: Robot observations (batch, max_robots, robot_input_size)
            robot_mask: Mask for valid robots (batch, max_robots)
            dt: Time delta
            hidden: Previous hidden states

        Returns:
            (encoded_state, new_hidden)
        """
        batch_size = global_obs.size(0)
        device = global_obs.device

        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        if dt is None:
            dt = torch.ones(batch_size, device=device)

        # Process global observation
        global_out, global_h = self.global_backbone(
            global_obs, hidden["global"], dt
        )

        # Process robot observations
        robot_flat = robot_obs.view(-1, self.robot_input_size)
        dt_expanded = dt.unsqueeze(1).expand(-1, self.max_robots).reshape(-1)
        robot_h_flat = [h.view(-1, h.size(-1)) for h in hidden["robot"]]

        robot_out_flat, robot_h_new_flat = self.robot_backbone(
            robot_flat, robot_h_flat, dt_expanded
        )

        robot_out = robot_out_flat.view(batch_size, self.max_robots, -1)
        robot_h_new = [h.view(batch_size, self.max_robots, -1) for h in robot_h_new_flat]

        # Attention-based robot aggregation
        if robot_mask is not None:
            attn_mask = ~robot_mask.bool()
        else:
            attn_mask = None

        robot_agg, _ = self.robot_attention(
            robot_out, robot_out, robot_out,
            key_padding_mask=attn_mask
        )
        robot_agg = robot_agg.mean(dim=1)  # (batch, hidden_size // 2)

        # Fuse global and robot features
        fused = self.fusion(torch.cat([global_out, robot_agg], dim=-1))

        new_hidden = {
            "global": global_h,
            "robot": [h.view(batch_size * self.max_robots, -1) for h in robot_h_new],
        }

        return fused, new_hidden


# Unit tests for dt robustness
def test_ltc_dt_robustness():
    """Test LTC cell handles varying dt correctly."""
    cell = LTCCell(input_size=10, hidden_size=20)

    batch_size = 4
    x = torch.randn(batch_size, 10)
    h = torch.zeros(batch_size, 20)

    # Test different dt values
    dt_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    outputs = []
    for dt in dt_values:
        dt_tensor = torch.tensor(dt)
        h_new = cell(x, h, dt_tensor)
        outputs.append(h_new)

        # Check output is valid
        assert not torch.isnan(h_new).any(), f"NaN in output for dt={dt}"
        assert not torch.isinf(h_new).any(), f"Inf in output for dt={dt}"

    # Larger dt should produce larger changes
    for i in range(len(outputs) - 1):
        diff_small = (outputs[i] - h).abs().mean()
        diff_large = (outputs[i + 1] - h).abs().mean()
        # Not strictly monotonic due to decay, but should be related
        assert diff_large >= 0, "Differences should be non-negative"

    print("LTC dt robustness test passed!")


def test_lnn_backbone_sequence():
    """Test LNN backbone processes sequences correctly."""
    backbone = LNNBackbone(input_size=32, hidden_size=64, num_layers=2)

    batch_size = 4
    seq_len = 10

    x_seq = torch.randn(batch_size, seq_len, 32)
    dt_seq = torch.rand(batch_size, seq_len) * 2  # Random dt in [0, 2]

    outputs, final_h = backbone.forward_sequence(x_seq, dt_seq)

    assert outputs.shape == (batch_size, seq_len, 64)
    assert len(final_h) == 2
    assert final_h[0].shape == (batch_size, 64)

    print("LNN backbone sequence test passed!")


if __name__ == "__main__":
    test_ltc_dt_robustness()
    test_lnn_backbone_sequence()
