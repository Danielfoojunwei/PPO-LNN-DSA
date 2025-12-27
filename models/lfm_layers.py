"""
Liquid Foundation Model (LFM) Layers

This module implements the core LFM components that replace LSTM in the PPO architecture:
- Adaptive Linear Operators: Context-aware transformations
- Token Mixing: Sequential dependency modeling
- Channel Mixing: Feature transformation
- LFM Block: Complete LFM layer replacing LSTM cells
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveLinearOperator(nn.Module):
    """
    Adaptive Linear Operator that adjusts computation based on input context.
    Replaces static weight matrices with context-dependent transformations.
    """
    def __init__(self, input_dim, output_dim, context_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim or input_dim

        # Static component (base transformation)
        self.static_weight = nn.Parameter(torch.randn(output_dim, input_dim) / math.sqrt(input_dim))
        self.static_bias = nn.Parameter(torch.zeros(output_dim))

        # Adaptive component (context-dependent modulation)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, output_dim),
            nn.Tanh()
        )

        # Gating mechanism to blend static and adaptive
        self.gate = nn.Sequential(
            nn.Linear(self.context_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, context=None):
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim] or [batch, input_dim]
            context: Context tensor for adaptation (defaults to x if None)
        Returns:
            Adaptively transformed output
        """
        if context is None:
            context = x

        # Static transformation
        static_out = F.linear(x, self.static_weight, self.static_bias)

        # Adaptive modulation based on context
        # Use mean pooling if sequence dimension exists
        if len(context.shape) == 3:
            context_pooled = context.mean(dim=1)  # [batch, context_dim]
        else:
            context_pooled = context

        adaptive_scale = self.context_encoder(context_pooled)
        gate_value = self.gate(context_pooled)

        # Apply adaptive scaling
        if len(x.shape) == 3:
            adaptive_scale = adaptive_scale.unsqueeze(1)  # [batch, 1, output_dim]
            gate_value = gate_value.unsqueeze(1)

        # Blend static and adaptive transformations
        output = static_out * (1 + gate_value * adaptive_scale)

        return output


class TokenMixing(nn.Module):
    """
    Token Mixing layer that models dependencies across sequence positions.
    This replaces LSTM's sequential processing with parallel attention-like mixing.
    """
    def __init__(self, hidden_dim, seq_len=None, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Multi-head token mixing
        self.query = AdaptiveLinearOperator(hidden_dim, hidden_dim)
        self.key = AdaptiveLinearOperator(hidden_dim, hidden_dim)
        self.value = AdaptiveLinearOperator(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
        Returns:
            Token-mixed output [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V with adaptive operators
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        return output


class ChannelMixing(nn.Module):
    """
    Channel Mixing layer that transforms features across channels.
    Uses adaptive operators for context-aware feature transformation.
    """
    def __init__(self, hidden_dim, expansion_factor=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = hidden_dim * expansion_factor

        # Two-layer MLP with adaptive operators
        self.fc1 = AdaptiveLinearOperator(hidden_dim, self.intermediate_dim)
        self.fc2 = AdaptiveLinearOperator(self.intermediate_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
        Returns:
            Channel-mixed output [batch, seq_len, hidden_dim]
        """
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x, context=residual)  # Use residual as context
        x = self.dropout(x)
        return x


class LFMBlock(nn.Module):
    """
    Complete LFM Block that replaces an LSTM layer.
    Combines Token Mixing and Channel Mixing with residual connections.
    """
    def __init__(self, hidden_dim, num_heads=4, expansion_factor=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token mixing (replaces LSTM's temporal modeling)
        self.token_mixing = TokenMixing(hidden_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Channel mixing (replaces LSTM's feature transformation)
        self.channel_mixing = ChannelMixing(hidden_dim, expansion_factor=expansion_factor)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
        Returns:
            Transformed output [batch, seq_len, hidden_dim]
        """
        # Token mixing with pre-norm and residual
        x = x + self.token_mixing(self.norm1(x))

        # Channel mixing with pre-norm and residual
        x = x + self.channel_mixing(self.norm2(x))

        return x


class LFMEncoder(nn.Module):
    """
    Multi-layer LFM Encoder that replaces stacked LSTM layers.
    This is the main component used in the PPO-LFM actor-critic architecture.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of LFM blocks
        self.layers = nn.ModuleList([
            LFMBlock(hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]
        Returns:
            encoded: Encoded sequence [batch, seq_len, hidden_dim]
            pooled: Pooled representation [batch, hidden_dim]
        """
        # Project input to hidden dimension
        x = self.input_proj(x)

        # Pass through LFM blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        encoded = self.norm(x)

        # Global pooling for sequence representation (replaces LSTM final hidden state)
        pooled = encoded.mean(dim=1)

        return encoded, pooled


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) layer for efficient computation.
    Dynamically selects a subset of experts based on input.
    Optional component for scaling efficiency.
    """
    def __init__(self, hidden_dim, num_experts=8, num_active=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_active = num_active

        # Router network
        self.router = nn.Linear(hidden_dim, num_experts)

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
        Returns:
            Expert-mixed output [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Compute routing scores
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.num_active, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize

        # Compute expert outputs
        output = torch.zeros_like(x)
        for i in range(self.num_active):
            expert_idx = top_k_indices[..., i]  # [batch, seq_len]
            expert_weight = top_k_probs[..., i].unsqueeze(-1)  # [batch, seq_len, 1]

            # Apply corresponding expert (simplified - in practice would batch this)
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    expert_id = expert_idx[batch_idx, seq_idx].item()
                    expert_out = self.experts[expert_id](x[batch_idx:batch_idx+1, seq_idx:seq_idx+1])
                    output[batch_idx, seq_idx] += expert_weight[batch_idx, seq_idx] * expert_out.squeeze()

        return output
