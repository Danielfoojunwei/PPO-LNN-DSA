"""Model definitions for the empirical DSA benchmark suite."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelSpec:
    model_type: str
    hidden_dim: int = 64
    seq_model_dim: int = 64
    num_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.1


class LTCCell(nn.Module):
    """A lightweight dt-aware liquid time-constant cell."""

    def __init__(self, input_dim: int, hidden_dim: int, tau_min: float = 0.1, tau_max: float = 10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.w_x = nn.Linear(input_dim, hidden_dim * 4)
        self.w_h = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(hidden_dim))
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        gates = self.w_x(x) + self.w_h(h)
        i, f, o, g = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        candidate = i * g + f * h
        tau = torch.clamp(torch.exp(self.log_tau), self.tau_min, self.tau_max).unsqueeze(0)
        decay = torch.exp(-dt / tau)
        h_new = decay * h + (1.0 - decay) * candidate
        h_new = o * torch.tanh(self.ln(h_new))
        return h_new


class SequenceBackbone(nn.Module):
    """Shared feature extractor for all sequence-based baselines."""

    def __init__(self, obs_dim: int, spec: ModelSpec):
        super().__init__()
        self.model_type = spec.model_type
        self.obs_dim = obs_dim
        self.hidden_dim = spec.hidden_dim
        input_dim = obs_dim + 1  # append dt feature

        if spec.model_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, spec.hidden_dim),
                nn.Tanh(),
                nn.Linear(spec.hidden_dim, spec.hidden_dim),
                nn.Tanh(),
            )
        elif spec.model_type == "lstm":
            self.input_proj = nn.Linear(input_dim, spec.hidden_dim)
            self.rnn = nn.LSTM(
                input_size=spec.hidden_dim,
                hidden_size=spec.hidden_dim,
                num_layers=spec.num_layers,
                batch_first=True,
                dropout=spec.dropout if spec.num_layers > 1 else 0.0,
            )
        elif spec.model_type == "gru":
            self.input_proj = nn.Linear(input_dim, spec.hidden_dim)
            self.rnn = nn.GRU(
                input_size=spec.hidden_dim,
                hidden_size=spec.hidden_dim,
                num_layers=spec.num_layers,
                batch_first=True,
                dropout=spec.dropout if spec.num_layers > 1 else 0.0,
            )
        elif spec.model_type == "ltc":
            self.input_proj = nn.Linear(input_dim, spec.hidden_dim)
            self.ltc_layers = nn.ModuleList(
                [LTCCell(spec.hidden_dim, spec.hidden_dim) for _ in range(spec.num_layers)]
            )
            self.ltc_norms = nn.ModuleList([nn.LayerNorm(spec.hidden_dim) for _ in range(spec.num_layers)])
        elif spec.model_type == "lfm":
            self.input_proj = nn.Linear(input_dim, spec.hidden_dim)
            self.encoder_layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "attn": nn.MultiheadAttention(
                                embed_dim=spec.hidden_dim,
                                num_heads=spec.num_heads,
                                dropout=spec.dropout,
                                batch_first=True,
                            ),
                            "ff": nn.Sequential(
                                nn.Linear(spec.hidden_dim, spec.hidden_dim * 2),
                                nn.GELU(),
                                nn.Dropout(spec.dropout),
                                nn.Linear(spec.hidden_dim * 2, spec.hidden_dim),
                            ),
                            "ln1": nn.LayerNorm(spec.hidden_dim),
                            "ln2": nn.LayerNorm(spec.hidden_dim),
                        }
                    )
                    for _ in range(spec.num_layers)
                ]
            )
        elif spec.model_type == "ltc_lfm":
            self.input_proj = nn.Linear(input_dim, spec.hidden_dim)
            self.ltc_layers = nn.ModuleList(
                [LTCCell(spec.hidden_dim, spec.hidden_dim) for _ in range(spec.num_layers)]
            )
            self.ltc_norms = nn.ModuleList([nn.LayerNorm(spec.hidden_dim) for _ in range(spec.num_layers)])
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=spec.hidden_dim,
                num_heads=spec.num_heads,
                dropout=spec.dropout,
                batch_first=True,
            )
            self.ff = nn.Sequential(
                nn.Linear(spec.hidden_dim, spec.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(spec.dropout),
                nn.Linear(spec.hidden_dim * 2, spec.hidden_dim),
            )
            self.ln1 = nn.LayerNorm(spec.hidden_dim)
            self.ln2 = nn.LayerNorm(spec.hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {spec.model_type}")

    def forward(self, obs: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, dt.unsqueeze(-1)], dim=-1)
        if self.model_type == "mlp":
            pooled = self.mlp(x).reshape(x.shape[0], -1)
            return pooled
        if self.model_type in {"lstm", "gru"}:
            x = self.input_proj(x)
            out, _ = self.rnn(x)
            return out[:, -1, :]
        if self.model_type == "ltc":
            x = self.input_proj(x)
            batch_size, seq_len, hidden_dim = x.shape
            h_states = [x.new_zeros(batch_size, hidden_dim) for _ in range(len(self.ltc_layers))]
            outputs = []
            for t in range(seq_len):
                out_t = x[:, t, :]
                dt_t = dt[:, t]
                for i, cell in enumerate(self.ltc_layers):
                    h_states[i] = cell(out_t, h_states[i], dt_t)
                    out_t = self.ltc_norms[i](out_t + h_states[i])
                outputs.append(out_t)
            return torch.stack(outputs, dim=1)[:, -1, :]
        if self.model_type == "lfm":
            x = self.input_proj(x)
            for layer in self.encoder_layers:
                y = layer["ln1"](x)
                attn_out, _ = layer["attn"](y, y, y, need_weights=False)
                x = x + attn_out
                y = layer["ln2"](x)
                x = x + layer["ff"](y)
            return x.mean(dim=1)
        if self.model_type == "ltc_lfm":
            x = self.input_proj(x)
            batch_size, seq_len, hidden_dim = x.shape
            h_states = [x.new_zeros(batch_size, hidden_dim) for _ in range(len(self.ltc_layers))]
            temporal = []
            for t in range(seq_len):
                out_t = x[:, t, :]
                dt_t = dt[:, t]
                for i, cell in enumerate(self.ltc_layers):
                    h_states[i] = cell(out_t, h_states[i], dt_t)
                    out_t = self.ltc_norms[i](out_t + h_states[i])
                temporal.append(out_t)
            temporal = torch.stack(temporal, dim=1)
            y = self.ln1(temporal)
            attn_out, _ = self.temporal_attn(y, y, y, need_weights=False)
            temporal = temporal + attn_out
            temporal = temporal + self.ff(self.ln2(temporal))
            return temporal.mean(dim=1)
        raise RuntimeError("Unreachable backbone branch")


class ActorCriticModel(nn.Module):
    """Unified actor-critic model for all empirical PPO baselines."""

    def __init__(self, obs_dim: int, seq_len: int, action_dim: int, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.action_dim = action_dim

        self.backbone = SequenceBackbone(obs_dim=obs_dim, spec=spec)
        backbone_out = spec.hidden_dim if spec.model_type != "mlp" else seq_len * spec.hidden_dim

        self.actor = nn.Sequential(
            nn.Linear(backbone_out, spec.hidden_dim),
            nn.Tanh(),
            nn.Linear(spec.hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(backbone_out, spec.hidden_dim),
            nn.Tanh(),
            nn.Linear(spec.hidden_dim, 1),
        )

    def encode(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.backbone(state["obs"], state["dt"])

    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(state)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

    def act(self, state: Dict[str, torch.Tensor], deterministic: bool = False):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy

    def evaluate_actions(self, state: Dict[str, torch.Tensor], actions: torch.Tensor):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value


MODEL_REGISTRY = {
    "ppo_mlp": ModelSpec(model_type="mlp", hidden_dim=64, num_layers=1),
    "ppo_lstm": ModelSpec(model_type="lstm", hidden_dim=64, num_layers=1),
    "ppo_gru": ModelSpec(model_type="gru", hidden_dim=64, num_layers=1),
    "ppo_ltc": ModelSpec(model_type="ltc", hidden_dim=64, num_layers=1),
    "ppo_lfm": ModelSpec(model_type="lfm", hidden_dim=64, num_layers=2, num_heads=4),
    "ppo_ltc_lfm": ModelSpec(model_type="ltc_lfm", hidden_dim=64, num_layers=1, num_heads=4),
}


def build_model(model_name: str, obs_dim: int, seq_len: int, action_dim: int) -> ActorCriticModel:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {model_name}")
    return ActorCriticModel(obs_dim=obs_dim, seq_len=seq_len, action_dim=action_dim, spec=MODEL_REGISTRY[model_name])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
