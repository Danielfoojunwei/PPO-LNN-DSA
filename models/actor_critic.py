"""
Actor-Critic Network with Liquid Foundation Models

This module implements the actor-critic architecture for PPO using LFM instead of LSTM.
The architecture is designed for dynamic spectrum access in IoT environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models.lfm_layers import LFMEncoder, AdaptiveLinearOperator


class LFMActorCritic(nn.Module):
    """
    Actor-Critic network using LFM for temporal modeling.

    This replaces the traditional PPO-LSTM architecture with PPO-LFM.

    Architecture:
    - Shared LFM encoder for feature extraction
    - Actor head: Outputs action probabilities for spectrum access decisions
    - Critic head: Outputs state value estimates
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_lfm_layers=2,
        num_heads=4,
        use_moe=False
    ):
        """
        Args:
            state_dim: Dimension of state space (e.g., spectrum measurements)
            action_dim: Dimension of action space (e.g., number of channels)
            hidden_dim: Hidden dimension for LFM layers
            num_lfm_layers: Number of LFM blocks
            num_heads: Number of attention heads in token mixing
            use_moe: Whether to use Mixture of Experts
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared LFM encoder (replaces LSTM backbone)
        self.encoder = LFMEncoder(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_lfm_layers,
            num_heads=num_heads
        )

        # Actor head: Policy network
        self.actor = nn.Sequential(
            AdaptiveLinearOperator(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic head: Value network
        self.critic = nn.Sequential(
            AdaptiveLinearOperator(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Optional: Mixture of Experts for efficiency
        self.use_moe = use_moe
        if use_moe:
            from models.lfm_layers import MixtureOfExperts
            self.moe = MixtureOfExperts(hidden_dim, num_experts=8, num_active=2)

    def forward(self, state, deterministic=False):
        """
        Forward pass through the actor-critic network.

        Args:
            state: State tensor [batch, seq_len, state_dim]
            deterministic: If True, select action deterministically (for evaluation)

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
            entropy: Action distribution entropy
        """
        # Encode state with LFM
        encoded_seq, pooled = self.encoder(state)

        # Apply MoE if enabled
        if self.use_moe:
            pooled_seq = self.moe(encoded_seq)
            pooled = pooled_seq.mean(dim=1)

        # Actor: Compute action probabilities
        action_logits = self.actor(pooled)
        action_dist = Categorical(logits=action_logits)

        # Sample or select deterministic action
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # Critic: Estimate state value
        value = self.critic(pooled).squeeze(-1)

        return action, log_prob, value, entropy

    def evaluate_actions(self, state, action):
        """
        Evaluate actions for PPO training.

        Args:
            state: State tensor [batch, seq_len, state_dim]
            action: Action tensor [batch]

        Returns:
            log_prob: Log probability of actions
            value: State value estimates
            entropy: Action distribution entropy
        """
        # Encode state
        encoded_seq, pooled = self.encoder(state)

        # Apply MoE if enabled
        if self.use_moe:
            pooled_seq = self.moe(encoded_seq)
            pooled = pooled_seq.mean(dim=1)

        # Actor evaluation
        action_logits = self.actor(pooled)
        action_dist = Categorical(logits=action_logits)

        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # Critic evaluation
        value = self.critic(pooled).squeeze(-1)

        return log_prob, value, entropy

    def get_value(self, state):
        """
        Get value estimate for a state (used in advantage computation).

        Args:
            state: State tensor [batch, seq_len, state_dim]

        Returns:
            value: State value estimate
        """
        with torch.no_grad():
            _, pooled = self.encoder(state)

            if self.use_moe:
                # Note: This is a simplified version; full implementation would reuse encoded_seq
                pass

            value = self.critic(pooled).squeeze(-1)

        return value


class MultiAgentLFMActorCritic(nn.Module):
    """
    Multi-agent extension for federated learning scenarios.
    Each agent has a local actor-critic with shared encoder architecture.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        num_agents,
        hidden_dim=256,
        num_lfm_layers=2,
        shared_encoder=True
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            num_agents: Number of agents in the system
            hidden_dim: Hidden dimension for LFM
            num_lfm_layers: Number of LFM layers
            shared_encoder: Whether to share encoder across agents
        """
        super().__init__()
        self.num_agents = num_agents
        self.shared_encoder = shared_encoder

        if shared_encoder:
            # Single shared encoder for all agents
            self.encoder = LFMEncoder(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                num_layers=num_lfm_layers
            )
            self.encoders = None
        else:
            # Separate encoder for each agent
            self.encoder = None
            self.encoders = nn.ModuleList([
                LFMEncoder(
                    input_dim=state_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_lfm_layers
                )
                for _ in range(num_agents)
            ])

        # Separate actor-critic heads for each agent
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim)
            )
            for _ in range(num_agents)
        ])

        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_agents)
        ])

    def forward(self, state, agent_id, deterministic=False):
        """
        Forward pass for a specific agent.

        Args:
            state: State tensor [batch, seq_len, state_dim]
            agent_id: ID of the agent (0 to num_agents-1)
            deterministic: Whether to select actions deterministically

        Returns:
            action, log_prob, value, entropy
        """
        # Encode with shared or agent-specific encoder
        if self.shared_encoder:
            _, pooled = self.encoder(state)
        else:
            _, pooled = self.encoders[agent_id](state)

        # Agent-specific actor
        action_logits = self.actors[agent_id](pooled)
        action_dist = Categorical(logits=action_logits)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # Agent-specific critic
        value = self.critics[agent_id](pooled).squeeze(-1)

        return action, log_prob, value, entropy

    def get_shared_parameters(self):
        """
        Get shared parameters for federated averaging.
        Returns encoder parameters if shared, otherwise all parameters.
        """
        if self.shared_encoder:
            return self.encoder.parameters()
        else:
            return self.parameters()
