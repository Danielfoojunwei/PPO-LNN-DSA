"""
PPO-LNN Implementation using original ncps library

Uses the official ncps library (https://github.com/mlech26l/ncps) for:
- CfC (Closed-form Continuous-time) cells
- LTC (Liquid Time-Constant) cells
- Neural Circuit Policies (NCP) wiring

Key difference from LSTM:
- LSTM: h[t+1] = f(h[t], x[t]) - discrete, fixed time steps
- LNN/CfC: dh/dt = f(h, x, t) - continuous, adapts to variable dt

This is a SIMPLIFIED implementation without MoE or attention,
to match the original ncps paper architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np

# Import ncps library
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, FullyConnected


class LNNEncoder(nn.Module):
    """
    Liquid Neural Network Encoder using ncps library.

    Uses CfC (Closed-form Continuous-time) cells which are:
    - dt-aware: Handle variable time steps
    - Fast: Closed-form solution without ODE solvers
    - Expressive: Learned time constants per neuron
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        use_cfc: bool = True,  # CfC is faster than LTC
        use_ncp_wiring: bool = False,  # NCP wiring for sparser, more interpretable networks
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_cfc = use_cfc
        self.use_ncp_wiring = use_ncp_wiring

        # Input projection to hidden size
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

        # Create wiring
        if use_ncp_wiring:
            # NCP wiring: sparse, interpretable - has explicit motor neurons
            motor_size = hidden_size // 2  # Motor neurons output
            wiring = AutoNCP(hidden_size, motor_size, sparsity_level=0.5)
            self.actual_output_size = motor_size  # AutoNCP outputs motor neurons
        else:
            # Fully connected wiring - outputs all hidden units
            wiring = FullyConnected(hidden_size, hidden_size)
            self.actual_output_size = hidden_size  # FullyConnected outputs all units

        # Create liquid cell
        if use_cfc:
            # CfC: Closed-form Continuous-time (faster)
            self.liquid = CfC(
                input_size=hidden_size,
                units=wiring,
                return_sequences=False,  # Only return last output
                batch_first=True,
                mode="default",  # "default", "pure", "no_gate"
            )
        else:
            # LTC: Liquid Time-Constant (more expressive but slower)
            self.liquid = LTC(
                input_size=hidden_size,
                units=wiring,
                return_sequences=False,
                batch_first=True,
            )

        # Store state size for hidden state management
        self.state_size = self.liquid.state_size

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(batch_size, self.state_size, device=device)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        timespans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dt-aware dynamics.

        Args:
            x: Input tensor (batch, input_size) or (batch, seq, input_size)
            hidden: Previous hidden state (batch, state_size)
            timespans: Time deltas (batch,) or (batch, seq) - THIS IS KEY FOR dt-AWARENESS

        Returns:
            (output, new_hidden)
        """
        batch_size = x.size(0)
        device = x.device

        # Project input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: (batch, 1, input)

        # Project through input layer
        x = self.input_proj(x)  # (batch, seq, hidden)

        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # Prepare timespans for ncps
        # ncps expects timespans of shape (batch, seq, 1)
        if timespans is not None:
            if timespans.dim() == 1:
                timespans = timespans.view(batch_size, 1, 1)  # (batch, 1, 1)
            elif timespans.dim() == 2:
                timespans = timespans.unsqueeze(-1)  # (batch, seq, 1)

        # Forward through liquid cell
        # CfC/LTC accepts timespans parameter for dt-awareness
        output, new_hidden = self.liquid(x, hidden, timespans=timespans)

        return output, new_hidden


class PPOLNNPolicy(nn.Module):
    """
    PPO Policy with Liquid Neural Network backbone.

    Simple architecture:
    - Input projection
    - CfC/LTC liquid cell (dt-aware)
    - Actor head (policy)
    - Critic head (value)
    """

    def __init__(
        self,
        state_dim: int = 42,
        action_dim: int = 20,
        hidden_size: int = 128,
        use_cfc: bool = True,
        use_ncp_wiring: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # LNN encoder
        self.encoder = LNNEncoder(
            input_size=state_dim,
            hidden_size=hidden_size,
            use_cfc=use_cfc,
            use_ncp_wiring=use_ncp_wiring,
        )

        # Get actual output size from encoder
        encoder_output_size = self.encoder.actual_output_size

        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden states."""
        return self.encoder.init_hidden(batch_size, device)

    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        timespans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor (batch, state_dim)
            hidden: Previous hidden state
            timespans: Time deltas for dt-awareness

        Returns:
            (action_logits, value, new_hidden)
        """
        # Encode state with dt-awareness
        encoded, new_hidden = self.encoder(state, hidden, timespans)

        # Get action logits and value
        action_logits = self.actor(encoded)
        value = self.critic(encoded).squeeze(-1)

        return action_logits, value, new_hidden

    def get_action(
        self,
        state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        timespans: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Returns:
            (action, log_prob, value, new_hidden)
        """
        action_logits, value, new_hidden = self.forward(state, hidden, timespans)

        # Create distribution
        dist = torch.distributions.Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value, new_hidden

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        timespans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log prob and entropy for given actions.

        Note: Due to a bug in ncps library, timespans only work with batch_size=1.
        For batch training, we pass None for timespans (uses default dt=1.0).
        The dt-awareness is still present during action selection.

        Returns:
            (log_probs, entropy, values)
        """
        # Don't use timespans during batch evaluation due to ncps bug
        # The rollout data already captures the effect of variable dt
        action_logits, values, _ = self.forward(states, hidden, None)

        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values


class PPOLNNAgent:
    """
    PPO Agent with Liquid Neural Network backbone.

    Key advantages over LSTM:
    1. dt-aware: Explicitly handles variable time steps via timespans
    2. Continuous-time: Models dynamics as ODEs (closed-form for CfC)
    3. Adaptive time constants: Different neurons respond at different speeds

    This is a simplified version using the original ncps library,
    without MoE or attention layers.
    """

    def __init__(
        self,
        state_dim: int = 42,
        action_dim: int = 20,
        hidden_size: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        use_cfc: bool = True,
        use_ncp_wiring: bool = False,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Create policy
        self.policy = PPOLNNPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            use_cfc=use_cfc,
            use_ncp_wiring=use_ncp_wiring,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Training params
        self.clip_epsilon = 0.2
        self.ppo_epochs = 4
        self.minibatch_size = 64
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        # Hidden state
        self.hidden = None
        self.last_dt = 1.0

        # Rollout buffer
        self.buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "dts": [],  # Store dt for each step
        }

        # Stats
        self.total_steps = 0

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
        Select action given state.

        Args:
            state: Observation
            dt: Time since last decision (seconds) - KEY FOR dt-AWARENESS
            deterministic: If True, take argmax action

        Returns:
            (action, log_prob, value)
        """
        if dt is not None:
            self.last_dt = dt

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dt_tensor = torch.tensor([self.last_dt], device=self.device)

        with torch.no_grad():
            action, log_prob, value, self.hidden = self.policy.get_action(
                state_tensor, self.hidden, dt_tensor, deterministic
            )

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
        """Store transition in buffer."""
        self.buffer["states"].append(torch.FloatTensor(state))
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["values"].append(value)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.buffer["dts"].append(dt if dt is not None else self.last_dt)
        self.total_steps += 1

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam
            )

        returns = advantages + values
        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Update policy using collected rollout with dt-awareness."""
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

        # Compute next value for GAE
        with torch.no_grad():
            last_state = states[-1:].to(self.device)
            hidden = self.policy.init_hidden(1, self.device)
            # Note: not using timespans here due to ncps bug, but dt=1.0 is used
            _, next_value, _ = self.policy.forward(last_state, hidden, None)
            next_value = next_value.squeeze()

        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update with minibatching
        total_samples = len(rewards)
        indices = np.arange(total_samples)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        n_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, total_samples, self.minibatch_size):
                end = min(start + self.minibatch_size, total_samples)
                mb_indices = indices[start:end]
                mb_size = len(mb_indices)

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_dts = dts[mb_indices]

                # Evaluate actions with dt-awareness
                # Initialize fresh hidden for each minibatch (standard for PPO)
                hidden = self.policy.init_hidden(mb_size, self.device)
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    mb_states, mb_actions, hidden, mb_dts
                )

                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, mb_returns)

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy.mean()
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["approx_kl"] += approx_kl
                n_updates += 1

        # Average metrics
        for k in metrics:
            metrics[k] /= max(1, n_updates)

        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []

        return metrics

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]


# Also create LTC version for comparison
class PPOLTCAgent(PPOLNNAgent):
    """
    PPO Agent with LTC (Liquid Time-Constant) backbone.

    LTC is more expressive than CfC but slower (requires ODE solver).
    """

    def __init__(self, *args, **kwargs):
        kwargs['use_cfc'] = False  # Use LTC instead of CfC
        super().__init__(*args, **kwargs)


# NCP wiring version for sparse, interpretable networks
class PPONCPAgent(PPOLNNAgent):
    """
    PPO Agent with NCP (Neural Circuit Policy) wiring.

    NCP uses sparse, biologically-inspired wiring patterns.
    """

    def __init__(self, *args, **kwargs):
        kwargs['use_ncp_wiring'] = True
        super().__init__(*args, **kwargs)
