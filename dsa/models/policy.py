"""The shared actor-critic wrapper.  Identical for every model in the registry.

Spec section 1.7::

    obs (B, obs_dim), dt (B,)
      u0      = tanh( Linear(obs_dim + 1, H) @ [obs ; dt] )   # every model sees dt as a FEATURE
      y1, s1' = block1(u0, s1, dt)
      u1      = LayerNorm_1(y1)
      y2, s2' = block2(u1, s2, dt)
      z       = LayerNorm_2(y2)
      logits  = Linear(H, action_dim)( tanh( Linear(H, H)(z) ) )
      value   = Linear(H, 1)( tanh( Linear(H, H)(z) ) ).squeeze(-1)

Depth is exactly two blocks for every model, no exceptions -- that is what makes the
ablation single-factor (defect D11).  The two paths by which ``dt`` can reach a model
are (a) the encoder feature, available to *all* models, and (b) the block's internal
state update, only for dt-aware blocks.

Hidden state is **carried across environment steps** (spec section 1.2).  There is no
frame stack.  ``initial_state(batch)`` is both the constructor of a fresh state and
the episode-boundary reset.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

from .cells import CELL_REGISTRY

__all__ = ["RecurrentActorCritic", "deterministic_init_"]


def deterministic_init_(module: nn.Module, generator: torch.Generator) -> None:
    """Re-initialise every parameter of ``module`` from ``generator``.

    Spec section 9.2 "deterministic init recipe".  Traversal is ``nn.Module.modules()``
    pre-order DFS, which is a fixed order for a fixed module tree, so the sequence of
    draws -- and therefore the resulting weights -- is a pure function of
    ``(module structure, generator seed)``.  The global RNG is never consulted.
    """
    handled: set[int] = set()
    for m in module.modules():
        if id(m) in handled:
            continue
        with torch.no_grad():
            if isinstance(m, nn.MultiheadAttention):
                bound = 1.0 / math.sqrt(m.embed_dim)
                if m.in_proj_weight is not None:
                    m.in_proj_weight.uniform_(-bound, bound, generator=generator)
                else:  # pragma: no cover - not reachable with equal q/k/v dims
                    for w in (m.q_proj_weight, m.k_proj_weight, m.v_proj_weight):
                        w.uniform_(-bound, bound, generator=generator)
                if m.in_proj_bias is not None:
                    m.in_proj_bias.zero_()
                if getattr(m, "bias_k", None) is not None:
                    m.bias_k.zero_()
                    m.bias_v.zero_()
                m.out_proj.weight.uniform_(-bound, bound, generator=generator)
                if m.out_proj.bias is not None:
                    m.out_proj.bias.zero_()
                handled.add(id(m.out_proj))
            elif isinstance(m, nn.Linear):
                bound = 1.0 / math.sqrt(m.in_features)
                m.weight.uniform_(-bound, bound, generator=generator)
                if m.bias is not None:
                    m.bias.uniform_(-bound, bound, generator=generator)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    m.weight.fill_(1.0)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, (nn.GRUCell, nn.LSTMCell)):
                bound = 1.0 / math.sqrt(m.hidden_size)
                for p in m.parameters():
                    p.uniform_(-bound, bound, generator=generator)

    # Cell-specific initialisation (LTC log_tau / A, CfC b_tau, attention rel_bias).
    for m in module.modules():
        hook = getattr(m, "custom_init_", None)
        if callable(hook):
            hook(generator)


class RecurrentActorCritic(nn.Module):
    """Two-block recurrent actor-critic.  Frozen signature, spec section 9.2."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cell_kinds: tuple[str, str],
        hidden_dim: int,
        generator: torch.Generator | None = None,
        cell_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        cell_kinds = tuple(cell_kinds)
        if len(cell_kinds) != 2:
            raise ValueError(f"every model has exactly 2 blocks; got {cell_kinds!r}")
        unknown = [k for k in cell_kinds if k not in CELL_REGISTRY]
        if unknown:
            raise KeyError(f"unknown cell kind(s): {unknown!r}")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.cell_kinds = cell_kinds
        self.cell_kwargs = dict(cell_kwargs or {})

        # (a) dt as an input feature -- identical for every model.
        self.encoder = nn.Linear(obs_dim + 1, hidden_dim)
        # (b) dt inside the state update -- only dt-aware blocks use it.
        self.blocks = nn.ModuleList(
            [
                CELL_REGISTRY[kind](hidden_dim, hidden_dim, **self.cell_kwargs)
                for kind in cell_kinds
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in cell_kinds])
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        if generator is not None:
            deterministic_init_(self, generator)

    # -- state ---------------------------------------------------------------
    @property
    def state_sizes(self) -> tuple[int, ...]:
        return tuple(int(b.state_size) for b in self.blocks)

    @property
    def dt_aware_blocks(self) -> tuple[bool, ...]:
        return tuple(bool(getattr(b, "dt_aware", False)) for b in self.blocks)

    def initial_state(self, batch: int) -> tuple[torch.Tensor, ...]:
        """Zero state for each block.  Also the episode-boundary reset."""
        ref = self.encoder.weight
        return tuple(
            b.initial_state(batch, device=ref.device, dtype=ref.dtype) for b in self.blocks
        )

    # -- forward -------------------------------------------------------------
    def _encode(self, obs: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(torch.cat([obs, dt.unsqueeze(-1)], dim=-1)))

    def _heads(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(z), self.critic(z).squeeze(-1)

    def step(
        self, obs: torch.Tensor, dt: torch.Tensor, state: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """``obs (B, obs_dim)``, ``dt (B,)`` -> ``logits (B, A)``, ``value (B,)``, new state."""
        u = self._encode(obs, dt)
        new_state = []
        for block, norm, s in zip(self.blocks, self.norms, state):
            y, s_next = block(u, s, dt)
            u = norm(y)
            new_state.append(s_next)
        logits, value = self._heads(u)
        return logits, value, tuple(new_state)

    def unroll(
        self, obs: torch.Tensor, dt: torch.Tensor, state: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """``obs (B, T, obs_dim)``, ``dt (B, T)`` -> ``logits (B, T, A)``, ``values (B, T)``.

        Numerically identical to ``T`` successive :meth:`step` calls.  Recomputes the
        entire hidden trajectory from ``state`` with no ``detach`` -- this is the PPO-RNN
        update contract (spec section 4.4 rule 3).
        """
        u = self._encode(obs, dt)
        new_state = []
        for block, norm, s in zip(self.blocks, self.norms, state):
            y, s_next = block.unroll(u, s, dt)
            u = norm(y)
            new_state.append(s_next)
        logits, values = self._heads(u)
        return logits, values, tuple(new_state)

    # -- convenience ---------------------------------------------------------
    def act(
        self,
        obs: torch.Tensor,
        dt: torch.Tensor,
        state: tuple[torch.Tensor, ...],
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return ``(action, log_prob, value, new_state)``.

        With ``deterministic=True`` the action is ``argmax(logits)`` and the call is a
        pure function of the weights and inputs -- no RNG is touched at all.  With
        ``deterministic=False`` sampling uses the explicitly supplied ``generator``,
        never the global RNG (spec section 4.4 rule 8).

        Defect D10 was caused by dropout being live during rollout, which made two
        successive ``deterministic=True`` calls disagree.  This method never changes
        the module's train/eval mode; the caller is responsible for that, and
        ``tests/test_registry.py::test_deterministic_act`` pins the behaviour in
        ``eval()`` mode for every registry key.
        """
        logits, value, new_state = self.step(obs, dt, state)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            probs = dist.probs
            action = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        return action, dist.log_prob(action), value, new_state
