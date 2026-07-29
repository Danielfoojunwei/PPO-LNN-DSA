"""Recurrent blocks used by :class:`dsa.models.policy.RecurrentActorCritic`.

Every block implements the same contract (spec section 9.2)::

    state_size : int
    initial_state(batch, device, dtype) -> Tensor (batch, state_size)
    forward(u, state, dt)               -> (output (B, hidden_dim), new_state (B, state_size))
    unroll(u_seq, state, dt_seq)        -> (outputs (B, T, hidden_dim), final_state)

``unroll`` is required to be numerically identical to ``T`` successive ``forward``
calls.  The default implementation *is* that loop; blocks that override it
(``MLPBlock``, ``CausalAttentionBlock``) are covered by
``tests/test_cells.py::test_block_step_unroll_equivalence`` and
``tests/test_registry.py::test_step_unroll_equivalence``.

Naming honesty (spec section 1.1, defects D5/D6): the block that a previous version
of this repository called "LFM" was a stock pre-norm transformer.  It is called
``CausalAttentionBlock`` here and is registered as ``"attn"``.  The
liquid-foundation primitive is the closed-form continuous-time cell
(:class:`CfCCell`, registered as ``"cfc"``).
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from torch import nn

__all__ = [
    "RecurrentBlock",
    "RecurrentBlockBase",
    "MLPBlock",
    "GRUBlock",
    "LSTMBlock",
    "CausalAttentionBlock",
    "LTCCell",
    "CfCCell",
    "CfCDtBlindCell",
    "CELL_REGISTRY",
    "ATTENTION_WINDOW",
    "DEFAULT_LTC_UNFOLDS",
    "DEFAULT_TAU_MIN",
    "DEFAULT_TAU_MAX",
]

#: Width of the banded-causal attention window (spec section 1.5).  Changing this
#: invalidates the throughput budget and must be recorded in ``results/manifest.json``.
ATTENTION_WINDOW = 8

#: Number of fused-Euler sub-steps in the LTC solver (spec section 1.3).
DEFAULT_LTC_UNFOLDS = 6

#: Log-spaced time-constant initialisation range for the LTC (spec section 1.3).
DEFAULT_TAU_MIN = 0.5
DEFAULT_TAU_MAX = 8.0


@runtime_checkable
class RecurrentBlock(Protocol):
    """Structural type every registered block satisfies."""

    input_dim: int
    hidden_dim: int
    state_size: int

    def initial_state(self, batch: int, device=None, dtype=None) -> torch.Tensor: ...

    def forward(
        self, u: torch.Tensor, state: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class RecurrentBlockBase(nn.Module):
    """Shared plumbing: zero initial state and a reference ``unroll`` loop."""

    #: Set by subclasses.  ``0`` means the block is memoryless.
    state_size: int = 0
    #: ``True`` only for blocks in which ``dt`` enters the state update itself.
    dt_aware: bool = False

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

    # -- state ----------------------------------------------------------------
    def initial_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        """Zero state of shape ``(batch, state_size)``.

        This is also the episode-boundary reset: the learner calls it at the top of
        every rollout (spec section 4.4 rule 1).
        """
        return torch.zeros(
            int(batch),
            self.state_size,
            device=device,
            dtype=torch.float32 if dtype is None else dtype,
        )

    # -- sequence -------------------------------------------------------------
    def unroll(
        self, u_seq: torch.Tensor, state: torch.Tensor, dt_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference implementation: ``T`` successive :meth:`forward` calls.

        No ``detach`` anywhere -- the whole hidden trajectory stays in the graph
        (spec section 4.4 rule 3).
        """
        outputs = []
        for t in range(u_seq.shape[1]):
            y, state = self.forward(u_seq[:, t], state, dt_seq[:, t])
            outputs.append(y)
        return torch.stack(outputs, dim=1), state


# ---------------------------------------------------------------------------
# Honest baselines
# ---------------------------------------------------------------------------
class MLPBlock(RecurrentBlockBase):
    """Memoryless block: ``Linear -> Tanh -> Linear -> Tanh``.  ``state_size == 0``."""

    def __init__(self, input_dim: int, hidden_dim: int, **_: object) -> None:
        super().__init__(input_dim, hidden_dim)
        self.state_size = 0
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, u, state, dt):
        return torch.tanh(self.fc2(torch.tanh(self.fc1(u)))), state

    def unroll(self, u_seq, state, dt_seq):
        # Exactly equivalent to the loop because the block carries no state.
        return torch.tanh(self.fc2(torch.tanh(self.fc1(u_seq)))), state


class GRUBlock(RecurrentBlockBase):
    """``nn.GRUCell``.  ``dt`` never enters the state update."""

    def __init__(self, input_dim: int, hidden_dim: int, **_: object) -> None:
        super().__init__(input_dim, hidden_dim)
        self.state_size = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, u, state, dt):
        h = self.cell(u, state)
        return h, h


class LSTMBlock(RecurrentBlockBase):
    """``nn.LSTMCell``.  State is ``cat([h, c], -1)`` so ``state_size == 2H``."""

    def __init__(self, input_dim: int, hidden_dim: int, **_: object) -> None:
        super().__init__(input_dim, hidden_dim)
        self.state_size = 2 * hidden_dim
        self.cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, u, state, dt):
        h_prev, c_prev = torch.split(state, self.hidden_dim, dim=-1)
        h, c = self.cell(u, (h_prev.contiguous(), c_prev.contiguous()))
        return h, torch.cat([h, c], dim=-1)


class CausalAttentionBlock(RecurrentBlockBase):
    """A transformer block.  Called what it is (spec section 1.1, defect D6).

    Pre-norm self-attention over a **banded-causal window** of the last ``K``
    block inputs, followed by a pre-norm ``Linear(H, 2H) -> GELU -> Linear(2H, H)``
    feed-forward, both with residual connections.  Relative position enters as a
    learned additive attention bias indexed by the distance to the query, so the
    block is position-aware while remaining exactly reproducible in both ``step``
    and ``unroll`` mode.

    State is the ring buffer of the last ``K`` inputs (most-recent-first) plus a
    fill counter, hence ``state_size = K * H + 1``.  The buffer only ever holds
    *past and present* inputs, which is what
    ``tests/test_cells.py::test_attention_is_causal`` and
    ``tests/test_registry.py::test_no_model_sees_the_future`` verify: the
    transformer baseline receives exactly the same information budget as the
    recurrent models, never more.

    ``dt`` is accepted and ignored -- this block is not dt-aware.  ``dt`` still
    reaches it through the shared encoder feature (spec section 1.7 path (a)).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        window: int = ATTENTION_WINDOW,
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__(input_dim, hidden_dim)
        if input_dim != hidden_dim:
            raise ValueError(
                "CausalAttentionBlock needs input_dim == hidden_dim for its residual "
                f"path; got {input_dim} != {hidden_dim}"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {hidden_dim} is not divisible by num_heads {num_heads}"
            )
        self.num_heads = int(num_heads)
        self.window = int(window)
        self.dropout_p = float(dropout)
        self.state_size = self.window * hidden_dim + 1

        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, self.num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        # Additive relative-position bias, indexed by distance-to-query 0..K-1.
        self.rel_bias = nn.Parameter(torch.zeros(self.num_heads, self.window))

    # -- helpers --------------------------------------------------------------
    def _split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, h = self.window, self.hidden_dim
        buffer = state[:, : k * h].reshape(state.shape[0], k, h)
        fill = state[:, k * h]
        return buffer, fill

    def _pack_state(self, buffer: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
        return torch.cat([buffer.reshape(buffer.shape[0], -1), fill.unsqueeze(-1)], dim=-1)

    def _apply_ff(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm_ff(x))

    # -- single step ----------------------------------------------------------
    def forward(self, u, state, dt):
        b = u.shape[0]
        k = self.window
        buffer, fill = self._split_state(state)
        # push the current input to the front (most-recent-first ordering)
        buffer = torch.cat([u.unsqueeze(1), buffer[:, : k - 1]], dim=1)
        fill = torch.clamp(fill + 1.0, max=float(k))

        normed = self.norm_attn(buffer)                       # (B, K, H)
        query = normed[:, :1]                                 # (B, 1, H)

        distance = torch.arange(k, device=u.device)           # (K,) distance to query
        valid = distance.unsqueeze(0) < fill.unsqueeze(1)     # (B, K)
        bias = self.rel_bias.unsqueeze(0).expand(b, -1, -1)   # (B, heads, K)
        mask = bias.masked_fill(~valid.unsqueeze(1), float("-inf"))
        mask = mask.reshape(b * self.num_heads, 1, k)

        attended, _ = self.attn(query, normed, normed, attn_mask=mask, need_weights=False)
        h = self._apply_ff(u + attended.squeeze(1))
        return h, self._pack_state(buffer, fill)

    # -- sequence -------------------------------------------------------------
    def unroll(self, u_seq, state, dt_seq):
        b, t_len, _ = u_seq.shape
        k = self.window
        buffer, fill = self._split_state(state)
        # buffer is most-recent-first; flip to chronological order for the prefix
        prefix = buffer.flip(1)                               # (B, K, H)
        seq = torch.cat([prefix, u_seq], dim=1)               # (B, K + T, H)
        normed = self.norm_attn(seq)
        query = normed[:, k:]                                 # (B, T, H)

        device = u_seq.device
        key_index = torch.arange(k + t_len, device=device).unsqueeze(0)      # (1, K+T)
        query_abs = (torch.arange(t_len, device=device) + k).unsqueeze(1)    # (T, 1)
        distance = query_abs - key_index                                     # (T, K+T)
        in_window = (distance >= 0) & (distance < k)

        prefix_pos = torch.arange(k, device=device).unsqueeze(0)             # (1, K)
        prefix_valid = prefix_pos >= (k - fill).unsqueeze(1)                 # (B, K)
        key_valid = torch.cat(
            [prefix_valid, torch.ones(b, t_len, dtype=torch.bool, device=device)], dim=1
        )                                                                    # (B, K+T)
        allowed = in_window.unsqueeze(0) & key_valid.unsqueeze(1)            # (B, T, K+T)

        bias = self.rel_bias[:, distance.clamp(0, k - 1)]                    # (heads, T, K+T)
        mask = bias.unsqueeze(0).masked_fill(~allowed.unsqueeze(1), float("-inf"))
        mask = mask.reshape(b * self.num_heads, t_len, k + t_len)

        attended, _ = self.attn(query, normed, normed, attn_mask=mask, need_weights=False)
        h = self._apply_ff(u_seq + attended)

        new_buffer = seq[:, -k:].flip(1)
        new_fill = torch.clamp(fill + float(t_len), max=float(k))
        return h, self._pack_state(new_buffer, new_fill)


# ---------------------------------------------------------------------------
# Liquid cells
# ---------------------------------------------------------------------------
class LTCCell(RecurrentBlockBase):
    r"""Liquid Time-constant network cell (Hasani et al., AAAI 2021).

    The ODE (paper eq. 1), elementwise over the ``H`` hidden units::

        dx/dt = -[ 1/tau + f(x, I, t, theta) ] * x(t)  +  f(x, I, t, theta) * A
        f(x, I, t, theta) = sigmoid( W_x @ I(t) + W_h @ x(t) + b )

    whose **defining property** is the input-dependent system time constant
    (paper eq. 5)::

        tau_sys = tau / ( 1 + tau * f(x, I, t, theta) )

    ``tau_sys`` is a function of the input and of the state.  A cell whose
    effective decay is a learned constant independent of ``(x, I)`` is a sigmoid
    forget gate, not an LTC -- that was defect D5, and
    ``tests/test_cells.py::test_ltc_time_constant_is_input_dependent`` fails
    against such a cell.  :meth:`system_time_constant` exposes the quantity so
    the test can measure it directly.

    The integrator is the **fused / semi-implicit Euler solver** of paper eq. 6.
    With ``K = unfolds`` sub-steps of size ``h = dt / K``::

        for k in range(K):
            f_k = sigmoid( W_x @ u + W_h @ x + b )
            x   = ( x + h * f_k * A ) / ( 1 + h * ( 1/tau + f_k ) )

    ``dt`` enters *only* through ``h``, multiplying both the numerator input term
    and the denominator decay term.  That is the entire continuous-time content of
    the cell, and it gives two properties the tests check: the state change goes to
    zero as ``dt -> 0``, and the update is unconditionally stable for large ``dt``
    (the semi-implicit denominator grows with ``h``).

    ``u`` is held constant across the ``K`` sub-steps.  Output is ``x`` after the
    sub-steps -- no extra activation, no output gate; normalisation lives in the
    policy wrapper (spec section 1.7).

    Initialisation: ``log_tau = linspace(log(tau_min), log(tau_max), H)`` for
    genuine log-spaced timescale diversity across units, and ``A ~ N(0, 0.1^2)``
    drawn from the model's explicit generator.
    """

    dt_aware = True

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        unfolds: int = DEFAULT_LTC_UNFOLDS,
        tau_min: float = DEFAULT_TAU_MIN,
        tau_max: float = DEFAULT_TAU_MAX,
        **_: object,
    ) -> None:
        super().__init__(input_dim, hidden_dim)
        if unfolds < 1:
            raise ValueError("unfolds must be >= 1")
        if not 0.0 < tau_min < tau_max:
            raise ValueError("require 0 < tau_min < tau_max")
        self.state_size = hidden_dim
        self.unfolds = int(unfolds)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        # W_x and b
        self.input_map = nn.Linear(input_dim, hidden_dim, bias=True)
        # W_h  (the bias of the pre-activation lives in input_map)
        self.recurrent_map = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(hidden_dim))
        self.A = nn.Parameter(torch.zeros(hidden_dim))

    # -- exposed quantities ---------------------------------------------------
    def tau(self) -> torch.Tensor:
        """The learned per-unit time constants ``tau = exp(log_tau)``, shape ``(H,)``."""
        return torch.exp(self.log_tau)

    def synaptic_activation(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``f(x, I, t, theta) = sigmoid(W_x @ I + W_h @ x + b)``, shape ``(B, H)``."""
        return torch.sigmoid(self.input_map(u) + self.recurrent_map(x))

    def system_time_constant(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``tau_sys = tau / (1 + tau * f(x, I, t, theta))`` (paper eq. 5), ``(B, H)``.

        Depends on both the input ``u`` and the state ``x``.  This is the property
        that distinguishes an LTC from a gated RNN.
        """
        tau = self.tau()
        return tau / (1.0 + tau * self.synaptic_activation(u, x))

    # -- dynamics -------------------------------------------------------------
    def forward(self, u, state, dt):
        x = state
        pre_input = self.input_map(u)                      # constant across sub-steps
        step_h = (dt / self.unfolds).unsqueeze(-1)         # (B, 1)
        inv_tau = torch.exp(-self.log_tau)                 # 1 / tau, shape (H,)
        # Loop-invariant terms hoisted out of the sub-step loop.  Algebraically
        # identical to  x = (x + h * f * A) / (1 + h * (1/tau + f))  -- only the
        # association of the float multiplications changes.
        h_times_a = step_h * self.A                        # (B, H)
        denom_base = 1.0 + step_h * inv_tau                # (B, H)
        for _ in range(self.unfolds):
            f = torch.sigmoid(pre_input + self.recurrent_map(x))
            x = torch.addcmul(x, h_times_a, f) / torch.addcmul(denom_base, step_h, f)
        return x, x

    def custom_init_(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            self.log_tau.copy_(
                torch.linspace(
                    math.log(self.tau_min), math.log(self.tau_max), self.hidden_dim
                )
            )
            self.A.copy_(
                torch.randn(self.hidden_dim, generator=generator, dtype=self.A.dtype) * 0.1
            )


class CfCCell(RecurrentBlockBase):
    r"""Closed-form Continuous-time cell (Hasani et al., Nature MI 2022).

    This is the liquid-foundation primitive.  The "LFM" in the repository title
    maps here, **not** to attention (spec section 1.1, defect D6).

    The closed-form solution (paper eq. 10)::

        x(t) = sigmoid(-f(x, I) * t) * g(x, I)  +  [1 - sigmoid(-f(x, I) * t)] * h(x, I)

    implemented per decision step with ``t := dt`` and the shared backbone of the
    official cell::

        z     = tanh( W_bb @ [x_prev ; u] + b_bb )      # shared backbone, one layer
        g     = tanh( W_g @ z + b_g )                   # initial-condition branch
        h     = tanh( W_h @ z + b_h )                   # steady-state branch
        f     =       W_f @ z + b_f                     # unconstrained time-constant logit
        gate  = sigmoid( -( f * dt + b_tau ) )          # closed-form time gate
        x_new = gate * g + (1 - gate) * h

    ``dt`` enters *only* through ``f * dt`` in the gate exponent.  ``b_tau`` is a
    learned ``(H,)`` time offset initialised to zeros; with ``b_tau = 0`` this is
    literally paper eq. 10.  The offset is the official ``ncps`` "explicit
    time-gating bias" form.

    There is no state-space discretisation, no attention and no mean-pooling in
    this cell.

    Note on limits: eq. 10 is a closed-form *solution*, not an incremental
    integrator, so ``x_new`` does not tend to ``x_prev`` as ``dt -> 0`` (it tends
    to the ``b_tau``-gated mixture of ``g`` and ``h``).  What it does satisfy, and
    what the tests check, is that it solves the logistic ODE it was derived from
    (``tests/test_cells.py::test_cfc_matches_numerical_ode_integration``), that it
    relaxes monotonically to the steady-state branch as ``dt`` grows, and that it
    is bounded for arbitrarily large ``dt``.
    """

    dt_aware = True
    #: ``True`` in the control variant only.
    dt_blind = False

    def __init__(self, input_dim: int, hidden_dim: int, **_: object) -> None:
        super().__init__(input_dim, hidden_dim)
        self.state_size = hidden_dim
        self.backbone = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.to_g = nn.Linear(hidden_dim, hidden_dim)
        self.to_h = nn.Linear(hidden_dim, hidden_dim)
        self.to_f = nn.Linear(hidden_dim, hidden_dim)
        self.b_tau = nn.Parameter(torch.zeros(hidden_dim))

    def branches(
        self, u: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(g, h, f)``, each ``(B, H)``.  None of them depends on ``dt``."""
        z = torch.tanh(self.backbone(torch.cat([state, u], dim=-1)))
        return torch.tanh(self.to_g(z)), torch.tanh(self.to_h(z)), self.to_f(z)

    def effective_dt(self, dt: torch.Tensor) -> torch.Tensor:
        """The interval the time gate actually sees.  Overridden by the control."""
        return dt

    def time_gate(self, f: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """``sigmoid(-(f * dt + b_tau))``, shape ``(B, H)``."""
        return torch.sigmoid(-(f * dt.unsqueeze(-1) + self.b_tau))

    def forward(self, u, state, dt):
        g, h, f = self.branches(u, state)
        gate = self.time_gate(f, self.effective_dt(dt))
        x = gate * g + (1.0 - gate) * h
        return x, x

    def custom_init_(self, generator: torch.Generator) -> None:  # noqa: ARG002
        with torch.no_grad():
            self.b_tau.zero_()


class CfCDtBlindCell(CfCCell):
    """:class:`CfCCell` with ``dt`` pinned to ``1.0`` inside ``forward``.

    Architecturally identical, same parameter count, same initialisation.  It still
    receives ``dt`` as an encoder input feature (spec section 1.7 path (a)); what is
    removed is path (b), ``dt`` entering the state update.  This is the
    exactly-one-factor control that carries the pre-registered primary comparison.
    """

    dt_aware = False
    dt_blind = True

    def effective_dt(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(dt)


CELL_REGISTRY: dict[str, type] = {
    "mlp": MLPBlock,
    "gru": GRUBlock,
    "lstm": LSTMBlock,
    "attn": CausalAttentionBlock,
    "ltc": LTCCell,
    "cfc": CfCCell,
    "cfc_dtblind": CfCDtBlindCell,
}
