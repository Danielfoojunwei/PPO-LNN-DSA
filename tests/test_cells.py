"""Cell-level tests.  These test the science, not the shapes.

Each test names the defect it is a regression test for.  Where a test would have
caught a defect, there is an explicit *negative control*: a deliberately broken cell
reproducing the old behaviour, asserted to fail the same check.  A test that cannot
fail is not evidence.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from dsa.models.cells import (
    ATTENTION_WINDOW,
    CELL_REGISTRY,
    CausalAttentionBlock,
    CfCCell,
    CfCDtBlindCell,
    GRUBlock,
    LSTMBlock,
    LTCCell,
    MLPBlock,
    RecurrentBlockBase,
)
from dsa.models.policy import deterministic_init_

torch.set_num_threads(1)

H = 16
BATCH = 6


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _make(cls, seed: int = 0, **kwargs):
    g = _gen(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        cell = cls(H, H, **kwargs)
    deterministic_init_(cell, g)
    cell.eval()
    return cell


def _dt(value: float, batch: int = BATCH, dtype=torch.float32) -> torch.Tensor:
    return torch.full((batch,), float(value), dtype=dtype)


# ===========================================================================
# D5 -- the LTC must actually be an LTC
# ===========================================================================
class _ConstantTauForgetGate(nn.Module):
    """Negative control: the old `empirical/models.py` cell.

    ``log_tau`` is a constant parameter with no dependence on ``x`` or ``I``, so the
    effective time constant is a learned constant -- algebraically a sigmoid forget
    gate (defect D5).  Any test that claims to detect D5 must fail against this.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(hidden_dim))

    def system_time_constant(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_tau).expand(u.shape[0], -1)


def _time_constant_variation(cell, inputs, states) -> float:
    """Max relative spread of ``tau_sys`` across the supplied ``(u, x)`` pairs."""
    with torch.no_grad():
        taus = torch.stack([cell.system_time_constant(u, x) for u, x in zip(inputs, states)])
    spread = (taus.max(dim=0).values - taus.min(dim=0).values) / taus.mean(dim=0).clamp_min(1e-9)
    return float(spread.max())


def _assert_time_constant_is_input_dependent(cell, hidden_dim: int) -> None:
    """Shared check, applied to both the real LTC and the negative control."""
    g = _gen(1)
    x = torch.zeros(BATCH, hidden_dim)
    inputs = [torch.randn(BATCH, hidden_dim, generator=g) * 3.0 for _ in range(6)]
    variation = _time_constant_variation(cell, inputs, [x] * len(inputs))
    assert variation > 0.05, (
        "tau_sys does not vary with the input: this is a gated RNN, not an LTC "
        f"(relative spread {variation:.3e})"
    )


def test_ltc_time_constant_is_input_dependent():
    """tau_sys = tau / (1 + tau * f(x, I, t, theta)) must depend on the input (D5)."""
    _assert_time_constant_is_input_dependent(_make(LTCCell), H)


def test_ltc_time_constant_check_rejects_a_constant_tau_cell():
    """The negative control: the same check FAILS against the old constant-tau cell."""
    broken = _ConstantTauForgetGate(H)
    with pytest.raises(AssertionError, match="not an LTC"):
        _assert_time_constant_is_input_dependent(broken, H)


def test_ltc_time_constant_depends_on_state():
    """tau_sys is a function of x as well as I -- the 'liquid' feedback path."""
    cell = _make(LTCCell)
    g = _gen(2)
    u = torch.randn(BATCH, H, generator=g)
    states = [torch.randn(BATCH, H, generator=g) * 2.0 for _ in range(6)]
    assert _time_constant_variation(cell, [u] * len(states), states) > 0.05


def test_ltc_time_constant_matches_paper_equation_five():
    """tau_sys computed independently from f and tau agrees with the exposed method."""
    cell = _make(LTCCell)
    g = _gen(3)
    u = torch.randn(BATCH, H, generator=g)
    x = torch.randn(BATCH, H, generator=g)
    with torch.no_grad():
        f = cell.synaptic_activation(u, x)
        tau = cell.tau()
        expected = tau / (1.0 + tau * f)
        assert torch.allclose(cell.system_time_constant(u, x), expected, atol=0, rtol=0)
    assert bool(((f > 0.0) & (f < 1.0)).all()), "f must be a sigmoid, i.e. in (0, 1)"


def test_ltc_forward_dynamics_use_the_exposed_time_constant():
    """Close the loop: tau_sys must govern the *actual* update, not just be reported.

    A cell could expose a textbook-looking ``system_time_constant`` while integrating
    something else entirely.  The LTC ODE rearranges to

        dx/dt = -(1 / tau_sys) * x  +  f * A

    so we measure dx/dt numerically from the cell's own forward pass at tiny dt and
    compare it against that expression built from the exposed tau_sys.
    """
    cell = _make(LTCCell, seed=9).double()
    g = _gen(38)
    u = torch.randn(BATCH, H, generator=g, dtype=torch.float64)
    x = torch.randn(BATCH, H, generator=g, dtype=torch.float64) * 0.5
    eps = 1e-7
    with torch.no_grad():
        moved, _ = cell(u, x, _dt(eps, dtype=torch.float64))
        measured = (moved - x) / eps
        tau_sys = cell.system_time_constant(u, x)
        f = cell.synaptic_activation(u, x)
        predicted = -x / tau_sys + f * cell.A
    assert torch.allclose(measured, predicted, atol=1e-5), (
        "the exposed tau_sys does not govern the forward dynamics"
    )


def test_ltc_timescale_diversity():
    """All units initialised to tau = 1.0 was the other half of D5.  Require spread."""
    cell = _make(LTCCell)
    tau = cell.tau().detach()
    assert float(tau.max() / tau.min()) >= 5.0
    # log-spaced over the documented range
    assert math.isclose(float(tau.min()), cell.tau_min, rel_tol=1e-5)
    assert math.isclose(float(tau.max()), cell.tau_max, rel_tol=1e-5)


# ===========================================================================
# D12 / continuous-time properties of the LTC solver
# ===========================================================================
def test_ltc_dt_changes_the_state_update():
    cell = _make(LTCCell)
    g = _gen(4)
    u = torch.randn(BATCH, H, generator=g)
    x = torch.randn(BATCH, H, generator=g) * 0.3
    with torch.no_grad():
        a, _ = cell(u, x, _dt(0.25))
        b, _ = cell(u, x, _dt(2.5))
    assert float((a - b).abs().max()) > 1e-4


def test_ltc_zero_dt_is_exactly_a_no_op():
    """dt enters only through h = dt / K, so dt = 0 must leave the state untouched."""
    cell = _make(LTCCell)
    g = _gen(5)
    u = torch.randn(BATCH, H, generator=g)
    x = torch.randn(BATCH, H, generator=g) * 0.3
    with torch.no_grad():
        out, new_state = cell(u, x, _dt(0.0))
    assert torch.equal(out, x)
    assert torch.equal(new_state, x)


def test_ltc_state_change_vanishes_linearly_as_dt_goes_to_zero():
    """||x(dt) - x(0)|| must be O(dt): a real continuous-time property."""
    cell = _make(LTCCell).double()
    g = _gen(6)
    u = torch.randn(BATCH, H, generator=g, dtype=torch.float64)
    x = torch.randn(BATCH, H, generator=g, dtype=torch.float64) * 0.3
    deltas = []
    for dt in (1e-3, 5e-4, 2.5e-4):
        with torch.no_grad():
            out, _ = cell(u, x, _dt(dt, dtype=torch.float64))
        deltas.append(float((out - x).norm()))
    assert deltas[0] > 0.0
    # halving dt halves the state change, to within 1 %
    assert abs(deltas[0] / deltas[1] - 2.0) < 0.01
    assert abs(deltas[1] / deltas[2] - 2.0) < 0.01


def test_ltc_is_unconditionally_stable_for_large_dt():
    """The semi-implicit denominator grows with h, so |x| stays bounded by max|A|.

    This is a property of the fused solver of paper eq. 6, and it is why an explicit
    Euler discretisation would not do.
    """
    cell = _make(LTCCell).double()
    g = _gen(7)
    u = torch.randn(BATCH, H, generator=g, dtype=torch.float64)
    x = torch.zeros(BATCH, H, dtype=torch.float64)
    bound = float(cell.A.detach().abs().max())
    for dt in (1e2, 1e6, 1e12):
        with torch.no_grad():
            out, _ = cell(u, x, _dt(dt, dtype=torch.float64))
        assert bool(torch.isfinite(out).all()), f"LTC diverged at dt={dt}"
        assert float(out.abs().max()) <= bound + 1e-9, f"LTC unbounded at dt={dt}"


def test_ltc_fused_solver_converges_to_the_ltc_ode():
    """The solver must actually integrate dx/dt = -[1/tau + f] x + f A.

    Reference: RK4 in float64 (h = 5e-4, error far below the quantity measured).  The fused semi-implicit Euler of
    paper eq. 6 is first order, so the error must fall by roughly 10x for each 10x
    increase in the number of unfolds.  A cell that merely *looked* like an ODE
    solver would not converge to this reference at all.
    """
    base = _make(LTCCell).double()
    g = _gen(8)
    u = torch.randn(BATCH, H, generator=g, dtype=torch.float64)
    x0 = torch.randn(BATCH, H, generator=g, dtype=torch.float64) * 0.3
    dt = 1.0

    def reference(steps: int) -> torch.Tensor:
        inv_tau = 1.0 / base.tau()
        pre = base.input_map(u)
        step = dt / steps

        def deriv(x):
            f = torch.sigmoid(pre + base.recurrent_map(x))
            return -(inv_tau + f) * x + f * base.A

        x = x0.clone()
        for _ in range(steps):
            k1 = deriv(x)
            k2 = deriv(x + step / 2 * k1)
            k3 = deriv(x + step / 2 * k2)
            k4 = deriv(x + step * k3)
            x = x + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    with torch.no_grad():
        ref = reference(2_000)
        errors = []
        for unfolds in (6, 60, 600):
            cell = LTCCell(H, H, unfolds=unfolds).double()
            cell.load_state_dict(base.state_dict())
            out, _ = cell(u, x0, _dt(dt, dtype=torch.float64))
            errors.append(float((out - ref).abs().max()))

    scale = float(ref.abs().max())
    assert errors[0] < 0.15 * scale, f"default solver is not tracking the ODE: {errors}"
    assert errors[1] < 0.2 * errors[0], f"no first-order convergence: {errors}"
    assert errors[2] < 0.2 * errors[1], f"no first-order convergence: {errors}"


# ===========================================================================
# D6 -- the CfC is the liquid-foundation primitive, not attention
# ===========================================================================
def test_cfc_dt_changes_the_state_update():
    cell = _make(CfCCell)
    g = _gen(10)
    u = torch.randn(BATCH, H, generator=g)
    x = torch.randn(BATCH, H, generator=g) * 0.3
    with torch.no_grad():
        a, _ = cell(u, x, _dt(0.2))
        b, _ = cell(u, x, _dt(3.0))
    assert float((a - b).abs().max()) >= 1e-4


def test_cfc_dtblind_is_bitwise_identical_when_only_dt_changes():
    """The exactly-one-factor control for the whole continuous-time thesis."""
    cell = _make(CfCDtBlindCell)
    g = _gen(11)
    u = torch.randn(BATCH, H, generator=g)
    x = torch.randn(BATCH, H, generator=g) * 0.3
    with torch.no_grad():
        outs = [cell(u, x, _dt(v))[0] for v in (0.05, 1.0, 7.5, 100.0)]
    for other in outs[1:]:
        assert torch.equal(outs[0], other)


def test_cfc_dtblind_is_architecturally_identical_to_cfc():
    a, b = _make(CfCCell, seed=12), _make(CfCDtBlindCell, seed=12)
    assert sorted(a.state_dict()) == sorted(b.state_dict())
    for key, value in a.state_dict().items():
        assert value.shape == b.state_dict()[key].shape
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    # identical seed -> identical weights: the only difference is the dt path
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    # and at dt == 1 they are the same function
    g = _gen(13)
    u = torch.randn(BATCH, H, generator=g)
    x = torch.randn(BATCH, H, generator=g)
    with torch.no_grad():
        assert torch.equal(a(u, x, _dt(1.0))[0], b(u, x, _dt(1.0))[0])


def test_cfc_matches_numerical_ode_integration():
    r"""The closed form must solve the ODE it is the closed-form solution *of*.

    Differentiating x(t) = sigma(-f t) g + (1 - sigma(-f t)) h gives

        dx/dt = -f (x - h) (1 - (x - h) / (g - h)),   x(0) = (g + h) / 2

    a logistic relaxation from the gated mixture toward the steady-state branch h.
    We integrate that with RK4 in float64 and compare against the cell's own output.
    """
    cell = _make(CfCCell, seed=14).double()
    g = _gen(15)
    u = torch.randn(BATCH, H, generator=g, dtype=torch.float64)
    x0 = torch.randn(BATCH, H, generator=g, dtype=torch.float64) * 0.3
    with torch.no_grad():
        branch_g, branch_h, branch_f = cell.branches(u, x0)

    # b_tau is zero at initialisation, so x(0) is exactly the (g + h) / 2 mixture.
    assert torch.equal(cell.b_tau.detach(), torch.zeros_like(cell.b_tau))
    usable = (branch_g - branch_h).abs() > 1e-2
    assert float(usable.float().mean()) > 0.5, "degenerate test fixture"

    def integrate(dt: float, steps: int = 2_000) -> torch.Tensor:
        x = (branch_g + branch_h) / 2
        step = dt / steps

        def deriv(x):
            excess = x - branch_h
            return -branch_f * excess * (1.0 - excess / (branch_g - branch_h))

        for _ in range(steps):
            k1 = deriv(x)
            k2 = deriv(x + step / 2 * k1)
            k3 = deriv(x + step / 2 * k2)
            k4 = deriv(x + step * k3)
            x = x + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    for dt in (0.2, 1.0, 3.0):
        with torch.no_grad():
            closed_form, _ = cell(u, x0, _dt(dt, dtype=torch.float64))
        error = float((closed_form - integrate(dt))[usable].abs().max())
        assert error < 1e-9, f"closed form does not solve its own ODE at dt={dt}: {error:.3e}"


def test_cfc_time_gate_limits():
    """dt -> 0 gives the b_tau-gated mixture; dt -> inf relaxes to the steady state."""
    cell = _make(CfCCell, seed=16).double()
    g = _gen(17)
    u = torch.randn(BATCH, H, generator=g, dtype=torch.float64)
    x0 = torch.randn(BATCH, H, generator=g, dtype=torch.float64) * 0.3
    with torch.no_grad():
        branch_g, branch_h, branch_f = cell.branches(u, x0)
        near_zero, _ = cell(u, x0, _dt(1e-9, dtype=torch.float64))
        assert torch.allclose(near_zero, (branch_g + branch_h) / 2, atol=1e-7)

        # For units with f > 0 the gate closes and x -> h (the steady-state branch).
        positive = branch_f > 0.1
        assert bool(positive.any())
        far, _ = cell(u, x0, _dt(1e4, dtype=torch.float64))
        assert torch.allclose(far[positive], branch_h[positive], atol=1e-7)


def test_cfc_is_bounded_for_large_dt():
    cell = _make(CfCCell, seed=18)
    g = _gen(19)
    u = torch.randn(BATCH, H, generator=g)
    x0 = torch.zeros(BATCH, H)
    for dt in (1e3, 1e6, 1e12):
        with torch.no_grad():
            out, _ = cell(u, x0, _dt(dt))
        assert bool(torch.isfinite(out).all())
        # both branches are tanh, and the gate is a convex combination
        assert float(out.abs().max()) <= 1.0 + 1e-6


# ===========================================================================
# The transformer baseline
# ===========================================================================
def test_attention_block_is_causal():
    """Changing a future input must not change any earlier output."""
    block = _make(CausalAttentionBlock, seed=20, num_heads=1)
    g = _gen(21)
    seq = torch.randn(BATCH, 12, H, generator=g)
    dt_seq = torch.rand(BATCH, 12, generator=g) + 0.5
    state = block.initial_state(BATCH)
    with torch.no_grad():
        base, _ = block.unroll(seq, state, dt_seq)
        perturbed_seq = seq.clone()
        perturbed_seq[:, 7:] += 5.0
        perturbed, _ = block.unroll(perturbed_seq, state, dt_seq)
    assert torch.allclose(base[:, :7], perturbed[:, :7], atol=1e-6)
    assert float((base[:, 7:] - perturbed[:, 7:]).abs().max()) > 1e-3


def test_attention_block_horizon_is_the_declared_window():
    """Inputs older than K steps must not influence the current output."""
    block = _make(CausalAttentionBlock, seed=22, num_heads=1)
    k = ATTENTION_WINDOW
    total = 2 * k + 3
    g = _gen(23)
    seq = torch.randn(BATCH, total, H, generator=g)
    dt_seq = torch.ones(BATCH, total)
    state = block.initial_state(BATCH)
    with torch.no_grad():
        base, _ = block.unroll(seq, state, dt_seq)
        far_past = seq.clone()
        far_past[:, 0] += 9.0  # older than K relative to the final step
        perturbed, _ = block.unroll(far_past, state, dt_seq)
    assert torch.allclose(base[:, -1], perturbed[:, -1], atol=1e-6)
    assert float((base[:, 0] - perturbed[:, 0]).abs().max()) > 1e-3


def test_attention_state_size_matches_the_declared_contract():
    block = _make(CausalAttentionBlock, seed=24, num_heads=1)
    assert block.state_size == ATTENTION_WINDOW * H + 1


# ===========================================================================
# D10 -- dropout must be inert in eval, for every block
# ===========================================================================
def test_dropout_is_inert_in_eval_and_live_in_train():
    """Regression test for D10, which was caused by dropout live during rollout.

    The registry ships ``dropout = 0.0`` everywhere, so this test deliberately builds
    a dropout-carrying attention block to prove the eval/train switch is wired up at
    all.  Without the switch, the eval branch below would be non-deterministic.
    """
    block = _make(CausalAttentionBlock, seed=25, num_heads=1, dropout=0.5)
    g = _gen(26)
    u = torch.randn(BATCH, H, generator=g)
    state = block.initial_state(BATCH)
    dt = _dt(1.0)

    block.eval()
    with torch.no_grad():
        reference, _ = block(u, state, dt)
        for _ in range(30):
            out, _ = block(u, state, dt)
            assert torch.equal(out, reference), "eval() output is not deterministic"

    block.train()
    with torch.no_grad():
        draws = torch.stack([block(u, state, dt)[0] for _ in range(30)])
    assert float(draws.std(dim=0).max()) > 0.0, "dropout is not actually active in train()"


# ===========================================================================
# Cross-cell contracts
# ===========================================================================
ALL_KINDS = tuple(CELL_REGISTRY)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_block_step_unroll_equivalence(kind):
    """unroll() must equal T successive forward() calls for every block."""
    cls = CELL_REGISTRY[kind]
    block = _make(cls, seed=27, num_heads=1)
    g = _gen(28)
    length = 14
    seq = torch.randn(BATCH, length, H, generator=g)
    dt_seq = torch.rand(BATCH, length, generator=g) * 2.5 + 0.1
    state = block.initial_state(BATCH)
    with torch.no_grad():
        bulk, bulk_state = block.unroll(seq, state, dt_seq)
        stepwise, running = [], state
        for t in range(length):
            out, running = block(seq[:, t], running, dt_seq[:, t])
            stepwise.append(out)
        stepwise = torch.stack(stepwise, dim=1)
    assert torch.allclose(bulk, stepwise, atol=1e-5), kind
    assert torch.allclose(bulk_state, running, atol=1e-5), kind


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_block_shapes_and_state_contract(kind):
    block = _make(CELL_REGISTRY[kind], seed=29, num_heads=1)
    state = block.initial_state(BATCH)
    assert state.shape == (BATCH, block.state_size)
    assert torch.equal(state, torch.zeros_like(state))
    g = _gen(30)
    u = torch.randn(BATCH, H, generator=g)
    with torch.no_grad():
        out, new_state = block(u, state, _dt(1.0))
    assert out.shape == (BATCH, H)
    assert new_state.shape == (BATCH, block.state_size)


DT_AWARE_KINDS = {"ltc", "cfc"}


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_dt_enters_the_state_update_only_for_dt_aware_cells(kind):
    """Path (b) of spec section 1.7, isolated at the block level.

    ``mlp``/``gru``/``lstm``/``attn``/``cfc_dtblind`` must be bitwise invariant to dt
    here; they still see dt through the shared encoder feature (path (a)).
    """
    block = _make(CELL_REGISTRY[kind], seed=31, num_heads=1)
    g = _gen(32)
    u = torch.randn(BATCH, H, generator=g)
    state = block.initial_state(BATCH)
    with torch.no_grad():
        low, _ = block(u, state, _dt(0.2))
        high, _ = block(u, state, _dt(4.0))
    if kind in DT_AWARE_KINDS:
        assert float((low - high).abs().max()) > 1e-4, f"{kind} claims dt-awareness"
        assert getattr(block, "dt_aware", False) is True
    else:
        assert torch.equal(low, high), f"{kind} secretly uses dt"
        assert getattr(block, "dt_aware", False) is False


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_blocks_are_differentiable_end_to_end(kind):
    block = _make(CELL_REGISTRY[kind], seed=33, num_heads=1)
    block.train()
    g = _gen(34)
    seq = torch.randn(BATCH, 5, H, generator=g)
    dt_seq = torch.rand(BATCH, 5, generator=g) + 0.5
    out, _ = block.unroll(seq, block.initial_state(BATCH), dt_seq)
    out.sum().backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads, f"{kind} produced no gradients"
    assert any(float(gr.abs().sum()) > 0 for gr in grads), f"{kind} gradients are all zero"


def test_registry_covers_the_declared_cell_kinds():
    assert set(CELL_REGISTRY) == {
        "mlp",
        "gru",
        "lstm",
        "attn",
        "ltc",
        "cfc",
        "cfc_dtblind",
    }
    assert CELL_REGISTRY["mlp"] is MLPBlock
    assert CELL_REGISTRY["gru"] is GRUBlock
    assert CELL_REGISTRY["lstm"] is LSTMBlock
    assert CELL_REGISTRY["attn"] is CausalAttentionBlock
    assert CELL_REGISTRY["ltc"] is LTCCell
    assert CELL_REGISTRY["cfc"] is CfCCell
    assert CELL_REGISTRY["cfc_dtblind"] is CfCDtBlindCell


def test_mlp_block_is_stateless():
    block = _make(MLPBlock, seed=35)
    assert block.state_size == 0
    assert block.initial_state(BATCH).shape == (BATCH, 0)


def test_lstm_state_is_h_and_c():
    block = _make(LSTMBlock, seed=36)
    assert block.state_size == 2 * H
    g = _gen(37)
    u = torch.randn(BATCH, H, generator=g)
    with torch.no_grad():
        out, state = block(u, block.initial_state(BATCH), _dt(1.0))
    assert torch.equal(state[:, :H], out)


# ===========================================================================
# M29b -- the reference unroll must keep the hidden trajectory in the graph
# ===========================================================================
# `RecurrentBlockBase.unroll` is the only place BPTT can be severed, and severing
# it is invisible to every value- or shape-based assertion: outputs, states, step /
# unroll equivalence and even "every parameter has a non-zero gradient" all survive
# `state.detach()` untouched, because the last timestep still reaches the loss
# directly.  What detach destroys is the *path through time*, so these tests assert
# on the gradient of an early input and of the initial state, which can only reach
# a late output through the carried state.


class _DetachingUnrollBlock(GRUBlock):
    """Negative control: the mutation this section exists to catch.

    Identical to :meth:`RecurrentBlockBase.unroll` except for the ``state.detach()``
    that severs backpropagation through time.  Any test claiming to detect a severed
    BPTT must fail against this block, and
    ``test_bptt_check_rejects_a_detaching_unroll`` asserts exactly that.
    """

    def unroll(self, u_seq, state, dt_seq):
        outputs = []
        for t in range(u_seq.shape[1]):
            y, state = self.forward(u_seq[:, t], state.detach(), dt_seq[:, t])
            outputs.append(y)
        return torch.stack(outputs, dim=1), state


#: Stateful blocks that inherit the reference ``unroll``.  ``mlp`` is memoryless and
#: ``attn`` overrides ``unroll`` with a batched attention path, so neither is served
#: by the loop under test here.
BASE_UNROLL_STATEFUL_KINDS = tuple(
    kind
    for kind, cls in CELL_REGISTRY.items()
    if cls.unroll is RecurrentBlockBase.unroll
)


def _bptt_grads(block, length: int = 6, dt_value: float = 0.3):
    """Gradients of a *last-timestep* loss w.r.t. the initial state and the inputs.

    ``dt`` is small so that continuous-time cells retain a long memory; a large
    ``dt`` closes the CfC time gate and makes the through-time gradient physically
    (not structurally) small.
    """
    g = _gen(41)
    u_seq = torch.randn(BATCH, length, H, generator=g, requires_grad=True)
    dt_seq = torch.full((BATCH, length), float(dt_value))
    state = block.initial_state(BATCH).clone().requires_grad_(True)
    outputs, _final = block.unroll(u_seq, state, dt_seq)
    loss = outputs[:, -1].sum()
    grad_state, grad_u = torch.autograd.grad(
        loss, [state, u_seq], allow_unused=True, retain_graph=True
    )
    return loss, grad_state, grad_u


@pytest.mark.parametrize("kind", BASE_UNROLL_STATEFUL_KINDS)
def test_unroll_backpropagates_through_time(kind):
    """The last output must be differentiable w.r.t. the *first* input and the
    initial state.  ``state.detach()`` in the unroll loop makes both exactly zero
    (the initial state leaves the graph entirely), while leaving every other
    property this suite checks intact."""
    block = _make(CELL_REGISTRY[kind], seed=40)
    block.train()
    assert block.state_size > 0, kind
    _loss, grad_state, grad_u = _bptt_grads(block)

    assert grad_state is not None, (
        f"{kind}: the initial state is not in the autograd graph of the final "
        "output at all -- the recurrent path has been severed"
    )
    assert float(grad_state.abs().max()) > 0.0, kind
    # The earliest input can only reach the final output through the carried state.
    assert float(grad_u[:, 0].abs().max()) > 0.0, f"{kind}: no gradient through time"
    # Sanity: the direct (same-timestep) path is intact too, so a failure above is
    # specifically a through-time failure and not a dead block.
    assert float(grad_u[:, -1].abs().max()) > 0.0, kind


def test_bptt_check_rejects_a_detaching_unroll():
    """Teeth for ``test_unroll_backpropagates_through_time``.

    The detaching block reproduces the mutation exactly: it still yields non-zero
    parameter gradients and a live same-timestep input gradient, so the pre-existing
    tests would pass it -- and the two through-time quantities are exactly dead.
    """
    block = _make(_DetachingUnrollBlock, seed=40)
    block.train()
    loss, grad_state, grad_u = _bptt_grads(block)

    assert grad_state is None, "the detaching control kept the initial state in the graph"
    assert float(grad_u[:, 0].abs().max()) == 0.0
    assert float(grad_u[:, -1].abs().max()) > 0.0

    # ... and it passes the checks the suite already had.
    params = [p for p in block.parameters()]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    assert any(gr is not None and float(gr.abs().sum()) > 0.0 for gr in grads)


@pytest.mark.parametrize("kind", BASE_UNROLL_STATEFUL_KINDS)
def test_detaching_unroll_is_numerically_invisible(kind):
    """Why a value assertion cannot catch M29b, stated as a test.

    Detaching changes no output value anywhere -- forward numerics are identical --
    so the whole of this file's step/unroll-equivalence machinery is blind to it.
    This is the reason the gradient assertions above exist.
    """
    block = _make(CELL_REGISTRY[kind], seed=42)
    block.eval()
    g = _gen(43)
    length = 7
    seq = torch.randn(BATCH, length, H, generator=g)
    dt_seq = torch.rand(BATCH, length, generator=g) + 0.2
    state = block.initial_state(BATCH)
    with torch.no_grad():
        reference, ref_state = block.unroll(seq, state, dt_seq)
        detached_out, detached_state = [], state
        for t in range(length):
            y, detached_state = block.forward(seq[:, t], detached_state.detach(), dt_seq[:, t])
            detached_out.append(y)
        detached_out = torch.stack(detached_out, dim=1)
    assert torch.equal(reference, detached_out), kind
    assert torch.equal(ref_state, detached_state), kind
