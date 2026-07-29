"""Registry-level invariants: capacity matching, determinism, and information parity.

These are the tests that make the ablation interpretable.  Defect D11 was that the
two "liquid" winners were simply the two biggest networks (parameter count vs reward
correlated at Spearman 0.829) and that the ablation also differed by a whole
transformer layer.  ``test_param_budget``, ``test_param_spread``, ``test_equal_depth``
and ``test_ablation_single_factor`` make both of those impossible.

Defect D10 was dropout live during rollout, which made two successive greedy actions
disagree.  ``test_eval_mode_is_bitwise_deterministic`` and ``test_deterministic_act``
are its regression tests.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from dsa.models.cells import RecurrentBlockBase
from dsa.models import (
    ATTENTION_HEAD_CANDIDATES,
    MODEL_REGISTRY,
    PARAM_TARGET,
    PARAM_TOLERANCE,
    PRIMARY_MODELS,
    RecurrentActorCritic,
    build_model,
    count_parameters,
    model_summary,
    solve_hidden_dim,
    solve_width_and_heads,
)

torch.set_num_threads(1)

OBS_DIM = 40
ACTION_DIM = 8
KEYS = tuple(MODEL_REGISTRY)


@pytest.fixture(scope="module")
def summaries() -> dict[str, dict]:
    return {k: model_summary(k, OBS_DIM, ACTION_DIM) for k in KEYS}


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ===========================================================================
# Registry shape
# ===========================================================================
def test_registry_keys_are_frozen():
    assert set(KEYS) == {
        "ppo_mlp",
        "ppo_gru",
        "ppo_lstm",
        "ppo_transformer",
        "ppo_ltc",
        "ppo_cfc",
        "ppo_cfc_dtblind",
        "ppo_ltc_cfc",
    }
    for key, spec in MODEL_REGISTRY.items():
        assert spec.key == key


def test_primary_models_excludes_only_lstm():
    assert set(PRIMARY_MODELS) <= set(KEYS)
    assert set(KEYS) - set(PRIMARY_MODELS) == {"ppo_lstm"}


def test_no_model_is_named_lfm():
    """Vocabulary correction, spec section 1.1 / defect D6.

    The attention block is a transformer and is named one.  The liquid-foundation
    primitive is the CfC cell.
    """
    for key, spec in MODEL_REGISTRY.items():
        assert "lfm" not in key.lower(), key
        assert "lfm" not in spec.family.lower(), key
    assert MODEL_REGISTRY["ppo_transformer"].cell_kinds == ("attn", "attn")
    assert MODEL_REGISTRY["ppo_transformer"].family == "attention"
    assert MODEL_REGISTRY["ppo_cfc"].family == "liquid"


# ===========================================================================
# D11 -- capacity matching
# ===========================================================================
def test_param_budget(summaries):
    for key, summary in summaries.items():
        error = abs(summary["parameter_count"] - PARAM_TARGET) / PARAM_TARGET
        assert error <= PARAM_TOLERANCE, (
            f"{key}: {summary['parameter_count']} params, "
            f"{error:.3%} from the {PARAM_TARGET} budget"
        )


def test_param_spread(summaries):
    counts = [s["parameter_count"] for s in summaries.values()]
    assert max(counts) / min(counts) <= 1.11, dict(zip(summaries, counts))


def test_equal_depth():
    for key, spec in MODEL_REGISTRY.items():
        assert len(spec.cell_kinds) == 2, key


def test_ablation_single_factor():
    """ppo_ltc_cfc differs from each parent in exactly one block position."""

    def positions_differing(a: str, b: str) -> int:
        ka, kb = MODEL_REGISTRY[a].cell_kinds, MODEL_REGISTRY[b].cell_kinds
        return sum(x != y for x, y in zip(ka, kb))

    assert positions_differing("ppo_ltc_cfc", "ppo_ltc") == 1
    assert positions_differing("ppo_ltc_cfc", "ppo_cfc") == 1

    # The dt-blind control differs from ppo_cfc only by the blinding flag.
    blind = MODEL_REGISTRY["ppo_cfc_dtblind"].cell_kinds
    sighted = MODEL_REGISTRY["ppo_cfc"].cell_kinds
    assert tuple(k.replace("_dtblind", "") for k in blind) == sighted
    assert not MODEL_REGISTRY["ppo_cfc_dtblind"].dt_aware
    assert MODEL_REGISTRY["ppo_cfc"].dt_aware


def test_cfc_and_dtblind_have_equal_parameter_counts(summaries):
    assert (
        summaries["ppo_cfc"]["parameter_count"]
        == summaries["ppo_cfc_dtblind"]["parameter_count"]
    )
    assert summaries["ppo_cfc"]["hidden_dim"] == summaries["ppo_cfc_dtblind"]["hidden_dim"]


def test_cfc_and_dtblind_have_identical_initial_weights():
    """Same architecture, same parameter count, same initialisation distribution.

    That is what makes the pre-registered primary comparison a one-factor experiment.
    """
    a = build_model("ppo_cfc", OBS_DIM, ACTION_DIM, seed=1234)
    b = build_model("ppo_cfc_dtblind", OBS_DIM, ACTION_DIM, seed=1234)
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert sorted(sd_a) == sorted(sd_b)
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), key


def test_encoder_and_heads_identical_shapes():
    """For a fixed H, only the two blocks may differ between models."""
    hidden = 32
    shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    for key, spec in MODEL_REGISTRY.items():
        kwargs = dict(spec.cell_kwargs)
        if "attn" in spec.cell_kinds:
            kwargs["num_heads"] = 1
        model = RecurrentActorCritic(
            OBS_DIM, ACTION_DIM, spec.cell_kinds, hidden, generator=_gen(0), cell_kwargs=kwargs
        )
        shapes[key] = {
            name: tuple(p.shape)
            for name, p in model.named_parameters()
            if not name.startswith("blocks.")
        }
    reference = shapes[KEYS[0]]
    for key, shape in shapes.items():
        assert shape == reference, key


def test_solve_hidden_dim_is_pure_and_order_independent():
    """Principle P4: no caching that depends on call order."""
    forward = {k: solve_hidden_dim(MODEL_REGISTRY[k], OBS_DIM, ACTION_DIM) for k in KEYS}
    backward = {
        k: solve_hidden_dim(MODEL_REGISTRY[k], OBS_DIM, ACTION_DIM) for k in reversed(KEYS)
    }
    assert forward == backward
    assert all(v > 0 for v in forward.values())


def test_solve_hidden_dim_finds_the_argmin(summaries):
    """The solved width really is the best integer, not merely a feasible one."""
    for key, spec in MODEL_REGISTRY.items():
        hidden, heads = solve_width_and_heads(spec, OBS_DIM, ACTION_DIM)
        best_error = abs(summaries[key]["parameter_count"] - PARAM_TARGET)
        step = 1 if heads is None else heads
        for candidate in (hidden - step, hidden + step):
            if candidate < step:
                continue
            kwargs = dict(spec.cell_kwargs)
            if heads is not None:
                kwargs["num_heads"] = heads
            model = RecurrentActorCritic(
                OBS_DIM, ACTION_DIM, spec.cell_kinds, candidate, cell_kwargs=kwargs
            )
            assert abs(count_parameters(model) - PARAM_TARGET) >= best_error, (key, candidate)


def test_attention_head_count_is_solved_not_hardcoded(summaries):
    summary = summaries["ppo_transformer"]
    heads = summary["num_heads"]
    assert heads in ATTENTION_HEAD_CANDIDATES
    assert summary["hidden_dim"] % heads == 0
    for key in KEYS:
        if "attn" not in MODEL_REGISTRY[key].cell_kinds:
            assert summaries[key]["num_heads"] is None, key


def test_model_summary_matches_the_built_model(summaries):
    for key, summary in summaries.items():
        model = build_model(key, OBS_DIM, ACTION_DIM, seed=7)
        assert count_parameters(model) == summary["parameter_count"]
        assert model.hidden_dim == summary["hidden_dim"]
        assert list(model.state_sizes) == summary["state_sizes"]
        assert model.obs_dim == OBS_DIM
        assert model.action_dim == ACTION_DIM


def test_build_model_rejects_unknown_keys():
    with pytest.raises(KeyError):
        build_model("ppo_lfm", OBS_DIM, ACTION_DIM, seed=0)


# ===========================================================================
# Determinism of construction
# ===========================================================================
@pytest.mark.parametrize("key", KEYS)
def test_build_model_is_deterministic(key):
    a = build_model(key, OBS_DIM, ACTION_DIM, seed=99)
    b = build_model(key, OBS_DIM, ACTION_DIM, seed=99)
    for name, param in a.state_dict().items():
        assert torch.equal(param, b.state_dict()[name]), name


@pytest.mark.parametrize("key", KEYS)
def test_build_model_depends_on_the_seed(key):
    a = build_model(key, OBS_DIM, ACTION_DIM, seed=1)
    b = build_model(key, OBS_DIM, ACTION_DIM, seed=2)
    differing = [
        n for n, p in a.state_dict().items() if not torch.equal(p, b.state_dict()[n])
    ]
    assert differing, f"{key}: changing the seed changed nothing"


def test_build_model_ignores_the_global_rng():
    """Principle P4 / spec section 3.3: the global RNG is never load-bearing."""
    reference = {k: build_model(k, OBS_DIM, ACTION_DIM, seed=5).state_dict() for k in KEYS}
    torch.manual_seed(987654321)
    np.random.seed(13)
    random.seed(13)
    _ = torch.randn(1000)
    for key in reversed(KEYS):  # also a different construction order
        rebuilt = build_model(key, OBS_DIM, ACTION_DIM, seed=5).state_dict()
        for name, param in reference[key].items():
            assert torch.equal(param, rebuilt[name]), (key, name)


def test_build_model_does_not_disturb_the_global_rng():
    torch.manual_seed(4242)
    expected = torch.randn(8)
    torch.manual_seed(4242)
    for key in KEYS:
        build_model(key, OBS_DIM, ACTION_DIM, seed=3)
    assert torch.equal(torch.randn(8), expected)


# ===========================================================================
# Sequence contract
# ===========================================================================
@pytest.mark.parametrize("key", KEYS)
def test_step_unroll_equivalence(key):
    """unroll() must be numerically identical to T successive step() calls (1e-5)."""
    model = build_model(key, OBS_DIM, ACTION_DIM, seed=17)
    model.eval()
    batch, horizon = 5, 24
    g = _gen(18)
    obs = torch.randn(batch, horizon, OBS_DIM, generator=g)
    dt = torch.rand(batch, horizon, generator=g) * 2.5 + 0.2
    state = model.initial_state(batch)
    with torch.no_grad():
        logits_bulk, values_bulk, state_bulk = model.unroll(obs, dt, state)
        logits, values, running = [], [], state
        for t in range(horizon):
            lg, vl, running = model.step(obs[:, t], dt[:, t], running)
            logits.append(lg)
            values.append(vl)
        logits_step = torch.stack(logits, dim=1)
        values_step = torch.stack(values, dim=1)
    assert torch.allclose(logits_bulk, logits_step, atol=1e-5), key
    assert torch.allclose(values_bulk, values_step, atol=1e-5), key
    for a, b in zip(state_bulk, running):
        assert torch.allclose(a, b, atol=1e-5), key


@pytest.mark.parametrize("key", KEYS)
def test_initial_state_is_a_zero_reset(key):
    model = build_model(key, OBS_DIM, ACTION_DIM, seed=19)
    state = model.initial_state(4)
    assert len(state) == 2
    for tensor, size in zip(state, model.state_sizes):
        assert tensor.shape == (4, size)
        assert torch.equal(tensor, torch.zeros_like(tensor))


@pytest.mark.parametrize("key", KEYS)
def test_shapes(key):
    model = build_model(key, OBS_DIM, ACTION_DIM, seed=20)
    g = _gen(21)
    obs = torch.randn(3, OBS_DIM, generator=g)
    dt = torch.rand(3, generator=g) + 0.5
    with torch.no_grad():
        logits, value, state = model.step(obs, dt, model.initial_state(3))
    assert logits.shape == (3, ACTION_DIM)
    assert value.shape == (3,)
    assert len(state) == 2


@pytest.mark.parametrize("key", KEYS)
def test_gradients_reach_every_parameter(key):
    model = build_model(key, OBS_DIM, ACTION_DIM, seed=22)
    model.train()
    g = _gen(23)
    obs = torch.randn(4, 6, OBS_DIM, generator=g)
    dt = torch.rand(4, 6, generator=g) + 0.5
    logits, values, _ = model.unroll(obs, dt, model.initial_state(4))
    (logits.sum() + values.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, (key, missing)
    dead = [n for n, p in model.named_parameters() if float(p.grad.abs().sum()) == 0.0]
    assert not dead, (key, dead)


# ===========================================================================
# D10 -- eval determinism
# ===========================================================================
@pytest.mark.parametrize("key", KEYS)
def test_eval_mode_is_bitwise_deterministic(key):
    """30 forward passes on frozen weights must give bitwise-identical outputs.

    The old repository measured ppo_lfm log-prob sd 0.0166 and value sd 0.0201 here,
    because dropout was live in rollout, update and eval (defect D10).
    """
    model = build_model(key, OBS_DIM, ACTION_DIM, seed=24)
    model.eval()
    g = _gen(25)
    obs = torch.randn(7, OBS_DIM, generator=g)
    dt = torch.rand(7, generator=g) + 0.5
    state = model.initial_state(7)
    with torch.no_grad():
        ref_logits, ref_value, ref_state = model.step(obs, dt, state)
        for _ in range(30):
            logits, value, new_state = model.step(obs, dt, state)
            assert torch.equal(logits, ref_logits), key
            assert torch.equal(value, ref_value), key
            for a, b in zip(new_state, ref_state):
                assert torch.equal(a, b), key


@pytest.mark.parametrize("key", KEYS)
def test_deterministic_act(key):
    """Two successive greedy actions on identical input must be identical."""
    model = build_model(key, OBS_DIM, ACTION_DIM, seed=26)
    model.eval()
    g = _gen(27)
    obs = torch.randn(9, OBS_DIM, generator=g)
    dt = torch.rand(9, generator=g) + 0.5
    state = model.initial_state(9)
    with torch.no_grad():
        first = model.act(obs, dt, state, deterministic=True)
        second = model.act(obs, dt, state, deterministic=True)
    assert torch.equal(first[0], second[0]), key
    assert torch.equal(first[1], second[1]), key
    assert torch.equal(first[2], second[2]), key


def test_eval_determinism_test_has_teeth():
    """Negative control for D10: with dropout live, eval() is what saves us.

    The shipped registry uses dropout = 0.0 everywhere, so the test above would pass
    even if train/eval mode were ignored.  Here we build a dropout-carrying model and
    show that (a) eval() is bitwise deterministic and (b) train() is not -- i.e. the
    mode switch is genuinely wired up, which is what D10 lacked.
    """
    model = RecurrentActorCritic(
        OBS_DIM,
        ACTION_DIM,
        ("attn", "attn"),
        32,
        generator=_gen(28),
        cell_kwargs={"num_heads": 1, "dropout": 0.5},
    )
    g = _gen(29)
    obs = torch.randn(6, OBS_DIM, generator=g)
    dt = torch.rand(6, generator=g) + 0.5
    state = model.initial_state(6)

    model.eval()
    with torch.no_grad():
        reference, _, _ = model.step(obs, dt, state)
        for _ in range(30):
            assert torch.equal(model.step(obs, dt, state)[0], reference)
        greedy = model.act(obs, dt, state, deterministic=True)[0]
        assert torch.equal(model.act(obs, dt, state, deterministic=True)[0], greedy)

    model.train()
    with torch.no_grad():
        draws = torch.stack([model.step(obs, dt, state)[0] for _ in range(30)])
    assert float(draws.std(dim=0).max()) > 1e-4, "dropout never fired; test is vacuous"


def test_stochastic_action_sampling_uses_the_supplied_generator():
    """Spec section 4.4 rule 8: never the global RNG, never Categorical.sample()."""
    model = build_model("ppo_gru", OBS_DIM, ACTION_DIM, seed=30)
    model.eval()
    g = _gen(31)
    obs = torch.randn(64, OBS_DIM, generator=g)
    dt = torch.rand(64, generator=g) + 0.5
    state = model.initial_state(64)
    with torch.no_grad():
        a = model.act(obs, dt, state, deterministic=False, generator=_gen(5))[0]
        b = model.act(obs, dt, state, deterministic=False, generator=_gen(5))[0]
        c = model.act(obs, dt, state, deterministic=False, generator=_gen(6))[0]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ===========================================================================
# Information parity: no model gets privileged inputs
# ===========================================================================
def test_every_model_has_the_same_input_contract():
    """The transformer baseline must not receive anything the recurrent models don't.

    Every model consumes exactly ``(obs, dt, state)`` through an encoder of identical
    shape ``Linear(obs_dim + 1, H)``.  There is no extra observation feature, no frame
    stack and no privileged channel anywhere in the registry.
    """
    for key in KEYS:
        model = build_model(key, OBS_DIM, ACTION_DIM, seed=32)
        assert isinstance(model.encoder, nn.Linear)
        assert model.encoder.in_features == OBS_DIM + 1
        assert model.encoder.out_features == model.hidden_dim
        # only the two blocks and the two LayerNorms sit between encoder and heads
        top_level = {name.split(".")[0] for name, _ in model.named_parameters()}
        assert top_level == {"encoder", "blocks", "norms", "actor", "critic"}, key


@pytest.mark.parametrize("key", KEYS)
def test_no_model_sees_the_future(key):
    """Perturbing observations after step t must not change outputs at or before t."""
    model = build_model(key, OBS_DIM, ACTION_DIM, seed=33)
    model.eval()
    batch, horizon, cut = 4, 16, 9
    g = _gen(34)
    obs = torch.randn(batch, horizon, OBS_DIM, generator=g)
    dt = torch.rand(batch, horizon, generator=g) + 0.5
    state = model.initial_state(batch)
    perturbed = obs.clone()
    perturbed[:, cut:] += 4.0
    with torch.no_grad():
        base_logits, base_values, _ = model.unroll(obs, dt, state)
        new_logits, new_values, _ = model.unroll(perturbed, dt, state)
    assert torch.allclose(base_logits[:, :cut], new_logits[:, :cut], atol=1e-5), key
    assert torch.allclose(base_values[:, :cut], new_values[:, :cut], atol=1e-5), key
    assert float((base_logits[:, cut:] - new_logits[:, cut:]).abs().max()) > 1e-3, key


@pytest.mark.parametrize("key", KEYS)
def test_dt_reaches_every_model_through_the_encoder(key):
    """Path (a) of spec section 1.7: dt is an input feature for ALL models.

    The ppo_cfc vs ppo_cfc_dtblind contrast isolates path (b) with path (a) held
    fixed, which is only meaningful if path (a) is genuinely present everywhere.
    """
    model = build_model(key, OBS_DIM, ACTION_DIM, seed=35)
    model.eval()
    g = _gen(36)
    obs = torch.randn(6, OBS_DIM, generator=g)
    state = model.initial_state(6)
    with torch.no_grad():
        low, _, _ = model.step(obs, torch.full((6,), 0.25), state)
        high, _, _ = model.step(obs, torch.full((6,), 2.5), state)
    assert float((low - high).abs().max()) > 1e-5, key


# ===========================================================================
# M17 -- the capacity solver must be load-bearing
# ===========================================================================
# `solve_hidden_dim` is the public name of the capacity-matching solver, but the
# registry reaches the width through `solve_width_and_heads`.  Nothing previously
# tied the two together, so `solve_hidden_dim` could return a constant and the whole
# suite stayed green: the function was documentation, not machinery.  These tests
# make it machinery.


def test_solved_width_is_the_width_the_registry_builds(summaries):
    """Every built model's ``hidden_dim`` must be exactly what ``solve_hidden_dim``
    returns for its spec, and that width must be what puts it on the shared
    parameter budget.  A solver returning a constant fails on the first model."""
    widths: dict[str, int] = {}
    for key, spec in MODEL_REGISTRY.items():
        solved = solve_hidden_dim(spec, OBS_DIM, ACTION_DIM)
        assert solved == solve_width_and_heads(spec, OBS_DIM, ACTION_DIM)[0], key
        assert summaries[key]["hidden_dim"] == solved, key

        model = build_model(key, OBS_DIM, ACTION_DIM, seed=5)
        assert model.hidden_dim == solved, (
            f"{key}: built width {model.hidden_dim} != solved width {solved}"
        )
        error = abs(count_parameters(model) - PARAM_TARGET) / PARAM_TARGET
        assert error <= PARAM_TOLERANCE, (key, solved, error)
        widths[key] = solved

    # Non-degenerate: capacity matching means different cell families need
    # different widths, so the solver cannot be a constant function.
    assert len(set(widths.values())) > 1, widths


def test_no_single_width_can_capacity_match_every_model():
    """Teeth for the test above.

    If some constant ``H`` did put every model on budget, then a constant solver
    would be harmless and the test above would be pinning nothing.  It does not:
    for every candidate width -- including each model's own solved width -- at
    least one other model lands outside ``PARAM_TOLERANCE``.
    """
    candidates = sorted(
        {solve_hidden_dim(spec, OBS_DIM, ACTION_DIM) for spec in MODEL_REGISTRY.values()}
        | {128}
    )
    assert len(candidates) > 1
    for hidden in candidates:
        off_budget = []
        for key, spec in MODEL_REGISTRY.items():
            kwargs = dict(spec.cell_kwargs)
            if "attn" in spec.cell_kinds:
                kwargs["num_heads"] = 1  # keep every width feasible for attention
            model = RecurrentActorCritic(
                OBS_DIM, ACTION_DIM, spec.cell_kinds, hidden, cell_kwargs=kwargs
            )
            if abs(count_parameters(model) - PARAM_TARGET) / PARAM_TARGET > PARAM_TOLERANCE:
                off_budget.append(key)
        assert off_budget, (
            f"hidden_dim={hidden} puts every model on budget, so the per-model "
            "solver would not be load-bearing"
        )


# ===========================================================================
# M29b -- backpropagation through time, at the whole-model level
# ===========================================================================


def _reference_unroll_blocks(model) -> bool:
    """True iff every block of ``model`` uses ``RecurrentBlockBase.unroll``."""
    return all(type(b).unroll is RecurrentBlockBase.unroll for b in model.blocks)


@pytest.mark.parametrize("key", KEYS)
def test_bptt_reaches_the_first_timestep(key):
    """``RecurrentActorCritic.unroll`` is the PPO update's only view of the policy.

    If the hidden trajectory is detached inside it, the update still runs, still
    produces finite losses and still moves every parameter -- it just stops being a
    recurrent update.  The observable is that the observation at ``t=0`` no longer
    influences the outputs at ``t=T-1``, a path that exists only through the carried
    state.  ``ppo_mlp`` is the positive control: it is memoryless by construction, so
    that influence is exactly zero for it and must be.
    """
    model = build_model(key, OBS_DIM, ACTION_DIM, seed=5)
    model.train()
    batch, length = 4, 6
    g = _gen(51)
    obs = torch.randn(batch, length, OBS_DIM, generator=g, requires_grad=True)
    dt = torch.full((batch, length), 0.3)
    state = tuple(s.clone().requires_grad_(True) for s in model.initial_state(batch))

    logits, values, _ = model.unroll(obs, dt, state)
    loss = logits[:, -1].sum() + values[:, -1].sum()
    grads = torch.autograd.grad(loss, [obs, *state], allow_unused=True)
    grad_obs, grad_state = grads[0], grads[1:]

    assert float(grad_obs[:, -1].abs().max()) > 0.0, key  # the direct path is alive
    memoryless = all(s == 0 for s in model.state_sizes)
    if memoryless:
        assert float(grad_obs[:, 0].abs().max()) == 0.0, key
        return

    assert float(grad_obs[:, 0].abs().max()) > 0.0, (
        f"{key}: the observation at t=0 does not reach the output at t=T-1 -- "
        "backpropagation through time has been severed"
    )
    if _reference_unroll_blocks(model):
        # Blocks on the reference unroll additionally carry a live gradient path
        # back to the state they were handed, which is what the PPO update replays.
        for i, gs in enumerate(grad_state):
            assert gs is not None, f"{key}: block {i} dropped its initial state from the graph"
            assert float(gs.abs().max()) > 0.0, f"{key} block {i}"
