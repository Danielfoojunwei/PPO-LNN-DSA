"""Tests for :mod:`dsa.federated`.

These tests exist because the previous implementation passed a test suite that did
not contain any of them.  Each group below pins one of the confirmed defects:

* **D7 (the hierarchy was a no-op)** -- ``test_hierarchical_is_not_flat_fedavg``,
  with ``test_one_edge_round_collapses_to_flat`` as the positive control that
  proves the comparison is sensitive to the mechanism rather than to noise.
* **D7 (communication was closed-form arithmetic)** -- ``test_no_closed_form_bytes``,
  ``test_ledger_measures_the_actual_tensors``, ``test_cloud_traffic_is_reduced``.
* **D8 (recorded 3072 while executing 18 432)** -- ``test_train_steps_equals_reality``.
* **Heterogeneity** -- ``test_edges_are_behaviourally_separable``, which checks the
  clustering in the *realised dynamics* and not merely in the config, because a
  delta that lands inside an environment clip exists only on paper.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
import torch

import dsa.federated.runners as runners
from dsa.envs.scenarios import get_scenario
from dsa.envs.spectrum import SpectrumEnv
from dsa.federated.aggregate import (
    BYTES_PER_MEGABYTE,
    CLOUD_LINKS,
    CommunicationLedger,
    Transfer,
    fedavg,
    state_dict_bytes,
)
from dsa.federated.runners import (
    FEDERATED_ARMS,
    RESULT_KEYS,
    PooledVectorEnv,
    client_env_seed,
    federated_row,
    flat_federated_core,
    hierarchical_federated_core,
    lane_seeds,
    run_arm,
)
from dsa.federated.topology import (
    EDGE_DRIFT,
    EDGE_INTERFERENCE,
    MIN_BACKGROUND_USERS,
    ClientSpec,
    FederatedConfig,
    build_topology,
    clients_of_edge,
)
from dsa.seeding import derive_seed

torch.set_num_threads(1)

FED_PKG = pathlib.Path(__file__).resolve().parents[1] / "dsa" / "federated"


# --------------------------------------------------------------------------- #
# Shared fixtures.  Federated runs are expensive; each is executed once.
# --------------------------------------------------------------------------- #


def _tiny(**overrides) -> FederatedConfig:
    """A federated config small enough for CI, structurally identical to the real one."""
    kwargs = dict(
        model_key="ppo_mlp",
        num_clients=4,
        num_edges=2,
        cloud_rounds=2,
        edge_rounds_per_cloud_round=2,
        local_steps=64,
        local_num_envs=1,
        horizon=64,
        eval_episodes=4,
    )
    kwargs.update(overrides)
    return FederatedConfig(**kwargs)


@pytest.fixture(scope="module")
def tiny_fed() -> FederatedConfig:
    return _tiny()


@pytest.fixture(scope="module")
def arm_results(tiny_fed: FederatedConfig) -> dict[str, dict]:
    """One run of each arm at the same base seed."""
    return {arm: run_arm(arm, tiny_fed, 7) for arm in FEDERATED_ARMS}


@pytest.fixture(scope="module")
def controlled_pair() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Hierarchical and flat final parameters on *byte-identical* client streams.

    ``seed_arm_override`` pins both arms to the same environment-seed label, so the
    clients see the same data in both arms and every difference in the resulting
    global model is attributable to topology alone.  ``edge_rounds=3`` gives the
    edge aggregates three rounds to diverge from the cloud before they are folded
    back in, which is where a hierarchy's effect lives.
    """
    fed = _tiny(cloud_rounds=2, edge_rounds_per_cloud_round=3)
    _, theta_hier = hierarchical_federated_core(fed, 7, seed_arm_override="controlled")
    _, theta_flat = flat_federated_core(fed, 7, seed_arm_override="controlled")
    return theta_hier, theta_flat


@pytest.fixture(scope="module")
def inert_control_pair() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The same comparison with ``edge_rounds_per_cloud_round = 1``.

    With one edge round per cloud round, edge aggregation happens immediately
    before cloud aggregation with no local training in between, and with balanced
    edges an average of edge averages is exactly the average over all clients.
    The hierarchy is therefore a provable no-op here, which makes this pair the
    float-noise floor that :func:`controlled_pair` has to clear.
    """
    fed = _tiny(cloud_rounds=3, edge_rounds_per_cloud_round=1)
    _, theta_hier = hierarchical_federated_core(fed, 7, seed_arm_override="controlled")
    _, theta_flat = flat_federated_core(fed, 7, seed_arm_override="controlled")
    return theta_hier, theta_flat


def _max_abs_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    assert set(a) == set(b)
    return max(float((a[k] - b[k]).abs().max()) for k in a)


# --------------------------------------------------------------------------- #
# aggregate.py -- averaging
# --------------------------------------------------------------------------- #


def test_state_dict_bytes_measures_the_tensors():
    sd = {
        "a": torch.zeros(10, dtype=torch.float32),  # 40 bytes
        "b": torch.zeros(3, 4, dtype=torch.float64),  # 96 bytes
        "c": torch.zeros(5, dtype=torch.int64),  # 40 bytes
    }
    assert state_dict_bytes(sd) == 40 + 96 + 40


def test_state_dict_bytes_rejects_non_tensors():
    with pytest.raises(TypeError):
        state_dict_bytes({"a": [1.0, 2.0]})  # type: ignore[dict-item]


def test_fedavg_is_the_arithmetic_mean():
    dicts = [{"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([3.0, 6.0])}]
    out = fedavg(dicts)
    assert torch.allclose(out["w"], torch.tensor([2.0, 4.0]))


def test_fedavg_honours_weights_and_normalises_them():
    dicts = [{"w": torch.tensor([0.0])}, {"w": torch.tensor([10.0])}]
    # Weights need not sum to 1; they are normalised internally.
    assert torch.allclose(fedavg(dicts, [3.0, 1.0])["w"], torch.tensor([2.5]))
    assert torch.allclose(fedavg(dicts, [30.0, 10.0])["w"], torch.tensor([2.5]))
    # A zero weight excludes a contributor entirely.
    assert torch.allclose(fedavg(dicts, [1.0, 0.0])["w"], torch.tensor([0.0]))


def test_fedavg_is_order_invariant():
    """Client ordering must not move the aggregate beyond storage precision.

    The accumulator is float64, so a float64 payload is order-invariant to
    ~1e-15.  A float32 payload inherits one final rounding when the float64 sum is
    cast back, which is ~6e-8 -- float32 resolution, not an algorithmic
    dependence.  Both bounds are asserted so the distinction stays visible: this
    residual is the same magnitude as the 1.19e-07 the audit measured for the old
    hierarchy, which is exactly why "differs by ~1e-7" has to be read as "does not
    differ".
    """
    f64 = [{"w": torch.randn(16, dtype=torch.float64)} for _ in range(5)]
    assert _max_abs_diff(fedavg(f64), fedavg(list(reversed(f64)))) < 1e-14

    f32 = [{"w": t["w"].to(torch.float32)} for t in f64]
    residual = _max_abs_diff(fedavg(f32), fedavg(list(reversed(f32))))
    assert residual < 1e-6
    assert fedavg(f32)["w"].dtype == torch.float32


def test_fedavg_preserves_dtype_and_passes_through_matching_integers():
    dicts = [
        {"w": torch.tensor([1.0], dtype=torch.float32), "n": torch.tensor([7])},
        {"w": torch.tensor([2.0], dtype=torch.float32), "n": torch.tensor([7])},
    ]
    out = fedavg(dicts)
    assert out["w"].dtype == torch.float32
    assert out["n"].dtype == torch.int64
    assert int(out["n"]) == 7


def test_fedavg_refuses_to_average_disagreeing_integer_buffers():
    dicts = [{"n": torch.tensor([1])}, {"n": torch.tensor([2])}]
    with pytest.raises(ValueError, match="cannot be averaged"):
        fedavg(dicts)


@pytest.mark.parametrize(
    "dicts, weights",
    [
        ([], None),
        ([{"a": torch.zeros(1)}, {"b": torch.zeros(1)}], None),
        ([{"a": torch.zeros(1)}], [1.0, 1.0]),
        ([{"a": torch.zeros(1)}], [-1.0]),
        ([{"a": torch.zeros(1)}], [0.0]),
    ],
)
def test_fedavg_rejects_malformed_input(dicts, weights):
    with pytest.raises(ValueError):
        fedavg(dicts, weights)


# --------------------------------------------------------------------------- #
# aggregate.py -- the ledger
# --------------------------------------------------------------------------- #


def test_ledger_measures_the_actual_tensors():
    """The recorded size must come from the payload, not from a formula."""
    ledger = CommunicationLedger()
    small = {"w": torch.zeros(10, dtype=torch.float32)}  # 40 bytes
    big = {"w": torch.zeros(1000, dtype=torch.float32)}  # 4000 bytes
    ledger.record("cloud_up", "client0", "cloud", small, 0)
    ledger.record("cloud_down", "cloud", "client0", big, 0)
    totals = ledger.totals()
    assert totals["cloud_bytes"] == 4040.0
    assert totals["edge_local_bytes"] == 0.0
    assert totals["total_bytes"] == 4040.0
    assert totals["num_transfers"] == 2.0
    assert totals["cloud_megabytes"] == 4040.0 / BYTES_PER_MEGABYTE


def test_ledger_link_taxonomy():
    ledger = CommunicationLedger()
    payload = {"w": torch.zeros(1)}
    for link in ("cloud_up", "cloud_down", "edge_up", "edge_down"):
        ledger.record(link, "a", "b", payload, 0)
    totals = ledger.totals()
    assert totals["cloud_bytes"] == 8.0  # two cloud links x 4 bytes
    assert totals["edge_local_bytes"] == 8.0
    assert totals["total_bytes"] == totals["cloud_bytes"] + totals["edge_local_bytes"]
    assert sorted(ledger.counts_by_link()) == ["cloud_down", "cloud_up", "edge_down", "edge_up"]


def test_ledger_rejects_unknown_link():
    with pytest.raises(ValueError, match="link must be one of"):
        CommunicationLedger().record("satellite_up", "a", "b", {"w": torch.zeros(1)}, 0)


def test_empty_ledger_reports_zeros_not_missing_keys():
    totals = CommunicationLedger().totals()
    assert set(totals) == {
        "cloud_bytes",
        "edge_local_bytes",
        "total_bytes",
        "cloud_megabytes",
        "edge_local_megabytes",
        "total_megabytes",
        "num_transfers",
    }
    assert all(v == 0.0 for v in totals.values())


def test_transfer_is_immutable():
    t = Transfer("cloud_up", "a", "b", 4, 0)
    with pytest.raises(Exception):
        t.num_bytes = 8  # type: ignore[misc]


def test_no_closed_form_bytes():
    """No module outside ``aggregate.py`` may do arithmetic on a byte count.

    Defect D7's communication numbers were hand-added constants at four call
    sites.  A grep is not enough -- ``21.5005`` never appears literally -- so this
    walks the AST of every other module in the package and fails on any binary
    operation whose operands mention ``bytes``.  Reading a byte count out of
    ``CommunicationLedger.totals()`` and putting it in a row is fine; multiplying
    or adding one is not.
    """
    offenders: list[str] = []
    for path in sorted(FED_PKG.glob("*.py")):
        if path.name == "aggregate.py":
            continue  # the one sanctioned home for byte arithmetic
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                src = ast.unparse(node)
                if "bytes" in src.lower() and "state_dict_bytes" not in src:
                    offenders.append(f"{path.name}:{node.lineno}: {src}")
    assert offenders == [], "byte arithmetic outside aggregate.py:\n" + "\n".join(offenders)

    # And inside aggregate.py, tensor sizes may only be read in one function.
    agg = (FED_PKG / "aggregate.py").read_text()
    tree = ast.parse(agg)
    fns = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and "element_size" in ast.unparse(n)
    }
    assert fns == {"state_dict_bytes"}, f"element_size() used outside state_dict_bytes: {fns}"


# --------------------------------------------------------------------------- #
# topology.py
# --------------------------------------------------------------------------- #


def test_total_env_steps_per_arm_arithmetic():
    fed = FederatedConfig()
    assert fed.total_env_steps_per_arm() == (
        fed.num_clients
        * fed.cloud_rounds
        * fed.edge_rounds_per_cloud_round
        * fed.local_steps
    )
    # The flat arm runs R*E rounds precisely so its budget matches.
    assert fed.flat_rounds() == fed.cloud_rounds * fed.edge_rounds_per_cloud_round
    assert (
        fed.num_clients * fed.flat_rounds() * fed.local_steps
        == fed.total_env_steps_per_arm()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_clients": 7, "num_edges": 2},  # not divisible
        {"local_steps": 700, "local_num_envs": 12, "horizon": 64},  # not a whole rollout
        {"cloud_rounds": 0},
        {"edge_rounds_per_cloud_round": 0},
        {"num_edges": 99},
        {"eval_episodes": 0},
    ],
)
def test_federated_config_validation(kwargs):
    with pytest.raises(ValueError):
        FederatedConfig(**kwargs)


def test_build_topology_is_deterministic_and_needs_no_seed():
    fed = FederatedConfig()
    base = get_scenario(fed.scenario)
    assert build_topology(base, fed) == build_topology(base, fed)


def test_edge_clustering_in_config():
    """Same edge => same dynamics regime.  Different edge => different regime."""
    fed = FederatedConfig()
    topo = build_topology(get_scenario(fed.scenario), fed)
    for edge in range(fed.num_edges):
        peers = clients_of_edge(topo, edge)
        assert len(peers) == fed.clients_per_edge()
        assert len({c.scenario_config.drift_strength for c in peers}) == 1
        assert len({c.scenario_config.interference_scale for c in peers}) == 1
    for a in clients_of_edge(topo, 0):
        for b in clients_of_edge(topo, 1):
            assert a.scenario_config.drift_strength != b.scenario_config.drift_strength
            assert (
                a.scenario_config.interference_scale
                != b.scenario_config.interference_scale
            )


def test_heterogeneity_deltas_survive_into_the_config():
    """No silent clamp on the between-edge axes."""
    fed = FederatedConfig()
    base = get_scenario(fed.scenario)
    topo = build_topology(base, fed)
    for edge in range(fed.num_edges):
        cfg = clients_of_edge(topo, edge)[0].scenario_config
        assert cfg.drift_strength == pytest.approx(base.drift_strength + EDGE_DRIFT[edge])
        assert cfg.interference_scale == pytest.approx(
            base.interference_scale + EDGE_INTERFERENCE[edge]
        )


def test_within_edge_nuisance_axis_varies_and_is_floored():
    """Clients on an edge are similar, not identical, and never contention-free."""
    fed = FederatedConfig()
    topo = build_topology(get_scenario(fed.scenario), fed)
    for edge in range(fed.num_edges):
        loads = {c.scenario_config.num_background_users for c in clients_of_edge(topo, edge)}
        assert len(loads) > 1, "within-edge nuisance variation was erased"
    for c in topo:
        assert c.scenario_config.num_background_users >= MIN_BACKGROUND_USERS
        # fixed-load clients must keep load_min == load_max == the constant, or the
        # config would describe a process the environment does not run.
        assert (
            c.scenario_config.load_min
            == c.scenario_config.load_max
            == c.scenario_config.num_background_users
        )


def _probe(cfg, seeds, key: str) -> float:
    """Mean of a ground-truth diagnostic over a fixed action sequence."""
    vals: list[float] = []
    for seed in seeds:
        env = SpectrumEnv(cfg, seed)
        for t in range(cfg.max_steps):
            _obs, _r, _term, _trunc, info = env.step(t % cfg.action_dim)
            vals.append(float(info[key]))
    return float(np.mean(vals))


def test_edges_are_behaviourally_separable():
    """The clustering must exist in the realised dynamics, not just in the config.

    Every client is probed on the *same* environment seeds, so the probe is paired
    and the only thing that differs between two clients is their config.  Two
    statistics are used, chosen so that the between-edge axes can be read
    independently of the within-edge nuisance axis:

    ``base_busy_std``
        Cross-channel dispersion of latent occupancy.  Background load does not
        enter the occupancy process at all, so this isolates ``drift_strength``.
    ``interference_mean``
        Load *does* enter interference, so this one is compared **within each load
        stratum**, which is what makes ``interference_scale`` identifiable.

    If either edge delta were being flattened by one of the environment's clips,
    the corresponding margin would collapse to zero and this test would fail.
    """
    fed = FederatedConfig()
    topo = build_topology(get_scenario(fed.scenario), fed)
    seeds = [derive_seed(1234, "probe", i) for i in range(8)]

    busy = {c.client_id: _probe(c.scenario_config, seeds, "base_busy_std") for c in topo}
    interference = {
        c.client_id: _probe(c.scenario_config, seeds, "interference_mean") for c in topo
    }
    edge_of = {c.client_id: c.edge_id for c in topo}
    load_of = {c.client_id: c.scenario_config.num_background_users for c in topo}

    # Drift axis: load-independent, so clients on one edge are bit-identical and
    # the edges are cleanly ordered.
    for edge in range(fed.num_edges):
        vals = {busy[c.client_id] for c in clients_of_edge(topo, edge)}
        assert len(vals) == 1, "occupancy dispersion varies within an edge"
    e0 = busy[clients_of_edge(topo, 0)[0].client_id]
    e1 = busy[clients_of_edge(topo, 1)[0].client_id]
    assert e1 - e0 > 1e-3, f"drift heterogeneity did not survive into the dynamics ({e0}, {e1})"

    # Interference axis: stratify on the confounding load level.
    strata = sorted(set(load_of.values()))
    assert len(strata) >= 2
    for load in strata:
        a = [interference[c] for c in interference if edge_of[c] == 0 and load_of[c] == load]
        b = [interference[c] for c in interference if edge_of[c] == 1 and load_of[c] == load]
        if not a or not b:
            continue
        assert min(b) - max(a) > 5e-3, (
            f"at load={load} the interference axis is not separable: "
            f"edge0={sorted(a)} edge1={sorted(b)}"
        )


def test_load_axis_is_only_partially_balanced_across_edges():
    """Documents a real residual confound rather than asserting it away.

    The specification fixes the within-edge nuisance offset at
    ``(client_id % 3) - 1``.  Over contiguous edge blocks of four that period-3
    pattern does not balance: edge 1 ends up carrying slightly more background
    load than edge 0.  The imbalance is small next to the deliberate
    ``interference_scale`` delta, but it is not zero, so it is pinned here -- if
    a future change makes it larger, this test fails and the stratified
    separability test above stops being sufficient.
    """
    fed = FederatedConfig()
    topo = build_topology(get_scenario(fed.scenario), fed)
    means = [
        float(
            np.mean(
                [c.scenario_config.num_background_users for c in clients_of_edge(topo, e)]
            )
        )
        for e in range(fed.num_edges)
    ]
    imbalance = abs(means[1] - means[0])
    assert imbalance < 0.5, (
        f"between-edge load imbalance {imbalance} is no longer a minor residual; "
        "the load axis has become a second between-edge factor"
    )


def test_clients_of_edge_partitions_the_topology():
    fed = FederatedConfig()
    topo = build_topology(get_scenario(fed.scenario), fed)
    seen: list[ClientSpec] = []
    for edge in range(fed.num_edges):
        seen.extend(clients_of_edge(topo, edge))
    assert sorted(c.client_id for c in seen) == list(range(fed.num_clients))


# --------------------------------------------------------------------------- #
# runners -- compute matching (D8)
# --------------------------------------------------------------------------- #


def test_train_steps_equals_reality(arm_results, tiny_fed):
    """Recorded steps == declared steps == steps actually executed, for all arms.

    The old repository recorded ``3072`` in all fifteen rows of its federated CSV
    while its round loop executed ``6 rounds x 6 clients x 512 = 18 432`` real
    environment steps -- a 6x compute advantage for the federated arms, invisible
    in the results.  ``train_steps`` here is summed from
    ``RecurrentPPO.total_env_steps`` over every agent each runner actually
    trained, so the two can only agree if the loop really ran the declared budget.
    """
    declared = tiny_fed.total_env_steps_per_arm()
    for arm, result in arm_results.items():
        assert result["declared_train_steps"] == declared
        assert result["train_steps"] == declared, (
            f"{arm}: recorded {result['train_steps']} vs declared {declared}"
        )


def test_arms_are_matched_on_optimizer_work_too(arm_results):
    """Env steps, rollouts and gradient steps all match; only topology differs."""
    updates = {arm: r["num_updates"] for arm, r in arm_results.items()}
    assert len(set(updates.values())) == 1, updates


def test_runner_result_keys_are_exact(arm_results):
    for arm, result in arm_results.items():
        assert tuple(result.keys()) == RESULT_KEYS, arm


def test_round_history_is_emitted(arm_results, tiny_fed):
    for arm, result in arm_results.items():
        history = result["round_history"]
        assert history, f"{arm} emitted no per-round metrics"
        for row in history:
            assert "env_steps_cumulative" in row
            assert "mean_rollout_return" in row
            assert "mean_policy_entropy_nats" in row
            assert "total_bytes_cumulative" in row
            # Averaging an already-averaged diagnostic must not double-prefix the
            # column, or the quantity disappears from every downstream table.
            assert not [k for k in row if k.startswith("mean_mean_")]
        # cumulative counters are monotone
        steps = [r["env_steps_cumulative"] for r in history]
        assert steps == sorted(steps)
        assert steps[-1] == float(tiny_fed.total_env_steps_per_arm())


def test_ppo_ratio_sanity_survives_the_federated_layer(arm_results):
    """Spec 4.4 rule 6, carried through to federated rows.

    On epoch 0 / minibatch 0 the recomputed policy *is* the behaviour policy, so
    the PPO ratio must be 1.  A federated wrapper that reloaded parameters into
    the wrong agent, or left dropout live, would show up here and nowhere else.
    """
    for arm, result in arm_results.items():
        row = federated_row(result)
        assert row["max_first_epoch_ratio_deviation"] < 1e-4, arm


# --------------------------------------------------------------------------- #
# runners -- the hierarchy is real (D7)
# --------------------------------------------------------------------------- #


def test_hierarchical_is_not_flat_fedavg(controlled_pair):
    """The regression test the old repository needed and did not have.

    Both arms here are driven by byte-identical client environment streams and an
    identical initial model, so the *only* difference between them is the
    aggregation topology.  If the resulting global parameters come out equal to
    within float32 summation noise, the hierarchy computed nothing: edge
    aggregation would be a pure relabelling of a flat average, and every
    edge-local byte the arm was charged would have bought exactly zero change in
    the model.  That is precisely what the old code did -- the live repro measured
    ``max|hier - flat| = 1.19e-07`` over all thirty tensors, which is the
    magnitude of float32 rounding, not of an algorithm.

    The difference has to come from somewhere real, and it does: at edge round
    ``k >= 1`` a hierarchical client starts from its *edge's* aggregate, which no
    other edge and not the cloud has seen, whereas a flat client always starts
    from the global average.  See ``test_one_edge_round_collapses_to_flat`` for
    the control that shows this test is measuring that mechanism.
    """
    theta_hier, theta_flat = controlled_pair
    diff = _max_abs_diff(theta_hier, theta_flat)
    assert diff > 1e-3, (
        f"hierarchical and flat global models agree to {diff:.3e} -- the hierarchy "
        "is inert (defect D7 has regressed)"
    )


def test_one_edge_round_collapses_to_flat(inert_control_pair):
    """Positive control: with E=1 the hierarchy is *provably* a no-op.

    The arms must agree to float noise here -- and they do, at ~1e-7, the same
    magnitude the old broken implementation produced at E=2.  That is what makes
    ``test_hierarchical_is_not_flat_fedavg`` meaningful: the same comparison, on
    the same machine, in the same dtype, returns "equal" when the hierarchy
    genuinely does nothing and "different" when it does something.  Without this
    control, a passing non-equivalence test could just mean the tolerance was set
    below the noise floor.
    """
    inert = _max_abs_diff(*inert_control_pair)
    assert inert < 1e-5, f"E=1 should collapse to flat FedAvg, got {inert:.3e}"


def test_active_hierarchy_dwarfs_the_inert_control(controlled_pair, inert_control_pair):
    """The E>1 difference must be orders of magnitude above the E=1 float floor."""
    floor = _max_abs_diff(*inert_control_pair)
    active = _max_abs_diff(*controlled_pair)
    assert active > 100 * max(floor, 1e-12), (
        f"active hierarchy diff {active:.3e} is not clearly above the inert "
        f"float floor {floor:.3e}"
    )


def test_edge_aggregation_is_recorded_in_the_round_history(arm_results, tiny_fed):
    """Edge rounds that do not sync to the cloud must be visible as such."""
    history = arm_results["hierarchical_federated"]["round_history"]
    levels = [row["aggregation_level"] for row in history]
    assert "edge" in levels, "no edge-only aggregation round was recorded"
    assert levels.count("cloud") == tiny_fed.cloud_rounds
    flat_levels = {
        row["aggregation_level"] for row in arm_results["flat_federated"]["round_history"]
    }
    assert flat_levels == {"cloud"}


# --------------------------------------------------------------------------- #
# runners -- communication (D7)
# --------------------------------------------------------------------------- #


def test_cloud_traffic_is_reduced(arm_results):
    """The entire justification for the topology, asserted.

    If the hierarchy did not cut wide-area traffic there would be no reason to pay
    for it, and the correct response would be to change the design rather than the
    metric.
    """
    hier = arm_results["hierarchical_federated"]["communication"]
    flat = arm_results["flat_federated"]["communication"]
    assert hier["cloud_bytes"] < flat["cloud_bytes"]
    assert hier["total_bytes"] > 0
    assert hier["edge_local_bytes"] > 0


def test_hierarchical_moves_more_total_bytes(arm_results):
    """The cost side of the same trade, reported with equal prominence."""
    hier = arm_results["hierarchical_federated"]["communication"]
    flat = arm_results["flat_federated"]["communication"]
    assert hier["total_bytes"] > flat["total_bytes"]


def test_cloud_bytes_equal_the_sum_over_cloud_links(tiny_fed):
    """``totals()`` must be a sum over recorded transfers, with nothing added."""
    result, _theta = hierarchical_federated_core(tiny_fed, 7)
    # Rebuild the ledger's own accounting from a fresh run of the same arm and
    # check the aggregate against the individual transfers.
    ledger = CommunicationLedger()
    payloads = [{"w": torch.zeros(n, dtype=torch.float32)} for n in (5, 9, 17)]
    for i, p in enumerate(payloads):
        ledger.record("cloud_up", "edge0", "cloud", p, i)
        ledger.record("edge_down", "edge0", "client0", p, i)
    expected_cloud = sum(
        t.num_bytes for t in ledger.transfers if t.link.startswith("cloud")
    )
    assert ledger.totals()["cloud_bytes"] == float(expected_cloud)
    assert all(link in CLOUD_LINKS for link in ("cloud_up", "cloud_down"))
    assert result["communication"]["cloud_bytes"] > 0


def test_transfer_counts_match_the_topology(tiny_fed):
    """Every transfer the protocol implies is recorded, and no extras."""
    hier, _ = hierarchical_federated_core(tiny_fed, 7)
    flat, _ = flat_federated_core(tiny_fed, 7)
    c, g = tiny_fed.num_clients, tiny_fed.num_edges
    r, e = tiny_fed.cloud_rounds, tiny_fed.edge_rounds_per_cloud_round

    # hierarchical: initial cloud->edge, then per cloud round one up and one down
    # per edge; client<->edge traffic every local phase.
    expected_hier = (g + r * g) + (r * g) + 2 * (r * e * c)
    assert hier["communication"]["num_transfers"] == float(expected_hier)

    # flat: initial cloud->client, then per round one up and one down per client.
    expected_flat = (c + tiny_fed.flat_rounds() * c) + tiny_fed.flat_rounds() * c
    assert flat["communication"]["num_transfers"] == float(expected_flat)


def test_centralized_transmits_nothing(arm_results):
    """A performance reference, not a communication baseline."""
    comm = arm_results["centralized"]["communication"]
    assert comm["total_bytes"] == 0.0
    assert comm["num_transfers"] == 0.0


def test_parameter_bytes_matches_the_model(arm_results):
    for arm, result in arm_results.items():
        assert result["parameter_bytes"] > 0
        # dense float32 parameters: 4 bytes each (plus any non-parameter buffers)
        assert result["parameter_bytes"] >= 4 * result["parameter_count"]


# --------------------------------------------------------------------------- #
# runners -- seeding and determinism
# --------------------------------------------------------------------------- #


def test_client_env_seed_excludes_the_model_key():
    """Env streams must not move when the backbone changes (spec 3.2)."""
    a = client_env_seed(7, "hierarchical_federated", 1, 0, 3)
    b = client_env_seed(7, "hierarchical_federated", 1, 0, 3)
    assert a == b
    assert a == derive_seed(7, "fed", "hierarchical_federated", 1, 0, 3, "env")
    # distinct roles give distinct streams
    assert len(
        {
            client_env_seed(7, "hierarchical_federated", r, k, c)
            for r in range(2)
            for k in range(2)
            for c in range(3)
        }
    ) == 12


def test_lane_seeds_are_distinct_and_pure():
    seeds = lane_seeds(12345, 12, 0)
    assert len(set(seeds)) == 12
    assert seeds == lane_seeds(12345, 12, 0)
    assert set(seeds).isdisjoint(lane_seeds(12345, 12, 1))


def test_seed_arm_override_pins_two_arms_to_one_stream(tiny_fed):
    """Without the override the arm name is in the env seed; with it, it is not."""
    assert client_env_seed(7, "flat_federated", 0, 0, 0) != client_env_seed(
        7, "hierarchical_federated", 0, 0, 0
    )
    assert client_env_seed(7, "ctl", 0, 0, 0) == client_env_seed(7, "ctl", 0, 0, 0)


def test_runs_are_deterministic_except_wall_clock(tiny_fed):
    """Same seed + same config => byte-identical row, modulo ``wall_`` columns."""
    a = federated_row(run_arm("hierarchical_federated", tiny_fed, 7))
    b = federated_row(run_arm("hierarchical_federated", tiny_fed, 7))
    assert set(a) == set(b)
    for key in a:
        if key.startswith("wall_"):
            continue
        assert a[key] == b[key], f"{key} differs between identical runs: {a[key]} vs {b[key]}"


def test_wall_clock_columns_carry_the_wall_prefix(arm_results):
    """Spec 3.5: every wall-clock column is named ``wall_*``.

    ``mean_sim_time`` / ``std_sim_time`` are *simulated* time -- the sum of the
    environment's decision intervals -- which is a deterministic function of the
    seed and therefore correctly has no ``wall_`` prefix.  The substantive claim
    (nothing outside ``^wall_`` varies between identical runs) is proved
    empirically by ``test_runs_are_deterministic_except_wall_clock``; this test
    only pins the naming convention that the reproducibility gate greps for.
    """
    for result in arm_results.values():
        row = federated_row(result)
        assert {k for k in row if k.startswith("wall_")} == {"wall_seconds"}
        assert isinstance(row["wall_seconds"], float)


def test_federated_row_reports_all_three_byte_columns(arm_results):
    for result in arm_results.values():
        row = federated_row(result)
        for key in ("cloud_megabytes", "edge_local_megabytes", "total_megabytes"):
            assert key in row
        assert row["train_steps_match_declared"] is True
        assert "mean_eval_return" in row


def test_federated_row_can_carry_a_config_snapshot(arm_results, tiny_fed):
    """A row must be readable without consulting a separate manifest."""
    result = arm_results["hierarchical_federated"]
    bare = federated_row(result)
    snap = federated_row(result, tiny_fed)
    assert set(bare).issubset(set(snap))
    for key in (
        "config_num_clients",
        "config_num_edges",
        "config_cloud_rounds",
        "config_edge_rounds_per_cloud_round",
        "config_local_steps",
        "config_total_env_steps_per_arm",
    ):
        assert key in snap
    assert snap["config_num_clients"] == float(tiny_fed.num_clients)
    assert snap["config_total_env_steps_per_arm"] == float(
        tiny_fed.total_env_steps_per_arm()
    )
    # the snapshot must not introduce a second non-deterministic column
    assert {k for k in snap if k.startswith("wall_")} == {"wall_seconds"}


def test_row_exposes_both_entropies_under_distinct_names(arm_results):
    """Greedy return alone cannot detect a topology difference; entropy can.

    Measured at the production config, hierarchical and flat end with genuinely
    different parameters (max abs difference 1.3e-3 over 27 of 28 tensors) and yet
    *bit-identical* greedy evaluation returns, because at this budget the argmax
    policy is degenerate -- it selects one channel for every step of every
    episode, so ``action_histogram_entropy`` is 0 while the underlying
    distribution is still near-uniform at ~2.06 nats against ``ln 8 = 2.079``.

    A reader who saw only ``mean_eval_return`` would conclude the two topologies
    are the same computation, which is exactly the wrong conclusion.  Both
    entropies must therefore reach the CSV, under the two distinct names, so the
    degeneracy is visible rather than inferred.
    """
    for arm, result in arm_results.items():
        row = federated_row(result)
        assert "mean_action_histogram_entropy" in row, arm
        assert "mean_policy_entropy_nats" in row, arm
        assert row["mean_policy_entropy_nats"] >= 0.0


def test_run_arm_rejects_unknown_arm(tiny_fed):
    with pytest.raises(KeyError):
        run_arm("federated_but_better", tiny_fed, 7)


# --------------------------------------------------------------------------- #
# PooledVectorEnv
# --------------------------------------------------------------------------- #


def test_pooled_lane_stream_depends_only_on_its_own_config_and_seed():
    """No construction-order effects, mirroring the ``SyncVectorEnv`` guarantee."""
    fed = FederatedConfig()
    topo = build_topology(get_scenario(fed.scenario), fed)
    cfg_a = topo[0].scenario_config
    cfg_b = topo[-1].scenario_config

    solo = PooledVectorEnv([cfg_a], [99])
    obs_solo, _ = solo.reset()

    # Same config and seed, but now in lane 2 of a heterogeneous batch, and after
    # 20 decoy environments have been constructed.
    for _ in range(20):
        PooledVectorEnv([cfg_b], [1])
    mixed = PooledVectorEnv([cfg_b, cfg_b, cfg_a], [1, 2, 99])
    obs_mixed, _ = mixed.reset()
    assert np.array_equal(obs_solo[0], obs_mixed[2])


def test_pooled_env_rejects_mismatched_lane_dimensions():
    fed = FederatedConfig()
    base = get_scenario(fed.scenario)
    from dataclasses import replace

    other = replace(base, name="narrow", num_channels=4)
    with pytest.raises(ValueError, match="pooled lanes must agree"):
        PooledVectorEnv([base, other], [1, 2])


def test_pooled_env_does_not_auto_reset():
    fed = FederatedConfig()
    cfg = get_scenario(fed.scenario)
    env = PooledVectorEnv([cfg], [1])
    env.reset()
    for _ in range(cfg.max_steps):
        env.step(np.zeros(1, dtype=np.int64))
    with pytest.raises(RuntimeError, match="does not auto-reset"):
        env.step(np.zeros(1, dtype=np.int64))


# --------------------------------------------------------------------------- #
# D7 / M13 -- the client-initialisation mechanism itself
# --------------------------------------------------------------------------- #
# `test_hierarchical_is_not_flat_fedavg` compares the two arms' *outcomes*, and it
# survives reverting `init_state=theta_edge[e]` to `init_state=theta_cloud`: a
# hierarchy whose clients all re-initialise from the cloud still differs from flat
# FedAvg, for the incidental reason that it aggregates to the cloud `cloud_rounds`
# times where flat aggregates `cloud_rounds * edge_rounds_per_cloud_round` times.
# The tests below therefore observe the parameters actually handed to each client,
# which is the line the defect lived on.


@pytest.fixture(scope="module")
def hierarchical_init_trace() -> list[dict]:
    """Every ``init_state`` a client received during a real hierarchical run.

    Two cloud rounds of two edge rounds is the smallest topology containing all
    three distinct sources of a client's starting parameters: the seed cloud model
    at ``(r=0, k=0)``, the client's own edge aggregate at ``(r=0, k=1)``, and the
    re-synchronised cloud model at ``(r=1, k=0)``.
    """
    fed = _tiny(cloud_rounds=2, edge_rounds_per_cloud_round=2)
    real = runners._train_local_phase
    trace: list[dict] = []

    def spy(fed_, base_seed, hp, init_state, client, arm_seed_label, cloud_round, edge_round):
        out = real(
            fed_, base_seed, hp, init_state, client, arm_seed_label, cloud_round, edge_round
        )
        trace.append(
            {
                "cloud_round": int(cloud_round),
                "edge_round": int(edge_round),
                "client_id": int(client.client_id),
                "edge_id": int(client.edge_id),
                "init": {k: v.detach().clone() for k, v in dict(init_state).items()},
                "out": {k: v.detach().clone() for k, v in out[0].items()},
            }
        )
        return out

    runners._train_local_phase = spy
    try:
        runners.hierarchical_federated_core(fed, 7)
    finally:
        runners._train_local_phase = real
    return trace


def _phase(trace: list[dict], cloud_round: int, edge_round: int) -> list[dict]:
    return sorted(
        (
            t
            for t in trace
            if t["cloud_round"] == cloud_round and t["edge_round"] == edge_round
        ),
        key=lambda t: t["client_id"],
    )


def _max_abs_diff(a, b) -> float:
    assert sorted(a) == sorted(b)
    return max(
        float((a[k].to(torch.float64) - b[k].to(torch.float64)).abs().max()) for k in a
    )


def test_a_client_starts_from_its_own_edge_not_from_the_cloud(hierarchical_init_trace):
    """Defect D7, part one, pinned at the mechanism.

    At edge round ``k = 1`` a client must be handed its **own edge's** aggregate of
    the ``k = 0`` client models -- not the cloud model, and not the other edge's
    aggregate.  The three are constructed to be genuinely different points, which is
    asserted before the equality is, so the test cannot pass vacuously.
    """
    trace = hierarchical_init_trace
    first, second = _phase(trace, 0, 0), _phase(trace, 0, 1)
    assert len(first) == 4 and len(second) == 4

    # At (0, 0) every client legitimately starts from the one cloud model.
    cloud = first[0]["init"]
    for t in first:
        assert _max_abs_diff(t["init"], cloud) == 0.0, t["client_id"]

    # The edge aggregates that phase (0, 0) produced.
    edge_agg = {
        e: fedavg([t["out"] for t in first if t["edge_id"] == e]) for e in (0, 1)
    }
    # Non-vacuity: cloud, edge 0 and edge 1 are three genuinely different points.
    assert _max_abs_diff(edge_agg[0], edge_agg[1]) > 1e-5
    assert _max_abs_diff(edge_agg[0], cloud) > 1e-5
    assert _max_abs_diff(edge_agg[1], cloud) > 1e-5

    for t in second:
        own, other = t["edge_id"], 1 - t["edge_id"]
        assert _max_abs_diff(t["init"], edge_agg[own]) == 0.0, (
            f"client {t['client_id']} did not start from edge {own}'s aggregate"
        )
        assert _max_abs_diff(t["init"], edge_agg[other]) > 1e-5, t["client_id"]
        assert _max_abs_diff(t["init"], cloud) > 1e-5, (
            f"client {t['client_id']} started from the cloud model (defect D7)"
        )


def test_the_edge_aggregate_persists_across_edge_rounds(hierarchical_init_trace):
    """Defect D7, part two: ``theta_edge[e]`` must survive the ``k`` loop.

    Stated as a property of what the clients received: at ``k = 1`` the two edges'
    clients start from two different points, and the clients on one edge all start
    from the same point.  A hierarchy that recomputed the edge aggregate and dropped
    it would put all four clients back on one shared model.
    """
    second = _phase(hierarchical_init_trace, 0, 1)
    by_edge = {e: [t for t in second if t["edge_id"] == e] for e in (0, 1)}
    for e, members in by_edge.items():
        assert len(members) == 2, e
        assert _max_abs_diff(members[0]["init"], members[1]["init"]) == 0.0, e
    assert _max_abs_diff(by_edge[0][0]["init"], by_edge[1][0]["init"]) > 1e-5


def test_the_cloud_round_resynchronises_every_edge(hierarchical_init_trace):
    """The complement, so that "start from your edge" cannot be satisfied by an arm
    that never synchronises: at the top of cloud round 1 every client starts from the
    same model again, and that model is the client-count-weighted average of the two
    edge aggregates the previous phase produced.
    """
    trace = hierarchical_init_trace
    second, third = _phase(trace, 0, 1), _phase(trace, 1, 0)
    assert len(third) == 4

    resync = third[0]["init"]
    for t in third:
        assert _max_abs_diff(t["init"], resync) == 0.0, t["client_id"]

    edge_agg = [fedavg([t["out"] for t in second if t["edge_id"] == e]) for e in (0, 1)]
    expected = fedavg(edge_agg, weights=[2.0, 2.0])
    assert _max_abs_diff(resync, expected) == 0.0
    # ... and the re-synchronised model is not simply one of the edges.
    assert _max_abs_diff(resync, edge_agg[0]) > 1e-5
    assert _max_abs_diff(resync, edge_agg[1]) > 1e-5
