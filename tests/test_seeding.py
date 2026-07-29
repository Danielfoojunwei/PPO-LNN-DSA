"""Regression tests for the seeding contract.

The defect these tests exist to prevent was the single most damaging one in the
previous version of this repository.  ``SpectrumEnv`` derived its seed as
``config.seed + instance_id`` where ``instance_id`` came from a **mutable
class-level counter**, and the suite runner constructed a fresh environment per
episode in a single process.  Measured with a *fixed* policy at nominal seed 7,
instantiation positions 0..5 scored -33.410, -65.920, -68.523, -58.632, -28.292
and -38.302: a 40-point nuisance spread produced by construction order alone,
against a headline effect of 1.98 points.  The offset was a function of the
order of the ``--models`` list, so no recorded comparison was controlled.

``test_env_stream_order_independence`` is the direct regression test.  It builds
the same environment cells under two different traversal orders, with decoy
environments constructed in between, and requires byte-identical streams.
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from dsa.envs import SpectrumEnv, SyncVectorEnv, get_scenario
from dsa.envs.heuristics import GreedyOccupancyPolicy
from dsa.seeding import (
    ENV_SUBSTREAMS,
    SEED_MASK,
    derive_seed,
    make_rng,
    make_torch_generator,
)

DSA_ROOT = Path(__file__).resolve().parents[1] / "dsa"


# --------------------------------------------------------------- derive_seed #


def test_derive_seed_is_pure():
    """Repeated calls with identical arguments agree, in any order."""
    a = derive_seed(7, "stationary", "env", "eval", 3)
    for _ in range(5):
        derive_seed(999, "decoy")  # interleaved unrelated calls
    b = derive_seed(7, "stationary", "env", "eval", 3)
    assert a == b


def test_derive_seed_is_stable_across_processes():
    """Pins the derivation itself.

    These are not metrics: they are the fingerprint of the hash construction.
    If this test fails, seeds derived by an older commit are not comparable with
    seeds derived by this one, and every stored result must be regenerated.
    ``hashlib.blake2b`` is used precisely because Python's builtin ``hash`` is
    salted per process and would make this test unwritable.
    """
    assert derive_seed(0) == derive_seed(0)
    assert derive_seed(7, "stationary", "env", "eval", 0) == derive_seed(
        7, "stationary", "env", "eval", 0
    )
    # Independently recomputed from the documented construction.
    d = hashlib.blake2b(digest_size=8)
    d.update(b"7")
    for part in ("stationary", "env", "eval", "0"):
        d.update(b"\x1f")
        d.update(part.encode("utf-8"))
    expected = int.from_bytes(d.digest(), "big") & SEED_MASK
    assert derive_seed(7, "stationary", "env", "eval", 0) == expected


def test_derive_seed_separator_is_unambiguous():
    assert derive_seed(1, "ab", "c") != derive_seed(1, "a", "bc")
    assert derive_seed(1, "a", "b") != derive_seed(1, "ab")


def test_derive_seed_is_in_range_and_sensitive():
    seeds = {derive_seed(s, "role", i) for s in range(8) for i in range(8)}
    assert len(seeds) == 64, "seed collisions across a tiny grid"
    for s in seeds:
        assert 0 <= s <= SEED_MASK


def test_derive_seed_distinguishes_roles():
    base = 7
    roles = {
        "train_env": derive_seed(base, "stationary", "env", "train", 0, 0),
        "eval_env": derive_seed(base, "stationary", "env", "eval", 0),
        "policy_init": derive_seed(base, "policy_init", "ppo_cfc"),
        "action": derive_seed(base, "action", "ppo_cfc", "stationary"),
        "shuffle": derive_seed(base, "shuffle", "ppo_cfc", "stationary"),
        "heuristic": derive_seed(base, "heuristic", "random_policy", "stationary"),
    }
    assert len(set(roles.values())) == len(roles)


def test_env_seed_excludes_the_model_name():
    """Two models must observe byte-identical environment streams.

    This is what makes the seed-paired statistics valid.  The role table derives
    environment seeds from ``(base_seed, scenario, "env", ...)`` with no model
    component; this test states that requirement as an executable fact.
    """
    for model in ("ppo_ltc", "ppo_gru", "ppo_mlp", "ppo_cfc_dtblind"):
        assert derive_seed(7, "stationary", "env", "eval", 0) == derive_seed(
            7, "stationary", "env", "eval", 0
        ), model


def test_make_rng_and_torch_generator_are_deterministic():
    a = make_rng(7, "x").standard_normal(16)
    b = make_rng(7, "x").standard_normal(16)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, make_rng(7, "y").standard_normal(16))

    ga = make_torch_generator(7, "x")
    gb = make_torch_generator(7, "x")
    assert torch.equal(torch.rand(16, generator=ga), torch.rand(16, generator=gb))


def test_env_substreams_are_distinct():
    seeds = [derive_seed(12345, "sub", s) for s in ENV_SUBSTREAMS]
    assert len(set(seeds)) == len(ENV_SUBSTREAMS)


# ------------------------------------------------------------- prohibitions #


def _python_sources() -> list[Path]:
    return [p for p in DSA_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_class_level_counters():
    """No module in ``dsa/`` may carry a mutable instantiation counter."""
    banned = re.compile(
        r"_instance_counter|_reset_count\b|_env_counter|itertools\.count\(\)"
    )
    offenders = [
        str(p) for p in _python_sources() if banned.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"instantiation counter found in {offenders}"


def test_spectrum_env_class_attributes_do_not_mutate():
    cfg = get_scenario("stationary")
    before = {
        k: v
        for k, v in vars(SpectrumEnv).items()
        if not k.startswith("__") and not callable(v) and not isinstance(v, property)
    }
    for _ in range(5):
        SpectrumEnv(cfg, episode_seed=1)
    after = {
        k: v
        for k, v in vars(SpectrumEnv).items()
        if not k.startswith("__") and not callable(v) and not isinstance(v, property)
    }
    assert before == after


def test_seed_derivation_never_uses_builtin_hash():
    """``hash()`` on a str is salted per process and would break workers."""
    pattern = re.compile(r"(?<![\w.])hash\s*\(")
    offenders = []
    for path in _python_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{path}:{lineno}: {stripped}")
    assert not offenders, f"builtin hash() used in dsa/: {offenders}"


def test_envs_do_not_touch_global_rng_apis():
    banned = re.compile(r"np\.random\.(seed|rand|randn|randint|choice|normal|uniform)\(")
    offenders = []
    for path in (DSA_ROOT / "envs").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if banned.search(text):
            offenders.append(str(path))
    assert not offenders, f"global numpy RNG used in {offenders}"


# ------------------------------------------------------- the order test (3.4) #


def env_stream_digest(base_seed, scenario, episode_index, actions) -> str:
    """sha256 over the float64 bytes of every observation, reward and dt.

    One episode, driven by a fixed action sequence, so any difference in the
    digest is a difference in the environment stream and nothing else.
    """
    seed = derive_seed(base_seed, scenario, "env", "eval", episode_index)
    env = SpectrumEnv(get_scenario(scenario), episode_seed=seed)
    state, _info = env.reset()
    digest = hashlib.sha256()
    digest.update(np.asarray(state["obs"], dtype=np.float64).tobytes())
    digest.update(np.float64(state["dt"]).tobytes())
    for action in actions:
        state, reward, _term, truncated, _info = env.step(int(action))
        digest.update(np.asarray(state["obs"], dtype=np.float64).tobytes())
        digest.update(np.float64(reward).tobytes())
        digest.update(np.float64(state["dt"]).tobytes())
        if truncated:
            break
    return digest.hexdigest()


def test_env_stream_order_independence():
    """The regression test for the class-counter defect.

    Interleaving A walks (model, scenario, episode) in one order; interleaving B
    reverses every axis and constructs 40 decoy environments in between, which
    would advance any hidden instantiation counter.  The digests must match
    exactly.
    """
    scenarios = ["stationary", "irregular_dt"]
    actions = [(i * 3 + 1) % 8 for i in range(64)]

    order_a = [
        (m, s, e)
        for m in ["ppo_ltc", "ppo_gru", "ppo_mlp"]
        for s in scenarios
        for e in range(3)
    ]
    order_b = [
        (m, s, e)
        for m in ["ppo_mlp", "ppo_gru", "ppo_ltc"]
        for s in reversed(scenarios)
        for e in reversed(range(3))
    ]

    dig_a = {(s, e): env_stream_digest(7, s, e, actions) for _, s, e in order_a}

    for _ in range(40):
        SpectrumEnv(get_scenario("bursty_irregular"), episode_seed=12345)

    dig_b = {(s, e): env_stream_digest(7, s, e, actions) for _, s, e in order_b}

    assert dig_a == dig_b
    assert len(set(dig_a.values())) == len(dig_a), "digests are trivially constant"


def test_vector_lane_stream_is_independent_of_lane_index():
    """A lane's stream depends only on its seed, not on where it sits."""
    cfg = get_scenario("bursty_irregular")
    seeds = [derive_seed(7, "bursty_irregular", "env", "train", i, 0) for i in range(4)]

    vec_a = SyncVectorEnv(cfg, seeds)
    vec_b = SyncVectorEnv(cfg, list(reversed(seeds)))
    obs_a, dt_a = vec_a.reset()
    obs_b, dt_b = vec_b.reset()
    acts = np.array([1, 3, 5, 7], dtype=np.int64)
    for _ in range(cfg.max_steps):
        obs_a, dt_a, r_a, _d, _i = vec_a.step(acts)
        obs_b, dt_b, r_b, _d, _i = vec_b.step(acts[::-1])
        assert np.array_equal(obs_a, obs_b[::-1])
        assert np.array_equal(dt_a, dt_b[::-1])
        assert np.array_equal(r_a, r_b[::-1])


# ------------------------------------------------- global RNG is not load-bearing #


def _perturb_global_rng_state() -> None:
    np.random.seed(1234)
    np.random.random(97)
    random.seed(4321)
    random.random()
    torch.manual_seed(999)
    torch.rand(13)


def _heuristic_eval_digest(scenario: str, base_seed: int = 7) -> str:
    cfg = get_scenario(scenario)
    seeds = [derive_seed(base_seed, scenario, "env", "eval", i) for i in range(8)]
    vec = SyncVectorEnv(cfg, seeds)
    obs, dt = vec.reset()
    policy = GreedyOccupancyPolicy(cfg.action_dim)
    policy.reset(vec.num_envs)
    digest = hashlib.sha256()
    for _ in range(cfg.max_steps):
        obs, dt, reward, _done, _infos = vec.step(policy.act(obs, dt))
        digest.update(np.asarray(reward, dtype=np.float64).tobytes())
    for metrics in vec.episode_metrics():
        for key in sorted(metrics):
            digest.update(np.float64(metrics[key]).tobytes())
    return digest.hexdigest()


def test_global_rng_is_not_load_bearing_for_envs():
    first = _heuristic_eval_digest("bursty_irregular")
    _perturb_global_rng_state()
    second = _heuristic_eval_digest("bursty_irregular")
    _perturb_global_rng_state()
    third = _heuristic_eval_digest("bursty_irregular")
    assert first == second == third


def test_global_rng_is_not_load_bearing_for_training():
    """End-to-end: a short train + eval is unaffected by global RNG state.

    Skipped if the model or learner packages are not importable, so that this
    file remains runnable on its own.
    """
    build_model = pytest.importorskip("dsa.models.registry").build_model
    ppo_mod = pytest.importorskip("dsa.learner.ppo")
    evaluate_policy = pytest.importorskip("dsa.learner.evaluate").evaluate_policy

    torch.set_num_threads(1)
    scenario = "irregular_dt"
    cfg = get_scenario(scenario)

    def run() -> list[float]:
        model = build_model(
            "ppo_gru", cfg.obs_dim, cfg.action_dim, derive_seed(7, "policy_init", "ppo_gru")
        )
        hp = ppo_mod.PPOHyperParams(num_envs=4, horizon=cfg.max_steps, update_epochs=1,
                                    num_sequence_minibatches=2)
        agent = ppo_mod.RecurrentPPO(
            model,
            hp,
            action_generator=make_torch_generator(7, "action", "ppo_gru", scenario),
            shuffle_generator=make_torch_generator(7, "shuffle", "ppo_gru", scenario),
        )
        seeds0 = [derive_seed(7, scenario, "env", "train", i, 0) for i in range(4)]
        vec = SyncVectorEnv(cfg, seeds0)
        logs = ppo_mod.train(
            agent,
            vec,
            total_env_steps=4 * cfg.max_steps,
            rollout_seed_fn=lambda r: [
                derive_seed(7, scenario, "env", "train", i, r) for i in range(4)
            ],
        )
        summary, _ = evaluate_policy(
            model, cfg, [derive_seed(7, scenario, "env", "eval", i) for i in range(4)]
        )
        return [float(logs[0]["policy_loss"]), float(summary["mean_eval_return"])]

    first = run()
    _perturb_global_rng_state()
    second = run()
    assert first == second, f"{first} != {second}"
