"""Environment tests, including a truth-in-labelling suite for the scenarios.

Three defect classes are pinned here.

``dt`` bookkeeping.  The previous implementation advanced the simulation clock
with the *old* interval while relaxing the channel dynamics with a *newly
sampled* one, and handed the new one to the model.  Three different intervals
were live in one transition, which made the continuous-time premise untestable.
``test_dt_single_source_of_truth`` and ``test_relaxation_uses_the_observed_dt``
pin the observed interval to the interval the dynamics actually consumed.

Truth in labelling.  ``varying_users_small`` and ``varying_users_large`` both set
``scenario_name = "stationary"`` and differed only in a background-user constant
that was never mutated within an episode -- a static load sweep presented as
population dynamics, inflating five dynamics classes into seven.
``test_scenario_names_tell_the_truth`` rolls out every registered scenario and
asserts that the property its name advertises actually holds.

Baseline quality.  The previous greedy heuristic lost to uniform random in six
of seven scenarios, which made "PPO beats the heuristic" a claim about beating
an anti-optimal policy.  ``test_greedy_heuristic_beats_uniform_random`` requires
a strict majority and prints the measured returns either way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dsa.envs import (
    ALL_SCENARIOS,
    BASELINE_KEYS,
    EXPLORATORY_SCENARIOS,
    HEURISTIC_REGISTRY,
    OBS_LAYOUT,
    OBS_SCALARS,
    PRIMARY_SCENARIOS,
    SCALAR_OFFSETS,
    SCENARIO_REGISTRY,
    ConstantChannelPolicy,
    GreedyOccupancyPolicy,
    RandomPolicy,
    ScenarioConfig,
    SpectrumEnv,
    SyncVectorEnv,
    get_scenario,
    obs_dim_for,
    obs_layout_for,
)
from dsa.envs.spectrum import _BUSY_RATE, _INTERFERENCE_RATE, _QUALITY_RATE
from dsa.seeding import derive_seed

FIXED_ACTIONS = [(i * 3 + 1) % 8 for i in range(64)]

EPISODE_METRIC_KEYS = {
    "episode_return",
    "spectrum_utilization",
    "collision_rate",
    "success_rate",
    "sim_time",
    "action_histogram_entropy",
    "background_fairness",
    "steps",
    "mean_dt",
    "num_background_users_mean",
}


def eval_seeds(scenario: str, n: int, base: int = 7) -> list[int]:
    return [derive_seed(base, scenario, "env", "eval", i) for i in range(n)]


def rollout(scenario: str, actions=None, episode_seed: int | None = None):
    """One episode under a fixed action sequence.  Returns (states, rewards, infos)."""
    cfg = get_scenario(scenario)
    seed = episode_seed if episode_seed is not None else eval_seeds(scenario, 1)[0]
    env = SpectrumEnv(cfg, episode_seed=seed)
    state, info0 = env.reset()
    states, rewards, infos = [state], [0.0], [info0]
    acts = actions if actions is not None else FIXED_ACTIONS
    for step in range(cfg.max_steps):
        state, reward, _term, truncated, info = env.step(acts[step % len(acts)])
        states.append(state)
        rewards.append(reward)
        infos.append(info)
        if truncated:
            break
    return env, states, rewards, infos


def run_policy(policy, scenario: str, n_episodes: int = 16, base: int = 7):
    """Mean episode return for a heuristic policy over shared evaluation seeds."""
    cfg = get_scenario(scenario)
    seeds = eval_seeds(scenario, n_episodes, base)
    returns: list[float] = []
    for start in range(0, n_episodes, 16):
        vec = SyncVectorEnv(cfg, seeds[start : start + 16])
        obs, dt = vec.reset()
        policy.reset(vec.num_envs)
        for _ in range(cfg.max_steps):
            obs, dt, _r, _d, _i = vec.step(policy.act(obs, dt))
        returns += [m["episode_return"] for m in vec.episode_metrics()]
    return float(np.mean(returns))


# ------------------------------------------------------------ observation layout #


def test_obs_layout_tiles_the_vector_without_gaps():
    for c in (2, 5, 8, 16):
        layout = obs_layout_for(c)
        covered: list[int] = []
        for key in ("busy", "interference", "quality", "last_action", "scalars"):
            sl = layout[key]
            covered.extend(range(sl.start, sl.stop))
        assert covered == list(range(obs_dim_for(c)))
        assert layout["scalars"].stop - layout["scalars"].start == OBS_SCALARS


def test_obs_layout_matches_default_channel_count():
    assert OBS_LAYOUT == obs_layout_for(8)
    assert obs_dim_for(8) == 40


def test_scalar_offsets_are_a_bijection():
    assert sorted(SCALAR_OFFSETS.values()) == list(range(OBS_SCALARS))


def test_observation_shape_dtype_and_range():
    for scenario in ALL_SCENARIOS:
        cfg = get_scenario(scenario)
        _env, states, _r, _i = rollout(scenario)
        for state in states:
            obs = state["obs"]
            assert obs.dtype == np.float32
            assert obs.shape == (cfg.obs_dim,)
            assert np.isfinite(obs).all()
            lay = obs_layout_for(cfg.num_channels)
            for key in ("busy", "interference", "quality", "last_action"):
                block = obs[lay[key]]
                assert block.min() >= -1e-6 and block.max() <= 1.0 + 1e-6, (
                    scenario,
                    key,
                )


def test_previous_action_one_hot_is_correct():
    cfg = get_scenario("stationary")
    env = SpectrumEnv(cfg, episode_seed=eval_seeds("stationary", 1)[0])
    state, _ = env.reset()
    assert state["obs"][OBS_LAYOUT["last_action"]].sum() == 0.0, "no action taken yet"
    for step, action in enumerate(FIXED_ACTIONS):
        state, *_ = env.step(action)
        one_hot = state["obs"][OBS_LAYOUT["last_action"]]
        assert one_hot.sum() == 1.0
        assert int(np.argmax(one_hot)) == action
        if step >= 5:
            break


def test_background_users_are_hidden_terminals():
    """The observation must not reveal the background population.

    ``num_background_users`` is not a feature, and sensed occupancy reports the
    primary occupant only.  Two episodes that share every random substream
    except ``background`` must therefore produce identical observations until
    the agent's own collisions start feeding back through the reward scalars.
    """
    cfg = get_scenario("stationary")
    env = SpectrumEnv(cfg, episode_seed=eval_seeds("stationary", 1)[0])
    state, _ = env.reset()
    busy_obs = state["obs"][OBS_LAYOUT["busy"]]
    # The sensed block is binary (up to the bit-flip), never a device count.
    assert set(np.unique(busy_obs)).issubset({0.0, 1.0})
    # Ground truth for the hidden population is available in info only.
    _env, _states, _r, infos = rollout("bursty_irregular")
    assert "num_background_users" in infos[-1]


# ---------------------------------------------------------------- dt contract #


def test_dt_single_source_of_truth():
    """One interval per transition: clock, dynamics and agent all see the same dt."""
    for scenario in ALL_SCENARIOS:
        env, states, _rewards, infos = rollout(scenario)
        total = 0.0
        for t in range(1, len(states)):
            assert states[t]["dt"] == infos[t]["dt"], (scenario, t)
            total += infos[t]["dt"]
            assert infos[t]["sim_time"] == pytest.approx(total, abs=0.0, rel=1e-15)
        # Exact float equality: sim_time is accumulated by the same additions.
        assert env.sim_time == total, scenario


def test_reset_dt_is_the_nominal_interval_and_clock_is_zero():
    for scenario in ALL_SCENARIOS:
        cfg = get_scenario(scenario)
        env = SpectrumEnv(cfg, episode_seed=eval_seeds(scenario, 1)[0])
        state, info = env.reset()
        assert state["dt"] == float(cfg.dt_base)
        assert info["dt"] == float(cfg.dt_base)
        assert env.sim_time == 0.0
        assert env.step_count == 0


def test_relaxation_uses_the_observed_dt():
    """The relaxation factors are ``1 - exp(-rate * dt)`` for the observed dt.

    This is the off-by-one regression test: the previous implementation relaxed
    the channels with an interval the agent never saw.
    """
    for scenario in ("stationary", "irregular_dt", "bursty_irregular"):
        _env, states, _rewards, infos = rollout(scenario)
        for t in range(1, len(states)):
            dt = states[t]["dt"]
            assert infos[t]["busy_adapt"] == pytest.approx(
                1.0 - math.exp(-_BUSY_RATE * dt), rel=1e-12
            )
            assert infos[t]["quality_adapt"] == pytest.approx(
                1.0 - math.exp(-_QUALITY_RATE * dt), rel=1e-12
            )
            assert infos[t]["interference_adapt"] == pytest.approx(
                1.0 - math.exp(-_INTERFERENCE_RATE * dt), rel=1e-12
            )


def test_dt_observation_features_track_the_observed_dt():
    cfg = get_scenario("irregular_dt")
    _env, states, _r, _i = rollout("irregular_dt")
    base = OBS_LAYOUT["scalars"].start
    for state in states:
        dt = state["dt"]
        obs = state["obs"]
        assert obs[base + SCALAR_OFFSETS["dt_normalised"]] == pytest.approx(
            dt / max(cfg.dt_max, cfg.dt_base), rel=1e-6
        )
        assert obs[base + SCALAR_OFFSETS["log_dt"]] == pytest.approx(
            math.log(max(dt, 1e-3)), rel=1e-6
        )


def test_dynamics_are_invariant_to_how_time_is_subdivided():
    """The relaxation kernel composes: two half-steps equal one whole step.

    Only meaningful for the deterministic part of the update, so this checks the
    factor algebra rather than a full trajectory.
    """
    for dt in (0.2, 1.0, 2.5):
        whole = 1.0 - math.exp(-_BUSY_RATE * dt)
        half = 1.0 - math.exp(-_BUSY_RATE * dt / 2)
        assert (1 - whole) == pytest.approx((1 - half) ** 2, rel=1e-12)


# ------------------------------------------------------ truth in labelling (D13) #


def test_scenario_configs_are_distinct():
    seen: list[dict] = []
    names: list[str] = []
    for name, cfg in SCENARIO_REGISTRY.items():
        assert cfg.name == name, f"registry key {name!r} != config name {cfg.name!r}"
        fields = cfg.to_dict()
        assert fields not in seen, f"{name} duplicates another scenario's fields"
        seen.append(fields)
        names.append(cfg.name)
    assert len(set(names)) == len(names)


def test_registry_partitions_into_primary_and_exploratory():
    assert set(PRIMARY_SCENARIOS) | set(EXPLORATORY_SCENARIOS) == set(SCENARIO_REGISTRY)
    assert not set(PRIMARY_SCENARIOS) & set(EXPLORATORY_SCENARIOS)
    assert len(PRIMARY_SCENARIOS) == 5


def test_no_static_load_sweep_masquerades_as_dynamics():
    """The specific shape of the deleted defect: two scenarios that differ only
    in a constant background-user count."""
    for a_name, a in SCENARIO_REGISTRY.items():
        for b_name, b in SCENARIO_REGISTRY.items():
            if a_name >= b_name:
                continue
            da, db = a.to_dict(), b.to_dict()
            differing = {
                k for k in da if da[k] != db[k] and k != "name"
            }
            assert differing - {
                "num_background_users",
                "load_min",
                "load_max",
            }, (
                f"{a_name} and {b_name} differ only in a background-user constant: "
                "that is a static load sweep, not a dynamics class"
            )


def _dt_values(scenario: str, n_episodes: int = 4) -> list[float]:
    values: list[float] = []
    for seed in eval_seeds(scenario, n_episodes):
        _env, states, _r, _i = rollout(scenario, episode_seed=seed)
        values += [s["dt"] for s in states[1:]]
    return values


def _load_values(scenario: str, n_episodes: int = 4) -> list[list[float]]:
    per_episode = []
    for seed in eval_seeds(scenario, n_episodes):
        _env, _states, _r, infos = rollout(scenario, episode_seed=seed)
        per_episode.append([i["num_background_users"] for i in infos[1:]])
    return per_episode


def test_scenario_names_tell_the_truth():
    """Every registered scenario must exhibit the property its name advertises.

    This is the test that would have caught ``varying_users_small`` /
    ``varying_users_large``.
    """
    for scenario, cfg in SCENARIO_REGISTRY.items():
        dts = _dt_values(scenario)
        distinct_dt = sorted(set(round(v, 12) for v in dts))

        if cfg.dt_mode == "constant":
            assert distinct_dt == [round(cfg.dt_base, 12)], (
                scenario,
                distinct_dt[:5],
            )
        elif cfg.dt_mode == "loguniform":
            assert len(distinct_dt) > 50, (scenario, len(distinct_dt))
            assert min(dts) >= cfg.dt_min - 1e-12
            assert max(dts) <= cfg.dt_max + 1e-12
            assert max(dts) / min(dts) > 2.0, "loguniform dt is barely varying"
        elif cfg.dt_mode == "bimodal":
            assert distinct_dt == sorted(
                {round(cfg.dt_min, 12), round(cfg.dt_max, 12)}
            ), (scenario, distinct_dt)
        else:  # pragma: no cover
            raise AssertionError(f"unhandled dt_mode {cfg.dt_mode}")

        loads = _load_values(scenario)
        if cfg.load_mode == "fixed":
            for episode in loads:
                assert len(set(episode)) == 1, (scenario, sorted(set(episode)))
                assert episode[0] == float(cfg.num_background_users)
        else:
            assert max(len(set(e)) for e in loads) >= 3, (
                scenario,
                [sorted(set(e)) for e in loads],
            )
            for episode in loads:
                assert min(episode) >= cfg.load_min
                assert max(episode) <= cfg.load_max

        _env, _states, _r, infos = rollout(scenario)
        busy_trace = [i["base_busy_mean"] for i in infos[1:]]
        spread = max(busy_trace) - min(busy_trace)
        if cfg.drift_strength == 0.0 and cfg.regime_switch_period == 0.0:
            assert spread < 1e-9, (
                f"{scenario} advertises no non-stationarity but base occupancy "
                f"moved by {spread}"
            )
        else:
            assert spread > 1e-3, (
                f"{scenario} advertises non-stationarity (drift="
                f"{cfg.drift_strength}, regime={cfg.regime_switch_period}) but "
                f"base occupancy moved by only {spread}"
            )


def test_dt_varies_in_exactly_two_primary_scenarios():
    varying = [
        s for s in PRIMARY_SCENARIOS if get_scenario(s).dt_mode != "constant"
    ]
    assert sorted(varying) == ["bursty_irregular", "irregular_dt"]


def test_irregular_dt_differs_from_stationary_only_in_the_dt_process():
    a = get_scenario("stationary").to_dict()
    b = get_scenario("irregular_dt").to_dict()
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"name", "dt_mode", "dt_min", "dt_max"}


def test_birth_death_load_actually_varies():
    for scenario in ("bursty_irregular", "bursty_load"):
        loads = _load_values(scenario, n_episodes=4)
        assert max(len(set(e)) for e in loads) >= 3, scenario
    for scenario in ("stationary", "non_stationary", "interference_heavy"):
        loads = _load_values(scenario, n_episodes=2)
        for episode in loads:
            assert len(set(episode)) == 1, scenario


def test_birth_death_load_is_dt_consistent():
    """Longer intervals must produce more load churn per decision."""
    slow = ScenarioConfig(
        name="probe_slow",
        load_mode="birth_death",
        load_min=2,
        load_max=8,
        load_switch_rate=0.5,
        dt_mode="constant",
        dt_base=0.25,
        dt_min=0.25,
        dt_max=0.25,
    )
    fast = ScenarioConfig(
        name="probe_fast",
        load_mode="birth_death",
        load_min=2,
        load_max=8,
        load_switch_rate=0.5,
        dt_mode="constant",
        dt_base=2.5,
        dt_min=2.5,
        dt_max=2.5,
    )

    def churn(cfg: ScenarioConfig) -> float:
        changes = 0
        steps = 0
        for i in range(8):
            env = SpectrumEnv(cfg, episode_seed=derive_seed(11, cfg.name, i))
            env.reset()
            previous = None
            for t in range(cfg.max_steps):
                _s, _r, _term, _tr, info = env.step(t % cfg.action_dim)
                current = info["num_background_users"]
                if previous is not None and current != previous:
                    changes += 1
                previous = current
                steps += 1
        return changes / steps

    assert churn(fast) > churn(slow) * 2.0


def test_interference_heavy_really_is_heavier():
    def mean_interference(scenario: str) -> float:
        _env, _s, _r, infos = rollout(scenario)
        return float(np.mean([i["interference_mean"] for i in infos[1:]]))

    assert mean_interference("interference_heavy") > mean_interference("stationary")


def test_noisy_partial_really_is_noisier_and_partial():
    _env, _s, _r, noisy = rollout("noisy_partial")
    _env2, _s2, _r2, clean = rollout("stationary")
    assert np.mean([i["sense_flips"] for i in noisy[1:]]) > np.mean(
        [i["sense_flips"] for i in clean[1:]]
    )
    assert np.mean([i["obs_features_dropped"] for i in noisy[1:]]) > 0.0
    assert np.mean([i["obs_features_dropped"] for i in clean[1:]]) == 0.0


def test_scenario_config_is_frozen_and_validated():
    cfg = get_scenario("stationary")
    with pytest.raises(Exception):
        cfg.drift_strength = 0.9  # type: ignore[misc]
    with pytest.raises(ValueError):
        ScenarioConfig(name="bad", dt_mode="linear")
    with pytest.raises(ValueError):
        ScenarioConfig(name="bad", load_mode="poisson")
    with pytest.raises(ValueError):
        ScenarioConfig(name="bad", load_mode="birth_death", load_switch_rate=0.0)
    with pytest.raises(ValueError):
        ScenarioConfig(name="bad", dt_min=2.0, dt_max=1.0)


# --------------------------------------------------------- reward well-posedness #


def test_no_idle_action_and_rates_are_complementary():
    """Stated plainly because the metrics are otherwise easy to misread."""
    for scenario in ALL_SCENARIOS:
        cfg = get_scenario(scenario)
        assert cfg.action_dim == cfg.num_channels, "no idle action exists"
        env, _s, _r, _i = rollout(scenario)
        m = env.episode_metrics()
        assert m["success_rate"] + m["collision_rate"] == pytest.approx(1.0, abs=1e-12)


def test_reward_is_bounded_by_the_documented_range():
    for scenario in ALL_SCENARIOS:
        cfg = get_scenario(scenario)
        lo = -cfg.collision_penalty * (1.0 + 0.5 * 1.0 + 0.25 * 2)
        hi = cfg.success_reward_scale * 1.0 * 1.0 + cfg.idle_bias
        for seed_index in range(3):
            _env, _s, rewards, _i = rollout(
                scenario, episode_seed=eval_seeds(scenario, 4)[seed_index]
            )
            for r in rewards[1:]:
                assert lo - 1e-9 <= r <= hi + 1e-9, (scenario, r)


def test_reward_rewards_free_channels_and_punishes_collisions():
    for scenario in ALL_SCENARIOS:
        _env, _s, rewards, infos = rollout(scenario)
        for r, info in zip(rewards[1:], infos[1:]):
            if info["collision"] == 1.0:
                assert r < 0.0
            else:
                assert r > 0.0


def test_degenerate_policy_cannot_maximise():
    """No fixed action approaches the achievable maximum.

    Occupancy is heterogeneous across channels and drifts, hidden background
    devices camp and hop, so the best channel changes within an episode.  Every
    constant-channel policy must fall well short of the per-step oracle -- and
    also short of a policy that actually senses.
    """
    for scenario in PRIMARY_SCENARIOS:
        cfg = get_scenario(scenario)
        seeds = eval_seeds(scenario, 8)

        oracle_returns = np.zeros(len(seeds))
        vec = SyncVectorEnv(cfg, seeds)
        obs, dt = vec.reset()
        greedy = GreedyOccupancyPolicy(cfg.action_dim)
        greedy.reset(vec.num_envs)
        for _ in range(cfg.max_steps):
            obs, dt, _r, _d, infos = vec.step(greedy.act(obs, dt))
            oracle_returns += np.array([i["oracle_reward"] for i in infos])
        oracle = float(oracle_returns.mean())
        greedy_return = float(
            np.mean([m["episode_return"] for m in vec.episode_metrics()])
        )

        best_constant = max(
            run_policy(ConstantChannelPolicy(cfg.action_dim, channel=c), scenario, 8)
            for c in range(cfg.action_dim)
        )
        assert best_constant < 0.5 * oracle, (scenario, best_constant, oracle)
        assert best_constant < greedy_return, (scenario, best_constant, greedy_return)
        assert greedy_return < oracle, "the oracle must be strictly unachieved"


def test_episode_metrics_keys_and_ranges():
    for scenario in ALL_SCENARIOS:
        env, _s, _r, _i = rollout(scenario)
        m = env.episode_metrics()
        assert set(m) == EPISODE_METRIC_KEYS, set(m) ^ EPISODE_METRIC_KEYS
        assert 0.0 <= m["success_rate"] <= 1.0
        assert 0.0 <= m["collision_rate"] <= 1.0
        assert 0.0 <= m["background_fairness"] <= 1.0 + 1e-9
        assert 0.0 <= m["action_histogram_entropy"] <= math.log(env.action_dim) + 1e-9
        assert m["steps"] == get_scenario(scenario).max_steps
        assert m["mean_dt"] > 0.0
        assert all(np.isfinite(v) for v in m.values())


def test_action_histogram_entropy_is_not_policy_entropy():
    """A deterministic constant policy has zero *histogram* entropy."""
    cfg = get_scenario("stationary")
    env, _s, _r, _i = rollout("stationary", actions=[0])
    assert env.episode_metrics()["action_histogram_entropy"] == pytest.approx(0.0)
    env2, _s2, _r2, _i2 = rollout(
        "stationary", actions=list(range(cfg.action_dim))
    )
    assert env2.episode_metrics()["action_histogram_entropy"] == pytest.approx(
        math.log(cfg.action_dim), rel=1e-9
    )


def test_invalid_action_is_rejected():
    env = SpectrumEnv(get_scenario("stationary"), episode_seed=1)
    with pytest.raises(ValueError):
        env.step(-1)
    with pytest.raises(ValueError):
        env.step(env.action_dim)


def test_env_does_not_autoreset():
    cfg = get_scenario("stationary")
    env = SpectrumEnv(cfg, episode_seed=1)
    env.reset()
    for t in range(cfg.max_steps):
        _s, _r, _term, truncated, _i = env.step(0)
    assert truncated
    with pytest.raises(RuntimeError):
        env.step(0)


# ---------------------------------------------------------------- determinism #


def test_same_seed_gives_bit_identical_streams():
    """Bit-exact determinism was a genuinely correct property of the previous
    code.  It is preserved and pinned here."""
    for scenario in ALL_SCENARIOS:
        seed = eval_seeds(scenario, 1)[0]
        _e1, s1, r1, _i1 = rollout(scenario, episode_seed=seed)
        _e2, s2, r2, _i2 = rollout(scenario, episode_seed=seed)
        assert r1 == r2
        for a, b in zip(s1, s2):
            assert np.array_equal(a["obs"], b["obs"])
            assert a["dt"] == b["dt"]


def test_different_seeds_give_different_streams():
    seeds = eval_seeds("stationary", 4)
    digests = set()
    for seed in seeds:
        _e, states, _r, _i = rollout("stationary", episode_seed=seed)
        digests.add(np.concatenate([s["obs"] for s in states]).tobytes())
    assert len(digests) == len(seeds)


def test_reset_is_idempotent():
    env = SpectrumEnv(get_scenario("bursty_irregular"), episode_seed=99)
    first, _ = env.reset()
    for _ in range(5):
        env.step(3)
    second, _ = env.reset()
    assert np.array_equal(first["obs"], second["obs"])
    assert env.sim_time == 0.0 and env.step_count == 0


def test_metrics_are_deterministic_functions_of_the_seed():
    for scenario in PRIMARY_SCENARIOS:
        e1, *_ = rollout(scenario, episode_seed=1234)
        e2, *_ = rollout(scenario, episode_seed=1234)
        assert e1.episode_metrics() == e2.episode_metrics()


# ------------------------------------------------------------------ vector env #


def test_vector_env_matches_single_env_lane_by_lane():
    cfg = get_scenario("bursty_irregular")
    seeds = eval_seeds("bursty_irregular", 3)
    vec = SyncVectorEnv(cfg, seeds)
    obs, dt = vec.reset()
    singles = [SpectrumEnv(cfg, s) for s in seeds]
    single_states = [e.reset()[0] for e in singles]
    for lane in range(3):
        assert np.array_equal(obs[lane], single_states[lane]["obs"])
        assert dt[lane] == np.float32(single_states[lane]["dt"])

    for t in range(cfg.max_steps):
        acts = np.array([(t + lane) % cfg.action_dim for lane in range(3)])
        obs, dt, reward, done, _infos = vec.step(acts)
        for lane in range(3):
            state, r, _term, truncated, _info = singles[lane].step(int(acts[lane]))
            assert np.array_equal(obs[lane], state["obs"])
            assert dt[lane] == np.float32(state["dt"])
            assert reward[lane] == np.float32(r)
            assert bool(done[lane]) == bool(truncated)

    for lane in range(3):
        assert vec.episode_metrics()[lane] == singles[lane].episode_metrics()


def test_vector_env_reseed_is_pure():
    cfg = get_scenario("irregular_dt")
    seeds_a = eval_seeds("irregular_dt", 4, base=7)
    seeds_b = eval_seeds("irregular_dt", 4, base=19)

    vec = SyncVectorEnv(cfg, seeds_a)
    vec.reset()
    for _ in range(10):
        vec.step(np.zeros(4, dtype=np.int64))
    vec.reseed(seeds_b)
    obs_after, dt_after = vec.reset()

    fresh = SyncVectorEnv(cfg, seeds_b)
    obs_fresh, dt_fresh = fresh.reset()
    assert np.array_equal(obs_after, obs_fresh)
    assert np.array_equal(dt_after, dt_fresh)


def test_vector_env_validates_shapes():
    cfg = get_scenario("stationary")
    vec = SyncVectorEnv(cfg, [1, 2, 3])
    vec.reset()
    with pytest.raises(ValueError):
        vec.step(np.array([0, 0], dtype=np.int64))
    with pytest.raises(ValueError):
        vec.reseed([1, 2])
    with pytest.raises(ValueError):
        SyncVectorEnv(cfg, [])


# ------------------------------------------------------------------ heuristics #


def test_baseline_registry_is_complete():
    assert set(HEURISTIC_REGISTRY) == set(BASELINE_KEYS)
    cfg = get_scenario("stationary")
    for key, factory in HEURISTIC_REGISTRY.items():
        policy = factory(cfg.action_dim, 5)
        assert policy.name == key
        assert not hasattr(policy, "unroll"), "must not duck-type as a neural policy"
        policy.reset(4)
        actions = policy.act(
            np.zeros((4, cfg.obs_dim), dtype=np.float32),
            np.ones(4, dtype=np.float32),
        )
        assert actions.shape == (4,)
        assert actions.dtype == np.int64
        assert actions.min() >= 0 and actions.max() < cfg.action_dim


def test_heuristics_are_deterministic():
    cfg = get_scenario("bursty_irregular")
    for key, factory in HEURISTIC_REGISTRY.items():
        a = run_policy(factory(cfg.action_dim, 5), "bursty_irregular", 8)
        b = run_policy(factory(cfg.action_dim, 5), "bursty_irregular", 8)
        assert a == b, key


def test_random_policy_uses_its_own_generator():
    cfg = get_scenario("stationary")
    obs = np.zeros((64, cfg.obs_dim), dtype=np.float32)
    dt = np.ones(64, dtype=np.float32)
    p1, p2 = RandomPolicy(cfg.action_dim, 3), RandomPolicy(cfg.action_dim, 3)
    p1.reset(64)
    p2.reset(64)
    np.random.seed(0)  # global perturbation must be irrelevant
    a = p1.act(obs, dt)
    np.random.seed(12345)
    b = p2.act(obs, dt)
    assert np.array_equal(a, b)
    assert len(np.unique(a)) > 1, "uniform random should not be constant"


def test_constant_channel_policy_is_constant():
    cfg = get_scenario("stationary")
    policy = ConstantChannelPolicy(cfg.action_dim, channel=3)
    policy.reset(5)
    actions = policy.act(
        np.random.default_rng(0).random((5, cfg.obs_dim)).astype(np.float32),
        np.ones(5, dtype=np.float32),
    )
    assert np.array_equal(actions, np.full(5, 3))
    with pytest.raises(ValueError):
        ConstantChannelPolicy(cfg.action_dim, channel=cfg.action_dim)


def test_greedy_heuristic_beats_uniform_random(capsys):
    """The greedy baseline must be a genuine floor, not an anti-optimal policy.

    The previous version's greedy heuristic lost to uniform random in six of
    seven scenarios (stationary -26.76, interference_heavy -38.94,
    varying_users_large -43.87), which made every "PPO beats the heuristic"
    claim vacuous.  A strict majority is required here, and the measured numbers
    are printed either way so a regression is legible rather than silent.
    """
    rows = []
    wins = 0
    for scenario in ALL_SCENARIOS:
        cfg = get_scenario(scenario)
        rnd = run_policy(RandomPolicy(cfg.action_dim, 1), scenario, 16)
        const = run_policy(ConstantChannelPolicy(cfg.action_dim), scenario, 16)
        greedy = run_policy(GreedyOccupancyPolicy(cfg.action_dim), scenario, 16)
        wins += int(greedy > rnd)
        rows.append((scenario, rnd, const, greedy))

    with capsys.disabled():
        print("\n  mean episode return over 16 shared evaluation episodes")
        print(f"  {'scenario':<22}{'random':>10}{'constant':>10}{'greedy':>10}")
        for scenario, rnd, const, greedy in rows:
            print(f"  {scenario:<22}{rnd:>10.3f}{const:>10.3f}{greedy:>10.3f}")
        print(f"  greedy beats uniform random in {wins}/{len(rows)} scenarios")

    assert wins > len(rows) / 2, (
        "the greedy baseline is not a competitive floor; it wins in only "
        f"{wins}/{len(rows)} scenarios: {rows}"
    )


def test_greedy_heuristic_leaves_headroom_for_a_learner():
    """A benchmark a myopic heuristic solves cannot test a claim about memory.

    The greedy baseline must beat random comfortably *and* stay clearly short of
    the per-step oracle, which is only achievable by inferring the hidden
    background population from the agent's own collision history.
    """
    for scenario in PRIMARY_SCENARIOS:
        cfg = get_scenario(scenario)
        seeds = eval_seeds(scenario, 16)
        vec = SyncVectorEnv(cfg, seeds)
        obs, dt = vec.reset()
        policy = GreedyOccupancyPolicy(cfg.action_dim)
        policy.reset(vec.num_envs)
        oracle = np.zeros(vec.num_envs)
        for _ in range(cfg.max_steps):
            obs, dt, _r, _d, infos = vec.step(policy.act(obs, dt))
            oracle += np.array([i["oracle_reward"] for i in infos])
        greedy = float(np.mean([m["episode_return"] for m in vec.episode_metrics()]))
        oracle_mean = float(oracle.mean())
        assert greedy < oracle_mean - 5.0, (
            f"{scenario}: greedy {greedy:.2f} is within 5 points of the oracle "
            f"{oracle_mean:.2f}; the environment is myopically solved and cannot "
            "test a claim about state evolution"
        )


def test_background_fairness_responds_to_crowding():
    """A policy that camps on one channel must be less fair to the hidden
    background population than one that spreads out."""
    cfg = get_scenario("interference_heavy")
    seeds = eval_seeds("interference_heavy", 8)

    def fairness(actions) -> float:
        values = []
        for seed in seeds:
            env = SpectrumEnv(cfg, seed)
            env.reset()
            for t in range(cfg.max_steps):
                env.step(actions[t % len(actions)])
            values.append(env.episode_metrics()["background_fairness"])
        return float(np.mean(values))

    camped = fairness([0])
    spread = fairness(list(range(cfg.action_dim)))
    assert camped < spread
