"""Tests for the recurrent PPO learner (agent B).

These tests are the regression gate for audit defects D9 (nothing trained /
minibatching unreachable), D10 (dropout live in rollout, update and eval) and the
learner-visible half of D2 (construction-order-dependent seeding).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import numpy as np
import pytest
import torch

from dsa.envs.heuristics import HEURISTIC_REGISTRY
from dsa.envs.scenarios import get_scenario
from dsa.envs.spectrum import SpectrumEnv
from dsa.envs.vector import SyncVectorEnv
from dsa.learner.buffer import SequenceBatch, SequenceRolloutBuffer
from dsa.learner.evaluate import evaluate_policy
from dsa.learner.ppo import PPOHyperParams, RecurrentPPO, compute_gae, train
from dsa.models.policy import RecurrentActorCritic
from dsa.models.registry import MODEL_REGISTRY, build_model
from dsa.seeding import derive_seed, make_torch_generator

torch.set_num_threads(1)

SHORT_STEPS = 16


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def short_config(name: str = "stationary", max_steps: int = SHORT_STEPS):
    """A registered scenario shortened for test speed.  Semantics are unchanged."""
    return dataclasses.replace(get_scenario(name), max_steps=max_steps)


def train_seed_fn(base_seed: int, scenario: str, num_envs: int):
    def fn(rollout_index: int):
        return [
            derive_seed(base_seed, scenario, "env", "train", lane, rollout_index)
            for lane in range(num_envs)
        ]

    return fn


def make_agent(model_key: str, cfg, base_seed: int, scenario: str, **hp_overrides):
    hp = PPOHyperParams(
        num_envs=hp_overrides.pop("num_envs", 4),
        horizon=hp_overrides.pop("horizon", cfg.max_steps),
        update_epochs=hp_overrides.pop("update_epochs", 2),
        num_sequence_minibatches=hp_overrides.pop("num_sequence_minibatches", 2),
        **hp_overrides,
    )
    model = build_model(
        model_key,
        obs_dim=4 * cfg.num_channels + 8,
        action_dim=cfg.num_channels,
        seed=derive_seed(base_seed, "policy_init", model_key),
    )
    agent = RecurrentPPO(
        model,
        hp,
        action_generator=make_torch_generator(base_seed, "action", model_key, scenario),
        shuffle_generator=make_torch_generator(base_seed, "shuffle", model_key, scenario),
    )
    return agent


def make_vec(cfg, base_seed: int, scenario: str, num_envs: int):
    seeds = train_seed_fn(base_seed, scenario, num_envs)(0)
    return SyncVectorEnv(cfg, seeds)


def digest_logs(logs) -> str:
    h = hashlib.sha256()
    for entry in logs:
        for k in sorted(entry):
            h.update(k.encode())
            h.update(np.float64(entry[k]).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. GAE -- hand-computed
# --------------------------------------------------------------------------- #


def _gae_case(dones_list):
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 1.0, 1.5]])
    dones = torch.tensor([dones_list])
    last_value = torch.tensor([2.0])
    return compute_gae(rewards, values, dones, last_value, gamma=0.9, gae_lambda=0.5)


def test_gae_hand_computed_no_termination():
    """T=3, gamma=0.9, lambda=0.5, V(s_T)=2.0 bootstrapped at the horizon.

    delta_2 = 3.0 + 0.9*2.0 - 1.5 = 3.300      A_2 = 3.300
    delta_1 = 2.0 + 0.9*1.5 - 1.0 = 2.350      A_1 = 2.350 + 0.45*3.300 = 3.835
    delta_0 = 1.0 + 0.9*1.0 - 0.5 = 1.400      A_0 = 1.400 + 0.45*3.835 = 3.12575
    """
    adv, ret = _gae_case([0.0, 0.0, 0.0])
    expected_adv = torch.tensor([[3.12575, 3.835, 3.3]])
    assert torch.allclose(adv, expected_adv, atol=1e-6)
    assert torch.allclose(ret, expected_adv + torch.tensor([[0.5, 1.0, 1.5]]), atol=1e-6)


def test_gae_hand_computed_terminal_at_horizon():
    """dones[-1]=1 kills the bootstrap: delta_2 = 3.0 - 1.5 = 1.5."""
    adv, _ = _gae_case([0.0, 0.0, 1.0])
    expected = torch.tensor([[2.76125, 3.025, 1.5]])
    assert torch.allclose(adv, expected, atol=1e-6)


def test_gae_hand_computed_mid_episode_terminal():
    """A terminal at t=1 truncates the advantage recursion there."""
    adv, _ = _gae_case([0.0, 1.0, 0.0])
    expected = torch.tensor([[1.85, 1.0, 3.3]])
    assert torch.allclose(adv, expected, atol=1e-6)


def test_gae_time_limit_truncation_differs_from_termination():
    """Truncation must bootstrap; termination must not.  They must not coincide."""
    adv_trunc, _ = _gae_case([0.0, 0.0, 0.0])
    adv_term, _ = _gae_case([0.0, 0.0, 1.0])
    assert not torch.allclose(adv_trunc, adv_term)
    # The gap at the final step is exactly gamma * V(s_T).
    assert float(adv_trunc[0, 2] - adv_term[0, 2]) == pytest.approx(0.9 * 2.0, abs=1e-6)


def test_gae_returns_equal_advantages_plus_values():
    adv, ret = _gae_case([0.0, 0.0, 0.0])
    assert torch.allclose(ret - adv, torch.tensor([[0.5, 1.0, 1.5]]), atol=1e-7)


def test_compute_gae_rejects_bad_shapes():
    with pytest.raises(ValueError):
        compute_gae(
            torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2), 0.99, 0.95
        )
    with pytest.raises(ValueError):
        compute_gae(
            torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3), 0.99, 0.95
        )


# --------------------------------------------------------------------------- #
# 2. Buffer
# --------------------------------------------------------------------------- #


def test_buffer_produces_full_sequence_batch():
    n, t, obs_dim = 4, 5, 7
    buf = SequenceRolloutBuffer(n, t, obs_dim, (3, 0))
    init = (torch.zeros(n, 3), torch.zeros(n, 0))
    buf.start(init)
    assert len(buf) == 0
    for _ in range(t):
        buf.add(
            torch.zeros(n, obs_dim),
            torch.ones(n),
            torch.zeros(n, dtype=torch.int64),
            torch.zeros(n),
            torch.zeros(n),
            torch.ones(n),
            torch.zeros(n),
        )
    assert len(buf) == t
    batch = buf.finish(torch.zeros(n))
    assert batch.num_sequences() == n
    assert batch.horizon() == t
    assert batch.obs.shape == (n, t, obs_dim)
    assert batch.actions.dtype == torch.int64
    assert len(batch.init_state) == 2
    assert batch.init_state[1].shape == (n, 0)


def test_buffer_rejects_overflow_and_premature_finish():
    buf = SequenceRolloutBuffer(2, 2, 3, (1,))
    with pytest.raises(RuntimeError):
        buf.add(*(torch.zeros(2, 3), torch.ones(2), torch.zeros(2, dtype=torch.int64),
                  torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2)))
    buf.start((torch.zeros(2, 1),))
    with pytest.raises(RuntimeError):
        buf.finish(torch.zeros(2))
    for _ in range(2):
        buf.add(torch.zeros(2, 3), torch.ones(2), torch.zeros(2, dtype=torch.int64),
                torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2))
    with pytest.raises(RuntimeError):
        buf.add(torch.zeros(2, 3), torch.ones(2), torch.zeros(2, dtype=torch.int64),
                torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2))


def test_batch_stores_no_per_step_hidden_states():
    """D10/PPO-RNN guard: replaying from init_state is the only sanctioned path."""
    fields = {f.name for f in dataclasses.fields(SequenceBatch)}
    assert fields == {
        "obs", "dt", "actions", "log_probs", "values", "rewards",
        "dones", "init_state", "last_value",
    }
    assert not any("hidden" in f or f == "states" for f in fields)


# --------------------------------------------------------------------------- #
# 3. Minibatching is reachable (D9)
# --------------------------------------------------------------------------- #


def _synthetic_batch(model, n, t, obs_dim):
    init = model.initial_state(n)
    model.eval()
    with torch.no_grad():
        obs = torch.randn(n, t, obs_dim, generator=torch.Generator().manual_seed(0))
        dt = torch.ones(n, t)
        logits, values, _ = model.unroll(obs, dt, init)
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1)
        log_probs = dist.log_prob(actions)
    return SequenceBatch(
        obs=obs,
        dt=dt,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=torch.zeros(n, t),
        dones=torch.zeros(n, t),
        init_state=init,
        last_value=torch.zeros(n),
    )


@pytest.mark.parametrize(
    "num_envs,num_mb,epochs",
    [(16, 4, 6), (6, 4, 3), (8, 8, 2), (5, 2, 4)],
)
def test_minibatching_is_reachable(num_envs, num_mb, epochs):
    """The old buffer's flush condition could never fire, so minibatch_size collapsed
    to the full batch and only `epochs` gradient steps ran.  Here the number of
    optimizer steps must equal ceil(N / ceil(N/num_mb)) * epochs exactly."""
    cfg = short_config()
    obs_dim, action_dim = 4 * cfg.num_channels + 8, cfg.num_channels
    model = build_model("ppo_gru", obs_dim, action_dim, seed=1)
    hp = PPOHyperParams(
        num_envs=num_envs, horizon=4, update_epochs=epochs, num_sequence_minibatches=num_mb
    )
    agent = RecurrentPPO(
        model, hp, torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
    )
    batch = _synthetic_batch(model, num_envs, 4, obs_dim)

    mb_size = math.ceil(num_envs / num_mb)
    expected_mbs = math.ceil(num_envs / mb_size)
    expected = expected_mbs * epochs

    out = agent.update(batch)
    assert agent.total_gradient_steps == expected
    assert out["gradient_steps"] == float(expected)
    assert out["minibatch_size"] == float(mb_size)
    # Non-degenerate: the batch really was split.
    if num_mb > 1 and num_envs > 1:
        assert mb_size < num_envs


def test_spec_budget_is_24_gradient_steps_per_update():
    """The frozen suite config: 16 envs, 4 sequence minibatches, 6 epochs."""
    hp = PPOHyperParams(num_envs=16, num_sequence_minibatches=4, update_epochs=6)
    assert hp.minibatch_size() == 4
    assert hp.minibatches_per_epoch() == 4
    assert hp.gradient_steps_per_update() == 24


# --------------------------------------------------------------------------- #
# 4. Ratio sanity: exactly 1 on the first minibatch, diverging afterwards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_key", sorted(MODEL_REGISTRY))
def test_first_epoch_ratio_is_one(model_key):
    """Catches stale hidden states, dropout mismatch and train/eval mode bugs at once."""
    cfg = short_config()
    agent = make_agent(model_key, cfg, base_seed=7, scenario="stationary")
    vec = make_vec(cfg, 7, "stationary", agent.hparams.num_envs)
    batch = agent.collect(vec)
    out = agent.update(batch)
    assert out["first_epoch_max_ratio_deviation"] < 1e-4, model_key


def test_ratio_diverges_after_the_first_minibatch():
    """The other direction: if the ratio stayed 1.0 forever, log-probs were being
    read from storage instead of recomputed under the current policy."""
    cfg = short_config()
    agent = make_agent(
        "ppo_gru", cfg, base_seed=7, scenario="stationary",
        update_epochs=3, num_sequence_minibatches=2, learning_rate=1e-2,
    )
    vec = make_vec(cfg, 7, "stationary", agent.hparams.num_envs)
    out = agent.update(agent.collect(vec))
    assert out["first_epoch_max_ratio_deviation"] < 1e-4
    assert out["later_max_ratio_deviation"] > 1e-6


def test_unroll_reproduces_rollout_log_probs_and_values_before_any_update():
    """Epoch-0 state recomputation must match the rollout exactly, for every model."""
    cfg = short_config()
    for model_key in sorted(MODEL_REGISTRY):
        agent = make_agent(model_key, cfg, base_seed=13, scenario="stationary")
        vec = make_vec(cfg, 13, "stationary", agent.hparams.num_envs)
        batch = agent.collect(vec)
        agent.model.train()
        with torch.no_grad():
            logits, values, _ = agent.model.unroll(batch.obs, batch.dt, batch.init_state)
            lp = torch.distributions.Categorical(logits=logits).log_prob(batch.actions)
        assert torch.allclose(lp, batch.log_probs, atol=1e-5), model_key
        assert torch.allclose(values, batch.values, atol=1e-5), model_key


# --------------------------------------------------------------------------- #
# 5. Train / eval mode discipline (D10)
# --------------------------------------------------------------------------- #


def test_eval_mode_during_rollout_and_train_mode_during_update():
    cfg = short_config()
    agent = make_agent("ppo_gru", cfg, base_seed=7, scenario="stationary")
    vec = make_vec(cfg, 7, "stationary", agent.hparams.num_envs)
    model = agent.model

    real_step, real_unroll = model.step, model.unroll

    # Probe the rollout only.  `unroll` may legitimately call `step` internally, so
    # the probe is uninstalled before the update phase is measured separately.
    seen_step: list[bool] = []
    model.step = lambda *a, **k: (seen_step.append(model.training), real_step(*a, **k))[1]
    model.train()  # deliberately hostile starting mode
    batch = agent.collect(vec)
    model.step = real_step
    assert seen_step, "collect() never called model.step"
    assert all(flag is False for flag in seen_step), "rollout ran in training mode"

    seen_unroll: list[bool] = []
    model.unroll = lambda *a, **k: (seen_unroll.append(model.training), real_unroll(*a, **k))[1]
    agent.update(batch)
    model.unroll = real_unroll
    assert seen_unroll, "update() never called model.unroll"
    assert all(flag is True for flag in seen_unroll), "update ran in eval mode"


def test_no_model_in_the_registry_has_active_dropout():
    cfg = short_config()
    obs_dim, action_dim = 4 * cfg.num_channels + 8, cfg.num_channels
    for key in sorted(MODEL_REGISTRY):
        model = build_model(key, obs_dim, action_dim, seed=3)
        for mod in model.modules():
            if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d)):
                assert float(mod.p) == 0.0, f"{key}: live dropout p={mod.p}"


def test_greedy_action_is_repeatable_for_every_model():
    """Two successive deterministic acts on identical input must agree (D10)."""
    cfg = short_config()
    obs_dim, action_dim = 4 * cfg.num_channels + 8, cfg.num_channels
    obs = torch.zeros(3, obs_dim)
    dt = torch.ones(3)
    for key in sorted(MODEL_REGISTRY):
        model = build_model(key, obs_dim, action_dim, seed=5)
        model.eval()
        state = model.initial_state(3)
        with torch.no_grad():
            a1 = torch.argmax(model.step(obs, dt, state)[0], dim=-1)
            a2 = torch.argmax(model.step(obs, dt, state)[0], dim=-1)
        assert torch.equal(a1, a2), key


# --------------------------------------------------------------------------- #
# 6. Real step accounting (D8/D9)
# --------------------------------------------------------------------------- #


def test_train_steps_is_real():
    cfg = short_config()
    agent = make_agent("ppo_gru", cfg, base_seed=7, scenario="stationary")
    hp = agent.hparams
    vec = make_vec(cfg, 7, "stationary", hp.num_envs)
    num_rollouts = 3
    logs = train(
        agent, vec, num_rollouts * hp.num_envs * hp.horizon,
        train_seed_fn(7, "stationary", hp.num_envs),
    )
    assert len(logs) == num_rollouts
    assert agent.total_env_steps == num_rollouts * hp.num_envs * hp.horizon
    assert logs[-1]["total_env_steps"] == float(agent.total_env_steps)
    assert agent.total_gradient_steps == num_rollouts * hp.gradient_steps_per_update()


def test_train_rounds_partial_budget_up_to_whole_rollouts():
    cfg = short_config()
    agent = make_agent("ppo_gru", cfg, base_seed=7, scenario="stationary")
    hp = agent.hparams
    vec = make_vec(cfg, 7, "stationary", hp.num_envs)
    batch_steps = hp.num_envs * hp.horizon
    logs = train(agent, vec, batch_steps + 1, train_seed_fn(7, "stationary", hp.num_envs))
    assert len(logs) == 2
    assert agent.total_env_steps == 2 * batch_steps


# --------------------------------------------------------------------------- #
# 7. Seeding: D2 regression at the learner level
# --------------------------------------------------------------------------- #


def _run_short_training(model_key: str, base_seed: int, scenario: str = "stationary"):
    cfg = short_config(scenario)
    agent = make_agent(model_key, cfg, base_seed, scenario)
    vec = make_vec(cfg, base_seed, scenario, agent.hparams.num_envs)
    logs = train(
        agent, vec, 2 * agent.hparams.num_envs * agent.hparams.horizon,
        train_seed_fn(base_seed, scenario, agent.hparams.num_envs),
    )
    return logs


def test_training_is_independent_of_construction_order():
    """D2: the old env seeded itself from a mutable class-level instance counter, so
    the score of a run depended on how many envs the process had built first."""
    first = _run_short_training("ppo_gru", 7)

    # Decoys: build many environments and a competing agent in between.
    for i in range(40):
        SpectrumEnv(short_config("bursty_irregular"), episode_seed=12345 + i)
    _run_short_training("ppo_cfc", 7)
    for i in range(40):
        SpectrumEnv(short_config("irregular_dt"), episode_seed=999 + i)

    second = _run_short_training("ppo_gru", 7)
    assert digest_logs(first) == digest_logs(second)


def test_global_rng_perturbation_does_not_change_training():
    first = _run_short_training("ppo_cfc", 19)
    np.random.seed(1234)
    torch.manual_seed(4321)
    _ = np.random.rand(1000), torch.randn(1000)
    second = _run_short_training("ppo_cfc", 19)
    assert digest_logs(first) == digest_logs(second)


def test_env_stream_is_model_independent_at_reset():
    """Env seeds exclude the model name, so every model starts from the same states."""
    cfg = short_config()
    seeds = train_seed_fn(31, "stationary", 4)(0)
    a_obs, a_dt = SyncVectorEnv(cfg, seeds).reset()
    b_obs, b_dt = SyncVectorEnv(cfg, seeds).reset()
    assert np.array_equal(a_obs, b_obs)
    assert np.array_equal(a_dt, b_dt)


def test_action_and_shuffle_generators_are_separate_streams():
    cfg = short_config()
    agent = make_agent("ppo_gru", cfg, base_seed=7, scenario="stationary")
    assert agent.action_generator is not agent.shuffle_generator
    assert (
        agent.action_generator.initial_seed() != agent.shuffle_generator.initial_seed()
    )


# --------------------------------------------------------------------------- #
# 8. Evaluation
# --------------------------------------------------------------------------- #


def eval_seeds(base_seed: int, scenario: str, n: int):
    return [derive_seed(base_seed, scenario, "env", "eval", i) for i in range(n)]


def test_evaluation_does_not_contaminate_the_training_rng_stream():
    """Rule 7 of the brief: evaluating between rollouts must not move the training
    trajectory by a single bit."""
    cfg = short_config()
    seeds = eval_seeds(7, "stationary", 4)

    def run(with_eval: bool):
        agent = make_agent("ppo_gru", cfg, base_seed=7, scenario="stationary")
        vec = make_vec(cfg, 7, "stationary", agent.hparams.num_envs)
        seed_fn = train_seed_fn(7, "stationary", agent.hparams.num_envs)
        logs = []
        for r in range(2):
            if with_eval:
                evaluate_policy(agent.model, cfg, seeds)
            vec.reseed(seed_fn(r))
            logs.append(agent.update(agent.collect(vec)))
        return logs

    assert digest_logs(run(False)) == digest_logs(run(True))


def test_evaluate_policy_is_deterministic_and_repeatable():
    cfg = short_config()
    model = build_model("ppo_cfc", 4 * cfg.num_channels + 8, cfg.num_channels, seed=11)
    seeds = eval_seeds(7, "stationary", 6)
    s1, e1 = evaluate_policy(model, cfg, seeds)
    s2, e2 = evaluate_policy(model, cfg, seeds)
    assert s1 == s2
    assert e1 == e2
    assert len(e1) == 6
    assert s1["num_eval_episodes"] == 6.0


def test_evaluate_policy_restores_prior_training_mode():
    cfg = short_config()
    model = build_model("ppo_gru", 4 * cfg.num_channels + 8, cfg.num_channels, seed=11)
    model.train()
    evaluate_policy(model, cfg, eval_seeds(7, "stationary", 2))
    assert model.training is True
    model.eval()
    evaluate_policy(model, cfg, eval_seeds(7, "stationary", 2))
    assert model.training is False


def test_evaluate_policy_reports_both_entropies_under_distinct_names():
    """The old repo labelled the argmax action histogram 'mean_policy_entropy'."""
    cfg = short_config()
    model = build_model("ppo_gru", 4 * cfg.num_channels + 8, cfg.num_channels, seed=11)
    summary, _ = evaluate_policy(model, cfg, eval_seeds(7, "stationary", 4))
    assert "mean_action_histogram_entropy" in summary
    assert "eval_action_histogram_entropy" in summary
    assert "mean_policy_entropy_nats" in summary
    assert summary["eval_action_histogram_entropy"] == summary["mean_action_histogram_entropy"]
    # A real distributional entropy is strictly positive and bounded by ln(A).
    assert 0.0 < summary["mean_policy_entropy_nats"] <= math.log(cfg.num_channels) + 1e-6
    assert "mean_eval_return" in summary
    assert summary["mean_eval_return"] == summary["mean_episode_return"]


def test_evaluate_policy_handles_heuristics_and_chunking():
    cfg = short_config()
    policy = HEURISTIC_REGISTRY["random_policy"](cfg.num_channels, derive_seed(7, "heuristic"))
    summary, per_episode = evaluate_policy(policy, cfg, eval_seeds(7, "stationary", 20))
    assert len(per_episode) == 20  # 16 + 4, chunked
    assert summary["mean_policy_entropy_nats"] == 0.0
    assert "mean_eval_return" in summary


def test_stochastic_evaluation_requires_an_explicit_generator():
    cfg = short_config()
    model = build_model("ppo_gru", 4 * cfg.num_channels + 8, cfg.num_channels, seed=11)
    with pytest.raises(ValueError):
        evaluate_policy(model, cfg, eval_seeds(7, "stationary", 2), deterministic=False)
    g = make_torch_generator(7, "action", "ppo_gru", "stationary")
    summary, _ = evaluate_policy(
        model, cfg, eval_seeds(7, "stationary", 2), deterministic=False, action_generator=g
    )
    assert "mean_eval_return" in summary


# --------------------------------------------------------------------------- #
# 9. Learning check plumbing (D9)
# --------------------------------------------------------------------------- #


def test_untrained_and_trained_evaluation_share_the_identical_stream():
    """F5 requires the two evaluations to be paired on the same environments."""
    cfg = short_config()
    scenario = "stationary"
    agent = make_agent("ppo_gru", cfg, base_seed=7, scenario=scenario)
    seeds = eval_seeds(7, scenario, 4)
    untrained, _ = evaluate_policy(agent.model, cfg, seeds)
    vec = make_vec(cfg, 7, scenario, agent.hparams.num_envs)
    train(
        agent, vec, 2 * agent.hparams.num_envs * agent.hparams.horizon,
        train_seed_fn(7, scenario, agent.hparams.num_envs),
    )
    trained, _ = evaluate_policy(agent.model, cfg, seeds)
    # Same stream, different weights => the metrics must actually be comparable and
    # the parameters must have moved.
    assert untrained["num_eval_episodes"] == trained["num_eval_episodes"]
    assert agent.total_gradient_steps > 0


def test_collect_records_truncation_as_bootstrap_not_as_termination():
    """The spectrum environment truncates at max_steps and never terminates.  If
    collect() wrote the vector env's episode-end flag into `dones`, the final step
    of every sequence would lose its V(s_T) bootstrap and every return estimate at
    the horizon boundary would be biased low."""
    cfg = short_config()
    agent = make_agent("ppo_gru", cfg, base_seed=7, scenario="stationary")
    vec = make_vec(cfg, 7, "stationary", agent.hparams.num_envs)
    batch = agent.collect(vec)

    # The lanes really did hit the time limit ...
    assert batch.horizon() == cfg.max_steps
    # ... yet no step is marked terminal ...
    assert float(batch.dones.abs().max()) == 0.0
    # ... so the bootstrap is live and it actually changes the returns.
    _, ret_with = compute_gae(
        batch.rewards, batch.values, batch.dones, batch.last_value, 0.99, 0.95
    )
    _, ret_without = compute_gae(
        batch.rewards, batch.values, batch.dones, torch.zeros_like(batch.last_value), 0.99, 0.95
    )
    assert not torch.allclose(ret_with, ret_without)


def test_update_metrics_contract():
    cfg = short_config()
    agent = make_agent("ppo_gru", cfg, base_seed=7, scenario="stationary")
    vec = make_vec(cfg, 7, "stationary", agent.hparams.num_envs)
    out = agent.update(agent.collect(vec))
    for key in (
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
        "explained_variance", "grad_norm", "first_epoch_max_ratio_deviation",
        "policy_entropy_mean",
    ):
        assert key in out, key
        assert isinstance(out[key], float)
    assert out["policy_entropy_mean"] > 0.0
    assert not math.isnan(out["policy_entropy_mean"])


def test_update_time_state_replay_matches_the_rollout_trajectory():
    """Replaying the stored sequence through unroll() from init_state must reproduce
    the exact hidden trajectory the rollout carried -- otherwise the update is
    optimising a different policy than the one that generated the data."""
    cfg = short_config()
    for model_key in ("ppo_gru", "ppo_lstm", "ppo_ltc", "ppo_cfc", "ppo_ltc_cfc", "ppo_transformer"):
        agent = make_agent(model_key, cfg, base_seed=17, scenario="stationary")
        model = agent.model
        batch = agent.collect(make_vec(cfg, 17, "stationary", agent.hparams.num_envs))

        model.eval()
        with torch.no_grad():
            # Sequential replay, exactly as the rollout did it.
            state = batch.init_state
            step_logits = []
            for t in range(batch.horizon()):
                lo, _v, state = model.step(batch.obs[:, t], batch.dt[:, t], state)
                step_logits.append(lo)
            step_logits_t = torch.stack(step_logits, dim=1)
            step_final = state
            # Batched replay, exactly as update() does it.
            unroll_logits, _uv, unroll_final = model.unroll(batch.obs, batch.dt, batch.init_state)

        assert torch.allclose(step_logits_t, unroll_logits, atol=1e-5), model_key
        for blk, (a, b) in enumerate(zip(step_final, unroll_final)):
            assert torch.allclose(a, b, atol=1e-5), f"{model_key} block {blk}"


def test_run_snapshot_reports_measured_not_declared_steps():
    """D8 guard: the recorded train_steps must be the count actually executed."""
    cfg = short_config()
    agent = make_agent("ppo_gru", cfg, base_seed=7, scenario="stationary")
    hp = agent.hparams
    vec = make_vec(cfg, 7, "stationary", hp.num_envs)
    train(agent, vec, 3 * hp.num_envs * hp.horizon, train_seed_fn(7, "stationary", hp.num_envs))
    snap = agent.run_snapshot()
    assert snap["train_steps"] == agent.total_env_steps == 3 * hp.num_envs * hp.horizon
    assert snap["total_gradient_steps"] == agent.total_gradient_steps
    for k, v in hp.to_dict().items():
        assert snap[f"ppo_{k}"] == v
    assert snap["action_generator_seed"] != snap["shuffle_generator_seed"]


def test_agent_state_dict_roundtrip():
    cfg = short_config()
    agent = make_agent("ppo_cfc", cfg, base_seed=7, scenario="stationary")
    sd = agent.state_dict()
    other = make_agent("ppo_cfc", cfg, base_seed=99, scenario="stationary")
    other.load_state_dict(sd)
    for k, v in agent.state_dict().items():
        assert torch.equal(v, other.state_dict()[k])


def test_hparams_to_dict_is_serialisable():
    hp = PPOHyperParams()
    d = hp.to_dict()
    assert d["num_envs"] == 16 and d["horizon"] == 64
    assert d["update_epochs"] == 6 and d["num_sequence_minibatches"] == 4
    assert all(isinstance(v, (int, float)) for v in d.values())


# --------------------------------------------------------------------------- #
# 12. M24 -- evaluation must run the whole module tree in eval mode
# --------------------------------------------------------------------------- #
# `evaluate_policy` calling `policy.train()` instead of `policy.eval()` was invisible
# to the entire suite: the shipped registry uses dropout = 0.0 everywhere, so the
# mode flag has no numerical consequence *for the shipped models*.  It is still a
# contract -- the flag is what makes the guarantee robust to any future block that
# behaves differently in the two modes -- so it is pinned twice below: structurally,
# by watching `.training` on every submodule during a real evaluation, and
# numerically, on a deliberately dropout-carrying policy.


def _record_module_modes(policy) -> list[dict[str, bool]]:
    """Snapshot ``.training`` for every submodule on each ``policy.step`` call.

    Instrumenting ``step`` rather than the whole call means the modes are sampled
    exactly when the policy is being used to produce actions -- the moment at which
    a train/eval discrepancy would matter.
    """
    real_step = policy.step
    seen: list[dict[str, bool]] = []

    def spy(obs, dt, state):
        seen.append({(name or "<root>"): bool(m.training) for name, m in policy.named_modules()})
        return real_step(obs, dt, state)

    policy.step = spy
    return seen


@pytest.mark.parametrize("entry_mode", [True, False])
def test_evaluate_policy_runs_every_submodule_in_eval_mode(entry_mode):
    """Every submodule must be in eval mode for the whole evaluation, whichever mode
    the policy was handed in, and the caller's mode must be restored afterwards."""
    cfg = short_config()
    model = build_model("ppo_transformer", 4 * cfg.num_channels + 8, cfg.num_channels, seed=11)
    model.train(entry_mode)
    seen = _record_module_modes(model)

    evaluate_policy(model, cfg, eval_seeds(7, "stationary", 3))

    assert seen, "the probe never fired -- evaluate_policy did not call policy.step"
    offenders = sorted({name for snapshot in seen for name, training in snapshot.items() if training})
    assert not offenders, f"submodules left in training mode during evaluation: {offenders}"
    assert model.training is entry_mode


def test_evaluation_of_a_dropout_carrying_policy_is_deterministic():
    """The numerical consequence of the mode contract, made observable.

    The registry ships ``dropout = 0.0``, so this builds a policy that does behave
    differently in the two modes.  In training mode its dropout masks are drawn from
    the *global* torch RNG, so two evaluations under different global RNG states
    would disagree -- and evaluation would additionally be consuming the training
    stream, which spec section 4.4 rule 6 forbids.
    """
    cfg = short_config()
    model = RecurrentActorCritic(
        obs_dim=4 * cfg.num_channels + 8,
        action_dim=cfg.num_channels,
        cell_kinds=("attn", "attn"),
        hidden_dim=16,
        generator=torch.Generator().manual_seed(5),
        cell_kwargs={"num_heads": 1, "dropout": 0.5},
    )
    seeds = eval_seeds(7, "stationary", 4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        first, first_rows = evaluate_policy(model, cfg, seeds)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(987_654_321)
        second, second_rows = evaluate_policy(model, cfg, seeds)
    assert first == second
    assert first_rows == second_rows


# --------------------------------------------------------------------------- #
# 13. M28 -- advantage normalisation is over the whole batch
# --------------------------------------------------------------------------- #


def test_advantages_are_normalised_over_the_whole_batch_not_per_minibatch():
    """Rule 5 of the module contract, pinned by its observable consequence.

    With ``learning_rate = 0`` the parameters never move, so the PPO ratio is
    exactly 1 in every minibatch of every epoch, ``surr1 == surr2 == adv_mb`` and
    the policy loss of minibatch ``m`` collapses to ``-mean(adv_norm[idx_m])``.
    The reported ``policy_loss`` is then a closed form of whichever normalisation
    the update actually applied, and the two candidates disagree:

    * whole batch (the contract): each minibatch keeps its own offset from the
      batch mean, and those offsets do not cancel when the minibatches have
      unequal sizes;
    * per minibatch (the bug): every minibatch is re-centred, so every
      per-minibatch advantage mean -- and hence the reported policy loss -- is
      exactly 0.

    Hence 5 sequences split 3 + 2.  With equal-sized minibatches the per-epoch
    means would cancel to the batch mean under *both* schemes and this test would
    be vacuous, so the split is asserted.
    """
    cfg = short_config()
    agent = make_agent(
        "ppo_gru", cfg, base_seed=31, scenario="stationary",
        num_envs=5, num_sequence_minibatches=2, update_epochs=3, learning_rate=0.0,
    )
    hp = agent.hparams
    n_seq = hp.num_envs
    mb_size = hp.minibatch_size()
    n_mb = hp.minibatches_per_epoch(n_seq)
    assert (mb_size, n_mb) == (3, 2)
    assert mb_size * n_mb != n_seq, "the minibatch split must be unequal for this test to bite"

    batch = agent.collect(make_vec(cfg, 31, "stationary", n_seq))
    # A strong, known per-sequence advantage structure: sequence i earns reward i.
    batch.rewards = (
        torch.arange(n_seq, dtype=torch.float32)
        .unsqueeze(1)
        .expand(n_seq, batch.horizon())
        .contiguous()
    )

    replay = torch.Generator().manual_seed(agent.shuffle_generator.initial_seed())
    out = agent.update(batch)

    # The closed form above is only valid while the behaviour policy is unchanged.
    assert out["first_epoch_max_ratio_deviation"] < 1e-6
    assert out["later_max_ratio_deviation"] < 1e-5

    advantages, _returns = compute_gae(
        batch.rewards, batch.values, batch.dones, batch.last_value, hp.gamma, hp.gae_lambda
    )
    flat = advantages.reshape(-1)
    whole_batch_norm = (advantages - flat.mean()) / (flat.std(unbiased=False) + 1e-8)

    losses: list[float] = []
    for _epoch in range(hp.update_epochs):
        perm = torch.randperm(n_seq, generator=replay)
        for m in range(n_mb):
            idx = perm[m * mb_size : (m + 1) * mb_size]
            if idx.numel() == 0:
                continue
            losses.append(float(-whole_batch_norm[idx].mean()))
    expected = sum(losses) / len(losses)

    # Not vacuous: on this batch the two schemes genuinely give different answers.
    assert abs(expected) > 1e-3, expected
    assert abs(out["policy_loss"] - expected) < 1e-5, (out["policy_loss"], expected)
    # Per-minibatch normalisation would have re-centred every minibatch to zero mean.
    assert abs(out["policy_loss"]) > 1e-3, out["policy_loss"]


# --------------------------------------------------------------------------- #
# 14. M30 -- the critic loss is built from the recomputed values
# --------------------------------------------------------------------------- #


def test_the_critic_is_trained_by_the_recomputed_values():
    """``batch.values`` is collected under ``no_grad``.

    A critic loss built from it is a constant w.r.t. the parameters, so the critic
    head -- which influences nothing else in the loss -- would receive no gradient
    at all and would never move, while every reported diagnostic stayed finite and
    plausible.
    """
    cfg = short_config()
    agent = make_agent(
        "ppo_gru", cfg, base_seed=41, scenario="stationary",
        num_envs=4, num_sequence_minibatches=2, update_epochs=2, learning_rate=1e-2,
    )
    batch = agent.collect(make_vec(cfg, 41, "stationary", agent.hparams.num_envs))
    before = {n: p.detach().clone() for n, p in agent.model.critic.named_parameters()}

    agent.update(batch)

    for name, p in agent.model.critic.named_parameters():
        assert p.grad is not None, f"critic parameter {name} received no gradient at all"
        assert float(p.grad.abs().max()) > 0.0, name
        assert float((p.detach() - before[name]).abs().max()) > 0.0, name


def test_value_loss_responds_to_the_critic_parameters():
    """The same defect from the other side.

    Shifting the critic head's output bias changes the *recomputed* values and
    nothing else.  A value loss read off the stored rollout values would not notice,
    so the reported ``value_loss`` would be unmoved.  ``learning_rate = 0`` keeps the
    two updates comparable.
    """
    cfg = short_config()
    agent = make_agent(
        "ppo_gru", cfg, base_seed=43, scenario="stationary",
        num_envs=4, num_sequence_minibatches=2, update_epochs=2, learning_rate=0.0,
    )
    batch = agent.collect(make_vec(cfg, 43, "stationary", agent.hparams.num_envs))

    before = agent.update(batch)
    with torch.no_grad():
        agent.model.critic[-1].bias.add_(25.0)
    after = agent.update(batch)

    # The perturbation is confined to the value head ...
    assert abs(after["policy_loss"] - before["policy_loss"]) < 1e-6
    assert abs(after["entropy"] - before["entropy"]) < 1e-6
    # ... so the value loss is the one thing that must respond, and strongly.
    assert after["value_loss"] > 10.0 * before["value_loss"], (
        before["value_loss"], after["value_loss"]
    )
