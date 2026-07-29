"""The three federated arms: centralized, flat FedAvg, and a real hierarchy.

What was wrong before (defect D7)
---------------------------------
The previous implementation's hierarchy was inert.  Two lines did it:

* every client was initialised from ``global_agent`` rather than from its own
  edge, so the edge models never influenced anything downstream of them; and
* the edge aggregate was computed, averaged into the cloud model, and then
  **discarded** -- it did not persist into the next edge round.

With those two bugs, ``fedavg`` over edge means of client means is algebraically
the plain mean over all clients (the edges have equal client counts), so the
hierarchical global model came out bit-equal to flat FedAvg: the live repro
measured ``max|hierarchical - flat| = 1.19e-07`` across all thirty tensors, which
is float32 summation noise, not a difference.  Meanwhile the arm was charged 4/3
the communication of flat.  The repository was reporting a communication cost for
a computation that did nothing.

What is here now
----------------
:func:`run_hierarchical_federated` implements the loop the specification
describes, and the two repairs are the two lines you can point at:
``theta_c = _train_local_phase(init_state=theta_edge[edge_id], ...)`` -- clients
initialise from **their edge** -- and ``theta_edge[edge_id] = fedavg(...)``
assigned back into a dict that **persists across the ``k`` loop**.  Because a
client's starting point at edge round ``k >= 1`` is its edge's aggregate rather
than the cloud's, the arms genuinely diverge, and
``tests/test_federated.py::test_hierarchical_is_not_flat_fedavg`` fails loudly if
anyone reintroduces the old behaviour.

The controlled comparison
-------------------------
All three arms are matched on more than the specification requires, because the
claim under test is about *topology* and every unmatched axis is an alternative
explanation:

===========================  ===========  ==========  =====================
quantity                     centralized  flat        hierarchical
===========================  ===========  ==========  =====================
environment steps            36 864       36 864      36 864
PPO rollouts / local phases  48           48          48
sequences per rollout        12           12          12
optimizer steps              1 152        1 152       1 152
===========================  ===========  ==========  =====================

(The table above is arithmetic on the *config*, not a measured result; the
measured counts are asserted equal to it by
``tests/test_federated.py::test_train_steps_equals_reality``.)

One asymmetry remains and is not removable, so it is stated rather than hidden:
the centralized agent carries one Adam optimizer across all 48 rollouts, whereas
each federated local phase constructs a fresh optimizer from the parameters it
received.  That is what FedAvg *is* -- averaging Adam moments across clients is a
different algorithm -- so the centralized arm holds a genuine optimizer-state
advantage.  It is therefore labelled a **performance reference, not a
communication baseline**: it transmits no parameters and all of its ledger totals
are zero.
"""

from __future__ import annotations

import copy
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from ..envs.config import ScenarioConfig
from ..envs.scenarios import get_scenario
from ..envs.spectrum import SpectrumEnv
from ..learner.evaluate import evaluate_policy
from ..learner.ppo import PPOHyperParams, RecurrentPPO, train
from ..models.registry import build_model, count_parameters
from ..seeding import derive_seed, make_torch_generator
from .aggregate import CommunicationLedger, fedavg, state_dict_bytes
from .topology import ClientSpec, FederatedConfig, build_topology, clients_of_edge

__all__ = [
    "FEDERATED_ARMS",
    "run_arm",
    "run_centralized",
    "run_flat_federated",
    "run_hierarchical_federated",
    "flat_federated_core",
    "hierarchical_federated_core",
    "federated_row",
    "RESULT_KEYS",
]

#: The three arms, in increasing order of topological structure.
FEDERATED_ARMS: tuple[str, ...] = (
    "centralized",
    "flat_federated",
    "hierarchical_federated",
)

#: Exactly the keys every runner returns.  Asserted by
#: ``tests/test_federated.py::test_runner_result_keys_are_exact``.
RESULT_KEYS: tuple[str, ...] = (
    "arm",
    "scenario",
    "model_key",
    "seed",
    "parameter_count",
    "parameter_bytes",
    "train_steps",
    "declared_train_steps",
    "num_updates",
    "communication",
    "round_history",
    "final_eval",
    "wall_seconds",
)

_METRIC_KEYS: tuple[str, ...] = (
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "grad_norm",
    "mean_rollout_return",
    "policy_entropy_mean",
)


# --------------------------------------------------------------------------- #
# A vector env whose lanes may carry different scenario configs
# --------------------------------------------------------------------------- #


class PooledVectorEnv:
    """Duck-typed :class:`~dsa.envs.vector.SyncVectorEnv` with per-lane configs.

    ``SyncVectorEnv`` shares one :class:`ScenarioConfig` across all lanes, which is
    correct for the single-scenario suite but cannot express the centralized arm's
    pooled client distribution.  This class implements the same surface the
    learner consumes (``num_envs``, ``obs_dim``, ``action_dim``, ``reset``,
    ``step``, ``reseed``, ``episode_metrics``) over a heterogeneous lane list.

    It lives here, in agent D's file, rather than as an edit to
    :mod:`dsa.envs.vector`, because the pooling requirement is a federated
    concern.  Like ``SyncVectorEnv`` it holds **no** mutable counter: every lane is
    a pure function of its own ``(config, seed)`` pair, and ``reconfigure``
    replaces both explicitly.
    """

    def __init__(
        self, configs: Sequence[ScenarioConfig], episode_seeds: Sequence[int]
    ) -> None:
        cfgs = list(configs)
        seeds = [int(s) for s in episode_seeds]
        if not cfgs:
            raise ValueError("configs must be non-empty")
        if len(cfgs) != len(seeds):
            raise ValueError(
                f"got {len(cfgs)} configs for {len(seeds)} seeds; they must match"
            )
        dims = {(c.obs_dim, c.action_dim, c.max_steps) for c in cfgs}
        if len(dims) != 1:
            raise ValueError(
                "pooled lanes must agree on (obs_dim, action_dim, max_steps); got "
                f"{sorted(dims)}"
            )
        self.configs = cfgs
        self.episode_seeds = seeds
        self.num_envs = len(cfgs)
        self.obs_dim = cfgs[0].obs_dim
        self.action_dim = cfgs[0].action_dim
        self.max_steps = int(cfgs[0].max_steps)
        self.envs: list[SpectrumEnv] = []
        self._build()

    def _build(self) -> None:
        self.envs = [
            SpectrumEnv(cfg, seed) for cfg, seed in zip(self.configs, self.episode_seeds)
        ]

    def reconfigure(
        self, configs: Sequence[ScenarioConfig], episode_seeds: Sequence[int]
    ) -> None:
        """Replace both the per-lane configs and the per-lane seeds."""
        cfgs = list(configs)
        seeds = [int(s) for s in episode_seeds]
        if len(cfgs) != self.num_envs or len(seeds) != self.num_envs:
            raise ValueError(f"reconfigure expects {self.num_envs} configs and seeds")
        self.configs = cfgs
        self.episode_seeds = seeds
        self._build()

    def reseed(self, episode_seeds: Sequence[int]) -> None:
        self.reconfigure(self.configs, episode_seeds)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self._build()
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        dt = np.zeros(self.num_envs, dtype=np.float32)
        for i, env in enumerate(self.envs):
            state, _info = env.reset()
            obs[i] = state["obs"]
            dt[i] = state["dt"]
        return obs, dt

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        acts = np.asarray(actions, dtype=np.int64).reshape(-1)
        if acts.shape[0] != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {acts.shape[0]}")
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        dt = np.zeros(self.num_envs, dtype=np.float32)
        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []
        for i, env in enumerate(self.envs):
            state, r, terminated, truncated, info = env.step(int(acts[i]))
            obs[i] = state["obs"]
            dt[i] = state["dt"]
            reward[i] = r
            done[i] = bool(terminated or truncated)
            infos.append(info)
        return obs, dt, reward, done, infos

    def episode_metrics(self) -> list[dict[str, float]]:
        return [env.episode_metrics() for env in self.envs]


# --------------------------------------------------------------------------- #
# Seeding helpers -- every stream below is a pure function of its arguments
# --------------------------------------------------------------------------- #


def client_env_seed(
    base_seed: int, arm: str, cloud_round: int, edge_round: int, client_id: int
) -> int:
    """Seed for one client's local phase, per the frozen role table (spec 3.2)::

        derive_seed(base_seed, "fed", arm, r, k, c, "env")

    The model key is deliberately absent, so swapping the backbone does not move
    the environment stream.  The arm *is* present, which means two arms do not by
    default share client data; the runners therefore accept ``seed_arm_override``
    so a controlled experiment (and
    ``tests/test_federated.py::test_hierarchical_is_not_flat_fedavg``) can pin two
    arms to byte-identical client streams and attribute any difference purely to
    topology.
    """
    return derive_seed(
        base_seed, "fed", arm, int(cloud_round), int(edge_round), int(client_id), "env"
    )


def lane_seeds(client_seed: int, num_lanes: int, rollout_index: int) -> list[int]:
    """Per-lane episode seeds inside one client's local phase.

    A local phase runs ``local_num_envs`` independent episodes, so the single
    client seed is expanded into a lane substream the same way the environment
    expands an episode seed into its internal substreams.
    """
    return [
        derive_seed(client_seed, "lane", int(rollout_index), lane)
        for lane in range(int(num_lanes))
    ]


def _eval_seeds(base_seed: int, scenario: str, n: int) -> list[int]:
    """Evaluation-episode seeds, identical to the ones the main suite uses.

    All three arms are evaluated on this one stream, so the arm comparison is
    paired at the episode level.
    """
    return [derive_seed(base_seed, scenario, "env", "eval", i) for i in range(int(n))]


# --------------------------------------------------------------------------- #
# Local training
# --------------------------------------------------------------------------- #


def _hparams_for(fed: FederatedConfig, hparams: PPOHyperParams | None) -> PPOHyperParams:
    """PPO hyper-parameters with the batch shape pinned to the federated budget."""
    base = PPOHyperParams() if hparams is None else hparams
    return PPOHyperParams(
        learning_rate=base.learning_rate,
        gamma=base.gamma,
        gae_lambda=base.gae_lambda,
        clip_epsilon=base.clip_epsilon,
        value_coef=base.value_coef,
        entropy_coef=base.entropy_coef,
        max_grad_norm=base.max_grad_norm,
        update_epochs=base.update_epochs,
        num_sequence_minibatches=base.num_sequence_minibatches,
        num_envs=fed.local_num_envs,
        horizon=fed.horizon,
    )


def _make_agent(
    fed: FederatedConfig,
    base_seed: int,
    obs_dim: int,
    action_dim: int,
    hp: PPOHyperParams,
    role: tuple[object, ...],
) -> RecurrentPPO:
    """Construct a fresh agent whose RNG streams are a pure function of ``role``.

    Weight initialisation always uses the *same* seed
    (``derive_seed(base_seed, "policy_init", model_key)``) so that all three arms
    start from a byte-identical model.  Local phases immediately overwrite those
    weights with the received parameters; keeping the init seed fixed anyway means
    a run's starting point never depends on which arm is executing.
    """
    model = build_model(
        fed.model_key,
        obs_dim,
        action_dim,
        seed=derive_seed(base_seed, "policy_init", fed.model_key),
    )
    action_gen = make_torch_generator(
        base_seed, "action", fed.model_key, fed.scenario, *role
    )
    shuffle_gen = make_torch_generator(
        base_seed, "shuffle", fed.model_key, fed.scenario, *role
    )
    return RecurrentPPO(model, hp, action_gen, shuffle_gen, device=fed.device)


def _train_local_phase(
    fed: FederatedConfig,
    base_seed: int,
    hp: PPOHyperParams,
    init_state: Mapping[str, torch.Tensor],
    client: ClientSpec,
    arm_seed_label: str,
    cloud_round: int,
    edge_round: int,
) -> tuple[dict[str, torch.Tensor], int, int, list[dict[str, float]]]:
    """Train one client for ``fed.local_steps`` environment steps.

    ``init_state`` is the parameter vector the client *received* -- from its edge
    under the hierarchy, from the cloud under flat federated.  This argument is
    the entire fix for defect D7: the old code passed the global model here
    regardless of topology.

    Returns ``(trained_state_dict, measured_env_steps, gradient_steps, update_logs)``.
    The step count is read off the agent, never computed from the config.
    """
    agent = _make_agent(
        fed,
        base_seed,
        client.scenario_config.obs_dim,
        client.scenario_config.action_dim,
        hp,
        role=("fed", arm_seed_label, cloud_round, edge_round, client.client_id),
    )
    agent.load_state_dict(init_state)

    seed = client_env_seed(
        base_seed, arm_seed_label, cloud_round, edge_round, client.client_id
    )
    vec_env = PooledVectorEnv(
        [client.scenario_config] * hp.num_envs, lane_seeds(seed, hp.num_envs, 0)
    )
    logs = train(
        agent,
        vec_env,
        fed.local_steps,
        rollout_seed_fn=lambda r: lane_seeds(seed, hp.num_envs, r),
    )
    return (
        agent.state_dict(),
        int(agent.total_env_steps),
        int(agent.total_gradient_steps),
        logs,
    )


#: Diagnostics whose PPO-side name already encodes an average, mapped to the name
#: they keep after being averaged across clients.  Without this, ``_mean_key``
#: would emit ``mean_mean_rollout_return`` -- a column nobody would look for, so
#: the quantity would silently vanish from every federated table.
_MEAN_KEY_OVERRIDES: dict[str, str] = {
    "mean_rollout_return": "mean_rollout_return",
    "policy_entropy_mean": "mean_policy_entropy_nats",
}


def _mean_key(key: str) -> str:
    """Column name for ``key`` after averaging it over a phase's clients."""
    return _MEAN_KEY_OVERRIDES.get(key, f"mean_{key}")


def _summarise(logs: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean of each PPO diagnostic over a collection of update logs."""
    out: dict[str, float] = {}
    if not logs:
        return out
    for key in _METRIC_KEYS:
        vals = [float(log[key]) for log in logs if key in log and log[key] == log[key]]
        out[_mean_key(key)] = float(np.mean(vals)) if vals else float("nan")
    devs = [
        float(log["first_epoch_max_ratio_deviation"])
        for log in logs
        if "first_epoch_max_ratio_deviation" in log
    ]
    # The PPO ratio-sanity invariant (spec 4.4 rule 6).  Carried through the
    # federated layer because a stale-hidden-state or eval/train-mode bug in a
    # client update would otherwise be invisible in a federated row.
    out["max_first_epoch_ratio_deviation"] = float(max(devs)) if devs else float("nan")
    returns = [
        float(log["mean_rollout_return"])
        for log in logs
        if "mean_rollout_return" in log and log["mean_rollout_return"] == log["mean_rollout_return"]
    ]
    if returns:
        out["min_rollout_return"] = float(min(returns))
        out["max_rollout_return"] = float(max(returns))
    return out


def _final_eval(
    fed: FederatedConfig,
    base_seed: int,
    state: Mapping[str, torch.Tensor],
    obs_dim: int,
    action_dim: int,
) -> dict[str, float]:
    """Greedy evaluation of the final global model on the shared held-out stream.

    Every arm is evaluated on ``get_scenario(fed.scenario)`` -- the *unperturbed*
    base scenario, not any client's local variant -- with the same episode seeds.
    That makes the arms paired at the episode level and stops an arm from being
    rewarded for overfitting to whichever client distribution it happened to see
    last.
    """
    model = build_model(
        fed.model_key,
        obs_dim,
        action_dim,
        seed=derive_seed(base_seed, "policy_init", fed.model_key),
    )
    model.load_state_dict(dict(state))
    summary, _per_episode = evaluate_policy(
        model,
        get_scenario(fed.scenario),
        _eval_seeds(base_seed, fed.scenario, fed.eval_episodes),
        deterministic=True,
        device=fed.device,
    )
    return summary


def _result(
    arm: str,
    fed: FederatedConfig,
    base_seed: int,
    parameter_count: int,
    parameter_bytes: int,
    train_steps: int,
    num_updates: int,
    ledger: CommunicationLedger,
    round_history: list[dict[str, float]],
    final_eval: dict[str, float],
    wall_seconds: float,
) -> dict[str, object]:
    return {
        "arm": arm,
        "scenario": fed.scenario,
        "model_key": fed.model_key,
        "seed": int(base_seed),
        "parameter_count": int(parameter_count),
        "parameter_bytes": int(parameter_bytes),
        "train_steps": int(train_steps),
        "declared_train_steps": int(fed.total_env_steps_per_arm()),
        "num_updates": int(num_updates),
        "communication": ledger.totals(),
        "round_history": round_history,
        "final_eval": final_eval,
        "wall_seconds": float(wall_seconds),
    }


# --------------------------------------------------------------------------- #
# Arm: centralized
# --------------------------------------------------------------------------- #


def run_centralized(
    fed: FederatedConfig,
    base_seed: int,
    hparams: PPOHyperParams | None = None,
) -> dict[str, object]:
    """One agent trained on the pooled client distribution.

    Its ``local_num_envs`` lanes are assigned round-robin across all clients with
    an offset that advances each rollout, so over the run every client contributes
    exactly the same number of episodes -- a fixed assignment with
    ``num_envs % num_clients != 0`` would over-represent the low-numbered clients.

    This arm transmits no parameters.  Every ledger total is zero and it is a
    **performance reference, not a communication baseline**.
    """
    t0 = time.perf_counter()
    base_cfg = get_scenario(fed.scenario)
    topology = build_topology(base_cfg, fed)
    hp = _hparams_for(fed, hparams)
    obs_dim, action_dim = base_cfg.obs_dim, base_cfg.action_dim

    agent = _make_agent(
        fed, base_seed, obs_dim, action_dim, hp, role=("fed", "centralized")
    )
    ledger = CommunicationLedger()  # deliberately never written to

    batch_steps = hp.num_envs * hp.horizon
    total = fed.total_env_steps_per_arm()
    if total % batch_steps != 0:
        raise ValueError(
            f"centralized budget {total} is not a whole number of rollouts of "
            f"{batch_steps}; the arms would not be step-matched"
        )
    num_rollouts = total // batch_steps

    def assignment(rollout: int) -> list[ClientSpec]:
        return [
            topology[(rollout * hp.num_envs + lane) % fed.num_clients]
            for lane in range(hp.num_envs)
        ]

    def seeds_for(rollout: int) -> list[int]:
        return [
            derive_seed(
                client_env_seed(
                    base_seed, "centralized", rollout, 0, spec.client_id
                ),
                "lane",
                rollout,
                lane,
            )
            for lane, spec in enumerate(assignment(rollout))
        ]

    specs0 = assignment(0)
    vec_env = PooledVectorEnv(
        [s.scenario_config for s in specs0], seeds_for(0)
    )

    round_history: list[dict[str, float]] = []
    num_updates = 0
    # Explicit loop rather than dsa.learner.ppo.train: the pooled lane->client
    # assignment rotates per rollout, and train() only reseeds -- it cannot
    # reconfigure.  Nothing else differs from train().
    for r in range(num_rollouts):
        specs = assignment(r)
        vec_env.reconfigure([s.scenario_config for s in specs], seeds_for(r))
        batch = agent.collect(vec_env)
        metrics = agent.update(batch)
        num_updates += 1
        row: dict[str, float] = {
            "phase_index": float(r),
            "cloud_round": float(r // max(1, fed.edge_rounds_per_cloud_round)),
            "edge_round": float(r % max(1, fed.edge_rounds_per_cloud_round)),
            "aggregation_level": "none",
            "num_clients_trained": float(len({s.client_id for s in specs})),
            "env_steps_cumulative": float(agent.total_env_steps),
            "gradient_steps_cumulative": float(agent.total_gradient_steps),
            "cloud_bytes_cumulative": 0.0,
            "edge_local_bytes_cumulative": 0.0,
            "total_bytes_cumulative": 0.0,
        }
        row.update(_summarise([metrics]))
        round_history.append(row)

    final_state = agent.state_dict()
    result = _result(
        arm="centralized",
        fed=fed,
        base_seed=base_seed,
        parameter_count=count_parameters(agent.model),
        parameter_bytes=state_dict_bytes(final_state),
        train_steps=int(agent.total_env_steps),
        num_updates=num_updates,
        ledger=ledger,
        round_history=round_history,
        final_eval=_final_eval(fed, base_seed, final_state, obs_dim, action_dim),
        wall_seconds=time.perf_counter() - t0,
    )
    return result


# --------------------------------------------------------------------------- #
# Arm: flat federated
# --------------------------------------------------------------------------- #


def flat_federated_core(
    fed: FederatedConfig,
    base_seed: int,
    hparams: PPOHyperParams | None = None,
    seed_arm_override: str | None = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """:func:`run_flat_federated`, additionally returning the final global parameters.

    The public runner's return keys are frozen and do not include a parameter
    vector, but ``tests/test_federated.py::test_hierarchical_is_not_flat_fedavg``
    has to compare the two arms' *weights* -- comparing their returns would not
    distinguish "the topologies do different things" from "the topologies do the
    same thing and evaluation is noisy".  This core function is how the test gets
    at them.
    """
    t0 = time.perf_counter()
    arm_seed_label = "flat_federated" if seed_arm_override is None else seed_arm_override
    base_cfg = get_scenario(fed.scenario)
    topology = build_topology(base_cfg, fed)
    hp = _hparams_for(fed, hparams)
    obs_dim, action_dim = base_cfg.obs_dim, base_cfg.action_dim

    seed_agent = _make_agent(
        fed, base_seed, obs_dim, action_dim, hp, role=("fed", "init")
    )
    theta_cloud = seed_agent.state_dict()
    parameter_count = count_parameters(seed_agent.model)
    parameter_bytes = state_dict_bytes(theta_cloud)

    ledger = CommunicationLedger()
    # Initial distribution of the global model to every participant.  Recorded
    # once, here; the per-round redistribution below is recorded after each
    # aggregation, so no transfer is counted twice and none is omitted.
    for client in topology:
        ledger.record("cloud_down", "cloud", client.label(), theta_cloud, 0)

    train_steps = 0
    gradient_steps = 0
    num_updates = 0
    round_history: list[dict[str, float]] = []

    for phase in range(fed.flat_rounds()):
        cloud_round = phase // fed.edge_rounds_per_cloud_round
        edge_round = phase % fed.edge_rounds_per_cloud_round
        client_states: list[dict[str, torch.Tensor]] = []
        phase_logs: list[Mapping[str, float]] = []
        for client in topology:
            state, steps, grads, logs = _train_local_phase(
                fed,
                base_seed,
                hp,
                init_state=theta_cloud,  # flat: every client starts from the cloud
                client=client,
                arm_seed_label=arm_seed_label,
                cloud_round=cloud_round,
                edge_round=edge_round,
            )
            ledger.record("cloud_up", client.label(), "cloud", state, phase)
            client_states.append(state)
            phase_logs.extend(logs)
            train_steps += steps
            gradient_steps += grads
            num_updates += len(logs)

        theta_cloud = fedavg(client_states)
        for client in topology:
            ledger.record("cloud_down", "cloud", client.label(), theta_cloud, phase)

        totals = ledger.totals()
        row: dict[str, float] = {
            "phase_index": float(phase),
            "cloud_round": float(cloud_round),
            "edge_round": float(edge_round),
            "aggregation_level": "cloud",
            "num_clients_trained": float(len(topology)),
            "env_steps_cumulative": float(train_steps),
            "gradient_steps_cumulative": float(gradient_steps),
            "cloud_bytes_cumulative": totals["cloud_bytes"],
            "edge_local_bytes_cumulative": totals["edge_local_bytes"],
            "total_bytes_cumulative": totals["total_bytes"],
        }
        row.update(_summarise(phase_logs))
        round_history.append(row)

    result = _result(
        arm="flat_federated",
        fed=fed,
        base_seed=base_seed,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        train_steps=train_steps,
        num_updates=num_updates,
        ledger=ledger,
        round_history=round_history,
        final_eval=_final_eval(fed, base_seed, theta_cloud, obs_dim, action_dim),
        wall_seconds=time.perf_counter() - t0,
    )
    return result, theta_cloud


def run_flat_federated(
    fed: FederatedConfig,
    base_seed: int,
    hparams: PPOHyperParams | None = None,
    seed_arm_override: str | None = None,
) -> dict[str, object]:
    """Classic FedAvg: every client talks to the cloud, every round.

    Runs ``cloud_rounds * edge_rounds_per_cloud_round`` rounds so that its total
    environment-step budget equals the hierarchy's exactly.  Every transfer is
    ``cloud_up`` / ``cloud_down``: the wide-area link carries one payload per
    client per direction per round, which is the cost the hierarchy exists to
    avoid.

    ``seed_arm_override`` forces the arm label used in environment-seed derivation
    (see :func:`client_env_seed`).  It exists so a test can put this arm and the
    hierarchical arm on byte-identical client streams.
    """
    return flat_federated_core(fed, base_seed, hparams, seed_arm_override)[0]


# --------------------------------------------------------------------------- #
# Arm: hierarchical federated
# --------------------------------------------------------------------------- #


def hierarchical_federated_core(
    fed: FederatedConfig,
    base_seed: int,
    hparams: PPOHyperParams | None = None,
    seed_arm_override: str | None = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """:func:`run_hierarchical_federated`, additionally returning the final parameters.

    See :func:`flat_federated_core` for why the parameter vector is exposed.
    """
    t0 = time.perf_counter()
    arm_seed_label = (
        "hierarchical_federated" if seed_arm_override is None else seed_arm_override
    )
    base_cfg = get_scenario(fed.scenario)
    topology = build_topology(base_cfg, fed)
    hp = _hparams_for(fed, hparams)
    obs_dim, action_dim = base_cfg.obs_dim, base_cfg.action_dim

    seed_agent = _make_agent(
        fed, base_seed, obs_dim, action_dim, hp, role=("fed", "init")
    )
    theta_cloud = seed_agent.state_dict()
    parameter_count = count_parameters(seed_agent.model)
    parameter_bytes = state_dict_bytes(theta_cloud)

    edges = list(range(fed.num_edges))
    edge_clients = {e: clients_of_edge(topology, e) for e in edges}
    theta_edge: dict[int, dict[str, torch.Tensor]] = {
        e: copy.deepcopy(theta_cloud) for e in edges
    }

    ledger = CommunicationLedger()
    for e in edges:
        ledger.record("cloud_down", "cloud", f"edge{e}", theta_cloud, 0)

    train_steps = 0
    gradient_steps = 0
    num_updates = 0
    round_history: list[dict[str, float]] = []

    for r in range(fed.cloud_rounds):
        for k in range(fed.edge_rounds_per_cloud_round):
            phase = r * fed.edge_rounds_per_cloud_round + k
            phase_logs: list[Mapping[str, float]] = []
            for e in edges:
                client_states: list[dict[str, torch.Tensor]] = []
                for client in edge_clients[e]:
                    ledger.record(
                        "edge_down", f"edge{e}", client.label(), theta_edge[e], phase
                    )
                    state, steps, grads, logs = _train_local_phase(
                        fed,
                        base_seed,
                        hp,
                        # FIX 1 (defect D7): the client starts from ITS EDGE.
                        init_state=theta_edge[e],
                        client=client,
                        arm_seed_label=arm_seed_label,
                        cloud_round=r,
                        edge_round=k,
                    )
                    ledger.record(
                        "edge_up", client.label(), f"edge{e}", state, phase
                    )
                    client_states.append(state)
                    phase_logs.extend(logs)
                    train_steps += steps
                    gradient_steps += grads
                    num_updates += len(logs)
                # FIX 2 (defect D7): the edge aggregate PERSISTS into the next
                # edge round.  This assignment is what the old code discarded.
                theta_edge[e] = fedavg(client_states)

            totals = ledger.totals()
            row: dict[str, float] = {
                "phase_index": float(phase),
                "cloud_round": float(r),
                "edge_round": float(k),
                "aggregation_level": (
                    "cloud" if k == fed.edge_rounds_per_cloud_round - 1 else "edge"
                ),
                "num_clients_trained": float(len(topology)),
                "env_steps_cumulative": float(train_steps),
                "gradient_steps_cumulative": float(gradient_steps),
                "cloud_bytes_cumulative": totals["cloud_bytes"],
                "edge_local_bytes_cumulative": totals["edge_local_bytes"],
                "total_bytes_cumulative": totals["total_bytes"],
            }
            row.update(_summarise(phase_logs))
            round_history.append(row)

        last_phase = r * fed.edge_rounds_per_cloud_round + (
            fed.edge_rounds_per_cloud_round - 1
        )
        for e in edges:
            ledger.record("cloud_up", f"edge{e}", "cloud", theta_edge[e], last_phase)
        theta_cloud = fedavg(
            [theta_edge[e] for e in edges],
            weights=[float(len(edge_clients[e])) for e in edges],
        )
        for e in edges:
            theta_edge[e] = copy.deepcopy(theta_cloud)
            ledger.record("cloud_down", "cloud", f"edge{e}", theta_cloud, last_phase)
        # Keep the cumulative byte columns of the last row of this cloud round
        # consistent with the transfers that round actually produced.
        totals = ledger.totals()
        round_history[-1]["cloud_bytes_cumulative"] = totals["cloud_bytes"]
        round_history[-1]["edge_local_bytes_cumulative"] = totals["edge_local_bytes"]
        round_history[-1]["total_bytes_cumulative"] = totals["total_bytes"]

    result = _result(
        arm="hierarchical_federated",
        fed=fed,
        base_seed=base_seed,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        train_steps=train_steps,
        num_updates=num_updates,
        ledger=ledger,
        round_history=round_history,
        final_eval=_final_eval(fed, base_seed, theta_cloud, obs_dim, action_dim),
        wall_seconds=time.perf_counter() - t0,
    )
    return result, theta_cloud


def run_hierarchical_federated(
    fed: FederatedConfig,
    base_seed: int,
    hparams: PPOHyperParams | None = None,
    seed_arm_override: str | None = None,
) -> dict[str, object]:
    """Client -> edge -> cloud, with the edge model actually doing something.

    Two properties distinguish this from the previous implementation, and both are
    regression-tested:

    1. ``init_state=theta_edge[client.edge_id]`` -- a client starts from **its
       edge's** current parameters.  The old code passed the global model here.
    2. ``theta_edge[edge_id] = fedavg(client_states)`` is assigned back into a
       dict that survives the ``k`` loop, so at edge round ``k >= 1`` clients
       start from an aggregate the cloud has never seen.  The old code computed
       this value, folded it into the cloud average, and dropped it.

    Cloud aggregation weights edges by their client count, so with balanced edges
    the hierarchical average is an *unbiased* estimate of the flat average.  That
    matters for interpretation: any measured difference between the arms comes
    from the trajectory the clients took, not from a weighting artefact.

    ``seed_arm_override`` behaves as in :func:`run_flat_federated`.
    """
    return hierarchical_federated_core(fed, base_seed, hparams, seed_arm_override)[0]


# --------------------------------------------------------------------------- #
# Dispatch and row flattening
# --------------------------------------------------------------------------- #


_RUNNERS: dict[str, Callable[..., dict[str, object]]] = {
    "centralized": run_centralized,
    "flat_federated": run_flat_federated,
    "hierarchical_federated": run_hierarchical_federated,
}


def run_arm(
    arm: str,
    fed: FederatedConfig,
    base_seed: int,
    hparams: PPOHyperParams | None = None,
) -> dict[str, object]:
    """Run one arm by name.  ``torch.set_num_threads(1)`` is the caller's job."""
    if arm not in _RUNNERS:
        raise KeyError(f"unknown arm {arm!r}; known arms: {list(FEDERATED_ARMS)}")
    return _RUNNERS[arm](fed, base_seed, hparams)


def federated_row(
    result: Mapping[str, object], fed: FederatedConfig | None = None
) -> dict[str, float | str]:
    """Flatten one runner result into a single deterministic CSV row.

    Every timing column carries the ``wall_`` prefix and no other column may be
    non-deterministic -- the reproducibility gate drops exactly ``^wall_`` and
    requires bitwise equality on everything else (spec 3.5).

    Communication columns come straight from ``CommunicationLedger.totals()``.
    All three are emitted (``cloud_``, ``edge_local_``, ``total_``) so that the
    hierarchy's cost is reported alongside its benefit.

    Passing ``fed`` adds a ``config_``-prefixed snapshot of the topology and
    budget to the row.  It is optional because the runner return keys are frozen
    by the specification and carry no config slot; a reader of
    ``federated_all_runs.csv`` should not have to consult a separate manifest to
    learn how many clients or edges produced a row, so the runner script is
    expected to pass it.
    """
    comm = dict(result["communication"])  # type: ignore[arg-type]
    final_eval = dict(result["final_eval"])  # type: ignore[arg-type]
    history = list(result["round_history"])  # type: ignore[arg-type]

    row: dict[str, float | str] = {
        "arm": str(result["arm"]),
        "scenario": str(result["scenario"]),
        "model_key": str(result["model_key"]),
        "seed": int(result["seed"]),  # type: ignore[arg-type]
        "parameter_count": int(result["parameter_count"]),  # type: ignore[arg-type]
        "parameter_bytes": int(result["parameter_bytes"]),  # type: ignore[arg-type]
        "train_steps": int(result["train_steps"]),  # type: ignore[arg-type]
        "declared_train_steps": int(result["declared_train_steps"]),  # type: ignore[arg-type]
        "train_steps_match_declared": bool(
            int(result["train_steps"]) == int(result["declared_train_steps"])  # type: ignore[arg-type]
        ),
        "num_updates": int(result["num_updates"]),  # type: ignore[arg-type]
        "num_rounds_logged": len(history),
    }
    for key in (
        "cloud_bytes",
        "edge_local_bytes",
        "total_bytes",
        "cloud_megabytes",
        "edge_local_megabytes",
        "total_megabytes",
        "num_transfers",
    ):
        row[key] = float(comm[key])
    for key, value in sorted(final_eval.items()):
        row[key] = float(value)

    devs = [
        float(h["max_first_epoch_ratio_deviation"])
        for h in history
        if "max_first_epoch_ratio_deviation" in h
        and h["max_first_epoch_ratio_deviation"] == h["max_first_epoch_ratio_deviation"]
    ]
    row["max_first_epoch_ratio_deviation"] = float(max(devs)) if devs else float("nan")
    if history:
        last = history[-1]
        for key in ("mean_policy_loss", "mean_value_loss", "mean_entropy", "mean_rollout_return"):
            if key in last:
                row[f"final_round_{key}"] = float(last[key])

    if fed is not None:
        for key, value in sorted(fed.to_dict().items()):
            row[f"config_{key}"] = (
                value if isinstance(value, str) else float(value)  # type: ignore[assignment]
            )

    row["wall_seconds"] = float(result["wall_seconds"])  # type: ignore[arg-type]
    return row
