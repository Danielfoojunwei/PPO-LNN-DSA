"""Federated topology: the client population and the compute budget.

Two things live here, and they are the two things the previous version of this
repository got wrong before any learning happened.

**Compute matching (defect D8).**  :meth:`FederatedConfig.total_env_steps_per_arm`
is the *declared* budget, and it is the same integer for all three arms by
construction::

    hierarchical : num_clients * cloud_rounds * edge_rounds_per_cloud_round * local_steps
    flat         : num_clients * (cloud_rounds * edge_rounds_per_cloud_round) * local_steps
    centralized  : one agent trained for exactly that many steps

The old code declared ``rounds * local_timesteps = 3072`` while its round loop
actually executed ``rounds * clients * local_timesteps = 18432`` -- a silent 6x
compute advantage for the federated arms over the centralized one, recorded as
3072.0 in all fifteen rows of its results CSV.  The declaration above is only half
the fix; the other half is that every runner in :mod:`dsa.federated.runners`
reports the *measured* step count summed from the agents it actually trained, and
``tests/test_federated.py::test_train_steps_equals_reality`` asserts the two agree
for all three arms.

**Client heterogeneity.**  The audit accused the old repository of homogeneous
clients and then *refuted* its own accusation: the clustering was clean, with each
edge's clients sharing a non-stationarity level.  That property is deliberately
preserved here, because edge-local aggregation is only meaningful when clients on
an edge are more alike than clients across edges -- with i.i.d. clients a
hierarchy is pure overhead.

Heterogeneity is therefore two-level:

* **Between edges (the clustered axis).**  Every client on edge ``e`` shares
  ``drift_strength = base + EDGE_DRIFT[e]`` and
  ``interference_scale = base + EDGE_INTERFERENCE[e]``.
* **Within an edge (the nuisance axis).**  ``num_background_users`` varies by a
  deterministic function of ``client_id`` only, so clients on an edge are similar
  but not identical.

A heterogeneity axis is worthless if something downstream flattens it, so
``tests/test_federated.py`` checks the axes twice: once on the configs, and once
*behaviourally*, by rolling out each client's environment and confirming the edge
label is recoverable from the realised dynamics.  Environment clipping is the
specific hazard -- ``SpectrumEnv`` clips occupancy to ``[0.05, 0.95]`` and
interference to ``[0, 1]``, and a heterogeneity delta pushed into a clip is a
delta that exists in the config and nowhere in the data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Sequence

from ..envs.config import ScenarioConfig

__all__ = [
    "FederatedConfig",
    "ClientSpec",
    "build_topology",
    "clients_of_edge",
    "EDGE_DRIFT",
    "EDGE_INTERFERENCE",
    "MIN_BACKGROUND_USERS",
]

#: Additive occupancy-drift offset per edge index.  Edge 0 is the base regime.
EDGE_DRIFT: tuple[float, ...] = (0.00, 0.05)

#: Additive interference-scale offset per edge index.
EDGE_INTERFERENCE: tuple[float, ...] = (0.00, 0.10)

#: Floor on a client's background-user count.  A client with zero background
#: devices faces a qualitatively different (contention-free) problem, which would
#: be a heterogeneity axis of its own rather than a nuisance perturbation.
MIN_BACKGROUND_USERS: int = 1


@dataclass(frozen=True)
class FederatedConfig:
    """Topology, budget and evaluation settings for one federated experiment.

    The federated experiment varies **topology**, holding the model fixed.  Its
    conclusion is therefore about topology and says nothing about architecture;
    ``model_key`` is a CLI flag only so the finding can be spot-checked under a
    different backbone.
    """

    model_key: str = "ppo_ltc_cfc"
    scenario: str = "non_stationary"
    num_clients: int = 8
    num_edges: int = 2
    cloud_rounds: int = 3
    edge_rounds_per_cloud_round: int = 2
    local_steps: int = 768
    local_num_envs: int = 12
    horizon: int = 64
    eval_episodes: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.num_clients < 1:
            raise ValueError("num_clients must be >= 1")
        if self.num_edges < 1:
            raise ValueError("num_edges must be >= 1")
        if self.num_clients % self.num_edges != 0:
            raise ValueError(
                f"num_clients ({self.num_clients}) must be divisible by num_edges "
                f"({self.num_edges}) so that edge aggregates carry equal weight"
            )
        if self.num_edges > len(EDGE_DRIFT) or self.num_edges > len(EDGE_INTERFERENCE):
            raise ValueError(
                f"only {min(len(EDGE_DRIFT), len(EDGE_INTERFERENCE))} edge regimes are "
                f"defined; requested num_edges={self.num_edges}"
            )
        if self.cloud_rounds < 1:
            raise ValueError("cloud_rounds must be >= 1")
        if self.edge_rounds_per_cloud_round < 1:
            raise ValueError("edge_rounds_per_cloud_round must be >= 1")
        if self.horizon < 1 or self.local_num_envs < 1:
            raise ValueError("horizon and local_num_envs must be >= 1")
        if self.local_steps != self.local_num_envs * self.horizon:
            raise ValueError(
                "local_steps must equal local_num_envs * horizon so that a local "
                f"phase is a whole number of rollouts; got local_steps={self.local_steps}, "
                f"local_num_envs={self.local_num_envs}, horizon={self.horizon}"
            )
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be >= 1")

    # ------------------------------------------------------------- arithmetic #

    def clients_per_edge(self) -> int:
        return self.num_clients // self.num_edges

    def flat_rounds(self) -> int:
        """Rounds the flat arm runs so that it consumes the hierarchical budget."""
        return self.cloud_rounds * self.edge_rounds_per_cloud_round

    def local_phases_per_client(self) -> int:
        """Local training phases each client performs over the whole run."""
        return self.cloud_rounds * self.edge_rounds_per_cloud_round

    def total_env_steps_per_arm(self) -> int:
        """The environment-step budget every arm must consume, exactly."""
        return int(
            self.num_clients
            * self.cloud_rounds
            * self.edge_rounds_per_cloud_round
            * self.local_steps
        )

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = dict(asdict(self))
        d["clients_per_edge"] = self.clients_per_edge()
        d["flat_rounds"] = self.flat_rounds()
        d["local_phases_per_client"] = self.local_phases_per_client()
        d["total_env_steps_per_arm"] = self.total_env_steps_per_arm()
        return d


@dataclass(frozen=True)
class ClientSpec:
    """One client: its identity, its edge, and the environment it lives in."""

    client_id: int
    edge_id: int
    scenario_config: ScenarioConfig

    def label(self) -> str:
        return f"client{self.client_id}"


def _perturb_load(base: ScenarioConfig, offset: int) -> dict[str, object]:
    """Shift a client's background load by ``offset``, respecting the load mode.

    ``fixed`` clients get a shifted constant; ``birth_death`` clients get a shifted
    interval, so the *process* is perturbed rather than being silently converted
    into a constant.  Both are floored at :data:`MIN_BACKGROUND_USERS`.
    """
    if base.load_mode == "fixed":
        n = max(MIN_BACKGROUND_USERS, int(base.num_background_users) + offset)
        return {"num_background_users": n, "load_min": n, "load_max": n}
    lo = max(MIN_BACKGROUND_USERS, int(base.load_min) + offset)
    hi = max(lo, int(base.load_max) + offset)
    n = min(max(int(base.num_background_users) + offset, lo), hi)
    return {"num_background_users": n, "load_min": lo, "load_max": hi}


def build_topology(base: ScenarioConfig, fed: FederatedConfig) -> list[ClientSpec]:
    """Materialise the client population.

    Deterministic and free of randomness: the population is a *design*, not a
    sample, so two invocations with the same arguments return equal specs and no
    seed is involved.

    Edge assignment is contiguous (``edge_id = client_id // clients_per_edge``),
    which is what makes the edges "regional": neighbouring client ids share an
    edge and therefore share a dynamics regime.
    """
    per_edge = fed.clients_per_edge()
    specs: list[ClientSpec] = []
    for client_id in range(fed.num_clients):
        edge_id = client_id // per_edge
        # Within-edge nuisance variation: a deterministic function of client_id
        # only.  Note that a period-3 offset over contiguous blocks of 4 does NOT
        # balance perfectly across edges -- at the default topology the load
        # multisets are {3,4,5,3} on edge 0 and {4,5,3,4} on edge 1, so edge 1
        # carries 0.25 more background users on average.  That residual is small
        # relative to the deliberate between-edge deltas, but it is a genuine
        # partial confound with `interference_scale` rather than a clean nuisance
        # axis, and it is the reason
        # `tests/test_federated.py::test_edges_are_behaviourally_separable`
        # stratifies on load instead of comparing edge means directly.
        offset = (client_id % 3) - 1
        cfg = replace(
            base,
            name=f"{base.name}__edge{edge_id}_client{client_id}",
            drift_strength=float(base.drift_strength + EDGE_DRIFT[edge_id]),
            interference_scale=float(
                base.interference_scale + EDGE_INTERFERENCE[edge_id]
            ),
            **_perturb_load(base, offset),
        )
        specs.append(
            ClientSpec(client_id=client_id, edge_id=edge_id, scenario_config=cfg)
        )
    return specs


def clients_of_edge(
    topology: Sequence[ClientSpec], edge_id: int
) -> list[ClientSpec]:
    """The clients attached to ``edge_id``, in ascending ``client_id`` order."""
    return [c for c in topology if c.edge_id == int(edge_id)]
