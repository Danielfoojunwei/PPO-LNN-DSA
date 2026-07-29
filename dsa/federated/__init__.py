"""Hierarchical federated learning with measured communication and matched compute.

The three arms exported here differ in **topology only**.  Environment steps, PPO
rollouts, sequences per rollout and optimizer steps are matched by construction
and asserted equal by ``tests/test_federated.py``; the model is held fixed.  A
result from this module is therefore evidence about federated topology and is not
evidence about architecture.
"""

from __future__ import annotations

from .aggregate import (
    BYTES_PER_MEGABYTE,
    CLOUD_LINKS,
    EDGE_LINKS,
    LINK_KINDS,
    CommunicationLedger,
    Transfer,
    fedavg,
    state_dict_bytes,
)
from .runners import (
    FEDERATED_ARMS,
    RESULT_KEYS,
    PooledVectorEnv,
    client_env_seed,
    federated_row,
    flat_federated_core,
    hierarchical_federated_core,
    lane_seeds,
    run_arm,
    run_centralized,
    run_flat_federated,
    run_hierarchical_federated,
)
from .topology import (
    EDGE_DRIFT,
    EDGE_INTERFERENCE,
    MIN_BACKGROUND_USERS,
    ClientSpec,
    FederatedConfig,
    build_topology,
    clients_of_edge,
)

__all__ = [
    "fedavg",
    "state_dict_bytes",
    "Transfer",
    "CommunicationLedger",
    "CLOUD_LINKS",
    "EDGE_LINKS",
    "LINK_KINDS",
    "BYTES_PER_MEGABYTE",
    "FederatedConfig",
    "ClientSpec",
    "build_topology",
    "clients_of_edge",
    "EDGE_DRIFT",
    "EDGE_INTERFERENCE",
    "MIN_BACKGROUND_USERS",
    "FEDERATED_ARMS",
    "RESULT_KEYS",
    "run_arm",
    "run_centralized",
    "run_flat_federated",
    "run_hierarchical_federated",
    # Variants that also return the final global parameters, for controlled
    # topology experiments (the public runners' return keys are frozen).
    "flat_federated_core",
    "hierarchical_federated_core",
    "federated_row",
    "client_env_seed",
    "lane_seeds",
    "PooledVectorEnv",
]
