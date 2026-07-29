"""``dsa`` -- Dynamic Spectrum Access benchmark.

A seeded, capacity-matched, pre-registered benchmark comparing continuous-time
recurrent policies (LTC, CfC) against gated, attention and memoryless baselines
on a partially observable dynamic-spectrum-access task, plus a hierarchical
federated topology study.

Sub-packages
------------
``dsa.seeding``     pure, order-independent seed derivation (blake2b)
``dsa.envs``        the spectrum environment, scenarios and heuristic policies
``dsa.models``      recurrent cells, the shared actor-critic, the model registry
``dsa.learner``     sequence-based recurrent PPO and the evaluation harness
``dsa.federated``   aggregation, topology and the three federated arms
``dsa.analysis``    statistics, tables and figures -- the only place numbers are
                    turned into claims, and it never filters a policy out

Nothing in this package hardcodes a result.  Every number that reaches a
document is computed from a file under ``results/`` by a script in ``scripts/``.
"""

__all__ = ["__version__"]

__version__ = "2.0.0"
