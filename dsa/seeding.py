"""Pure, order-independent, process-independent seed derivation.

This module is the single fix for the most damaging defect in the previous
version of this repository: environment seeds were derived from a *mutable
class-level instance counter*, so the environment stream a model saw depended on
how many environments had been constructed earlier in the same process -- which
in turn depended on the order in which models were requested on the command
line.  A fixed policy scored between -68.5 and -28.3 across six instantiation
positions of the *same* nominal seed: a ~40-point nuisance spread against a
~2-point headline effect.  Nothing measured that way is a controlled comparison.

The contract here removes the possibility of that bug:

* ``derive_seed`` is a **pure function** of its arguments.  No globals, no
  counters, no call-order dependence, no process-local salt.
* It uses :func:`hashlib.blake2b`, never Python's builtin :func:`hash`, which is
  randomly salted per process for ``str``/``bytes`` and would silently break
  cross-process reproducibility (and therefore every multi-worker run).
* Consumers derive their own stream from an explicit *role* tuple.  Environment
  seeds deliberately **exclude the model name**, so every model observes a
  byte-identical environment stream for a given ``(base_seed, scenario)``.  That
  is what makes the seed-paired statistics in the analysis layer valid.

Role table (frozen; see the rebuild specification section 3.2)::

    training env, lane l, rollout r  derive_seed(base, scenario, "env", "train", l, r)
    evaluation env, episode i        derive_seed(base, scenario, "env", "eval", i)
    env internal substream s         derive_seed(episode_seed, "sub", s)
    policy weight init               derive_seed(base, "policy_init", model_key)
    action sampling                  derive_seed(base, "action", model_key, scenario)
    minibatch shuffling              derive_seed(base, "shuffle", model_key, scenario)
    heuristic policy RNG             derive_seed(base, "heuristic", policy_key, scenario)
    federated client c, round r/k    derive_seed(base, "fed", arm, r, k, c, "env")
    bootstrap / permutation          derive_seed(ANALYSIS_SEED, "boot", comparison_id)
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

__all__ = [
    "SEED_MASK",
    "derive_seed",
    "make_rng",
    "make_torch_generator",
    "ENV_SUBSTREAMS",
]

#: 63-bit mask.  Keeps every derived seed inside the non-negative int64 range
#: accepted by both ``numpy.random.default_rng`` and ``torch.Generator``.
SEED_MASK: int = (1 << 63) - 1

#: The environment's internal substreams.  Each is derived from the episode seed,
#: so the streams are independent and adding a new substream cannot perturb an
#: existing one.
ENV_SUBSTREAMS: tuple[str, ...] = (
    "init",
    "dt",
    "channel",
    "background",
    "sense",
    "load",
)

_SEPARATOR = b"\x1f"  # ASCII unit separator: cannot appear in str(int) or a key


def derive_seed(base_seed: int, *parts: object) -> int:
    """Return a stable 63-bit seed for ``(base_seed, *parts)``.

    Pure function.  Identical arguments produce an identical result in any
    process, on any machine, in any call order.

    ``parts`` are stringified and joined with an unambiguous separator byte, so
    ``derive_seed(1, "ab", "c") != derive_seed(1, "a", "bc")``.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        digest.update(_SEPARATOR)
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "big") & SEED_MASK


def make_rng(base_seed: int, *parts: object) -> np.random.Generator:
    """An explicit :class:`numpy.random.Generator` for the given role.

    Never touches ``numpy.random``'s global state.
    """
    return np.random.default_rng(derive_seed(base_seed, *parts))


def make_torch_generator(base_seed: int, *parts: object) -> torch.Generator:
    """An explicit CPU :class:`torch.Generator` for the given role.

    Never calls ``torch.manual_seed``.
    """
    generator = torch.Generator()
    generator.manual_seed(derive_seed(base_seed, *parts))
    return generator
