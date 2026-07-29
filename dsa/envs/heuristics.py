"""Zero-parameter baseline policies.

These live beside the environment rather than beside the neural models on
purpose: they index :data:`~dsa.envs.config.OBS_LAYOUT`, which is the
environment's contract.  Co-locating them means a change to the observation
layout cannot leave a heuristic silently reading the wrong columns.

All three baselines are **first-class citizens of every generated table**.  The
previous version of this repository defined a ``CORE_MODELS`` list that omitted
``random_policy`` and filtered it out of exactly the four tables the README
cited; restored, uniform random ranked fourth of seven, ahead of three of the
PPO variants.  Nothing in :mod:`dsa.analysis` may filter a policy out of a table.

On the greedy baseline
----------------------
The previous greedy heuristic *lost to uniform random in six of seven
scenarios*, which made the headline claim "PPO beats the heuristic" vacuous --
it was beating an anti-optimal policy.  :class:`GreedyOccupancyPolicy` is
rebuilt as a genuinely competitive floor: it combines the current (noisy)
occupancy reading with a dt-consistent exponential memory of past readings,
weights candidates by sensed quality and interference, and adds a small
persistence bonus for a channel that just worked.  It is deterministic and
consumes no randomness.

``tests/test_envs.py::test_greedy_heuristic_beats_uniform_random`` measures this
across every registered scenario and requires a strict majority.  If a future
change to the environment makes sensing uninformative, that test fails loudly
rather than letting a weak baseline flatter the learned policies.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .config import SCALAR_OFFSETS, obs_layout_for

__all__ = [
    "HeuristicPolicy",
    "RandomPolicy",
    "ConstantChannelPolicy",
    "GreedyOccupancyPolicy",
    "HEURISTIC_REGISTRY",
    "BASELINE_KEYS",
]


@runtime_checkable
class HeuristicPolicy(Protocol):
    """The duck type :func:`dsa.learner.evaluate.evaluate_policy` dispatches on.

    Deliberately *not* the neural policy interface: heuristics have no ``step``
    or ``unroll``, which is how the evaluator tells them apart.
    """

    name: str

    def reset(self, num_envs: int) -> None:
        """Prepare per-lane state for a fresh batch of ``num_envs`` episodes."""

    def act(self, obs: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """``(N, obs_dim) float32, (N,) float32 -> (N,) int64``."""


class RandomPolicy:
    """Uniform random channel selection.

    Draws from an explicit :class:`numpy.random.Generator` seeded at
    construction; never touches ``numpy.random``'s global state.  ``reset`` does
    **not** re-seed the generator -- it advances monotonically across evaluation
    chunks -- so the action stream is a pure function of the construction seed
    and the sequence of calls, and two identical invocations agree bitwise.
    """

    def __init__(self, action_dim: int, seed: int) -> None:
        self.name = "random_policy"
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.num_envs = 0

    def reset(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)

    def act(self, obs: np.ndarray, dt: np.ndarray) -> np.ndarray:
        n = np.asarray(obs).shape[0]
        return self._rng.integers(0, self.action_dim, size=n).astype(np.int64)


class ConstantChannelPolicy:
    """Always transmit on one fixed channel.

    The degenerate control.  It is in every table because it calibrates how much
    of a policy's return is explained by simply having a channel at all, and
    because it is the policy the reward function must *not* be maximisable by
    (see ``tests/test_envs.py::test_degenerate_policy_cannot_maximise``).
    """

    def __init__(self, action_dim: int, channel: int = 0) -> None:
        self.name = "constant_channel"
        self.action_dim = int(action_dim)
        if not (0 <= int(channel) < self.action_dim):
            raise ValueError(f"channel {channel} out of range")
        self.channel = int(channel)
        self.num_envs = 0

    def reset(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)

    def act(self, obs: np.ndarray, dt: np.ndarray) -> np.ndarray:
        n = np.asarray(obs).shape[0]
        return np.full(n, self.channel, dtype=np.int64)


class GreedyOccupancyPolicy:
    """Sensing plus a contention memory inferred from the agent's own collisions.

    Two memories, because the environment hides two different things:

    ``primary memory`` -- an exponential average of the *sensed* primary
        occupancy of each channel.  Sensing is bit-flipped at ``sensing_noise``,
        so averaging denoises it; under ``noisy_partial`` (20% flips plus 15%
        feature dropout) this is most of the policy's signal.

    ``contention memory`` -- an estimate, per channel, of whether a hidden
        background device is camped there.  Background devices are hidden
        terminals: they never appear in the observation.  The only evidence
        available is the agent's own outcome on the channel it last chose, read
        off the ``last_collision`` / ``last_success`` scalars.  This is literally
        the specification's "infer load from contention", and it is what makes
        this baseline a genuine floor rather than the anti-optimal policy the
        previous version of this repository shipped.

    Per decision, for lane ``i`` and channel ``c``::

        keep   = exp(-MEMORY_RATE * dt)                 # dt-consistent decay
        m      <- keep * m + (1 - keep) * busy          # primary memory
        cont   <- keep_c * cont + (1 - keep_c) * PRIOR  # relax toward the prior
        cont[last_action] <- (1 - BETA) * cont[last_action] + BETA * last_collision

        avail  = 1 - W_NOW * busy - W_MEMORY * m - W_CONTENTION * cont
        score  = avail * (quality + QUALITY_FLOOR) * (1 - interference)
                 + PERSISTENCE * last_success * one_hot(last_action)
        action = argmax(score)

    Every input is an observation feature.  The policy reads nothing the neural
    models cannot read, and it consumes no randomness: identical observations
    give identical actions.

    Both decay coefficients are ``exp(-rate * dt)``, so a long gap between
    decisions discounts stale evidence more than a short one.  A dt-blind
    baseline would make the irregular-interval scenarios look artificially
    favourable to any dt-aware learner; this one does not.
    """

    #: Weight on the current sensed primary-occupancy reading.
    W_NOW: float = 0.85
    #: Weight on the exponential memory of past primary-occupancy readings.
    W_MEMORY: float = 0.25
    #: Weight on the inferred hidden-terminal contention estimate.
    W_CONTENTION: float = 0.55
    #: Memory relaxation rate, per unit of simulated time.
    MEMORY_RATE: float = 0.5
    #: Relaxation rate of the contention estimate toward its prior.
    CONTENTION_RATE: float = 0.35
    #: Learning rate applied to the one channel the agent actually probed.
    CONTENTION_BETA: float = 0.7
    #: Uninformative prior probability that a channel carries a hidden device.
    CONTENTION_PRIOR: float = 0.5
    #: Bonus for repeating an action that just succeeded.
    PERSISTENCE: float = 0.12
    #: Keeps a fully dropped-out quality reading from zeroing an otherwise good
    #: candidate under ``observation_dropout``.
    QUALITY_FLOOR: float = 0.10

    def __init__(self, num_channels: int) -> None:
        self.name = "greedy_heuristic"
        self.num_channels = int(num_channels)
        self.action_dim = int(num_channels)
        self._layout = obs_layout_for(self.num_channels)
        self.num_envs = 0
        self._memory = np.zeros((0, self.num_channels), dtype=np.float64)
        self._contention = np.zeros((0, self.num_channels), dtype=np.float64)

    def reset(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        shape = (self.num_envs, self.num_channels)
        # 0.5 = maximally uninformative prior on per-channel occupancy.
        self._memory = np.full(shape, 0.5, dtype=np.float64)
        self._contention = np.full(shape, self.CONTENTION_PRIOR, dtype=np.float64)

    def act(self, obs: np.ndarray, dt: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        n = x.shape[0]
        if self._memory.shape[0] != n:
            self.reset(n)

        lay = self._layout
        busy = x[:, lay["busy"]]
        interference = x[:, lay["interference"]]
        quality = x[:, lay["quality"]]
        last_action = x[:, lay["last_action"]]
        scalars = x[:, lay["scalars"]]
        last_success = scalars[:, SCALAR_OFFSETS["last_success"]][:, None]
        last_collision = scalars[:, SCALAR_OFFSETS["last_collision"]][:, None]

        d = np.maximum(np.asarray(dt, dtype=np.float64).reshape(n, 1), 0.0)

        keep = np.exp(-self.MEMORY_RATE * d)
        self._memory = keep * self._memory + (1.0 - keep) * busy

        keep_c = np.exp(-self.CONTENTION_RATE * d)
        self._contention = (
            keep_c * self._contention + (1.0 - keep_c) * self.CONTENTION_PRIOR
        )
        # Update only the channel actually probed: last_action is a one-hot (all
        # zeros on the first decision of an episode, which leaves the prior).
        probed = last_action
        self._contention = (
            1.0 - probed * self.CONTENTION_BETA
        ) * self._contention + probed * self.CONTENTION_BETA * last_collision

        availability = (
            1.0
            - self.W_NOW * busy
            - self.W_MEMORY * self._memory
            - self.W_CONTENTION * self._contention
        )
        score = (
            availability
            * (quality + self.QUALITY_FLOOR)
            * (1.0 - np.clip(interference, 0.0, 1.0))
        )
        score = score + self.PERSISTENCE * last_success * last_action
        return np.argmax(score, axis=1).astype(np.int64)


#: name -> factory(action_dim, seed).  Consumed by the runner; every entry here
#: appears as a row in every generated table.
HEURISTIC_REGISTRY: dict[str, Callable[[int, int], "HeuristicPolicy"]] = {
    "random_policy": lambda action_dim, seed: RandomPolicy(action_dim, seed),
    "constant_channel": lambda action_dim, seed: ConstantChannelPolicy(
        action_dim, channel=0
    ),
    "greedy_heuristic": lambda action_dim, seed: GreedyOccupancyPolicy(
        num_channels=action_dim
    ),
}

BASELINE_KEYS: tuple[str, ...] = (
    "random_policy",
    "constant_channel",
    "greedy_heuristic",
)
