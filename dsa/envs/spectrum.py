"""The dynamic-spectrum-access environment.

Design commitments, each of which is a repair of a confirmed defect in the
previous version of this repository:

**1. The seed is an explicit constructor argument.**  ``SpectrumEnv`` has no
class-level instance counter, no module-level mutable state and no reliance on
any global RNG.  Two environments constructed with the same ``episode_seed``
produce byte-identical streams regardless of how many other environments were
constructed before them, in what order, or in which process.  Every stochastic
operation draws from an explicit :class:`numpy.random.Generator` derived from the
episode seed through :func:`dsa.seeding.derive_seed`.

**2. ``dt`` has a single source of truth.**  The previous implementation advanced
the simulation clock with the *old* interval while relaxing the channel dynamics
with a *newly sampled* one, and handed the new one to the model -- an off-by-one
that made the whole continuous-time premise untestable.  Here the interval is
sampled exactly once per transition, at the top of the transition, and the same
scalar (a) advances ``sim_time``, (b) drives the background-load birth-death
process, (c) drives the channel relaxation, and (d) is returned to the agent as
``obs["dt"]`` and recorded in ``info["dt"]``.

**3. Time-dependent processes are driven by ``sim_time``, never ``step_count``.**
Regime switching on a step index is not dt-consistent: it would make an episode
of long intervals and an episode of short intervals switch regime at the same
*decision*, not at the same *time*.

Action space and what it implies
--------------------------------
``action_dim == num_channels``.  The action is "transmit on channel ``a``".
**There is no idle / no-transmit action.**  Two consequences, stated plainly
because they matter when reading the metrics:

* ``success_rate + collision_rate == 1`` exactly, by construction.  The two are
  complements, not independent measurements, and a report that cites both is
  citing one number twice.
* The achievable collision rate is bounded below by the probability that *every*
  channel is simultaneously occupied at the decision instant.  A perfect policy
  does not reach zero collisions; it reaches the environment's floor.  Under the
  registered scenarios that floor is small but non-zero, and it rises with
  ``num_background_users`` and with occupancy drift.

Reward
------
Let ``a`` be the chosen channel, evaluated against the latent state *as it stands
at the moment of the decision* (i.e. the state the agent's most recent
observation was generated from)::

    primary_busy[a]  in {0, 1}      licensed/primary occupant present
    n_bg[a]          in {0, 1, ...} background IoT devices camped on channel a
    quality[a]       in [0.05, 1]   achievable rate multiplier
    interference[a]  in [0, 1]      aggregate interference level

    collision := (primary_busy[a] == 1) or (n_bg[a] > 0)

    if collision:
        severity = 1 + 0.5 * interference[a] + 0.25 * min(n_bg[a], 2)
        reward   = -collision_penalty * severity
    else:
        reward   = success_reward_scale * quality[a] * (1 - interference[a])
                   + idle_bias

With the registered constants (``collision_penalty=1.25``,
``success_reward_scale=2.0``, ``idle_bias=0.05``) the per-step reward lies in
``[-2.5, +2.05]`` and the 64-step episode return lies in ``[-160, +131.2]``.
``idle_bias`` is a small constant bonus for landing on a genuinely free channel;
it is named for the fact that the channel was idle, not for an idle action --
there is no idle action.

The reward is *well posed* in the sense the tests check: no fixed action can
approach the maximum, because per-channel occupancy is heterogeneous and drifts,
background devices camp and hop, and the best channel therefore changes within an
episode.  See ``tests/test_envs.py::test_degenerate_policy_cannot_maximise``.

Observability: background devices are hidden terminals
-----------------------------------------------------
The sensed-occupancy block of the observation reports the **primary (licensed)
occupant only**, bit-flipped with probability ``sensing_noise``.  Background IoT
devices are *hidden terminals*: they are low-power secondary users whose energy
does not reach the agent's detector, so their channel choices are never
observed.  This is the canonical partial-observability structure of the dynamic
spectrum access literature, and it is what gives the specification's instruction
-- "``num_background_users`` is not an observation feature; the agent must infer
load from contention" -- actual force.

This choice is load-bearing for the whole benchmark, so it is worth being
explicit about why.  An earlier draft of this environment let sensed occupancy
report *both* primary and background occupants.  Because the observation is
generated after the world advances, and the next action is resolved against that
same latent state, that draft was a fully observed MDP with a myopic optimum: a
memoryless sensing heuristic came within a few percent of the per-step oracle in
every primary scenario, leaving nothing for a recurrent policy to exploit.  An
environment with no hidden state cannot falsify a claim about state evolution.
The margin that must be preserved is asserted by
``tests/test_envs.py::test_greedy_heuristic_leaves_headroom_for_a_learner``
rather than quoted here.  With background devices hidden, contention must be
inferred from the agent's own collision history -- and because devices *camp* on a
channel and hop only at rate :data:`BG_HOP_RATE`, that history is genuinely
predictive.  That is the temporal structure the recurrent models are being
tested on.

Interference and quality are sensed with additive ``N(0, sensing_noise)`` noise
and clipped to ``[0, 1]``; ``observation_dropout`` optionally zeroes individual
sensed features.  The agent never observes ``num_background_users``, the latent
``base_busy`` process, the background channel assignments, or the oracle action.
Ground-truth diagnostics are exposed only through ``info``, for tests, metrics
and figures; no policy in this repository reads ``info``.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from ..seeding import make_rng
from .config import SCALAR_OFFSETS, ScenarioConfig, obs_dim_for, obs_layout_for

__all__ = ["Observation", "SpectrumEnv", "BG_HOP_RATE"]


#: Rate at which a background device abandons its current channel, per unit of
#: simulated time.  A device hops with probability ``1 - exp(-BG_HOP_RATE * dt)``.
#: Non-zero persistence is what makes sensed occupancy *predictive* rather than
#: pure noise, and therefore what makes a sensing-based heuristic meaningful.
BG_HOP_RATE: float = 0.25

#: Relaxation rates for the three latent channel processes (per unit time).
_BUSY_RATE: float = 0.60
_QUALITY_RATE: float = 0.45
_INTERFERENCE_RATE: float = 0.80

_REWARD_OBS_SCALE: float = 3.0
_LOG_DT_FLOOR: float = 1e-3


class Observation(TypedDict):
    """What the environment hands the agent at each decision instant."""

    obs: np.ndarray  # (obs_dim,) float32
    dt: float  # interval elapsed since the previous observation


class SpectrumEnv:
    """A single-agent dynamic-spectrum-access episode.

    Parameters
    ----------
    config
        A frozen :class:`~dsa.envs.config.ScenarioConfig`.  Never mutated.
    episode_seed
        The complete specification of this episode's randomness.  Every internal
        substream is ``derive_seed(episode_seed, "sub", name)``.

    The environment truncates at ``config.max_steps`` and never terminates
    early, so ``terminated`` is always ``False`` and ``truncated`` becomes
    ``True`` on the final step.
    """

    def __init__(self, config: ScenarioConfig, episode_seed: int) -> None:
        self.config = config
        self.episode_seed = int(episode_seed)
        self.num_channels = int(config.num_channels)
        self.max_steps = int(config.max_steps)
        self.obs_dim = obs_dim_for(self.num_channels)
        self.action_dim = int(config.num_channels)
        self._layout = obs_layout_for(self.num_channels)

        self.sim_time: float = 0.0
        self.step_count: int = 0
        self._started = False
        self.reset()

    # ------------------------------------------------------------------ setup #

    def _make_streams(self) -> None:
        seed = self.episode_seed
        self._rng_init = make_rng(seed, "sub", "init")
        self._rng_dt = make_rng(seed, "sub", "dt")
        self._rng_channel = make_rng(seed, "sub", "channel")
        self._rng_background = make_rng(seed, "sub", "background")
        self._rng_sense = make_rng(seed, "sub", "sense")
        self._rng_load = make_rng(seed, "sub", "load")

    def reset(self) -> tuple[Observation, dict[str, float]]:
        """Start a fresh episode.  Idempotent: two resets give identical states."""
        cfg = self.config
        c = self.num_channels
        self._make_streams()

        self.sim_time = 0.0
        self.step_count = 0

        # Latent channel state.
        self._base_busy = self._rng_init.uniform(0.15, 0.65, size=c)
        self._phase = self._rng_init.uniform(0.0, 2.0 * math.pi, size=c)
        self._quality = np.clip(self._rng_init.uniform(0.35, 0.85, size=c), 0.05, 1.0)
        self._interference = np.clip(
            self._rng_init.uniform(0.05, 0.30, size=c) * cfg.interference_scale, 0.0, 1.0
        )

        # Background population.
        if cfg.load_mode == "birth_death":
            self._num_bg = int(
                self._rng_load.integers(cfg.load_min, cfg.load_max + 1)
            )
        else:
            self._num_bg = int(cfg.num_background_users)
        self._max_bg_slots = max(
            int(cfg.load_max), int(cfg.num_background_users), self._num_bg, 1
        )
        # Slot i is occupied iff i < self._num_bg.  Channel assignments persist
        # across steps (devices camp) and hop at BG_HOP_RATE.
        self._bg_channel = self._rng_background.integers(
            0, c, size=self._max_bg_slots
        ).astype(np.int64)

        self._primary_busy = (
            self._rng_channel.random(size=c) < self._base_busy
        ).astype(np.int64)

        # Episode bookkeeping.
        self._last_action = -1
        self._last_reward = 0.0
        self._last_success = 0.0
        self._last_collision = 0.0
        self._successes = 0
        self._collisions = 0
        self._return = 0.0
        self._utilisation_sum = 0.0
        self._dt_history: list[float] = []
        self._bg_count_history: list[int] = []
        self._action_counts = np.zeros(c, dtype=np.int64)
        self._bg_success = np.zeros(self._max_bg_slots, dtype=np.int64)
        self._bg_present = np.zeros(self._max_bg_slots, dtype=np.int64)

        self._started = True
        obs = self._build_features(dt_elapsed=float(cfg.dt_base))
        info = self._diagnostics(dt_elapsed=float(cfg.dt_base))
        info["reward"] = 0.0
        info["collision"] = 0.0
        info["success"] = 0.0
        info["terminated"] = 0.0
        info["truncated"] = 0.0
        return {"obs": obs, "dt": float(cfg.dt_base)}, info

    # ------------------------------------------------------------- transition #

    def _sample_dt(self) -> float:
        cfg = self.config
        if cfg.dt_mode == "constant":
            return float(cfg.dt_base)
        if cfg.dt_mode == "loguniform":
            lo, hi = math.log(cfg.dt_min), math.log(cfg.dt_max)
            return float(math.exp(self._rng_dt.uniform(lo, hi)))
        if cfg.dt_mode == "bimodal":
            return float(cfg.dt_max if self._rng_dt.random() < 0.5 else cfg.dt_min)
        raise ValueError(f"unhandled dt_mode {cfg.dt_mode!r}")  # pragma: no cover

    def _bg_counts(self) -> np.ndarray:
        """Number of *active* background devices camped on each channel."""
        counts = np.zeros(self.num_channels, dtype=np.int64)
        if self._num_bg > 0:
            np.add.at(counts, self._bg_channel[: self._num_bg], 1)
        return counts

    def _channel_rewards(self) -> np.ndarray:
        """Reward the agent would receive for each channel, given the latent state.

        Used only for diagnostics (``info["oracle_reward"]`` /
        ``info["best_action"]``) and for the regret metric.  Never observed.
        """
        cfg = self.config
        counts = self._bg_counts()
        collide = (self._primary_busy > 0) | (counts > 0)
        severity = 1.0 + 0.5 * self._interference + 0.25 * np.minimum(counts, 2)
        loss = -cfg.collision_penalty * severity
        gain = (
            cfg.success_reward_scale * self._quality * (1.0 - self._interference)
            + cfg.idle_bias
        )
        return np.where(collide, loss, gain)

    def _resolve(self, action: int) -> tuple[float, dict[str, float]]:
        """Score ``action`` against the CURRENT latent state.  See module docstring."""
        cfg = self.config
        a = int(action)
        if not (0 <= a < self.action_dim):
            raise ValueError(f"action {a} out of range [0, {self.action_dim})")

        counts = self._bg_counts()
        n_bg_here = int(counts[a])
        collision = bool(self._primary_busy[a] > 0 or n_bg_here > 0)

        if collision:
            severity = (
                1.0
                + 0.5 * float(self._interference[a])
                + 0.25 * float(min(n_bg_here, 2))
            )
            reward = -cfg.collision_penalty * severity
            throughput = 0.0
        else:
            reward = (
                cfg.success_reward_scale
                * float(self._quality[a])
                * (1.0 - float(self._interference[a]))
                + cfg.idle_bias
            )
            throughput = float(self._quality[a]) * (1.0 - float(self._interference[a]))

        # Background outcomes, for the fairness metric: a background device
        # succeeds when the primary is absent, it is alone on its channel, and
        # the agent did not transmit there.
        if self._num_bg > 0:
            chans = self._bg_channel[: self._num_bg]
            self._bg_present[: self._num_bg] += 1
            alone = counts[chans] == 1
            free = self._primary_busy[chans] == 0
            not_taken = chans != a
            self._bg_success[: self._num_bg] += (alone & free & not_taken).astype(
                np.int64
            )

        channel_rewards = self._channel_rewards()
        best_action = int(np.argmax(channel_rewards))
        info: dict[str, float] = {
            "reward": float(reward),
            "collision": float(collision),
            "success": float(not collision),
            "throughput": float(throughput),
            "n_background_on_action": float(n_bg_here),
            "primary_busy_on_action": float(self._primary_busy[a]),
            "oracle_reward": float(channel_rewards[best_action]),
            "best_action": float(best_action),
            "regret": float(channel_rewards[best_action] - reward),
        }

        # Bookkeeping.
        self._last_action = a
        self._last_reward = float(reward)
        self._last_success = 0.0 if collision else 1.0
        self._last_collision = 1.0 if collision else 0.0
        self._successes += int(not collision)
        self._collisions += int(collision)
        self._return += float(reward)
        self._utilisation_sum += throughput
        self._action_counts[a] += 1
        return float(reward), info

    def _advance_background_load(self, dt: float) -> None:
        """Birth-death count update plus per-device channel hopping.

        Both are continuous-time consistent: the probability of an event over an
        interval ``dt`` is ``1 - exp(-rate * dt)``, so an episode of long
        intervals sees proportionally more churn per decision than an episode of
        short ones -- which is the entire point of modelling irregular sampling.
        """
        cfg = self.config
        if cfg.load_mode == "birth_death":
            p_switch = 1.0 - math.exp(-cfg.load_switch_rate * dt)
            if self._rng_load.random() < p_switch:
                delta = 1 if self._rng_load.random() < 0.5 else -1
                self._num_bg = int(
                    np.clip(self._num_bg + delta, cfg.load_min, cfg.load_max)
                )

        p_hop = 1.0 - math.exp(-BG_HOP_RATE * dt)
        hop = self._rng_background.random(size=self._max_bg_slots) < p_hop
        new_channels = self._rng_background.integers(
            0, self.num_channels, size=self._max_bg_slots
        )
        self._bg_channel = np.where(hop, new_channels, self._bg_channel).astype(np.int64)

    def _regime(self, t: float) -> float:
        period = self.config.regime_switch_period
        if period <= 0.0:
            return 0.0
        return 0.12 if int(math.floor(t / period)) % 2 == 1 else -0.08

    def _relax_channels(self, dt: float) -> dict[str, float]:
        """Advance the latent channel processes by exactly ``dt``.

        Every relaxation factor has the form ``1 - exp(-rate * dt)``, so the
        dynamics are invariant to how the same elapsed time is subdivided into
        decisions.  The factors are returned so ``tests/test_envs.py`` can pin
        them to the ``dt`` the agent actually observed.
        """
        cfg = self.config
        drift = cfg.drift_strength * np.sin(0.07 * self.sim_time + self._phase)
        target_busy = np.clip(
            self._base_busy + drift + self._regime(self.sim_time), 0.05, 0.95
        )
        adapt = 1.0 - math.exp(-_BUSY_RATE * dt)
        self._base_busy = np.clip(
            (1.0 - adapt) * self._base_busy + adapt * target_busy, 0.05, 0.95
        )
        self._primary_busy = (
            self._rng_channel.random(size=self.num_channels) < self._base_busy
        ).astype(np.int64)

        quality_target = 0.55 + 0.25 * np.sin(0.09 * self.sim_time + self._phase / 2.0)
        quality_adapt = 1.0 - math.exp(-_QUALITY_RATE * dt)
        quality_sigma = 0.04 + 0.02 * cfg.drift_strength
        self._quality = np.clip(
            (1.0 - quality_adapt) * self._quality
            + quality_adapt * quality_target
            + self._rng_channel.normal(0.0, quality_sigma, size=self.num_channels),
            0.05,
            1.0,
        )

        bg_load = self._num_bg / float(self.num_channels)
        interference_target = (
            0.08 + cfg.interference_scale * 0.45 * self._primary_busy + 0.12 * bg_load
        )
        int_adapt = 1.0 - math.exp(-_INTERFERENCE_RATE * dt)
        self._interference = np.clip(
            (1.0 - int_adapt) * self._interference
            + int_adapt * interference_target
            + self._rng_channel.normal(0.0, 0.03, size=self.num_channels),
            0.0,
            1.0,
        )
        return {
            "busy_adapt": float(adapt),
            "quality_adapt": float(quality_adapt),
            "interference_adapt": float(int_adapt),
        }

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, bool, dict[str, float]]:
        """Resolve ``action``, then advance the world by exactly one interval."""
        if not self._started:  # pragma: no cover - constructor always resets
            raise RuntimeError("reset() must be called before step()")
        if self.step_count >= self.max_steps:
            raise RuntimeError(
                "episode already truncated at max_steps; this environment does "
                "not auto-reset -- call reset() explicitly"
            )

        # 1. resolve the agent's action against the CURRENT latent state
        reward, info = self._resolve(action)

        # 2. draw the interval for THIS transition -- sampled exactly once, here
        dt = self._sample_dt()

        # 3. advance the world by exactly dt -- one dt, three consumers
        self.sim_time += dt
        self._advance_background_load(dt)
        adapts = self._relax_channels(dt)

        self.step_count += 1
        self._dt_history.append(float(dt))
        self._bg_count_history.append(int(self._num_bg))

        obs = self._build_features(dt_elapsed=dt)
        info.update(self._diagnostics(dt_elapsed=dt))
        info.update(adapts)
        truncated = self.step_count >= self.max_steps
        info["terminated"] = 0.0
        info["truncated"] = float(truncated)
        return {"obs": obs, "dt": float(dt)}, float(reward), False, truncated, info

    # ------------------------------------------------------------ observation #

    def _build_features(self, dt_elapsed: float) -> np.ndarray:
        cfg = self.config
        c = self.num_channels
        rng = self._rng_sense
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # Primary occupancy only.  Background devices are hidden terminals --
        # see the module docstring for why this is deliberate.
        true_busy = (self._primary_busy > 0).astype(np.float64)
        flips = rng.random(size=c) < cfg.sensing_noise
        sensed_busy = np.where(flips, 1.0 - true_busy, true_busy)

        sigma = cfg.sensing_noise
        sensed_interference = np.clip(
            self._interference + rng.normal(0.0, sigma, size=c), 0.0, 1.0
        )
        sensed_quality = np.clip(
            self._quality + rng.normal(0.0, sigma, size=c), 0.0, 1.0
        )

        n_dropped = 0
        if cfg.observation_dropout > 0.0:
            keep_busy = rng.random(size=c) >= cfg.observation_dropout
            keep_int = rng.random(size=c) >= cfg.observation_dropout
            keep_qual = rng.random(size=c) >= cfg.observation_dropout
            n_dropped = int(
                (~keep_busy).sum() + (~keep_int).sum() + (~keep_qual).sum()
            )
            sensed_busy = sensed_busy * keep_busy
            sensed_interference = sensed_interference * keep_int
            sensed_quality = sensed_quality * keep_qual

        lay = self._layout
        obs[lay["busy"]] = sensed_busy
        obs[lay["interference"]] = sensed_interference
        obs[lay["quality"]] = sensed_quality
        if self._last_action >= 0:
            obs[lay["last_action"].start + self._last_action] = 1.0

        base = lay["scalars"].start
        steps = max(self.step_count, 1)
        obs[base + SCALAR_OFFSETS["last_reward"]] = (
            self._last_reward / _REWARD_OBS_SCALE
        )
        obs[base + SCALAR_OFFSETS["last_success"]] = self._last_success
        obs[base + SCALAR_OFFSETS["last_collision"]] = self._last_collision
        obs[base + SCALAR_OFFSETS["success_rate"]] = self._successes / steps
        obs[base + SCALAR_OFFSETS["collision_rate"]] = self._collisions / steps
        obs[base + SCALAR_OFFSETS["progress"]] = self.step_count / self.max_steps
        obs[base + SCALAR_OFFSETS["dt_normalised"]] = dt_elapsed / max(
            cfg.dt_max, cfg.dt_base
        )
        obs[base + SCALAR_OFFSETS["log_dt"]] = math.log(
            max(float(dt_elapsed), _LOG_DT_FLOOR)
        )

        self._last_sense_flips = int(flips.sum())
        self._last_dropped = n_dropped
        return obs

    def _diagnostics(self, dt_elapsed: float) -> dict[str, float]:
        """Ground-truth latent state.  Exposed via ``info`` only; never observed."""
        return {
            "dt": float(dt_elapsed),
            "sim_time": float(self.sim_time),
            "step_count": float(self.step_count),
            "num_background_users": float(self._num_bg),
            "base_busy_mean": float(self._base_busy.mean()),
            "base_busy_std": float(self._base_busy.std()),
            "primary_busy_fraction": float(self._primary_busy.mean()),
            "interference_mean": float(self._interference.mean()),
            "quality_mean": float(self._quality.mean()),
            "free_channels": float(
                int(((self._primary_busy == 0) & (self._bg_counts() == 0)).sum())
            ),
            "sense_flips": float(getattr(self, "_last_sense_flips", 0)),
            "obs_features_dropped": float(getattr(self, "_last_dropped", 0)),
        }

    # ---------------------------------------------------------------- metrics #

    def _action_histogram_entropy(self) -> float:
        total = int(self._action_counts.sum())
        if total == 0:
            return 0.0
        p = self._action_counts / total
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum())

    def _background_fairness(self) -> float:
        """Jain's fairness index over per-device background success *rates*.

        1.0 means the agent's behaviour degraded every background device
        equally; values below 1 mean some devices were crowded out more than
        others.  Returns 1.0 when no background device was ever present.
        """
        present = self._bg_present
        active = present > 0
        if not active.any():
            return 1.0
        rates = self._bg_success[active] / present[active]
        denom = float((rates**2).sum()) * int(active.sum())
        if denom <= 0.0:
            return 1.0
        return float(rates.sum() ** 2 / denom)

    def episode_metrics(self) -> dict[str, float]:
        """Summary of the episode so far.

        ``success_rate + collision_rate == 1`` exactly: there is no idle action,
        so every step is one or the other.
        """
        steps = max(self.step_count, 1)
        return {
            "episode_return": float(self._return),
            "spectrum_utilization": float(self._utilisation_sum / steps),
            "collision_rate": float(self._collisions / steps),
            "success_rate": float(self._successes / steps),
            "sim_time": float(self.sim_time),
            "action_histogram_entropy": self._action_histogram_entropy(),
            "background_fairness": self._background_fairness(),
            "steps": float(self.step_count),
            "mean_dt": float(np.mean(self._dt_history)) if self._dt_history else 0.0,
            "num_background_users_mean": (
                float(np.mean(self._bg_count_history))
                if self._bg_count_history
                else float(self._num_bg)
            ),
        }
