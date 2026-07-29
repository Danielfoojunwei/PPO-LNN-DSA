"""Observation layout and the frozen scenario configuration dataclass.

The observation layout is a *contract*.  It is exported as :data:`OBS_LAYOUT`
(and :func:`obs_layout_for` for non-default channel counts) so that the
heuristic policies in :mod:`dsa.envs.heuristics` index named slices instead of
magic numbers.  When the layout changes, the heuristics follow automatically.

Observation vector, ``obs_dim = 4 * num_channels + OBS_SCALARS``::

    [0:C]      sensed occupancy, each entry bit-flipped with prob sensing_noise
    [C:2C]     sensed interference, + N(0, sensing_noise), clipped to [0, 1]
    [2C:3C]    sensed channel quality, + N(0, sensing_noise), clipped to [0, 1]
    [3C:4C]    one-hot of the previous action (all zeros at reset)
    [4C+0]     last_reward / 3.0
    [4C+1]     last_success in {0, 1}
    [4C+2]     last_collision in {0, 1}
    [4C+3]     running success rate
    [4C+4]     running collision rate
    [4C+5]     step_count / max_steps
    [4C+6]     dt_elapsed / dt_max
    [4C+7]     log(max(dt_elapsed, 1e-3))

``num_background_users`` is deliberately **not** an observation feature.  In the
previous version of this repository it was, and that is precisely what let two
"scenarios" that differed only in a constant background-user count masquerade as
different dynamics classes.  The agent must infer offered load from contention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = [
    "OBS_SCALARS",
    "DEFAULT_NUM_CHANNELS",
    "SCALAR_OFFSETS",
    "obs_dim_for",
    "obs_layout_for",
    "OBS_LAYOUT",
    "ScenarioConfig",
    "DT_MODES",
    "LOAD_MODES",
]

#: Number of scalar (non per-channel) observation features.
OBS_SCALARS: int = 8

#: The channel count every registered scenario uses.
DEFAULT_NUM_CHANNELS: int = 8

#: Index of each scalar feature *within* the ``scalars`` slice.
SCALAR_OFFSETS: dict[str, int] = {
    "last_reward": 0,
    "last_success": 1,
    "last_collision": 2,
    "success_rate": 3,
    "collision_rate": 4,
    "progress": 5,
    "dt_normalised": 6,
    "log_dt": 7,
}

#: Valid ``dt_mode`` values.
DT_MODES: tuple[str, ...] = ("constant", "loguniform", "bimodal")

#: Valid ``load_mode`` values.
LOAD_MODES: tuple[str, ...] = ("fixed", "birth_death")


def obs_dim_for(num_channels: int) -> int:
    """Observation dimensionality for ``num_channels`` channels."""
    return 4 * int(num_channels) + OBS_SCALARS


def obs_layout_for(num_channels: int) -> dict[str, slice]:
    """Named slices of the observation vector for ``num_channels`` channels."""
    c = int(num_channels)
    return {
        "busy": slice(0, c),
        "interference": slice(c, 2 * c),
        "quality": slice(2 * c, 3 * c),
        "last_action": slice(3 * c, 4 * c),
        "scalars": slice(4 * c, 4 * c + OBS_SCALARS),
    }


#: Layout at the default channel count.  ``obs_layout_for(8) == OBS_LAYOUT``.
OBS_LAYOUT: dict[str, slice] = obs_layout_for(DEFAULT_NUM_CHANNELS)


@dataclass(frozen=True)
class ScenarioConfig:
    """A fully specified scenario.

    Frozen on purpose.  The previous version of this repository mutated a shared
    config in a ``_scenario_defaults()`` helper *after* construction, which meant
    the config object a run recorded was not the config the run executed.  Every
    scenario in :mod:`dsa.envs.scenarios` is now specified in full, in one place,
    and cannot be edited in flight.

    Field meanings
    --------------
    num_channels
        Number of spectrum channels.  ``action_dim == num_channels``: the action
        space is "which channel do I transmit on".  **There is no idle /
        no-transmit action** -- see :class:`dsa.envs.spectrum.SpectrumEnv` for
        what that implies for the achievable collision rate.
    max_steps
        Decisions per episode.  Episodes truncate; they never terminate early.
    load_mode, num_background_users, load_min, load_max, load_switch_rate
        Background traffic.  ``"fixed"`` holds the count at
        ``num_background_users`` for the whole episode.  ``"birth_death"`` draws
        the initial count uniformly from ``[load_min, load_max]`` and moves it by
        +/-1 with probability ``1 - exp(-load_switch_rate * dt)`` per transition,
        which makes the load process continuous-time consistent.
    drift_strength
        Amplitude of the slow sinusoidal drift of per-channel occupancy.  Zero
        means the occupancy process has a genuine fixed point.
    regime_switch_period
        Period (in *simulated time*, not steps) of the square-wave occupancy
        regime.  Zero disables it.
    interference_scale
        Multiplier on the primary-user contribution to interference.
    collision_penalty, success_reward_scale, idle_bias
        Reward shape; see :meth:`dsa.envs.spectrum.SpectrumEnv._resolve`.
    sensing_noise
        Bit-flip probability for sensed occupancy and the standard deviation of
        the additive noise on sensed interference and quality.
    observation_dropout
        Probability that an individual sensed per-channel feature is zeroed.
    dt_mode, dt_base, dt_min, dt_max
        The decision-interval process.  ``"constant"`` always yields ``dt_base``;
        ``"loguniform"`` yields ``exp(U(log dt_min, log dt_max))``; ``"bimodal"``
        yields ``dt_min`` or ``dt_max`` on a fair coin.
    """

    name: str
    num_channels: int = 8
    max_steps: int = 64
    load_mode: str = "fixed"
    num_background_users: int = 4
    load_min: int = 4
    load_max: int = 4
    load_switch_rate: float = 0.0
    drift_strength: float = 0.0
    regime_switch_period: float = 0.0
    interference_scale: float = 1.0
    collision_penalty: float = 1.25
    success_reward_scale: float = 2.0
    idle_bias: float = 0.05
    sensing_noise: float = 0.03
    observation_dropout: float = 0.0
    dt_mode: str = "constant"
    dt_base: float = 1.0
    dt_min: float = 1.0
    dt_max: float = 1.0

    def __post_init__(self) -> None:
        if self.dt_mode not in DT_MODES:
            raise ValueError(f"dt_mode must be one of {DT_MODES}, got {self.dt_mode!r}")
        if self.load_mode not in LOAD_MODES:
            raise ValueError(
                f"load_mode must be one of {LOAD_MODES}, got {self.load_mode!r}"
            )
        if self.num_channels < 2:
            raise ValueError("num_channels must be >= 2")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not (0.0 < self.dt_min <= self.dt_max):
            raise ValueError("require 0 < dt_min <= dt_max")
        if self.dt_base <= 0.0:
            raise ValueError("dt_base must be > 0")
        if not (0 <= self.load_min <= self.load_max):
            raise ValueError("require 0 <= load_min <= load_max")
        if self.load_mode == "birth_death" and self.load_switch_rate <= 0.0:
            raise ValueError("birth_death load requires load_switch_rate > 0")
        if not (0.0 <= self.sensing_noise < 0.5):
            raise ValueError("sensing_noise must be in [0, 0.5)")
        if not (0.0 <= self.observation_dropout < 1.0):
            raise ValueError("observation_dropout must be in [0, 1)")

    @property
    def obs_dim(self) -> int:
        return obs_dim_for(self.num_channels)

    @property
    def action_dim(self) -> int:
        return int(self.num_channels)

    def to_dict(self) -> dict[str, object]:
        """Plain-dict view, suitable for manifests and equality checks."""
        return asdict(self)
