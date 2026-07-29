"""The scenario registry: five confirmatory dynamics classes, two exploratory.

Truth in labelling is the point of this file.  The previous version of this
repository registered seven "scenarios" of which two -- ``varying_users_small``
and ``varying_users_large`` -- both set ``scenario_name = "stationary"`` and
differed only in a background-user constant that was never mutated during an
episode.  That is a static load sweep advertised as population dynamics, and it
inflated the apparent breadth of the benchmark from five dynamics classes to
seven.  Both are deleted.  Genuine within-episode load variation now exists and
is called what it is: ``load_mode="birth_death"``.

Every property a scenario's *name* advertises is asserted against a rollout by
``tests/test_envs.py::test_scenario_names_tell_the_truth``.

Primary (confirmatory) scenarios
--------------------------------
``stationary``
    No drift, no regime switching, constant decision interval, fixed load.  The
    occupancy process has a genuine fixed point.
``non_stationary``
    Sinusoidal occupancy drift plus a square-wave regime, constant interval.
``interference_heavy``
    Elevated interference scale and a heavier fixed background load.
``irregular_dt``
    Identical to ``stationary`` in **every field except ``dt_mode``** (and the
    ``dt_min``/``dt_max`` the mode needs).  This is deliberate: it makes the
    pre-registered primary comparison an interaction test against a clean
    control.
``bursty_irregular``
    Bimodal decision intervals, drift, regime switching, and a birth-death
    background-load process.  The hardest registered class.

Exploratory scenarios (registered and runnable, excluded from the confirmatory
analysis): ``noisy_partial`` (heavy sensing noise plus feature dropout) and
``bursty_load`` (isolates the birth-death load factor at a constant interval).
"""

from __future__ import annotations

from .config import ScenarioConfig

__all__ = [
    "SCENARIO_REGISTRY",
    "PRIMARY_SCENARIOS",
    "EXPLORATORY_SCENARIOS",
    "ALL_SCENARIOS",
    "get_scenario",
]


_STATIONARY = ScenarioConfig(
    name="stationary",
    dt_mode="constant",
    dt_base=1.0,
    dt_min=1.0,
    dt_max=1.0,
    drift_strength=0.0,
    regime_switch_period=0.0,
    interference_scale=1.0,
    sensing_noise=0.03,
    observation_dropout=0.0,
    load_mode="fixed",
    num_background_users=4,
    load_min=4,
    load_max=4,
    load_switch_rate=0.0,
)

_NON_STATIONARY = ScenarioConfig(
    name="non_stationary",
    dt_mode="constant",
    dt_base=1.0,
    dt_min=1.0,
    dt_max=1.0,
    drift_strength=0.35,
    regime_switch_period=16.0,
    interference_scale=1.1,
    sensing_noise=0.03,
    observation_dropout=0.0,
    load_mode="fixed",
    num_background_users=4,
    load_min=4,
    load_max=4,
    load_switch_rate=0.0,
)

_INTERFERENCE_HEAVY = ScenarioConfig(
    name="interference_heavy",
    dt_mode="constant",
    dt_base=1.0,
    dt_min=1.0,
    dt_max=1.0,
    drift_strength=0.15,
    regime_switch_period=16.0,
    interference_scale=1.6,
    sensing_noise=0.03,
    observation_dropout=0.0,
    load_mode="fixed",
    num_background_users=7,
    load_min=7,
    load_max=7,
    load_switch_rate=0.0,
)

# Differs from ``stationary`` in exactly one *semantic* field: dt_mode.
# dt_min / dt_max only exist to parameterise that mode.
_IRREGULAR_DT = ScenarioConfig(
    name="irregular_dt",
    dt_mode="loguniform",
    dt_base=1.0,
    dt_min=0.2,
    dt_max=3.0,
    drift_strength=0.0,
    regime_switch_period=0.0,
    interference_scale=1.0,
    sensing_noise=0.03,
    observation_dropout=0.0,
    load_mode="fixed",
    num_background_users=4,
    load_min=4,
    load_max=4,
    load_switch_rate=0.0,
)

_BURSTY_IRREGULAR = ScenarioConfig(
    name="bursty_irregular",
    dt_mode="bimodal",
    dt_base=1.0,
    dt_min=0.25,
    dt_max=2.5,
    drift_strength=0.35,
    regime_switch_period=16.0,
    interference_scale=1.2,
    sensing_noise=0.03,
    observation_dropout=0.0,
    load_mode="birth_death",
    num_background_users=4,
    load_min=2,
    load_max=8,
    load_switch_rate=0.5,
)

_NOISY_PARTIAL = ScenarioConfig(
    name="noisy_partial",
    dt_mode="constant",
    dt_base=1.0,
    dt_min=1.0,
    dt_max=1.0,
    drift_strength=0.20,
    regime_switch_period=0.0,
    interference_scale=1.0,
    sensing_noise=0.20,
    observation_dropout=0.15,
    load_mode="fixed",
    num_background_users=4,
    load_min=4,
    load_max=4,
    load_switch_rate=0.0,
)

_BURSTY_LOAD = ScenarioConfig(
    name="bursty_load",
    dt_mode="constant",
    dt_base=1.0,
    dt_min=1.0,
    dt_max=1.0,
    drift_strength=0.0,
    regime_switch_period=0.0,
    interference_scale=1.0,
    sensing_noise=0.03,
    observation_dropout=0.0,
    load_mode="birth_death",
    num_background_users=4,
    load_min=2,
    load_max=8,
    load_switch_rate=0.5,
)


#: The five confirmatory dynamics classes, in canonical order.
PRIMARY_SCENARIOS: tuple[str, ...] = (
    "stationary",
    "non_stationary",
    "interference_heavy",
    "irregular_dt",
    "bursty_irregular",
)

#: Registered, runnable, and excluded from every confirmatory family.
EXPLORATORY_SCENARIOS: tuple[str, ...] = ("noisy_partial", "bursty_load")

SCENARIO_REGISTRY: dict[str, ScenarioConfig] = {
    "stationary": _STATIONARY,
    "non_stationary": _NON_STATIONARY,
    "interference_heavy": _INTERFERENCE_HEAVY,
    "irregular_dt": _IRREGULAR_DT,
    "bursty_irregular": _BURSTY_IRREGULAR,
    "noisy_partial": _NOISY_PARTIAL,
    "bursty_load": _BURSTY_LOAD,
}

#: Every registered scenario name, primary first.
ALL_SCENARIOS: tuple[str, ...] = PRIMARY_SCENARIOS + EXPLORATORY_SCENARIOS


def get_scenario(name: str) -> ScenarioConfig:
    """Look up a registered scenario by name.

    Returns the frozen registry object itself; ``ScenarioConfig`` is immutable so
    sharing it between callers is safe.
    """
    try:
        return SCENARIO_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown scenario {name!r}; registered: {sorted(SCENARIO_REGISTRY)}"
        ) from None
