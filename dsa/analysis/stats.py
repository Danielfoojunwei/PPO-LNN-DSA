"""Paired non-parametric statistics, implemented in numpy.

``scipy`` is deliberately **not** a dependency of this repository: it was pinned
in the previous ``requirements.txt`` with zero imports anywhere in the tree.
Everything here is numpy plus the standard library.

Why paired, and why non-parametric
----------------------------------
Environment seeds exclude the model name (spec 3.2), so for a given
``(base_seed, scenario)`` every policy is evaluated on a **byte-identical**
environment stream.  A comparison between two policies is therefore a set of
matched pairs, one per seed, and the correct analysis conditions on the pairing.
Twelve pairs is far too few to trust a normal approximation, so the test is an
exact sign-flip permutation test and the interval is a percentile bootstrap.

The exact test
--------------
Under the null hypothesis that the two arms are exchangeable within a pair, the
sign of each difference is equally likely to be ``+`` or ``-``.  With ``n <= 20``
we enumerate **all** ``2**n`` sign vectors rather than sampling them, so the
p-value is exact, deterministic and RNG-free.  At ``n = 12`` that is 4096
evaluations and the smallest attainable p-value is ``2 / 4096 = 4.883e-4``,
which is what buys the multiplicity headroom described in spec 4.3.

Enumeration is done with a subset-sum recursion rather than a ``2**n x n``
matrix.  Flipping the signs of a subset ``S`` changes the total from ``T`` to
``T - 2 * sum(d_i for i in S)``, so every one of the ``2**n`` flipped means is
recovered from the ``2**n`` subset sums, which are built in ``O(2**n)`` time and
memory instead of ``O(2**n * n)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from dsa.seeding import derive_seed

__all__ = [
    "ANALYSIS_SEED",
    "BOOTSTRAP_DRAWS",
    "CI_LEVEL",
    "EXACT_SIGNFLIP_MAX_N",
    "MONTE_CARLO_DRAWS",
    "NO_DIFFERENCE_PHRASE",
    "PairedResult",
    "cliffs_delta",
    "cliffs_delta_magnitude",
    "compare_paired",
    "describe_comparison",
    "derive_verdict",
    "holm_bonferroni",
    "paired_bootstrap_ci",
    "paired_sign_flip_test",
]


# --------------------------------------------------------------------------- #
# Frozen protocol constants -- changing any of these changes the pre-registered
# analysis and must be recorded in results/manifest.json.
# --------------------------------------------------------------------------- #

BOOTSTRAP_DRAWS: int = 20_000
MONTE_CARLO_DRAWS: int = 50_000
EXACT_SIGNFLIP_MAX_N: int = 20
ANALYSIS_SEED: int = 20260729
CI_LEVEL: float = 0.95

#: The exact wording required whenever an interval contains zero.  Principle P1:
#: an undetectable difference is never described as a gain.
NO_DIFFERENCE_PHRASE: str = "no detectable difference"

#: Verdict vocabulary.  ``to_row`` writes one of these three strings.
VERDICT_NO_DIFFERENCE = "no_detectable_difference"
VERDICT_FAVOURS_A = "favours_a"
VERDICT_FAVOURS_B = "favours_b"

#: Stored precision.  Six decimals is enough for an auditor to re-derive a claim
#: from the source CSV and far from the 17 significant digits the previous
#: ablation table reported for quantities smaller than their own noise.
ROUND_DECIMALS: int = 6


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairedResult:
    """One comparison between two arms, matched by seed.

    Field order follows spec 9.5.  ``cliffs_delta``, ``cliffs_delta_magnitude``
    and ``verdict`` are appended with defaults; appending defaulted fields keeps
    every positional construction in the frozen signature valid.
    """

    comparison_id: str
    family: str
    label_a: str
    label_b: str
    scenario: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    ci_level: float
    p_value: float
    p_method: str
    holm_adjusted_p: float | None = None
    significant: bool | None = None
    cliffs_delta: float = 0.0
    cliffs_delta_magnitude: str = "negligible"
    verdict: str = VERDICT_NO_DIFFERENCE

    # ------------------------------------------------------------------ views #

    def interval_contains_zero(self) -> bool:
        return bool(self.ci_low <= 0.0 <= self.ci_high)

    def to_row(self) -> dict[str, object]:
        """Flatten to one CSV row.  Every float is rounded to ``ROUND_DECIMALS``."""

        def r(x: float | None) -> float | None:
            if x is None:
                return None
            x = float(x)
            return x if math.isnan(x) or math.isinf(x) else round(x, ROUND_DECIMALS)

        return {
            "comparison_id": self.comparison_id,
            "family": self.family,
            "scenario": self.scenario,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_pairs": int(self.n_pairs),
            "mean_a": r(self.mean_a),
            "mean_b": r(self.mean_b),
            "mean_difference": r(self.mean_difference),
            "ci_low": r(self.ci_low),
            "ci_high": r(self.ci_high),
            "ci_level": r(self.ci_level),
            "p_value": r(self.p_value),
            "p_method": self.p_method,
            "holm_adjusted_p": r(self.holm_adjusted_p),
            "significant": self.significant,
            "cliffs_delta": r(self.cliffs_delta),
            "cliffs_delta_magnitude": self.cliffs_delta_magnitude,
            "verdict": self.verdict,
            "interval_contains_zero": self.interval_contains_zero(),
            "description": describe_comparison(self),
        }


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def _as_diff(diff: Sequence[float] | np.ndarray) -> np.ndarray:
    d = np.asarray(diff, dtype=np.float64).reshape(-1)
    if d.size == 0:
        raise ValueError("paired difference vector is empty")
    if not np.all(np.isfinite(d)):
        raise ValueError("paired difference vector contains non-finite values")
    return d


# --------------------------------------------------------------------------- #
# Confidence interval
# --------------------------------------------------------------------------- #


def paired_bootstrap_ci(
    diff: Sequence[float] | np.ndarray,
    level: float = CI_LEVEL,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = ANALYSIS_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean paired difference.

    Resamples the ``n`` differences with replacement ``draws`` times and returns
    the ``level`` percentile interval of the resampled means.  The generator is
    seeded explicitly, so two calls with the same arguments return bitwise
    identical endpoints.

    With ``n == 1`` every resample is the single observation, so the interval
    degenerates to that value.  That is the honest answer: one pair carries no
    information about spread.
    """
    d = _as_diff(diff)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    if draws < 1:
        raise ValueError(f"draws must be >= 1; got {draws}")

    n = d.size
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(draws), n), dtype=np.int64)
    means = d[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Hypothesis test
# --------------------------------------------------------------------------- #


def _all_signflip_means(d: np.ndarray) -> np.ndarray:
    """Means of ``s * d`` over all ``2**n`` sign vectors ``s in {-1, +1}**n``.

    Built from subset sums: flipping the subset ``S`` maps the total ``T`` to
    ``T - 2 * sum_{i in S} d_i``.  ``O(2**n)`` memory, not ``O(2**n * n)``.
    """
    n = d.size
    subset_sums = np.zeros(1, dtype=np.float64)
    for value in d:
        subset_sums = np.concatenate((subset_sums, subset_sums + value))
    total = float(d.sum())
    return (total - 2.0 * subset_sums) / n


def paired_sign_flip_test(
    diff: Sequence[float] | np.ndarray,
    seed: int = ANALYSIS_SEED,
) -> tuple[float, str]:
    """Two-sided paired sign-flip permutation test.

    Returns ``(p_value, p_method)`` where ``p_method`` is ``"exact_sign_flip"``
    when all ``2**n`` sign vectors were enumerated and
    ``"monte_carlo_sign_flip"`` when they were sampled.

    The p-value is ``#{ |mean(s * d)| >= |mean(d)| } / 2**n``.  The observed
    assignment is itself one of the enumerated sign vectors, so the statistic is
    never zero and the floor is ``2 / 2**n`` (the all-``+`` and all-``-``
    vectors, which tie with the observation whenever every difference shares a
    sign).

    An all-zero difference vector gives ``p == 1.0``: every flip reproduces the
    observation exactly.
    """
    d = _as_diff(diff)
    n = d.size
    observed = abs(float(d.mean()))
    # Absolute tolerance guards the >= comparison against the floating-point
    # asymmetry of the subset-sum recursion; without it a flip that is
    # algebraically equal to the observation can miss by one ulp.
    tol = 1e-12 * max(1.0, float(np.abs(d).sum()) / n)

    if n <= EXACT_SIGNFLIP_MAX_N:
        means = _all_signflip_means(d)
        count = int(np.count_nonzero(np.abs(means) >= observed - tol))
        return float(count) / float(2**n), "exact_sign_flip"

    rng = np.random.default_rng(int(seed))
    draws = MONTE_CARLO_DRAWS
    count = 0
    chunk = max(1, min(draws, 4_000_000 // max(1, n)))
    remaining = draws
    while remaining > 0:
        this = min(chunk, remaining)
        signs = rng.integers(0, 2, size=(this, n), dtype=np.int8) * 2 - 1
        means = (signs * d).mean(axis=1)
        count += int(np.count_nonzero(np.abs(means) >= observed - tol))
        remaining -= this
    # +1 in numerator and denominator: the observed assignment is a member of
    # the permutation set, so a Monte-Carlo p-value must never be able to reach
    # zero (Phipson & Smyth 2010).
    return float(count + 1) / float(draws + 1), "monte_carlo_sign_flip"


# --------------------------------------------------------------------------- #
# Effect size
# --------------------------------------------------------------------------- #


def cliffs_delta(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Cliff's delta: ``P(a > b) - P(a < b)`` over all cross pairs.

    A rank-based, distribution-free dominance measure in ``[-1, +1]``.  It is
    reported alongside the mean difference because a mean difference in reward
    units is not comparable across scenarios with different reward scales,
    whereas delta is.  ``+1`` means every ``a`` beats every ``b``; ``0`` means
    complete overlap.
    """
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("cliffs_delta requires two non-empty samples")
    comparison = np.sign(x[:, None] - y[None, :])
    return float(comparison.sum()) / float(x.size * y.size)


def cliffs_delta_magnitude(delta: float) -> str:
    """Romano et al. (2006) thresholds for |delta|: .147 / .33 / .474."""
    d = abs(float(delta))
    if d < 0.147:
        return "negligible"
    if d < 0.330:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Verdict derivation
# --------------------------------------------------------------------------- #


def derive_verdict(
    mean_difference: float,
    ci_low: float,
    ci_high: float,
    significant: bool | None = None,
) -> str:
    """The single place a directional verdict is allowed to come from.

    Two conditions must **both** hold before this function will name a winner:

    1. the bootstrap interval excludes zero, and
    2. the comparison survives multiplicity correction within its family.

    Before :func:`holm_bonferroni` has run, ``significant`` is ``None`` and only
    condition 1 is knowable, so the interval alone governs.  Once correction has
    run, ``significant is False`` demotes the verdict to
    ``no_detectable_difference`` **even when the raw interval excludes zero**.

    That demotion is the point of this function.  An audit of the previous
    output found six published rows carrying ``favours_a``/``favours_b`` beside
    Holm-adjusted p-values of 1.000, 0.245 and 0.250 — a directional claim on a
    comparison the pre-registered protocol had already declined to reject.  The
    protocol governs: if correction does not reject, the repository does not get
    to name a direction.  The raw interval and the raw p are still published in
    ``comparisons.csv`` for anyone re-analysing.
    """
    if ci_low <= 0.0 <= ci_high:
        return VERDICT_NO_DIFFERENCE
    if significant is False:
        return VERDICT_NO_DIFFERENCE
    return VERDICT_FAVOURS_A if float(mean_difference) > 0.0 else VERDICT_FAVOURS_B


# --------------------------------------------------------------------------- #
# The public comparison entry point
# --------------------------------------------------------------------------- #


def compare_paired(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    comparison_id: str,
    family: str,
    label_a: str,
    label_b: str,
    scenario: str,
) -> PairedResult:
    """Compare two seed-matched arms.  ``a`` and ``b`` must be aligned by seed.

    The bootstrap generator is derived from ``(ANALYSIS_SEED, "boot",
    comparison_id)``, so every comparison gets its own reproducible stream and
    no comparison's interval depends on how many other comparisons ran first.
    """
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(
            f"comparison {comparison_id!r}: arms are not seed-matched "
            f"({x.shape} vs {y.shape}); paired statistics require equal length"
        )
    d = _as_diff(x - y)

    boot_seed = derive_seed(ANALYSIS_SEED, "boot", comparison_id)
    ci_low, ci_high = paired_bootstrap_ci(d, level=CI_LEVEL, seed=boot_seed)
    p_value, p_method = paired_sign_flip_test(d, seed=boot_seed)
    delta = cliffs_delta(x, y)

    mean_difference = float(d.mean())
    # Uncorrected: no family is known yet, so only the interval can speak.
    # holm_bonferroni re-derives this with `significant` supplied.
    verdict = derive_verdict(mean_difference, ci_low, ci_high, significant=None)

    return PairedResult(
        comparison_id=comparison_id,
        family=family,
        label_a=label_a,
        label_b=label_b,
        scenario=scenario,
        n_pairs=int(d.size),
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        mean_difference=mean_difference,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=CI_LEVEL,
        p_value=p_value,
        p_method=p_method,
        cliffs_delta=delta,
        cliffs_delta_magnitude=cliffs_delta_magnitude(delta),
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# Multiplicity
# --------------------------------------------------------------------------- #


def holm_bonferroni(
    results: Sequence[PairedResult],
    alpha: float = 0.05,
) -> list[PairedResult]:
    """Holm step-down correction, applied **within** a declared family.

    Sorts ascending by raw p, multiplies the ``i``-th by ``m - i``, then enforces
    monotonicity by a running maximum so an adjusted p can never fall below the
    adjusted p of a smaller raw p.  Values are clipped at 1.0.  The returned
    list preserves the **input order**, not the sorted order.

    ``significant`` is ``holm_adjusted_p <= alpha``.  Holm controls the
    family-wise error rate without assuming independence, which matters here
    because comparisons inside a family share arms and are correlated.

    Correction is also where the **verdict** is finalised: a comparison whose
    interval excludes zero but whose adjusted p does not reject is re-derived to
    ``no_detectable_difference`` by :func:`derive_verdict`.  A directional
    verdict beside a non-rejecting adjusted p is an overclaim, and this is the
    only place with enough information to prevent it.
    """
    items = list(results)
    if not items:
        return []
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    families = {r.family for r in items}
    if len(families) > 1:
        raise ValueError(
            "holm_bonferroni corrects within one declared family; got "
            f"{sorted(families)}. Group by family before calling."
        )

    m = len(items)
    order = sorted(range(m), key=lambda i: (items[i].p_value, items[i].comparison_id))
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * items[idx].p_value)
        running = max(running, value)
        adjusted[idx] = running

    out: list[PairedResult] = []
    for i, r in enumerate(items):
        significant = bool(adjusted[i] <= alpha)
        out.append(
            replace(
                r,
                holm_adjusted_p=adjusted[i],
                significant=significant,
                verdict=derive_verdict(r.mean_difference, r.ci_low, r.ci_high, significant),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Wording
# --------------------------------------------------------------------------- #


def describe_comparison(result: PairedResult, decimals: int = 3) -> str:
    """Render a comparison as prose, in the only wording principle P1 permits.

    The sentence follows the **verdict**, not the raw interval, so a comparison
    demoted by :func:`derive_verdict` for failing multiplicity correction is
    described as ``"no detectable difference"`` too.  Whenever the verdict is
    not directional the sentence gives no direction: it is never called a gain,
    an improvement or a win, regardless of the sign of the point estimate.
    """
    fmt = f"{{:.{decimals}f}}"
    ci = f"95% CI [{fmt.format(result.ci_low)}, {fmt.format(result.ci_high)}]"
    tail = f"{ci}, p={fmt.format(result.p_value)} ({result.p_method}), n={result.n_pairs}"

    if result.verdict == VERDICT_NO_DIFFERENCE:
        return (
            f"{result.label_a} vs {result.label_b} on {result.scenario}: "
            f"{NO_DIFFERENCE_PHRASE} "
            f"(mean difference {fmt.format(result.mean_difference)}, {tail})."
        )

    direction = "higher than" if result.mean_difference > 0 else "lower than"
    return (
        f"{result.label_a} scored {fmt.format(abs(result.mean_difference))} "
        f"{direction} {result.label_b} on {result.scenario} "
        f"({tail}, Cliff's delta {fmt.format(result.cliffs_delta)} "
        f"[{result.cliffs_delta_magnitude}])."
    )
