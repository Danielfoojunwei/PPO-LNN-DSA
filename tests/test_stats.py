"""Statistics tested against analytically known answers, not against itself.

Every assertion here has a closed-form expected value that was worked out by
hand or is forced by the definition of the test.  A statistics module validated
only against its own output is not validated.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsa.analysis.stats import (
    ANALYSIS_SEED,
    BOOTSTRAP_DRAWS,
    EXACT_SIGNFLIP_MAX_N,
    NO_DIFFERENCE_PHRASE,
    PairedResult,
    cliffs_delta,
    cliffs_delta_magnitude,
    compare_paired,
    describe_comparison,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_sign_flip_test,
)


# --------------------------------------------------------------------------- #
# Exact sign-flip permutation test
# --------------------------------------------------------------------------- #


def test_signflip_hand_computed_four_element_example():
    """diff = [1, 2, 3, 4], worked out by hand.

    Total is 10, so the 16 flipped sums 4*mean are
    10, 8, 6, 4, 4, 2, 0, -2, 2, 0, -2, -4, -4, -6, -8, -10.
    Exactly two of them have absolute value >= 10 (the all-plus and all-minus
    vectors), so p = 2/16 = 0.125.
    """
    p, method = paired_sign_flip_test(np.array([1.0, 2.0, 3.0, 4.0]))
    assert method == "exact_sign_flip"
    assert p == pytest.approx(0.125, abs=1e-12)


def test_signflip_identical_samples_gives_p_one():
    """Two identical samples differ by exactly zero; every flip ties."""
    p, method = paired_sign_flip_test(np.zeros(8))
    assert method == "exact_sign_flip"
    assert p == 1.0


def test_signflip_symmetric_differences_give_p_one():
    """diff = [1, -1, 1, -1] has mean 0, so no flip can be more extreme."""
    p, _ = paired_sign_flip_test(np.array([1.0, -1.0, 1.0, -1.0]))
    assert p == 1.0


@pytest.mark.parametrize("n", [4, 8, 12])
def test_signflip_all_positive_hits_the_floor(n):
    """A maximally separated sample attains the minimum p, exactly 2 / 2**n.

    Only the all-plus and all-minus sign vectors reach the observed |mean|, so
    the floor is attained and never undershot.
    """
    p, method = paired_sign_flip_test(np.arange(1.0, n + 1.0))
    assert method == "exact_sign_flip"
    assert p == pytest.approx(2.0 / 2**n, rel=1e-12)


def test_signflip_p_value_is_a_probability():
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = rng.normal(size=10)
        p, _ = paired_sign_flip_test(d)
        assert 0.0 < p <= 1.0


def test_signflip_is_deterministic_and_rng_free_in_the_exact_regime():
    d = np.array([0.5, -1.25, 3.0, 2.0, -0.75])
    first = paired_sign_flip_test(d, seed=1)
    second = paired_sign_flip_test(d, seed=999999)
    assert first == second  # the exact branch must not consult the seed at all


def test_signflip_switches_to_monte_carlo_past_the_exact_limit():
    d = np.arange(1.0, EXACT_SIGNFLIP_MAX_N + 2.0)
    p, method = paired_sign_flip_test(d)
    assert method == "monte_carlo_sign_flip"
    assert p > 0.0  # +1 correction: a Monte-Carlo p-value can never be zero


def test_signflip_rejects_empty_and_non_finite_input():
    with pytest.raises(ValueError):
        paired_sign_flip_test(np.array([]))
    with pytest.raises(ValueError):
        paired_sign_flip_test(np.array([1.0, np.nan]))


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval
# --------------------------------------------------------------------------- #


def test_bootstrap_is_deterministic_across_calls():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert paired_bootstrap_ci(d) == paired_bootstrap_ci(d)


def test_bootstrap_seed_changes_the_interval_but_not_much():
    d = np.random.default_rng(3).normal(size=30)
    a = paired_bootstrap_ci(d, seed=1)
    b = paired_bootstrap_ci(d, seed=2)
    assert a != b                                  # genuinely resampled
    assert abs(a[0] - b[0]) < 0.2 * float(d.std())  # but the same interval, near enough


def test_bootstrap_brackets_the_sample_mean():
    d = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    lo, hi = paired_bootstrap_ci(d)
    assert lo < d.mean() < hi


def test_bootstrap_coverage_on_a_known_normal():
    """Nominal 95% coverage of a known mean, measured over 200 replications.

    The true mean is 0. Percentile bootstrap at n=40 is slightly liberal, so the
    bar is a realistic 88%, not 95% -- this test detects a broken interval, not
    a mildly anticonservative one.
    """
    rng = np.random.default_rng(20260729)
    covered = 0
    trials = 200
    for i in range(trials):
        sample = rng.normal(loc=0.0, scale=1.0, size=40)
        lo, hi = paired_bootstrap_ci(sample, draws=2000, seed=i)
        covered += int(lo <= 0.0 <= hi)
    assert covered / trials >= 0.88


def test_bootstrap_interval_narrows_with_more_data():
    rng = np.random.default_rng(11)
    small = paired_bootstrap_ci(rng.normal(size=10), seed=5)
    large = paired_bootstrap_ci(rng.normal(size=400), seed=5)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_rejects_bad_arguments():
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0, 2.0]), level=1.5)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0, 2.0]), draws=0)


# --------------------------------------------------------------------------- #
# Cliff's delta
# --------------------------------------------------------------------------- #


def test_cliffs_delta_extremes_and_centre():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == pytest.approx(-1.0)
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_cliffs_delta_hand_computed():
    """a=[1,3], b=[2,4]: pairs (1,2)-, (1,4)-, (3,2)+, (3,4)- => (1-3)/4 = -0.5."""
    assert cliffs_delta([1, 3], [2, 4]) == pytest.approx(-0.5)


def test_cliffs_delta_magnitude_thresholds():
    assert cliffs_delta_magnitude(0.10) == "negligible"
    assert cliffs_delta_magnitude(0.20) == "small"
    assert cliffs_delta_magnitude(0.40) == "medium"
    assert cliffs_delta_magnitude(-0.90) == "large"


# --------------------------------------------------------------------------- #
# Holm-Bonferroni
# --------------------------------------------------------------------------- #


def _result(cid: str, p: float, family: str = "F") -> PairedResult:
    return PairedResult(
        comparison_id=cid,
        family=family,
        label_a="a",
        label_b="b",
        scenario="s",
        n_pairs=12,
        mean_a=0.0,
        mean_b=0.0,
        mean_difference=1.0,
        ci_low=0.5,
        ci_high=1.5,
        ci_level=0.95,
        p_value=p,
        p_method="exact_sign_flip",
    )


def test_holm_worked_four_p_value_example():
    """p = [0.01, 0.02, 0.03, 0.04], m = 4.

    Step-down multipliers are 4, 3, 2, 1 giving 0.04, 0.06, 0.06, 0.04, then the
    running maximum enforces monotonicity: 0.04, 0.06, 0.06, 0.06.
    """
    results = [_result(f"C{i}", p) for i, p in enumerate([0.01, 0.02, 0.03, 0.04])]
    adjusted = [r.holm_adjusted_p for r in holm_bonferroni(results)]
    assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_never_reports_below_the_raw_p():
    rng = np.random.default_rng(7)
    results = [_result(f"C{i}", float(p)) for i, p in enumerate(rng.uniform(size=25))]
    for r in holm_bonferroni(results):
        assert r.holm_adjusted_p >= r.p_value - 1e-12


def test_holm_is_monotone_in_the_raw_p():
    results = [_result(f"C{i}", p) for i, p in enumerate([0.001, 0.2, 0.02, 0.9, 0.05])]
    corrected = holm_bonferroni(results)
    ordered = sorted(corrected, key=lambda r: r.p_value)
    values = [r.holm_adjusted_p for r in ordered]
    assert values == sorted(values)


def test_holm_clips_at_one_and_sets_significance():
    results = [_result(f"C{i}", 0.4) for i in range(10)]
    corrected = holm_bonferroni(results, alpha=0.05)
    assert all(r.holm_adjusted_p == 1.0 for r in corrected)
    assert all(r.significant is False for r in corrected)


def test_holm_preserves_input_order():
    results = [_result(f"C{i}", p) for i, p in enumerate([0.9, 0.01, 0.5])]
    corrected = holm_bonferroni(results)
    assert [r.comparison_id for r in corrected] == ["C0", "C1", "C2"]


def test_holm_refuses_to_mix_families():
    with pytest.raises(ValueError, match="one declared family"):
        holm_bonferroni([_result("C0", 0.01, "F1"), _result("C1", 0.02, "F2")])


def test_holm_on_an_empty_family_is_empty():
    assert holm_bonferroni([]) == []


# --------------------------------------------------------------------------- #
# compare_paired and the wording rule
# --------------------------------------------------------------------------- #


def test_compare_paired_requires_seed_matched_arms():
    with pytest.raises(ValueError, match="not seed-matched"):
        compare_paired([1, 2, 3], [1, 2], "C", "F", "a", "b", "s")


def test_compare_paired_is_deterministic():
    a, b = [1.0, 5.0, 3.0, 9.0], [0.0, 1.0, 2.0, 3.0]
    first = compare_paired(a, b, "C", "F", "a", "b", "s")
    second = compare_paired(a, b, "C", "F", "a", "b", "s")
    assert first == second


def test_compare_paired_verdict_matches_the_interval():
    identical = compare_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "C", "F", "a", "b", "s")
    assert identical.verdict == "no_detectable_difference"
    assert identical.interval_contains_zero()

    separated = compare_paired([10.0, 11.0, 12.0], [0.0, 1.0, 2.0], "C2", "F", "a", "b", "s")
    assert separated.verdict == "favours_a"
    assert not separated.interval_contains_zero()

    reversed_ = compare_paired([0.0, 1.0, 2.0], [10.0, 11.0, 12.0], "C3", "F", "a", "b", "s")
    assert reversed_.verdict == "favours_b"


def test_describe_says_no_detectable_difference_when_the_interval_covers_zero():
    """Principle P1, mechanically: an undetectable difference is never a gain."""
    result = compare_paired([1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0], "C", "F", "a", "b", "s")
    text = describe_comparison(result)
    assert NO_DIFFERENCE_PHRASE in text
    for forbidden in ("gain", "improvement", "wins", "better"):
        assert forbidden not in text.lower()


def test_describe_reports_direction_only_when_the_interval_excludes_zero():
    result = compare_paired([10.0] * 6, [0.0] * 6, "C", "F", "arm_a", "arm_b", "s")
    text = describe_comparison(result)
    assert "higher than" in text
    assert NO_DIFFERENCE_PHRASE not in text


def test_to_row_has_a_stable_schema_and_bounded_precision():
    result = compare_paired([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], "C", "F", "a", "b", "s")
    row = result.to_row()
    required = {
        "comparison_id", "family", "scenario", "label_a", "label_b", "n_pairs",
        "mean_a", "mean_b", "mean_difference", "ci_low", "ci_high", "ci_level",
        "p_value", "p_method", "holm_adjusted_p", "significant", "cliffs_delta",
        "cliffs_delta_magnitude", "verdict", "interval_contains_zero", "description",
    }
    assert required <= set(row)
    # No 17-significant-digit precision theatre.
    for key in ("mean_difference", "ci_low", "ci_high", "p_value"):
        assert len(str(row[key]).split(".")[-1]) <= 8


def test_module_constants_match_the_protocol():
    assert BOOTSTRAP_DRAWS == 20_000
    assert EXACT_SIGNFLIP_MAX_N == 20
    assert ANALYSIS_SEED == 20260729
