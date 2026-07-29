"""Analysis layer: statistics, tables, figures.

Three hard rules, each of which exists because the previous version of this
repository broke it.

1. **No policy filter, ever.**  Every table built here contains every policy
   present in ``all_runs.csv``, heuristic baselines included.  The previous
   analysis script defined a ``CORE_MODELS`` list that omitted ``random_policy``
   and filtered it out of exactly the four tables the README cited; restored,
   that baseline ranked 4th of 7.  ``tests/test_no_filter.py`` greps this
   package for any such list and fails the build.

2. **Every reported quantity carries an uncertainty.**  A mean without a
   standard deviation, standard error or confidence interval is not reported.

3. **A difference whose interval contains zero is "no detectable difference".**
   It is never called a gain.  ``describe_comparison`` enforces the wording.
"""

from dsa.analysis.stats import (
    ANALYSIS_SEED,
    BOOTSTRAP_DRAWS,
    EXACT_SIGNFLIP_MAX_N,
    MONTE_CARLO_DRAWS,
    PairedResult,
    cliffs_delta,
    compare_paired,
    describe_comparison,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_sign_flip_test,
)

__all__ = [
    "ANALYSIS_SEED",
    "BOOTSTRAP_DRAWS",
    "EXACT_SIGNFLIP_MAX_N",
    "MONTE_CARLO_DRAWS",
    "PairedResult",
    "cliffs_delta",
    "compare_paired",
    "describe_comparison",
    "holm_bonferroni",
    "paired_bootstrap_ci",
    "paired_sign_flip_test",
]
