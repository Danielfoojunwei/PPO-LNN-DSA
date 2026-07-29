# What this changes

<!-- One paragraph. What is different after this PR that was not true before? -->

## Type

- [ ] Bug fix
- [ ] New experiment / scenario / model
- [ ] Analysis or reporting change
- [ ] Infrastructure, CI or docs

## The five principles

Every PR is checked against these. Tick honestly; an unticked box with an
explanation is fine, a wrongly ticked box is not.

- [ ] **P1 — Honesty over flattery.** No claim here is stronger than the artifacts
      support. If a result got worse, the PR says so in the description.
- [ ] **P2 — Every number computed from `results/`.** I did not type a metric into
      `README.md`, `RESULTS.md` or `docs/*.md`. If numbers changed, I ran
      `make analyze && make report` and committed the regenerated output.
- [ ] **P3 — Name things what they are.** No block is named after an architecture
      it does not implement.
- [ ] **P4 — Determinism.** Same seed and config still give byte-identical
      metrics, independent of `--models` / `--scenarios` ordering.
- [ ] **P5 — Everything gets a unit test.** New public functions are covered by a
      test in the file that owns them.

## Checks

- [ ] `pytest -q` passes locally
- [ ] `make smoke` completes
- [ ] If I touched `dsa/analysis/` or `scripts/`: I added **no** model-name list
      used as a display filter. Every table still contains every policy that ran,
      baselines included.
- [ ] If I touched `configs/preregistration.yaml`: I explain below why, and I
      state which comparisons this reclassifies as exploratory.

## Anything that got worse

<!-- Required. Write "nothing" only if that is true. A PR that improves one
     metric and silently degrades another is the failure mode this section
     exists to catch. -->
