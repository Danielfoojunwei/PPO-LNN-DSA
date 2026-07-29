"""End-to-end determinism: the D1/D2 regression gate.

Rerunning the previous repository's own documented command moved its flagship
model from 1st to 6th of 7 on ``stationary`` (-48.447 -> -55.445).  The cause
was a mutable class-level ``_instance_counter`` feeding ``config.seed +
instance_id``, so a model's environment stream depended on its position in the
``--models`` list.  Measured with a *fixed* policy at one nominal seed, six
consecutive instances scored between -68.5 and -28.3: a 40-point nuisance
spread against a 1.98-point headline effect.

These tests run the real runner as a subprocess and require byte equality on
every column except the wall-clock ones.  They are marked ``slow`` because each
invocation trains; the fast lane is ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_SUITE = REPO_ROOT / "scripts" / "run_suite.py"

#: Spec 3.5: every non-deterministic column carries this prefix and nothing else may.
WALL_PREFIX = "wall_"

#: The smallest budget that still exercises rollout, update and both evaluations.
TINY = [
    "--smoke",
    "--no-guard",
    "--scenarios", "stationary",
    "--seeds", "7",
    "--training-steps", "1024",
    "--eval-episodes", "4",
    "--workers", "1",
]


def _run(args: list[str], out_dir: pathlib.Path) -> pd.DataFrame:
    result = subprocess.run(
        [sys.executable, str(RUN_SUITE), *args, "--output-dir", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, f"run_suite failed:\n{result.stdout}\n{result.stderr}"
    return pd.read_csv(out_dir / "all_runs.csv")


def _comparable(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in frame.columns if c.startswith(WALL_PREFIX)]
    return (
        frame.drop(columns=drop)
        .sort_values(["scenario", "model", "seed"])
        .reset_index(drop=True)
    )


@pytest.mark.slow
def test_model_order_invariance(tmp_path):
    """The headline gate: reordering ``--models`` may not change any metric.

    Two invocations request the same two models in opposite order.  Every
    non-wall-clock column must agree bitwise.
    """
    a = _run([*TINY, "--models", "ppo_gru", "ppo_cfc"], tmp_path / "a")
    b = _run([*TINY, "--models", "ppo_cfc", "ppo_gru"], tmp_path / "b")

    a, b = _comparable(a), _comparable(b)
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a, b, check_exact=True)


@pytest.mark.slow
def test_metrics_do_not_depend_on_which_other_models_were_requested(tmp_path):
    """A model's score must be identical whether or not it ran alongside others.

    This is the property the old instance counter destroyed: the seed offset was
    a function of the length and ordering of the ``--models`` list, so running a
    model alone gave a different answer than running it in company.
    """
    together = _run([*TINY, "--models", "ppo_gru", "ppo_cfc"], tmp_path / "together")
    alone = _run([*TINY, "--models", "ppo_cfc"], tmp_path / "alone")

    together, alone = _comparable(together), _comparable(alone)
    shared = alone["model"].unique()
    left = together[together["model"].isin(shared)].reset_index(drop=True)
    right = alone[alone["model"].isin(shared)].reset_index(drop=True)
    assert len(right) > 0
    pd.testing.assert_frame_equal(left, right, check_exact=True)


@pytest.mark.slow
def test_repeated_invocation_is_bitwise_identical(tmp_path):
    """Same command twice, same numbers -- the property the audit confirmed and
    which this rebuild must preserve."""
    a = _run([*TINY, "--models", "ppo_cfc"], tmp_path / "one")
    b = _run([*TINY, "--models", "ppo_cfc"], tmp_path / "two")
    pd.testing.assert_frame_equal(_comparable(a), _comparable(b), check_exact=True)


@pytest.mark.slow
def test_only_wall_prefixed_columns_are_non_deterministic(tmp_path):
    """The naming rule that makes the gate checkable.

    If any non-``wall_`` column ever differs between two identical runs, either
    that column is non-deterministic (a bug) or it is a timing column that was
    named wrongly (also a bug).
    """
    a = _run([*TINY, "--models", "ppo_gru"], tmp_path / "one")
    b = _run([*TINY, "--models", "ppo_gru"], tmp_path / "two")

    differing = [c for c in a.columns if not a[c].equals(b[c])]
    mislabelled = [c for c in differing if not c.startswith(WALL_PREFIX)]
    assert not mislabelled, f"non-deterministic columns without a wall_ prefix: {mislabelled}"
    assert any(c.startswith(WALL_PREFIX) for c in a.columns), "no timing columns emitted at all"


@pytest.mark.slow
def test_manifest_records_the_reproducibility_anchor(tmp_path):
    _run([*TINY, "--models", "ppo_cfc"], tmp_path / "m")
    manifest = json.loads((tmp_path / "m" / "manifest.json").read_text())

    for key in ("git_commit", "versions", "matrix", "models_solved", "preregistration_sha256", "outputs"):
        assert key in manifest, f"manifest is missing {key}"
    assert manifest["versions"]["torch"]
    assert manifest["matrix"]["seeds"]
    assert manifest["outputs"], "manifest records no output hashes"
    # Solved widths live in the manifest, never in source.
    for summary in manifest["models_solved"].values():
        assert summary["hidden_dim"] > 0
        assert summary["parameter_count"] > 0


@pytest.mark.slow
def test_report_generation_is_deterministic(tmp_path):
    """``make report`` twice on the same results gives byte-identical markdown."""
    from scripts.analyze import analyze
    from scripts.make_report import build_report

    out = tmp_path / "r"
    _run([*TINY, "--models", "ppo_gru", "ppo_cfc"], out)
    analyze(out, out, REPO_ROOT / "configs" / "preregistration.yaml", figures=False)

    first = build_report(out)
    second = build_report(out)
    assert first == second
    assert "make_report.py" in first
    # No wall-clock value may leak into the committed report.
    assert "wall_seconds" not in first


@pytest.mark.slow
def test_analyze_check_detects_stale_committed_tables(tmp_path):
    """``analyze --check`` must fail when the committed tables no longer match."""
    from scripts.analyze import analyze

    out = tmp_path / "c"
    _run([*TINY, "--models", "ppo_gru", "ppo_cfc"], out)
    analyze(out, out, REPO_ROOT / "configs" / "preregistration.yaml", figures=False)

    fresh = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze.py"),
         "--results-dir", str(out), "--output-dir", str(out), "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert fresh.returncode == 0, f"a freshly generated tree reported stale:\n{fresh.stdout}{fresh.stderr}"

    ranking = out / "tables" / "overall_ranking.csv"
    ranking.write_text(ranking.read_text().replace("ppo_gru", "ppo_TAMPERED", 1))

    tampered = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze.py"),
         "--results-dir", str(out), "--output-dir", str(out), "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert tampered.returncode == 1, "tampered evidence passed the --check gate"
    assert "STALE" in tampered.stderr.upper()


def test_seed_derivation_is_process_independent():
    """Cross-process seed stability: blake2b, never Python's salted ``hash()``.

    ``PYTHONHASHSEED`` is randomised per process by default, so a seed derived
    from ``hash("stationary")`` would differ between the parent and a worker.
    """
    code = (
        "import json;from dsa.seeding import derive_seed;"
        "print(json.dumps([derive_seed(7,'stationary','env','eval',i) for i in range(4)]))"
    )
    outputs = set()
    for hashseed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT, capture_output=True, text=True,
            env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin:/usr/local/bin"},
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"seed derivation varies with PYTHONHASHSEED: {outputs}"


def test_no_seed_derivation_uses_builtin_hash():
    """``hash()`` on a str is salted per process and must never seed anything."""
    import ast

    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "dsa").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "hash"
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"builtin hash() used in dsa/: {offenders}"
