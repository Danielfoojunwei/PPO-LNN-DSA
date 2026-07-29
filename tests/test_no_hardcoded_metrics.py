"""Principle P2, mechanically: no prose file may state a metric.

``README.md`` and everything under ``docs/`` are hand-written, so they may not
contain a number that looks like a measurement.  Numbers belong in
``RESULTS.md``, which is *generated* by ``scripts/make_report.py`` from files
under ``results/`` and is therefore exempt by construction.

Scope of the scan: prose only.  Fenced code blocks, inline code spans, markdown
tables, link targets and HTML comments are skipped -- those are commands,
configuration and citations, not claims.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Hand-written prose that must contain no measurements.
PROSE_FILES = [REPO_ROOT / "README.md"] + sorted((REPO_ROOT / "docs").glob("*.md"))

#: Generated from results/ by scripts/make_report.py; every number in it is
#: read from a CSV, which is the entire point of the file.
GENERATED_FILES = {"RESULTS.md"}

#: A float with two or more decimal places -- the shape a metric takes.
METRIC_LITERAL = re.compile(r"(?<![\w.])-?\d+\.\d{2,}")

#: Non-metric numerals that legitimately appear in prose.
ALLOWED = re.compile(
    r"""(
      Hasani\s+et\s+al\.|          # citations
      \b(19|20)\d{2}\b|            # years
      \bv?\d+\.\d+\.\d+\b|         # semantic versions
      \bAAAI\b|\barXiv\b
    )""",
    re.VERBOSE,
)


def _strip_non_prose(text: str) -> list[tuple[int, str]]:
    """Return (line_number, prose) for lines that carry hand-written claims."""
    out: list[tuple[int, str]] = []
    in_fence = False
    in_html_comment = False
    for i, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if "<!--" in stripped:
            in_html_comment = "-->" not in stripped
            continue
        if in_html_comment:
            if "-->" in stripped:
                in_html_comment = False
            continue

        # Markdown tables in a hand-written file are structure, and the ones we
        # care about live in generated files anyway.
        if stripped.startswith("|"):
            continue

        line = re.sub(r"`[^`]*`", "", raw)        # inline code spans
        line = re.sub(r"\]\([^)]*\)", "]()", line)  # link targets
        out.append((i, line))
    return out


@pytest.mark.parametrize("path", PROSE_FILES, ids=lambda p: p.name)
def test_prose_file_contains_no_metric_literal(path: pathlib.Path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    assert path.name not in GENERATED_FILES

    offenders: list[str] = []
    for lineno, line in _strip_non_prose(path.read_text()):
        for match in METRIC_LITERAL.finditer(line):
            window = line[max(0, match.start() - 30) : match.end() + 30]
            if ALLOWED.search(window):
                continue
            offenders.append(f"{path.name}:{lineno}: {match.group()}  <-  {line.strip()[:100]}")

    assert not offenders, (
        "hand-written prose contains what looks like a measurement.\n"
        "Every number a document states must be computed from results/ by a script.\n"
        "Put it in RESULTS.md via `make report`, or wrap it in backticks if it is a "
        "config value rather than a claim.\n" + "\n".join(offenders)
    )


def test_the_scan_actually_scans_something():
    """Guard against the scan silently covering zero files or zero lines."""
    assert PROSE_FILES, "no prose files found"
    existing = [p for p in PROSE_FILES if p.exists()]
    assert existing, "none of the prose files exist"
    total = sum(len(_strip_non_prose(p.read_text())) for p in existing)
    assert total > 20, f"only {total} prose lines scanned; the filter is too aggressive"


def test_the_detector_catches_a_planted_metric(tmp_path):
    """Negative control: the regex must actually fire on a planted claim."""
    planted = "Our model reached a mean return of -48.447, beating the baseline.\n"
    lines = _strip_non_prose(planted)
    assert any(METRIC_LITERAL.search(line) for _, line in lines)


def test_the_detector_ignores_code_fences_and_versions():
    text = "```\nreturn = -48.447\n```\nWe target Python `3.11` and cite Hasani et al. 2021.\n"
    for _, line in _strip_non_prose(text):
        for match in METRIC_LITERAL.finditer(line):
            window = line[max(0, match.start() - 30) : match.end() + 30]
            assert ALLOWED.search(window), f"false positive on {match.group()!r}"


def test_results_md_is_generated_not_handwritten():
    """If RESULTS.md exists it must declare its own provenance."""
    path = REPO_ROOT / "RESULTS.md"
    if not path.exists():
        pytest.skip("RESULTS.md not generated yet")
    head = path.read_text()[:1200]
    assert "make_report.py" in head, "RESULTS.md does not name the script that generates it"
    assert "generated" in head.lower()
