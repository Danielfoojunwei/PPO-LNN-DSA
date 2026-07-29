"""Principle P2, mechanically: no document may state an un-generated metric.

What this file used to do, and why that was not enough
-----------------------------------------------------
The previous version of this test scanned ``README.md`` and ``docs/*.md`` for
float literals and asserted there were none.  It also, in the same function,
``continue``\\ d on every line beginning with ``|`` and stripped every inline
code span before scanning::

    if stripped.startswith("|"):
        continue
    line = re.sub(r"`[^`]*`", "", raw)

Markdown table cells and inline code spans are exactly and exclusively where
this repository's README kept its ~35 mirrored metrics.  An auditor falsified
two of them -- the pre-registered primary comparison's mean difference
(-0.007 -> -9.007) and the greedy baseline's pooled return (8.291 -> 88.291) --
and the whole suite plus the CI report gate stayed green.  The detector was
structurally blind to the only places the numbers lived.

What it does now
----------------
Three properties, in increasing order of strength.

1. **Scope.**  Table cells and inline code spans are *scanned*.  Only fenced
   code blocks are exempt, because those hold commands and sample output rather
   than claims.

2. **Exemption is generation, not citation.**  A metric literal is allowed only
   where a generator wrote it: inside a ``<!-- BEGIN GENERATED: name -->``
   region, or inside a ``<!--v:key-->...<!--/v-->`` inline span.  Both are
   emitted by ``scripts/render_docs.py`` from ``results/``.  This is stricter
   than requiring a nearby ``claim_id``: citing a claim proves a number *has* a
   source, whereas generating it proves the number *equals* its source.

3. **Byte identity.**  Every generated region and span in the committed
   documents must equal what ``scripts/render_docs.py`` produces from the
   committed ``results/``.  That is the P2 analogue of ``make check`` for
   ``RESULTS.md``, and it is what makes a hand-edited number fail the build
   rather than merely fail to be flagged.

The negative controls at the bottom reproduce the auditor's two exact
falsifications and assert the scan and the byte-identity gate both go red.  A
gate nobody has watched fail is not a gate.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import shutil
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Documents that may contain hand-written prose.  All of them are scanned.
PROSE_FILES = [REPO_ROOT / "README.md"] + sorted((REPO_ROOT / "docs").glob("*.md"))

#: Generated end to end by scripts/make_report.py; every number in it is read
#: from a CSV, which is the entire point of the file.
GENERATED_FILES = {"RESULTS.md"}

#: A float with two or more decimal places, optionally in scientific notation --
#: the shape a metric takes.
METRIC_LITERAL = re.compile(r"(?<![\w.])-?\d+\.\d{2,}(?:[eE][-+]?\d+)?")

#: Numerals that are not measurements.  Matched against a window around the hit.
ALLOWED = re.compile(
    r"""(
      Hasani\s+et\s+al\.|          # citations
      \b(19|20)\d{2}\b|            # years
      \bv?\d+\.\d+\.\d+\b|         # semantic versions
      \bAAAI\b|\barXiv\b|
      \bPython\b|requires-python   # interpreter version constraints
    )""",
    re.VERBOSE,
)

BEGIN_RE = re.compile(r"^\s*<!--\s*BEGIN GENERATED:\s*([A-Za-z0-9_]+)\s*-->\s*$")
END_RE = re.compile(r"^\s*<!--\s*END GENERATED:\s*([A-Za-z0-9_]+)\s*-->\s*$")
SPAN_RE = re.compile(r"<!--v:([^>|]+?)(?:\|([^>]+?))?-->(.*?)<!--/v-->", re.DOTALL)


def _load_render_docs():
    path = REPO_ROOT / "scripts" / "render_docs.py"
    spec = importlib.util.spec_from_file_location("render_docs_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scannable_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, text)`` for every line that may carry a claim.

    Removed: fenced code blocks, generated regions, generated inline spans,
    link targets, and any other HTML comment.  **Not** removed: markdown table
    cells and inline code spans -- removing those is the bug this file exists
    to have fixed.
    """
    out: list[tuple[int, str]] = []
    in_fence = False
    in_generated: str | None = None
    in_comment = False

    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        begin = BEGIN_RE.match(raw)
        if begin:
            assert in_generated is None, f"line {lineno}: nested generated region"
            in_generated = begin.group(1)
            continue
        end = END_RE.match(raw)
        if end:
            assert in_generated == end.group(1), f"line {lineno}: mismatched END GENERATED"
            in_generated = None
            continue
        if in_generated is not None:
            continue

        line = SPAN_RE.sub("", raw)

        if in_comment:
            if "-->" not in line:
                continue
            line = line.split("-->", 1)[1]
            in_comment = False
        line = re.sub(r"<!--.*?-->", "", line)
        if "<!--" in line:
            line = line.split("<!--", 1)[0]
            in_comment = True

        line = re.sub(r"\]\([^)]*\)", "]()", line)  # link targets
        out.append((lineno, line))

    assert in_generated is None, "a generated region was never closed"
    return out


def offenders_in(text: str, name: str = "<text>") -> list[str]:
    found: list[str] = []
    for lineno, line in scannable_lines(text):
        for match in METRIC_LITERAL.finditer(line):
            window = line[max(0, match.start() - 30) : match.end() + 30]
            if ALLOWED.search(window):
                continue
            found.append(f"{name}:{lineno}: {match.group()}  <-  {line.strip()[:110]}")
    return found


# --------------------------------------------------------------------------- #
# 1. Scope: every metric in every document is generated
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", PROSE_FILES, ids=lambda p: p.name)
def test_document_contains_no_ungenerated_metric(path: pathlib.Path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    assert path.name not in GENERATED_FILES

    found = offenders_in(path.read_text(), path.name)
    assert not found, (
        "a document states what looks like a measurement outside a generated region.\n"
        "Every number a document states must be written by scripts/render_docs.py from a "
        "file under results/. Wrap it in a <!-- BEGIN GENERATED: name --> region or a "
        "<!--v:key--> span and add the key to build_values(); or, if it is genuinely not "
        "derivable from results/, register it in docs/historical_figures.yaml with its "
        "provenance.\n" + "\n".join(found)
    )


def test_the_scan_covers_table_cells_and_inline_code():
    """The exact regression: the scan must not skip where the metrics live.

    Both planted numbers are shaped like the two the auditor falsified, and both
    sit where the previous filter dropped the line entirely.
    """
    table_cell = "| mean paired difference | -9.007 | `CLAIM.PRIMARY.P1` |\n"
    inline_code = "The pooled return was `88.291` across scenarios.\n"
    both = table_cell + inline_code

    assert offenders_in(table_cell), "a metric in a markdown table cell was not scanned"
    assert offenders_in(inline_code), "a metric in an inline code span was not scanned"
    assert len(offenders_in(both)) == 2


def test_generated_regions_and_spans_are_the_only_exemption():
    inside_region = (
        "<!-- BEGIN GENERATED: p1_table -->\n"
        "| mean paired difference | -0.007 |\n"
        "<!-- END GENERATED: p1_table -->\n"
    )
    inside_span = "the difference was <!--v:claim.CLAIM.PRIMARY.P1.value-->-0.007<!--/v--> reward.\n"
    outside = "the difference was -0.007 reward.\n"

    assert not offenders_in(inside_region)
    assert not offenders_in(inside_span)
    assert offenders_in(outside), "an un-generated metric in plain prose must be caught"


def test_the_scan_actually_scans_something():
    """Guard against the scan silently covering zero files or zero lines."""
    assert PROSE_FILES, "no prose files found"
    existing = [p for p in PROSE_FILES if p.exists()]
    assert existing, "none of the prose files exist"
    total = sum(len(scannable_lines(p.read_text())) for p in existing)
    assert total > 200, f"only {total} lines scanned; the filter is too aggressive"

    # And it must still be reaching table rows and code spans in the real files.
    table_rows = sum(
        1
        for p in existing
        for _, line in scannable_lines(p.read_text())
        if line.strip().startswith("|")
    )
    assert table_rows > 0, "no hand-written table rows reached the scanner"


def test_the_detector_ignores_code_fences_versions_and_citations():
    text = (
        "```\nreturn = -48.447\n```\n"
        "We target Python `3.11`, torch `2.13.0`, and cite Hasani et al. 2021 "
        "(arXiv:2006.04439).\n"
    )
    assert not offenders_in(text)


# --------------------------------------------------------------------------- #
# 2. Byte identity: the committed documents match the committed results
# --------------------------------------------------------------------------- #


def _rendered(results_dir: pathlib.Path, repo_root: pathlib.Path, path: pathlib.Path) -> str:
    module = _load_render_docs()
    values = module.build_values(results_dir, repo_root)
    blocks = module.build_blocks(results_dir, repo_root)
    return module.render_text(path.read_text(), values, blocks, where=path.name)


@pytest.mark.parametrize("path", PROSE_FILES, ids=lambda p: p.name)
def test_generated_content_matches_the_committed_results(path: pathlib.Path):
    results = REPO_ROOT / "results"
    if not (results / "all_runs.csv").exists():
        pytest.skip("no committed results/all_runs.csv to render from")
    if not path.exists():
        pytest.skip(f"{path.name} not present")

    current = path.read_text()
    assert _rendered(results, REPO_ROOT, path) == current, (
        f"{path.name} is stale or was edited by hand inside a generated region.\n"
        "Run `make report` and commit the result."
    )


def test_a_corrupted_claim_is_caught_by_regeneration(tmp_path):
    """Negative control for the byte-identity gate, on the real README.

    Falsify the primary comparison's value in a *copy* of ``results/`` and the
    rendered README must stop matching the committed one.  This is the failure
    mode the CI job has to catch, exercised without touching the real tree.
    """
    results = REPO_ROOT / "results"
    if not (results / "claims.json").exists():
        pytest.skip("no committed results/claims.json")

    fake = tmp_path / "results"
    shutil.copytree(results, fake)
    doc = json.loads((fake / "claims.json").read_text())
    hits = [c for c in doc["claims"] if c["claim_id"] == "CLAIM.PRIMARY.P1"]
    assert hits, "CLAIM.PRIMARY.P1 is missing from claims.json"
    hits[0]["value"] = float(hits[0]["value"]) - 9.0
    (fake / "claims.json").write_text(json.dumps(doc, indent=2))

    readme = REPO_ROOT / "README.md"
    assert _rendered(fake, REPO_ROOT, readme) != readme.read_text(), (
        "corrupting CLAIM.PRIMARY.P1 did not change the rendered README -- "
        "the primary comparison is not actually generated from claims.json"
    )


def test_every_inline_span_key_resolves():
    """An unknown key must fail loudly rather than silently render nothing."""
    results = REPO_ROOT / "results"
    if not (results / "all_runs.csv").exists():
        pytest.skip("no committed results/all_runs.csv")
    module = _load_render_docs()
    values = module.build_values(results, REPO_ROOT)
    blocks = module.build_blocks(results, REPO_ROOT)

    with pytest.raises(module.RenderError):
        module.render_text("<!--v:no.such.key-->1.23<!--/v-->", values, blocks)
    with pytest.raises(module.RenderError):
        module.render_text(
            "<!-- BEGIN GENERATED: nope -->\n<!-- END GENERATED: nope -->", values, blocks
        )


def test_the_scan_and_the_generator_cover_the_same_files():
    """A document one covers and the other does not is a hole in the gate.

    If the generator skipped a file the scan covers, that file could contain no
    number at all; if the scan skipped a file the generator covers, a hand-typed
    number there would only be caught by the byte-identity check, and only if it
    happened to land inside a region.
    """
    module = _load_render_docs()
    assert sorted(module.DOC_PATHS(REPO_ROOT)) == sorted(PROSE_FILES)


def test_historical_registry_entries_all_carry_provenance():
    """The one non-``results/`` source is only usable with its provenance filled in."""
    module = _load_render_docs()
    figures = module.load_historical(REPO_ROOT)
    for key, entry in figures.items():
        for field in ("value", "measured", "source"):
            assert entry.get(field), f"historical figure {key!r} has no {field}"
        assert entry.get("derivable_from_results") is False, (
            f"historical figure {key!r} claims to be derivable from results/; "
            "if it is, generate it instead of registering it"
        )


# --------------------------------------------------------------------------- #
# 3. RESULTS.md keeps its own provenance banner
# --------------------------------------------------------------------------- #


def test_results_md_is_generated_not_handwritten():
    path = REPO_ROOT / "RESULTS.md"
    if not path.exists():
        pytest.skip("RESULTS.md not generated yet")
    head = path.read_text()[:1200]
    assert "make_report.py" in head, "RESULTS.md does not name the script that generates it"
    assert "generated" in head.lower()
