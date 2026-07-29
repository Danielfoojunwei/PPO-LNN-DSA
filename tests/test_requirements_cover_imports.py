"""``requirements.txt`` must match the tree's real imports, in both directions.

The previous pin list carried ``scipy``, ``gymnasium``, ``tensorboard`` and
``tqdm``, none of which was imported anywhere in the repository, and ``scipy``
was not even installed on the machine the results were produced on.  A
dependency list that nobody checks drifts immediately, so this test checks it:

* forward  -- every third-party module imported anywhere is pinned, and
* backward -- every pinned package is imported somewhere.

The backward direction is the one that catches the old failure.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"

#: Import name -> distribution name, where they differ.
IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "dateutil": "python-dateutil",
}

#: First-party packages, which are never pinned.
FIRST_PARTY = {"dsa", "scripts", "tests"}

#: Distributions that may be pinned without a direct ``import`` statement.
TOOLING = {"pytest"}


def _iter_python_files() -> list[pathlib.Path]:
    return [
        p
        for p in sorted(REPO_ROOT.rglob("*.py"))
        if ".git" not in p.parts and "__pycache__" not in p.parts
    ]


def _top_level_imports() -> dict[str, set[str]]:
    """Map top-level module name -> set of files importing it."""
    found: dict[str, set[str]] = {}
    for path in _iter_python_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:  # a file that will not parse is its own bug
            pytest.fail(f"{path.relative_to(REPO_ROOT)} does not parse: {exc}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import: first-party by definition
                    continue
                names = [node.module.split(".")[0]] if node.module else []
            else:
                continue
            for name in names:
                found.setdefault(name, set()).add(str(path.relative_to(REPO_ROOT)))
    return found


def _pinned() -> dict[str, str]:
    """Map lowercase distribution name -> the raw requirement line."""
    pins: dict[str, str] = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==")[0].split(">=")[0].split("[")[0].strip()
        pins[name.lower()] = line
    return pins


def _third_party_imports() -> dict[str, set[str]]:
    stdlib = set(sys.stdlib_module_names)
    return {
        name: files
        for name, files in _top_level_imports().items()
        if name not in stdlib and name not in FIRST_PARTY and not name.startswith("_")
    }


def test_requirements_file_exists_and_is_pinned():
    assert REQUIREMENTS.exists()
    pins = _pinned()
    assert pins, "requirements.txt declares nothing"
    unpinned = [line for line in pins.values() if "==" not in line]
    assert not unpinned, f"these requirements are not pinned to an exact version: {unpinned}"


def test_every_imported_third_party_package_is_pinned():
    """Forward direction: nothing is imported that is not declared."""
    pins = _pinned()
    missing: list[str] = []
    for name, files in sorted(_third_party_imports().items()):
        distribution = IMPORT_TO_DISTRIBUTION.get(name, name).lower()
        if distribution not in pins:
            missing.append(f"{name} (distribution {distribution!r}) imported by {sorted(files)[:3]}")
    assert not missing, "imported but not in requirements.txt:\n" + "\n".join(missing)


def test_every_pinned_package_is_actually_imported():
    """Backward direction: this is the check the old pin list failed.

    scipy, gymnasium, tensorboard and tqdm were all pinned with zero imports.
    """
    imported = set(_third_party_imports())
    distributions = {IMPORT_TO_DISTRIBUTION.get(n, n).lower() for n in imported}
    unused = [
        name for name in sorted(_pinned()) if name not in distributions and name not in TOOLING
    ]
    assert not unused, (
        "pinned but never imported anywhere in the tree: "
        + ", ".join(unused)
        + ". Remove them, or add the code that needs them."
    )


@pytest.mark.parametrize("removed", ["scipy", "gymnasium", "tensorboard", "tqdm", "seaborn"])
def test_the_previously_unused_pins_are_gone(removed):
    """Explicit regression: these five had zero imports and must stay unpinned."""
    assert removed not in _pinned(), f"{removed} is pinned again but nothing imports it"


def test_pinned_packages_are_importable_in_this_environment():
    """A pin that cannot be imported here would have failed the whole suite anyway."""
    distribution_to_import = {v: k for k, v in IMPORT_TO_DISTRIBUTION.items()}
    for name in sorted(_pinned()):
        module = distribution_to_import.get(name, name)
        __import__(module)


def test_no_python_file_imports_a_deleted_legacy_module():
    """Nothing may still reach into the demolished trees."""
    legacy = {
        "empirical", "preceptual_libiao", "environment", "baseline",
        "benchmark", "federated_legacy", "utils",
    }
    offenders = [
        f"{name}: {sorted(files)}"
        for name, files in _top_level_imports().items()
        if name in legacy
    ]
    assert not offenders, "import of a deleted legacy module:\n" + "\n".join(offenders)
