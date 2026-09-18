"""The two dependency lists must stay one list.

transcribe_server.py declares its dependencies inline (PEP 723), because that is
what `uv run transcribe_server.py`, the launcher and the Windows build resolve.
pyproject.toml repeats them so `uv sync` can build a locked dev environment. A
bump made in one place and not the other would test against one set of
packages and ship another, so this compares them exactly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 has tomli via the inline dependencies
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


def inline_metadata(script: Path) -> dict:
    """The script's `# /// script` block, parsed the way uv reads it."""
    text = script.read_text(encoding="utf-8")
    match = re.search(r"(?m)^# /// script\n((?:#.*\n)+?)^# ///$", text)
    assert match, f"{script.name} has no inline metadata block"
    body = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in match.group(1).splitlines(keepends=True)
    )
    return tomllib.loads(body)


def project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def test_pyproject_dependencies_match_the_inline_block():
    inline = inline_metadata(ROOT / "transcribe_server.py")
    assert project()["dependencies"] == inline["dependencies"], (
        "pyproject.toml and transcribe_server.py's inline block list different "
        "dependencies; change both"
    )


def test_pyproject_python_floor_matches_the_inline_block():
    inline = inline_metadata(ROOT / "transcribe_server.py")
    assert project()["requires-python"] == inline["requires-python"]
