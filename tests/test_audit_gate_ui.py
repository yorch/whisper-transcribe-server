"""Tests for the audit page's gate logic.

The gate is client-side JavaScript inside the `AUDIT_PAGE` string and there is
no headless browser in this environment, so these run the real `relock()` under
node with a small DOM stub. What they pin is the defect this page had: it
treated "the audit API is off" and "your token is wrong" as the same state, so
a server with no audit token showed a prompt that no token could ever satisfy.

Skipped, not failed, when node is not installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

import transcribe_server as s

NODE = shutil.which("node")
SCRIPT = re.compile(r"<script>\n(.*?)\n</script>", re.S)
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

# The three elements relock() touches, started in the state the HTML ships
# them in: the gate is visible, and the data and the switched-off notice are
# not. MODE comes from the server-rendered body attribute.
HARNESS = """
function makeNode(locked){
  const classes = new Set(locked ? ["locked"] : []);
  return {classes, classList: {
    add: (name) => classes.add(name),
    toggle: (name, on) => { on ? classes.add(name) : classes.delete(name); },
  }};
}
const nodes = {gate: makeNode(false), main: makeNode(true), off: makeNode(true)};
const el = (id) => nodes[id];
let unlocked = false, offset = 0, timer = null;
const MODE = "__MODE__";
"""


def page_script() -> str:
    match = SCRIPT.search(s.AUDIT_PAGE)
    assert match, "the audit page no longer has a <script> block to check"
    return match.group(1)


def node_argv(*args: str) -> list[str]:
    """node plus its arguments. Every caller is skipped when node is missing."""
    assert NODE is not None, "node is required here"
    return [NODE, *args]


def relock_source() -> str:
    match = re.search(r"function relock\(\)\{.*?\n\}", page_script(), re.S)
    assert match, "relock() is gone from the audit page script"
    return match.group(0)


def relock(mode: str) -> dict:
    """Run the real relock() for one mode, and report what ended up visible."""
    program = (
        HARNESS.replace("__MODE__", mode)
        + relock_source()
        + "\nrelock();"
        + "\nprocess.stdout.write(JSON.stringify({unlocked, gate: [...nodes.gate.classes],"
        + " main: [...nodes.main.classes], off: [...nodes.off.classes]}));"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        node_argv("-e", program), capture_output=True, text=True
    )
    assert result.returncode == 0, f"node failed:\n{result.stderr}"
    return json.loads(result.stdout)


def visible(state: dict, name: str) -> bool:
    return "locked" not in state[name]


@needs_node
def test_the_embedded_script_parses(tmp_path):
    path = tmp_path / "audit.js"
    path.write_text(page_script(), encoding="utf-8")
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        node_argv("--check", str(path)), capture_output=True, text=True
    )
    assert result.returncode == 0, f"the audit script does not parse:\n{result.stderr}"


@needs_node
def test_a_switched_off_api_shows_the_notice_and_never_the_gate():
    """The original bug: this state used to render a token prompt."""
    state = relock("off")
    assert visible(state, "off"), "a switched-off audit API must say so"
    assert not visible(state, "gate"), "no token can satisfy a switched-off API"
    assert not visible(state, "main")


@needs_node
def test_token_mode_shows_the_gate():
    state = relock("token")
    assert visible(state, "gate")
    assert not visible(state, "off"), "the notice is only for the switched-off API"
    assert not visible(state, "main")


@needs_node
def test_open_mode_only_falls_back_to_the_gate_if_a_token_appears_to_be_needed():
    """relock() in open mode means a 401 arrived anyway -- for instance the
    server was restarted with a token -- so asking for one is the useful move."""
    state = relock("open")
    assert visible(state, "gate")
    assert not visible(state, "off")
    assert not visible(state, "main")
    assert state["unlocked"] is False


@needs_node
def test_boot_probes_even_with_no_stored_token():
    """--audit-open has no token to have, so boot() must not bail out early."""
    script = page_script()
    assert "if(!TOKEN) return;" not in script, (
        "boot() returning early leaves /audit unusable under --audit-open"
    )
    assert 'const headers = TOKEN ? {"x-audit-token":TOKEN} : {};' in script
