"""Tests for the transcript preview in the served page.

The preview is client-side JavaScript inside the `PAGE` string and there is no
headless browser in this environment, so these tests do the next best thing:
the script must parse (node --check), the timestamp formatter must produce the
right clock, and the preview's own functions are executed under node with a
small DOM stub so the follow/pause/append behaviour is checked rather than
assumed. What still needs a browser is the visual result: layout, the actual
scrollbar, and whether the amber "paused" border reads well.

Skipped, not failed, when node is not installed: the rest of the suite has no
use for it.
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

# Enough of a DOM for the preview's functions: a transcript element with the
# handful of properties they touch, and a createElement that records what was
# built so a test can read the rows back.
PREVIEW_HARNESS = """
const document = {
  createElement: () => {
    const node = {className: "", textContent: "", children: [],
                  append: (...kids) => { node.children.push(...kids); }};
    return node;
  },
};
let following = true;
function followOn(){ return following; }
function makeView(){
  const rows = [];
  const marks = new Set();
  const transcript = {
    scrollHeight: 0, scrollTop: 0, clientHeight: 100, title: "",
    classList: {toggle: (name, on) => { on ? marks.add(name) : marks.delete(name); }},
    append: (row) => { rows.push(row); transcript.scrollHeight = rows.length * 20; },
    set textContent(value){ if(value === "") rows.length = 0; },
  };
  return {view: {transcript, shown: 0, paused: false, live: false}, rows, marks};
}
const seg = (start, text) => ({start, text});
/* Read the live rows, not every node ever built: a cleared transcript must not
   still show up in a result. */
const times = (rows) => rows.map(r => r.children[0].textContent);
const texts = (rows) => rows.map(r => r.children[1].textContent);
"""


def page_script() -> str:
    match = SCRIPT.search(s.PAGE)
    assert match, "the page no longer has a <script> block to check"
    return match.group(1)


def function_source(name: str) -> str:
    """One `function name(...) {...}` block, which is how the harness gets them."""
    match = re.search(rf"function {name}\(.*?\n\}}", page_script(), re.S)
    assert match, f"{name}() is gone from the page script"
    return match.group(0)


def node_argv(*args: str) -> list[str]:
    """node plus its arguments. Every caller is skipped when node is missing."""
    assert NODE is not None, "node is required here"
    return [NODE, *args]


def run_node(source: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        node_argv("-e", source), capture_output=True, text=True
    )
    assert result.returncode == 0, f"node failed:\n{result.stderr}"
    return result.stdout


def preview_probe(body: str) -> dict:
    """Run `body` against the real preview functions plus the DOM stub."""
    program = (
        PREVIEW_HARNESS
        + "\n".join(
            function_source(name)
            for name in ("fmtStamp", "stick", "setPaused", "appendSegments")
        )
        + "\nprocess.stdout.write(JSON.stringify((() => {"
        + body
        + "})()));"
    )
    return json.loads(run_node(program))


# --------------------------------------------------------------------------- #
# The page exposes what the feature needs
# --------------------------------------------------------------------------- #


def test_the_page_has_a_follow_switch_that_starts_on():
    assert re.search(r'id="follow"[^>]*checked', s.PAGE), (
        "auto-scroll needs an on-by-default Follow switch"
    )
    assert "> Follow<" in s.PAGE, "the switch must say what it does"


def test_the_follow_switch_is_remembered_for_the_session():
    script = page_script()
    assert 'FOLLOW_KEY = "follow"' in script
    assert "sessionStorage.setItem(FOLLOW_KEY" in script
    assert "sessionStorage.getItem(FOLLOW_KEY)" in script


def test_the_time_gutter_keeps_its_width_so_rows_line_up():
    assert 'ts.className = "ts"' in page_script()
    assert re.search(r"\.seg \.ts\{[^}]*width:9ch", s.PAGE), (
        "the gutter needs a fixed width"
    )
    assert "tabular-nums" in s.PAGE, "digits must not jitter as the clock advances"


def test_the_transcript_dom_survives_a_poll():
    """The point of the rewrite: no wholesale innerHTML on a live card."""
    script = page_script()
    assert "const views = new Map()" in script
    assert "views.get(id)" in script, "render() must reuse the card it already built"
    assert "view.transcript.append(row)" in script, "segments are appended, not rebuilt"
    assert "map(s => s.text).join" not in script, (
        "the preview must not collapse back into one run-on paragraph"
    )
    assert "views.delete(stale)" in script, "a removed card must not be remembered"


# --------------------------------------------------------------------------- #
# The script parses, and the clock is right
# --------------------------------------------------------------------------- #


@needs_node
def test_the_embedded_script_parses(tmp_path):
    path = tmp_path / "page.js"
    path.write_text(page_script(), encoding="utf-8")
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        node_argv("--check", str(path)), capture_output=True, text=True
    )
    assert result.returncode == 0, f"the page script does not parse:\n{result.stderr}"


@needs_node
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00"),
        (5, "00:00:05"),
        (59.9, "00:00:59"),  # floors, so a row never claims the next second
        (60, "00:01:00"),
        (61, "00:01:01"),
        (599, "00:09:59"),
        (3600, "01:00:00"),
        (3725, "01:02:05"),
        (36000, "10:00:00"),
        (-3, "00:00:00"),  # a negative start is not a crash
    ],
)
def test_fmt_stamp_matches_the_hh_mm_ss_clock(seconds, expected):
    out = run_node(
        f"{function_source('fmtStamp')}\nprocess.stdout.write(fmtStamp({seconds}));"
    )
    assert out == expected


@needs_node
def test_the_preview_clock_matches_the_timestamped_export():
    """The preview shows the same clock as Save .txt (timestamped), minus ms."""
    values = [0, 1, 59.9, 60, 61, 754.25, 3600, 3725, 59999.5]
    js = run_node(
        function_source("fmtStamp")
        + f"\nprocess.stdout.write(JSON.stringify({values}.map(fmtStamp)));"
    )
    assert json.loads(js) == [s._stamp(value)[:8] for value in values]


# --------------------------------------------------------------------------- #
# Behaviour: appending, following, pausing
# --------------------------------------------------------------------------- #


@needs_node
def test_a_followed_preview_appends_rows_with_timestamps_and_sticks_to_the_bottom():
    probe = preview_probe(
        """
        const h = makeView();
        appendSegments(h.view, [seg(3, "first"), seg(72, "second")], true);
        return {
          rows: h.rows.length,
          times: times(h.rows),
          texts: texts(h.rows),
          atBottom: h.view.transcript.scrollTop === h.view.transcript.scrollHeight,
        };
        """
    )
    assert probe["rows"] == 2
    assert probe["times"] == ["00:00:03", "00:01:12"]
    assert probe["texts"] == ["first", "second"]
    assert probe["atBottom"] is True


@needs_node
def test_an_empty_tail_adds_nothing():
    """A poll with no new text sends an empty list, and must add no rows.

    This used to be guaranteed by re-sending the whole transcript and letting
    appendSegments de-duplicate it. The guarantee now lives in the request
    (?since=<shown>), so the renderer only has to append what it is handed.
    """
    probe = preview_probe(
        """
        const h = makeView();
        appendSegments(h.view, [seg(3, "first"), seg(72, "second")], true, 2);
        appendSegments(h.view, [], true, 2);
        return {rows: h.rows.length, shown: h.view.shown};
        """
    )
    assert (probe["rows"], probe["shown"]) == (2, 2)


@needs_node
def test_a_preview_that_lost_segments_starts_over_instead_of_leaving_stale_rows():
    """A tail plus a smaller server total means those rows no longer exist."""
    probe = preview_probe(
        """
        const h = makeView();
        appendSegments(h.view, [seg(3, "first"), seg(72, "second")], true, 2);
        appendSegments(h.view, [seg(9, "replacement")], false, 1);
        return {rows: h.rows.length, times: times(h.rows)};
        """
    )
    assert (probe["rows"], probe["times"]) == (1, ["00:00:09"])


@needs_node
def test_scrolling_up_stops_the_chase_and_going_back_to_the_bottom_resumes_it():
    probe = preview_probe(
        """
        const h = makeView();
        appendSegments(h.view, [seg(1, "one")], true);
        const parked = h.view.transcript.scrollTop;
        setPaused(h.view, true);                       // the scroll handler does this
        appendSegments(h.view, [seg(2, "two")], true);
        const whilePaused = h.view.transcript.scrollTop;
        setPaused(h.view, false);                      // back at the bottom
        appendSegments(h.view, [seg(3, "three")], true);
        return {
          parked, whilePaused,
          resumed: h.view.transcript.scrollTop,
          height: h.view.transcript.scrollHeight,
        };
        """
    )
    assert probe["whilePaused"] == probe["parked"], (
        "a parked preview must not be yanked down"
    )
    assert probe["resumed"] == probe["height"], (
        "scrolling back to the bottom re-arms following"
    )


@needs_node
def test_the_switch_off_means_no_scrolling_even_at_the_bottom():
    probe = preview_probe(
        """
        following = false;
        const h = makeView();
        appendSegments(h.view, [seg(1, "one"), seg(2, "two")], true);
        return {rows: h.rows.length, scrollTop: h.view.transcript.scrollTop};
        """
    )
    assert probe["rows"] == 2, "text still arrives; only the scrolling is off"
    assert probe["scrollTop"] == 0


@needs_node
def test_a_finished_job_does_not_chase_its_tail():
    probe = preview_probe(
        """
        const h = makeView();
        appendSegments(h.view, [seg(1, "one"), seg(2, "two"), seg(3, "three")], false);
        return {rows: h.rows.length, scrollTop: h.view.transcript.scrollTop, live: h.view.live};
        """
    )
    assert probe["rows"] == 3
    assert probe["scrollTop"] == 0, "a finished transcript opens at its first line"
    assert probe["live"] is False


@needs_node
def test_pausing_marks_the_preview_and_says_why():
    probe = preview_probe(
        """
        const h = makeView();
        setPaused(h.view, true);
        const paused = {marks: [...h.marks], title: h.view.transcript.title};
        setPaused(h.view, false);
        return {paused, marks: [...h.marks], title: h.view.transcript.title};
        """
    )
    assert probe["paused"]["marks"] == ["paused"]
    assert "bottom" in probe["paused"]["title"]
    assert probe["marks"] == [] and probe["title"] == ""


# --------------------------------------------------------------------------- #
# The server side of the preview
# --------------------------------------------------------------------------- #


def test_the_api_gives_the_preview_what_it_renders(configured, client):
    """The preview reads seg.start and seg.text; keep that contract pinned."""
    job_id = configured.make_job()
    s.patch_job(
        job_id,
        state="done",
        language="en",
        duration=2.0,
        segments=[
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 1.5, "end": 2.0, "text": "second"},
        ],
    )

    body = client.get(f"/api/jobs/{job_id}").json()

    assert [seg["start"] for seg in body["segments"]] == [0.0, 1.5]
    assert [seg["text"] for seg in body["segments"]] == ["first", "second"]


# --------------------------------------------------------------------------- #
# The poll fetches only the tail, not the whole transcript every tick
# --------------------------------------------------------------------------- #


def _three_segment_job(configured) -> str:
    job_id = configured.make_job()
    s.patch_job(
        job_id,
        state="running",
        language="en",
        duration=3.0,
        segments=[
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 1.0, "end": 2.0, "text": "second"},
            {"start": 2.0, "end": 3.0, "text": "third"},
        ],
    )
    return job_id


def test_since_returns_only_the_segments_the_client_lacks(configured, client):
    """The whole point: a poll must not re-ship the transcript it already has."""
    job_id = _three_segment_job(configured)

    body = client.get(f"/api/jobs/{job_id}?since=2").json()

    assert [seg["text"] for seg in body["segments"]] == ["third"]
    assert body["segment_start"] == 2
    assert body["segment_count"] == 3, "the client needs the total to spot a reset"


def test_since_zero_still_returns_everything(configured, client):
    """Not every caller polls; the endpoint stays usable as a plain detail read."""
    job_id = _three_segment_job(configured)

    body = client.get(f"/api/jobs/{job_id}?since=0").json()

    assert [seg["text"] for seg in body["segments"]] == ["first", "second", "third"]
    assert body["segment_start"] == 0


def test_a_caught_up_client_gets_an_empty_tail(configured, client):
    job_id = _three_segment_job(configured)

    body = client.get(f"/api/jobs/{job_id}?since=3").json()

    assert body["segments"] == []
    assert body["segment_count"] == 3


@pytest.mark.parametrize("since", ["99", "-5"])
def test_an_out_of_range_since_is_clamped_not_fatal(configured, client, since):
    """A stale value must not 500, or duplicate the transcript."""
    job_id = _three_segment_job(configured)

    response = client.get(f"/api/jobs/{job_id}?since={since}")

    assert response.status_code == 200
    body = response.json()
    assert body["segment_count"] == 3
    assert len(body["segments"]) <= 3
    assert body["segment_start"] <= 3


def test_a_non_numeric_since_is_rejected_by_the_typed_param(configured, client):
    """`since` is a declared int, so FastAPI rejects junk before the handler.

    That is the same contract the other typed query params have (see
    audit_query's `limit`), and the only writer of this value is the page's own
    segment counter, so there is nothing to be lenient for.
    """
    job_id = _three_segment_job(configured)

    assert client.get(f"/api/jobs/{job_id}?since=abc").status_code == 422


def test_the_poll_asks_for_a_tail_and_the_renderer_passes_the_total():
    """Both halves have to hold, or the wire saving is silently lost."""
    script = page_script()
    assert '"?since=" + since' in script, "the poll must ask for a tail"
    assert "summary.segment_count < view.shown" in script, (
        "a card must notice when the job's segments went backwards"
    )
    assert "job.segment_count)" in function_source("render"), (
        "render() must hand appendSegments the server's total"
    )


def test_the_page_never_refetches_the_whole_transcript_on_a_tick():
    """Guards against the regression this change exists to fix."""
    tick = function_source("tick")
    assert "?since=" in tick, "tick() must be incremental"
    assert re.search(r'api\("/api/jobs/" \+ encodeURIComponent\([^)]*\)\)', tick) is None, (
        "tick() must not fetch the full detail endpoint any more"
    )
