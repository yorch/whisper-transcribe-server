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
from transcribe_server import STATIC_DIR

# The pages used to be embedded in the module as PAGE/AUDIT_PAGE. They are now
# files under static/, shared helpers split out into common.js and app.css, so
# the harness reads those instead of scraping a Python string literal.
STATIC = STATIC_DIR


def static(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def page_html() -> str:
    return static("index.html")


def page_css() -> str:
    """Everything that styles the main page: shared rules plus its own."""
    return static("app.css") + "\n" + static("index.css")


def page_script() -> str:
    """The main page's JS: the shared helpers plus its own script."""
    return static("common.js") + "\n" + static("index.js")


NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

# Enough of a DOM for the preview's functions: a transcript element with the
# handful of properties they touch, and a createElement that records what was
# built so a test can read the rows back.
PREVIEW_HARNESS = """
const document = {
  createElement: () => {
    const node = {className: "", textContent: "", children: [],
                  append: (...kids) => { node.children.push(...kids); }};
    // classList, because the speaker gutter colours itself by class.
    node.classList = {add: (name) => {
      node.className = (node.className ? node.className + " " : "") + name;
    }};
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
    // Clearing the content collapses the scroll box and clamps scrollTop, the
    // way a browser does. Without that the follow-after-redraw test below
    // would pass for the wrong reason.
    set textContent(value){
      if(value === ""){ rows.length = 0; transcript.scrollHeight = 0; transcript.scrollTop = 0; }
    },
  };
  return {view: {transcript, shown: 0, paused: false, live: false, labeled: false},
          rows, marks};
}
const seg = (start, text) => ({start, text});
/* Read the live rows, not every node ever built: a cleared transcript must not
   still show up in a result. Cells are found by class, not by index, so adding
   a column (the speaker gutter) cannot silently shift what a test reads. */
const cell = (row, cls) => {
  const found = row.children.find(
    (kid) => (kid.className || "").split(" ").includes(cls));
  return found ? found.textContent : "";
};
const times = (rows) => rows.map(r => cell(r, "ts"));
const texts = (rows) => rows.map(r => cell(r, "tx"));
const speakers = (rows) => rows.map(r => cell(r, "sp"));
"""


def function_source(name: str) -> str:
    """One `function name(...) {...}` block, which is how the harness gets them.

    The optional `async` matters: `startJobs` and `send` are async, and a
    harness that took the body without the keyword would not parse.
    """
    match = re.search(rf"(?:async )?function {name}\(.*?\n\}}", page_script(), re.S)
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


# A DOM big enough that createCard() and render() can really run: nodes record
# their children, and only the transcript behaves like a scroll box, which is
# what stick() reads. The jobs area is the one element looked up by id.
CARD_HARNESS = """
const TICKS = 40;
const views = new Map();
const RETRY_OK = true;
const jobsBox = {children: [], prepend: (kid) => { jobsBox.children.unshift(kid); }};
function makeNode(){
  const node = {
    className: "", id: "", title: "", dataset: {}, children: [],
    isConnected: true, scrollHeight: 0, scrollTop: 0, clientHeight: 100,
    _text: "",
    append(...kids){
      node.children.push(...kids);
      if(node.className === "transcript") node.scrollHeight = node.children.length * 20;
    },
    prepend(kid){ node.children.unshift(kid); },
    addEventListener(){},
    classList: {add(){}, remove(){}, toggle(name, on){ if(name === "paused") node.paused = on; }},
  };
  // A browser drops the children when textContent is set, which is how
  // renderStaged() redraws the list; the transcript also loses its scroll.
  Object.defineProperty(node, "textContent", {
    get(){ return node._text; },
    set(value){
      node._text = value;
      node.children.length = 0;
      if(node.className === "transcript"){ node.scrollHeight = 0; node.scrollTop = 0; }
    },
  });
  return node;
}
const document = {
  createElement: () => makeNode(),
  getElementById: (id) => (id === "jobs" ? jobsBox : {checked: true}),
};
const el = (id) => document.getElementById(id);
"""


def card_probe(body: str) -> dict:
    """Run `body` against the real card-building functions and a stub DOM."""
    program = (
        CARD_HARNESS
        + "\n".join(
            function_source(name)
            for name in (
                "viewKey",
                "fmtStamp",
                "fmtTime",
                "meter",
                "statusLine",
                "jobTags",
                "followOn",
                "stick",
                "setPaused",
                "appendSegments",
                "actions",
                "createCard",
                "render",
            )
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
    assert re.search(r'id="follow"[^>]*checked', page_html()), (
        "auto-scroll needs an on-by-default Follow switch"
    )
    assert "> Follow<" in page_html(), "the switch must say what it does"


def test_the_follow_switch_is_remembered_for_the_session():
    script = page_script()
    assert 'FOLLOW_KEY = "follow"' in script
    assert "sessionStorage.setItem(FOLLOW_KEY" in script
    assert "sessionStorage.getItem(FOLLOW_KEY)" in script


def test_the_time_gutter_keeps_its_width_so_rows_line_up():
    assert 'ts.className = "ts"' in page_script()
    assert re.search(r"\.seg \.ts\{[^}]*width:9ch", page_css()), (
        "the gutter needs a fixed width"
    )
    assert "tabular-nums" in page_css(), "digits must not jitter as the clock advances"


def test_the_transcript_dom_survives_a_poll():
    """The point of the rewrite: no wholesale innerHTML on a live card."""
    script = page_script()
    assert "const views = new Map()" in script
    assert "views.get(viewKey(job.id))" in script, (
        "render() must look up the card it already built"
    )
    assert "view.transcript.append(row)" in script, "segments are appended, not rebuilt"
    assert "map(s => s.text).join" not in script, (
        "the preview must not collapse back into one run-on paragraph"
    )
    assert "views.delete(viewKey(stale))" in script, (
        "a removed card must not be remembered"
    )


def test_every_view_lookup_keys_the_map_the_same_way():
    """The bug this pins: createCard() stored the card under the bare job id
    while render() and tick() looked it up as "job-" + id, so every lookup
    missed. render() then built a second card above the first on every progress
    step — meter, transcript and all — while the full transcript came back over
    the wire on every tick, and each stale card leaked out of the DOM for good
    because the sweep removes one node per id."""
    calls = re.findall(r"views\.(get|set|delete)\(([^)]*)\)", page_script())
    assert sorted(name for name, _ in calls) == ["delete", "get", "get", "set"], calls
    wrongly_keyed = [arg.strip() for _, arg in calls if "viewKey(" not in arg]
    assert not wrongly_keyed, (
        f"a view is keyed differently from the rest: {wrongly_keyed}"
    )


@needs_node
def test_a_progress_step_reuses_the_card_instead_of_building_another_one():
    """The regression, executed: two polls of the same running job must leave
    exactly one card in the DOM, holding one meter and both segments."""
    probe = card_probe(
        """
        const job = (extra) => Object.assign({
          id: "abc", filename: "a.wav", state: "running", progress: 0.1,
          elapsed: 5, duration: 60, segment_count: 1, speaker_labels: false,
          opts: {model: "small"}, segments: [{start: 0, text: "one"}],
        }, extra);
        render(job());
        const first = views.get(viewKey("abc"));
        // The next poll sends only the tail, the way tick() asks for it.
        render(job({progress: 0.4, elapsed: 20, segment_count: 2,
                    segments: [{start: 1, text: "two"}]}));
        const second = views.get(viewKey("abc"));
        return {
          cards: jobsBox.children.length,
          remembered: views.size,
          sameCard: first.node === second.node,
          rows: first.transcript.children.length,
          texts: first.transcript.children.map(r => r.children[2].textContent),
          cells: first.meter.children.length,
          lit: first.meter.children.filter(c => c.className === "lit").length,
          name: first.name.textContent,
          status: first.status.textContent,
        };
        """
    )
    assert probe["cards"] == 1, "a poll rebuilt the card instead of reusing it"
    assert probe["remembered"] == 1
    assert probe["sameCard"] is True
    assert probe["rows"] == 2, "the tail was not appended to the same transcript"
    assert probe["texts"] == ["one", "two"]
    assert probe["cells"] == 40 and probe["lit"] == 16, probe["lit"]
    assert probe["name"] == "a.wav"
    assert probe["status"].startswith("40% \u00b7 2 segments"), probe["status"]


@needs_node
def test_a_filename_cannot_become_markup_in_the_card():
    """The card is built as nodes, so the filename is a text node whatever it
    contains: no escaping step to forget, and nothing for the transcript's own
    textContent to leak either."""
    probe = card_probe(
        """
        const nasty = '<img src=x onerror="alert(1)">';
        render({id: "h", filename: nasty, state: "done", opts: {model: "small"},
                segments: [{start: 0, text: nasty}], segment_count: 1,
                elapsed: 3, duration: 3, language: "en"});
        const view = views.get(viewKey("h"));
        return {name: view.name.textContent, title: view.name.title,
                row: view.transcript.children[0].children[2].textContent};
        """
    )
    nasty = '<img src=x onerror="alert(1)">'
    assert probe == {"name": nasty, "title": nasty, "row": nasty}
    assert "innerHTML" not in function_source("render")
    assert "innerHTML" not in function_source("createCard")


@needs_node
def test_the_actions_are_buttons_a_handler_can_read_back():
    """The download/copy/retry/remove handlers read dataset, so the DOM builder
    has to put the same keys there the markup string used to."""
    probe = card_probe(
        """
        const buttons = (job) => {
          const view = createCard(job);
          render(job);
          return view.actions.children.map(b => [b.textContent,
            JSON.stringify(b.dataset), b.className]);
        };
        return {
          done: buttons({id: "d", filename: "x", state: "done", opts: {model: "m"},
                         segments: [], segment_count: 0, elapsed: 4, duration: 4,
                         language: "en"}),
          running: buttons({id: "r", filename: "x", state: "running", progress: 0.2,
                            opts: {model: "m"}, segments: [], segment_count: 0}),
        };
        """
    )
    labels = [row[0] for row in probe["done"]]
    assert labels == [
        "Copy text",
        "Save .txt",
        "Save timestamped",
        "Save .srt",
        "Save .vtt",
        "Save .json",
        "Remove",
    ], labels
    assert json.loads(probe["done"][1][1]) == {"dl": "txt", "id": "d"}
    assert json.loads(probe["done"][0][1]) == {"copy": "d"}
    # A live job is cancelled, not removed, and the ghost class rides along.
    assert [row[0] for row in probe["running"]] == ["Cancel"]
    assert probe["running"][0][2] == "ghost"
    assert json.loads(probe["running"][0][1]) == {"del": "r"}


# --------------------------------------------------------------------------- #
# The diarization controls
# --------------------------------------------------------------------------- #

# Enough of the control panel for syncDiarize: the two boxes it touches, plus
# the flag the status handler sets.
CONTROL_HARNESS = """
const controls = {
  diarize: {checked: true},
  words: {checked: false, disabled: false, title: ""},
};
const el = (id) => controls[id];
let DIARIZE_OK = false;
"""


def control_probe(body: str) -> dict:
    """Run `body` against the real syncDiarize and a two-element DOM stub."""
    program = (
        CONTROL_HARNESS
        + function_source("locksWordTimings")
        + function_source("syncDiarize")
        + "\nprocess.stdout.write(JSON.stringify((() => {"
        + body
        + "})()));"
    )
    return json.loads(run_node(program))


def test_the_page_ticks_identify_speakers_and_not_word_timings():
    """A dropped file should come back labelled, so that box ships ticked.

    Word timings do NOT: the script ticks them once the server has confirmed it
    offers diarization, so a --no-diarize server never forces them on. Both
    sides of that are covered by the control_probe tests below.
    """
    html = page_html()
    assert re.search(r'id="diarize"\s+checked', html), (
        "the Identify speakers box is no longer ticked by default"
    )
    assert not re.search(r'id="words"[^>]*checked', html), (
        "word timings must not be forced from the markup alone"
    )


@needs_node
def test_a_server_without_diarization_does_not_force_word_timings():
    """The bug this pins: the markup ships the box ticked so a dropped file is
    labelled, but a --no-diarize server hides that control and must not leave
    word timings ticked and disabled behind it."""
    probe = control_probe(
        """
        DIARIZE_OK = false;
        controls.diarize.checked = false;   // what the status handler does
        syncDiarize();
        return {words: controls.words.checked, disabled: controls.words.disabled,
                title: controls.words.title};
        """
    )
    assert probe == {"words": False, "disabled": False, "title": ""}, probe


@needs_node
def test_asking_for_speakers_ticks_and_locks_word_timings():
    probe = control_probe(
        """
        DIARIZE_OK = true;
        controls.diarize.checked = true;
        syncDiarize();
        return {words: controls.words.checked, disabled: controls.words.disabled,
                title: controls.words.title};
        """
    )
    assert probe["words"] is True
    assert probe["disabled"] is True, "the operator should see why it is on"
    assert "speaker" in probe["title"]


@needs_node
def test_unticking_speakers_releases_the_word_timing_box():
    """Unlocking must not also untick it: once the box is the operator's again,
    whatever they last chose should stand."""
    probe = control_probe(
        """
        DIARIZE_OK = true; controls.diarize.checked = true; syncDiarize();
        controls.diarize.checked = false; syncDiarize();
        return {disabled: controls.words.disabled, title: controls.words.title,
                words: controls.words.checked};
        """
    )
    assert probe["disabled"] is False and probe["title"] == ""
    assert probe["words"] is True, "the tick itself is left as it was"


@needs_node
def test_the_lock_is_applied_after_status_not_at_load():
    """DIARIZE_OK is unknown until /api/status answers, so an unconditional call
    at parse time would lock the box for a feature that may not exist."""
    script = page_script()
    assert "syncDiarize();" in script, "the status handler must still apply it"
    assert not re.search(
        r'addEventListener\("change", syncDiarize\);\s*syncDiarize\(\)', script
    ), "syncDiarize() must not run at load, before DIARIZE_OK is known"


# --------------------------------------------------------------------------- #
# The device readout
# --------------------------------------------------------------------------- #


def device_probe(body: str):
    """Run `body` against the real readout functions. They are pure, so this
    needs no DOM at all."""
    program = (
        "\n".join(
            function_source(name)
            for name in ("gb", "fmtGb", "deviceLabel", "deviceTooltip")
        )
        + "\nprocess.stdout.write(JSON.stringify((() => {"
        + body
        + "})()));"
    )
    return json.loads(run_node(program))


@needs_node
def test_the_device_cell_shows_the_card_and_how_much_of_it_is_in_use():
    labels = device_probe(
        """
        return [
          {device: "cuda", gpu: "1 CUDA device(s)",
           device_info: {name: "NVIDIA GeForce RTX 3060",
                         vram_total_mb: 12288, vram_used_mb: 1712}},
          // A card we could not name: the count is still better than nothing.
          {device: "cuda", gpu: "1 CUDA device(s)", device_info: {}},
          // No GPU anywhere.
          {device: "cpu", device_info: {python: "3.12.2", platform: "Linux"}},
        ].map(deviceLabel);
        """
    )
    assert labels == [
        "NVIDIA GeForce RTX 3060 \u00b7 1.7/12.0 GB",
        "1 CUDA device(s)",
        "cpu",
    ]


@needs_node
def test_a_named_card_without_memory_figures_still_reads_cleanly():
    """A driver that will not report memory must not produce "\u00b7 null/null"."""
    labels = device_probe(
        """
        return [
          {device: "cuda", device_info: {name: "Some Card", vram_total_mb: null}},
          {device: "cuda", device_info: {name: "Some Card", vram_total_mb: 8192}},
        ].map(deviceLabel);
        """
    )
    assert labels == ["Some Card", "Some Card"]


@needs_node
def test_the_tooltip_leads_with_why_the_gpu_is_not_being_used():
    lines = device_probe(
        """
        return deviceTooltip({
          device: "cpu", compute_type: "int8",
          cuda: {usable: false, reason: "cublas64_12.dll could not be loaded"},
          device_info: {name: "NVIDIA GeForce RTX 3060", vram_total_mb: 12288,
                        vram_used_mb: 1713, process_mb: 512, driver: "610.88",
                        compute_cap: "8.6", count: 1, python: "3.12.2",
                        platform: "Linux"},
        }).split("\\n");
        """
    )
    assert lines[0] == "cublas64_12.dll could not be loaded"
    assert "NVIDIA GeForce RTX 3060" in lines
    assert "VRAM 1.7 GB used of 12.0 GB" in lines
    assert "this server holds 0.5 GB" in lines
    assert "driver 610.88" in lines
    assert "compute capability 8.6" in lines
    assert "1 CUDA device(s)" in lines
    assert "running on cpu at int8" in lines
    assert "Python 3.12.2 on Linux" in lines


@needs_node
def test_pre_pascal_compute_capability_comes_with_the_fp16_warning():
    """Compute capability is the reason float16 is or is not a good idea, so the
    tooltip says which rather than leaving the number to be looked up."""
    lines = device_probe(
        """
        return deviceTooltip({device: "cuda", compute_type: "float16", cuda: null,
          device_info: {name: "GTX 1050 Ti", compute_cap: "6.1"}}).split("\\n");
        """
    )
    assert any("compute capability 6.1" in line and "int8" in line for line in lines)

    modern = device_probe(
        """
        return deviceTooltip({device: "cuda", compute_type: "float16", cuda: null,
          device_info: {name: "RTX 3060", compute_cap: "8.6"}}).split("\\n");
        """
    )
    assert "compute capability 8.6" in modern, "no warning where none is needed"


@needs_node
def test_the_tooltip_says_so_when_the_card_could_not_be_named():
    lines = device_probe(
        """
        return deviceTooltip({device: "cuda", compute_type: "float16",
          cuda: {usable: true}, device_info: {count: 1, python: "3.12.2"}
        }).split("\\n");
        """
    )
    assert any("nvidia-smi" in line for line in lines)


@needs_node
def test_the_tooltip_is_harmless_on_a_machine_with_no_gpu():
    lines = device_probe(
        """
        return deviceTooltip({device: "cpu", compute_type: "int8", cuda: null,
          device_info: {python: "3.12.2", platform: "Linux"}}).split("\\n");
        """
    )
    assert lines == ["running on cpu at int8", "Python 3.12.2 on Linux"]


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
def test_labels_arriving_at_the_end_redraw_the_transcript_once():
    """Diarization is a second pass, so every row on screen was drawn without a
    speaker. Counting new segments cannot notice that, so the server reports the
    shape and the client refetches from zero — a tail cannot rebuild rows that
    are already drawn."""
    probe = preview_probe(
        """
        const h = makeView();
        appendSegments(h.view, [seg(3, "first"), seg(72, "second")], true, 2, false);
        const before = {rows: h.rows.length, speakers: speakers(h.rows)};
        /* The labels landed, so the client asked for the whole transcript
           again and the server now reports it as labelled. */
        const labeled = [{start: 3, text: "first", speaker: 1},
                         {start: 72, text: "second", speaker: 2}];
        appendSegments(h.view, labeled, false, 2, true);
        const after = {rows: h.rows.length, speakers: speakers(h.rows),
                       texts: texts(h.rows), times: times(h.rows)};
        /* The next poll is caught up: an empty tail, still labelled. It must
           not rebuild again, or the preview would redraw on every tick. */
        appendSegments(h.view, [], false, 2, true);
        const again = {rows: h.rows.length, shown: h.view.shown};
        return {before, after, again};
        """
    )
    assert probe["before"] == {"rows": 2, "speakers": ["", ""]}
    assert probe["after"] == {
        "rows": 2,
        "speakers": ["Speaker 1", "Speaker 2"],
        "texts": ["first", "second"],
        "times": ["00:00:03", "00:01:12"],
    }
    assert probe["again"] == {"rows": 2, "shown": 2}


@needs_node
def test_the_speaker_gutter_does_not_disturb_unlabelled_rows():
    rows = preview_probe(
        """
        const h = makeView();
        appendSegments(h.view, [seg(3, "first"), seg(72, "second")], true);
        return {speakers: speakers(h.rows), texts: texts(h.rows),
                cells: h.rows.map(r => r.children.length)};
        """
    )
    assert rows["speakers"] == ["", ""]
    assert rows["texts"] == ["first", "second"]
    # The gutter is present but empty, so nothing shifts sideways when the
    # labels land.
    assert rows["cells"] == [3, 3]


@needs_node
def test_following_survives_the_redraw_that_adds_labels():
    """Rebuilding clears the transcript, which resets the scroll: someone who
    was following a live job must not be thrown back to the top."""
    probe = preview_probe(
        """
        const h = makeView();
        appendSegments(h.view, [seg(3, "a"), seg(4, "b")], true, 2, false);
        const wasAtBottom = h.view.transcript.scrollTop === h.view.transcript.scrollHeight;
        appendSegments(h.view, [{start: 3, text: "a", speaker: 1},
                                {start: 4, text: "b", speaker: 2}], false, 2, true);
        return {wasAtBottom,
                atBottom: h.view.transcript.scrollTop === h.view.transcript.scrollHeight,
                paused: h.view.paused};
        """
    )
    assert probe["wasAtBottom"] is True
    assert probe["atBottom"] is True, "the redraw lost the follower's position"
    assert probe["paused"] is False


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
# Staging: a dropped file waits for the button
# --------------------------------------------------------------------------- #

# The staging functions run against a small DOM and stub collaborators, so the
# state machine (what is staged, what the button says, what a failure does) is
# executed rather than read.
STAGE_HARNESS = """
const nodes = {};
function makeNode(tag){
  const node = {
    tag, className: "", id: "", textContent: "", title: "", dataset: {},
    disabled: undefined, children: [], classes: new Set(), _text: "",
    attributes: {},
    append(...kids){ node.children.push(...kids); },
    setAttribute(name, value){ node.attributes[name] = value; },
    classList: {
      add(name){ node.classes.add(name); },
      remove(name){ node.classes.delete(name); },
      toggle(name, on){ on ? node.classes.add(name) : node.classes.delete(name); },
    },
  };
  // Setting textContent drops the children, as it does in a browser: that is
  // what makes renderStaged() redraw the list rather than add to it.
  Object.defineProperty(node, "textContent", {
    get(){ return node._text; },
    set(value){ node._text = value; node.children.length = 0; },
  });
  return node;
}
const document = {createElement: (tag) => makeNode(tag)};
const el = (id) => nodes[id] || (nodes[id] = makeNode(id));
let MAX_MB = 0;
const staged = [];
let alerts = [], uploaded = [], uploadFails = false;
const alert = (message) => { alerts.push(message); };
async function api(path){
  uploaded.push(path);
  return {ok: !uploadFails, statusText: "500",
          json: async () => ({detail: "the server said no"})};
}
async function tick(){}
function currentSettings(){ return new FormData(); }
"""


def stage_probe(body: str) -> dict:
    """Run `body` against the real staging functions and the stub DOM.

    `body` is an async function body: it may await startJobs(), which awaits
    send().
    """
    program = (
        STAGE_HARNESS
        + "\n".join(
            function_source(name)
            for name in (
                "fmtSize",
                "startLabel",
                "stageNote",
                "renderStaged",
                "stage",
                "unstage",
                "startJobs",
                "send",
            )
        )
        + "\n(async () => { process.stdout.write(JSON.stringify(await (async () => {"
        + body
        + "})())); })();"
    )
    return json.loads(run_node(program))


def test_dropping_or_picking_a_file_stages_it_instead_of_transcribing_it():
    """The change: a file waits for the button, because the controls above it
    are read at press time and a wrong model used to cost a re-upload."""
    script = page_script()
    assert 'addEventListener("change", () => { stage([...picker.files]);' in script
    assert (
        'intake.addEventListener("drop", (e) => stage([...e.dataTransfer.files]))'
        in script
    )
    assert 'el("start").addEventListener("click", startJobs)' in script
    assert not re.search(r"\bsend\(\[", script), "only the button may upload"


def test_the_page_ships_the_button_disabled_with_the_initial_note():
    html = page_html()

    assert re.search(r'id="start"[^>]*disabled', html), (
        "nothing is staged on arrival, so the button starts disabled"
    )
    assert re.search(r'class="staged locked"', html), (
        "the empty staged list starts hidden"
    )
    assert re.search(r'id="run-note"[^>]*>Drop a recording above', html)
    # The controls are what the press reads, so the button belongs below them.
    assert html.index('id="hotwords"') < html.index('id="start"'), (
        "the button must sit after the controls it applies"
    )


@needs_node
def test_the_staged_list_arms_the_button_and_matches_the_shipped_markup():
    """The empty state renderStaged() produces has to equal the one in the
    markup, or the label would change the moment the first file lands."""
    html = page_html()

    # What the markup ships, so renderStaged()'s empty state can be held to it.
    start = re.search(r'id="start"[^>]*>([^<]*)<', html)
    note = re.search(r'id="run-note"[^>]*>([^<]*)<', html)
    assert start is not None, "the Start button is gone from the markup"
    assert note is not None, "the note beside it is gone from the markup"

    probe = stage_probe(
        """
        renderStaged();
        const snapshot = () => ({
          disabled: el("start").disabled,
          label: el("start").textContent,
          note: el("run-note").textContent,
          rows: el("staged").children.length,
          hidden: el("staged").classes.has("locked"),
        });
        const empty = snapshot();
        stage([{name: "standup.wav", size: 1024},
               {name: "demo.mkv", size: 1.5 * 1024 * 1024 * 1024}]);
        const two = snapshot();
        const names = el("staged").children.map(r => r.children[0].textContent);
        const sizes = el("staged").children.map(r => r.children[1].textContent);
        const removers = el("staged").children.map(r => r.children[2].textContent);
        const keys = el("staged").children.map(r => r.children[2].dataset.unstage);
        const labels = el("staged").children.map(
          r => r.children[2].attributes["aria-label"]);
        unstage(0);
        return {empty, two, names, sizes, removers, keys, labels, after: snapshot(),
                afterNames: el("staged").children.map(r => r.children[0].textContent)};
        """
    )
    assert probe["empty"] == {
        "disabled": True,
        "label": start.group(1),
        "note": note.group(1),
        "rows": 0,
        "hidden": True,
    }
    assert probe["empty"]["label"] == "Transcribe"
    assert probe["empty"]["note"] == "Drop a recording above to get started."

    assert probe["two"]["disabled"] is False
    assert probe["two"]["label"] == "Transcribe 2 files"
    assert probe["two"]["hidden"] is False
    assert "read when you press Transcribe" in probe["two"]["note"], (
        "the note has to say the controls apply at press time"
    )
    assert probe["names"] == ["standup.wav", "demo.mkv"]
    assert probe["sizes"] == ["1 kB", "1.5 GB"]
    assert probe["removers"] == ["Remove", "Remove"]
    assert probe["keys"] == ["0", "1"], "a row must know which file it removes"
    assert probe["labels"] == ["Remove standup.wav", "Remove demo.mkv"], (
        "five buttons all named Remove are five identical names to a screen reader"
    )

    assert probe["after"]["rows"] == 1
    assert probe["after"]["label"] == "Transcribe", "one file left is still one file"
    assert probe["afterNames"] == ["demo.mkv"]


@needs_node
def test_a_file_over_the_limit_is_refused_when_it_is_staged():
    """Refusing at press time would mean uploading a gigabyte to be told no."""
    probe = stage_probe(
        """
        MAX_MB = 10;
        stage([{name: "huge.wav", size: 20 * 1024 * 1024},
               {name: "fine.wav", size: 5 * 1024 * 1024}]);
        return {alerts, names: el("staged").children.map(r => r.children[0].textContent)};
        """
    )
    assert probe["alerts"] == ["huge.wav is larger than the 10 MB limit."]
    assert probe["names"] == ["fine.wav"]


@needs_node
def test_pressing_the_button_sends_what_was_staged_and_clears_it():
    probe = stage_probe(
        """
        stage([{name: "a.wav", size: 10}, {name: "b.wav", size: 10}]);
        await startJobs();
        await startJobs();   // nothing staged now: a second press is a no-op
        return {uploaded, rows: el("staged").children.length,
                disabled: el("start").disabled, alerts};
        """
    )
    assert probe["uploaded"] == ["/api/jobs", "/api/jobs"]
    assert probe["rows"] == 0
    assert probe["disabled"] is True, "an empty staging list disarms the button"
    assert probe["alerts"] == []


@needs_node
def test_a_failed_upload_goes_back_to_the_staging_list_in_order():
    """The point of staging: a blip costs a press, not a re-pick of an
    hour-long recording."""
    probe = stage_probe(
        """
        uploadFails = true;
        stage([{name: "a.wav", size: 10}, {name: "b.wav", size: 10}]);
        await startJobs();
        return {alerts, disabled: el("start").disabled,
                names: el("staged").children.map(r => r.children[0].textContent)};
        """
    )
    assert probe["alerts"] == [
        "Upload failed for a.wav: the server said no",
        "Upload failed for b.wav: the server said no",
    ]
    assert probe["names"] == ["a.wav", "b.wav"], "staged order survives a failure"
    assert probe["disabled"] is False, "there is something to retry"


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
    render = function_source("render")
    assert "job.segment_count" in render, (
        "render() must hand appendSegments the server's total"
    )
    assert "job.speaker_labels" in render, (
        "and the label shape, which a tail cannot reveal"
    )


def test_the_poll_refetches_from_zero_when_labels_land():
    """A tail cannot rebuild rows drawn without speakers, so the shape change
    has to trigger one full refetch."""
    tick = function_source("tick")
    assert "full.speaker_labels !== view.labeled" in tick, (
        "tick() must notice the label shape changing"
    )
    assert '"?since=0"' in tick, "and refetch the whole transcript once"


def test_the_server_reports_the_label_shape(configured, client):
    """The signal the client depends on; without it labels arrive invisibly."""
    job_id = configured.make_job()
    s.patch_job(
        job_id,
        state="done",
        segments=[{"start": 0.0, "end": 1.0, "text": "hi"}],
    )
    assert client.get(f"/api/jobs/{job_id}").json()["speaker_labels"] is False

    s.patch_job(
        job_id,
        segments=[{"start": 0.0, "end": 1.0, "text": "hi", "speaker": 1}],
    )
    body = client.get(f"/api/jobs/{job_id}?since=1").json()

    assert body["speaker_labels"] is True
    assert body["segments"] == [], "the tail is empty; the flag is the only signal"


def test_the_page_never_refetches_the_whole_transcript_on_a_tick():
    """Guards against the regression this change exists to fix."""
    tick = function_source("tick")
    assert "?since=" in tick, "tick() must be incremental"
    assert (
        re.search(r'api\("/api/jobs/" \+ encodeURIComponent\([^)]*\)\)', tick) is None
    ), "tick() must not fetch the full detail endpoint any more"
