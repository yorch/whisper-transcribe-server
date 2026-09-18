"""Tests for speaker diarization.

Three things are being pinned here, in descending order of how much they matter:

1. **Isolation.** `OfflineSpeakerDiarization.process()` holds the GIL for its
   entire run — measured, see docs/speaker-diarization.md section 5. Anything
   short of a separate process therefore freezes this server's event loop for
   the whole pass, taking the status poll, the live transcript and Cancel with
   it. `test_a_diarizing_child_cannot_starve_the_parent` is the regression test
   for that, and it fails if the isolation is ever "simplified" into a thread.

2. **Alignment.** Turns and Whisper segments do not share boundaries, and this
   is where a wrong answer is invisible. Every case here runs against
   hand-written turns, so it needs no model and no audio.

3. **The plain path stays plain.** A job that was not diarized must produce
   byte-identical exports to before this feature existed.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import tarfile
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

import transcribe_server as s

# Stand-ins: the stub workers never read them, and nothing here may download.
FAKE_MODELS = {
    "segmentation": Path("/nonexistent/segmentation.onnx"),
    "embedding": Path("/nonexistent/embedding.onnx"),
}


def worker(body: str) -> str:
    """A stub for DIARIZE_WORKER that speaks the child protocol."""
    return (
        "import json, sys, time\n"
        f'MARK = "{s.DIARIZE_MARK}"\n'
        "def emit(p):\n"
        "    print(MARK + json.dumps(p), flush=True)\n"
        f"{body}\n"
    )


def words(*pairs: tuple[float, float, str]) -> list[dict[str, Any]]:
    return [{"start": a, "end": b, "word": text} for a, b, text in pairs]


def segment(start: float, end: float, text: str, **extra: Any) -> dict[str, Any]:
    return {"start": start, "end": end, "text": text, **extra}


def turn(start: float, end: float, speaker: int) -> dict[str, Any]:
    return {"start": start, "end": end, "speaker": speaker}


# --------------------------------------------------------------------------- #
# Isolation: the whole reason this runs in a child process
# --------------------------------------------------------------------------- #


def test_a_diarizing_child_cannot_starve_the_parent(monkeypatch):
    """A child that never yields its own GIL must not stall this process.

    The child busy-loops, so it holds its interpreter's lock for the whole
    second and a half. If run_diarizer ever grows a thread-based fast path,
    the ticker below stops advancing and this test fails — which is exactly
    what an event loop would do in production.
    """
    monkeypatch.setattr(
        s,
        "DIARIZE_WORKER",
        worker(
            "end = time.time() + 1.5\n"
            "while time.time() < end:\n"
            "    pass\n"
            'emit({"t": "turns", "turns": []})'
        ),
    )

    ticks: list[float] = []
    stop = threading.Event()

    def ticker() -> None:
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.01)

    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        assert s.run_diarizer("meeting.wav", 0, FAKE_MODELS) == []
    finally:
        stop.set()
        thread.join(timeout=5)
    elapsed = time.perf_counter() - started

    # The child really did hold on for its full second and a half...
    assert elapsed >= 1.0, "the stub child exited early; the test proves nothing"
    # ...and this thread kept running throughout, which a GIL-holding call in
    # this process could not have allowed. ~150 is expected at 10 ms.
    assert len(ticks) > 50, (
        f"only {len(ticks)} ticks in {elapsed:.1f}s: diarization is no longer "
        "isolated in a child process"
    )


# --------------------------------------------------------------------------- #
# The child protocol
# --------------------------------------------------------------------------- #


def test_turns_are_parsed_and_renumbered_by_first_appearance(monkeypatch):
    monkeypatch.setattr(
        s,
        "DIARIZE_WORKER",
        worker(
            'emit({"t": "progress", "v": 40})\n'
            'emit({"t": "turns", "turns": [[0.0, 1.0, 7], [1.0, 2.0, 2]]})'
        ),
    )
    seen: list[int] = []
    turns = s.run_diarizer("meeting.wav", 0, FAKE_MODELS, on_progress=seen.append)

    # Cluster ids are arbitrary; Speaker 1 is whoever speaks first.
    assert turns == [turn(0.0, 1.0, 1), turn(1.0, 2.0, 2)]
    assert seen == [40]


def test_a_child_that_reports_an_error_raises_it(monkeypatch):
    monkeypatch.setattr(
        s, "DIARIZE_WORKER", worker('emit({"t": "error", "error": "no audio here"})')
    )
    with pytest.raises(RuntimeError, match="no audio here"):
        s.run_diarizer("meeting.wav", 0, FAKE_MODELS)


def test_a_child_that_says_nothing_useful_is_an_error(monkeypatch):
    """Exit 0 with no result is still a failure: do not invent an empty label set."""
    monkeypatch.setattr(s, "DIARIZE_WORKER", worker("pass"))
    with pytest.raises(RuntimeError, match="produced no result"):
        s.run_diarizer("meeting.wav", 0, FAKE_MODELS)


def test_a_wedged_child_is_killed_rather_than_waited_on(monkeypatch):
    """A child that stops emitting progress must not hang the job forever."""
    monkeypatch.setattr(s, "DIARIZE_STALL_SECONDS", 0.5)
    monkeypatch.setattr(s, "DIARIZE_WORKER", worker("time.sleep(60)"))
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="stopped responding"):
        s.run_diarizer("meeting.wav", 0, FAKE_MODELS)
    assert time.perf_counter() - started < 10, "the child was not killed"


def test_cancelling_stops_the_pass_and_returns_nothing(monkeypatch):
    monkeypatch.setattr(
        s,
        "DIARIZE_WORKER",
        worker(
            "for _ in range(200):\n"
            "    time.sleep(0.05)\n"
            '    emit({"t": "progress", "v": 1})'
        ),
    )
    cancelled = threading.Event()
    threading.Timer(0.3, cancelled.set).start()

    started = time.perf_counter()
    result = s.run_diarizer(
        "meeting.wav", 0, FAKE_MODELS, cancelled=cancelled.is_set
    )
    assert result is None
    assert time.perf_counter() - started < 5, "cancel did not stop the child"


def test_a_cancelled_pass_never_starts_a_child(monkeypatch):
    monkeypatch.setattr(s, "DIARIZE_WORKER", worker("raise SystemExit(3)"))
    assert s.run_diarizer("meeting.wav", 0, FAKE_MODELS, cancelled=lambda: True) is None


# --------------------------------------------------------------------------- #
# Model fetching
# --------------------------------------------------------------------------- #


def test_a_corrupt_download_is_rejected_and_leaves_nothing_behind(
    configured, monkeypatch
):
    """A truncated .onnx fails deep inside ONNX Runtime with nothing useful to
    say, which is why the hash is checked at download time instead."""

    def wrong_bytes(url: str, dest: Path) -> None:
        # A real tarball with the right member name and the wrong contents, so
        # the failure under test is the checksum and not an unreadable archive.
        member = str(s.DIARIZE_MODELS["segmentation"]["member"])
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
            payload = b"not an onnx file"
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        dest.write_bytes(buffer.getvalue())

    monkeypatch.setattr(s, "_download", wrong_bytes)
    with pytest.raises(RuntimeError, match="(?i)checksum"):
        s.ensure_diarize_models()

    directory = s.diarize_model_dir()
    leftovers = [p.name for p in directory.iterdir()] if directory.exists() else []
    assert leftovers == [], f"a failed fetch left {leftovers} in the model dir"


def test_a_failed_fetch_is_reported_with_the_way_out(configured, monkeypatch):
    def boom(url: str, dest: Path) -> None:
        raise OSError("no route to host")

    monkeypatch.setattr(s, "_download", boom)
    with pytest.raises(RuntimeError) as caught:
        s.fetch_diarize_models_or_explain()
    message = str(caught.value)
    assert "no route to host" in message
    assert "--no-diarize" in message, "the error should say what to do about it"


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


def test_a_segment_spanning_two_speakers_is_split_at_the_change():
    """The case that makes this feature worth having: one Whisper segment, two
    people, and a boundary that has to land between the right words."""
    turns = [turn(0.0, 2.0, 1), turn(2.0, 4.0, 2)]
    segments = [
        segment(
            0.0,
            4.0,
            "Hello there how are you",
            words=words(
                (0.0, 0.5, " Hello"), (0.5, 1.0, " there"),
                (2.0, 2.5, " how"), (2.5, 3.0, " are"), (3.0, 3.5, " you"),
            ),
        )
    ]

    out = s.align_speakers(segments, turns)

    assert [entry["speaker"] for entry in out] == [1, 2]
    # Spacing survives: faster-whisper's word tokens carry their own leading
    # space, so joining a run reproduces the original text.
    assert [entry["text"] for entry in out] == ["Hello there", "how are you"]
    assert (out[0]["start"], out[0]["end"]) == (0.0, 1.0)
    assert (out[1]["start"], out[1]["end"]) == (2.0, 3.5)


def test_a_segment_wholly_inside_one_turn_is_left_alone():
    turns = [turn(0.0, 5.0, 1), turn(5.0, 9.0, 2)]
    segments = [
        segment(1.0, 2.0, "just me", words=words((1.0, 2.0, " just me"))),
        segment(5.5, 6.0, "and me", words=words((5.5, 6.0, " and me"))),
    ]

    out = s.align_speakers(segments, turns)

    assert [(entry["text"], entry["speaker"]) for entry in out] == [
        ("just me", 1),
        ("and me", 2),
    ]


def test_without_word_timings_a_segment_goes_to_its_dominant_speaker():
    """The degraded path: still labelled, just not split. It must not drop text."""
    turns = [turn(0.0, 1.0, 1), turn(1.0, 10.0, 2)]
    segments = [segment(0.0, 10.0, "mostly the second person")]

    out = s.align_speakers(segments, turns)

    assert len(out) == 1
    assert out[0]["speaker"] == 2
    assert out[0]["text"] == "mostly the second person"


def test_a_word_in_a_gap_takes_the_nearest_turn_rather_than_no_speaker():
    """Silence the diarizer dropped should not create a phantom speaker."""
    turns = [turn(0.0, 1.0, 1), turn(5.0, 6.0, 2)]
    segments = [
        segment(0.0, 6.0, "before gap after", words=words((2.5, 2.6, " gap"))),
    ]

    out = s.align_speakers(segments, turns)

    assert [entry["speaker"] for entry in out] == [1]
    assert out[0]["text"] == "gap"


def test_no_turns_leaves_the_transcript_exactly_as_it_was():
    """A job that was not diarized must not gain a speaker key at all."""
    segments = [segment(0.0, 1.0, "plain")]

    assert s.align_speakers(segments, []) == segments
    assert "speaker" not in s.align_speakers(segments, [])[0]


def test_alignment_survives_turns_arriving_out_of_order():
    turns = [turn(4.0, 6.0, 9), turn(0.0, 2.0, 4), turn(2.0, 4.0, 9)]
    segments = [
        segment(0.0, 6.0, "a b", words=words((0.5, 1.0, " a"), (3.0, 3.5, " b"))),
    ]

    out = s.align_speakers(segments, turns)

    # align_speakers works in whatever ids it is handed — relabelling is the
    # diarizer's job, on the way out of run_diarizer. What matters here is that
    # an unsorted list still maps by time: 0.5s is in turn 0-2, 3.0s in 2-4.
    assert [entry["speaker"] for entry in out] == [4, 9]


def test_relabelling_is_stable_and_one_based():
    turns = [turn(10.0, 12.0, 55), turn(0.0, 2.0, 8), turn(20.0, 22.0, 55)]

    out = s.relabel_speakers(turns)

    assert [entry["speaker"] for entry in out] == [1, 2, 2]
    assert [entry["start"] for entry in out] == [0.0, 10.0, 20.0]


def test_speaker_summary_totals_time_per_speaker():
    segs = [
        {**segment(0.0, 2.0, "one"), "speaker": 1},
        {**segment(2.0, 3.0, "two"), "speaker": 2},
        {**segment(3.0, 5.0, "more"), "speaker": 1},
    ]

    assert s.speaker_summary(segs) == [
        {"speaker": 1, "label": "Speaker 1", "segments": 2, "seconds": 4.0},
        {"speaker": 2, "label": "Speaker 2", "segments": 1, "seconds": 1.0},
    ]
    assert s.speaker_summary([segment(0.0, 1.0, "plain")]) == []


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #


def test_asking_for_speakers_turns_on_word_timings(configured):
    opts = s.build_opts(
        None, None, "", "true", "balanced", "", "", "false", "false", "false",
        2000, 400, "true", 0,
    )
    assert opts["diarize"] is True
    assert opts["word_timestamps"] is True


def test_word_timings_are_left_alone_when_speakers_are_not_wanted(configured):
    opts = s.build_opts(
        None, None, "", "true", "balanced", "", "", "false", "false", "false",
        2000, 400, "false", 3,
    )
    assert opts["diarize"] is False
    assert opts["word_timestamps"] is False
    # Nobody asked, so no stale speaker count is carried into the export.
    assert opts["speakers"] == 0


def test_the_speaker_count_is_clamped(configured):
    high = s.build_opts(
        None, None, "", "true", "balanced", "", "", "false", "false", "false",
        2000, 400, "true", 9000,
    )
    low = s.build_opts(
        None, None, "", "true", "balanced", "", "", "false", "false", "false",
        2000, 400, "true", -5,
    )
    assert high["speakers"] == s.DIARIZE_MAX_SPEAKERS
    assert low["speakers"] == 0


def test_no_diarize_pins_the_feature_off(configured, monkeypatch):
    monkeypatch.setattr(configured_args(), "allow_diarize", False)
    opts = s.build_opts(
        None, None, "", "true", "balanced", "", "", "false", "false", "true",
        2000, 400, "true", 4,
    )
    assert opts["diarize"] is False
    assert opts["speakers"] == 0
    # The operator pinned the feature off, not word timings.
    assert opts["word_timestamps"] is True


def configured_args() -> Any:
    """The module's live ARGS, so a test can flip one field on it."""
    return s.ARGS


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #


def rendered(job: dict[str, Any], fmt: str) -> str:
    body, _ = s.render(job, fmt)
    return body


def labeled_job(**opts: Any) -> dict[str, Any]:
    return {
        "filename": "standup.m4a",
        "language": "en",
        "duration": 5.0,
        "opts": {**s.build_opts(
            "base", "int8", "", "true", "balanced", "", "", "false", "false",
            "true", 2000, 400, "true", 0,
        ), **opts},
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "morning", "speaker": 1},
            {"start": 2.0, "end": 3.5, "text": "morning", "speaker": 2},
            {"start": 3.5, "end": 5.0, "text": "shall we start", "speaker": 1},
        ],
    }


def plain_job() -> dict[str, Any]:
    return {
        "filename": "standup.m4a",
        "language": "en",
        "duration": 3.0,
        "opts": s.build_opts(
            "base", "int8", "", "true", "balanced", "", "", "false", "false",
            "false", 2000, 400,
        ),
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "plain one"},
            {"start": 1.5, "end": 3.0, "text": "plain two"},
        ],
    }


def test_txt_labels_each_line(configured):
    assert rendered(labeled_job(), "txt").splitlines() == [
        "Speaker 1: morning",
        "Speaker 2: morning",
        "Speaker 1: shall we start",
    ]


def test_timestamped_keeps_the_clock_and_the_label(configured):
    assert rendered(labeled_job(), "timestamped").splitlines() == [
        "[00:00:00.000] Speaker 1: morning",
        "[00:00:02.000] Speaker 2: morning",
        "[00:00:03.500] Speaker 1: shall we start",
    ]


def test_srt_and_vtt_carry_the_speaker(configured):
    srt = rendered(labeled_job(), "srt")
    assert "Speaker 2: morning" in srt
    assert "00:00:02,000 --> 00:00:03,500" in srt

    vtt = rendered(labeled_job(), "vtt")
    # The voice span is the native way to say this, and players style it.
    assert "<v Speaker 2>morning" in vtt


def test_json_gains_a_speaker_per_segment_and_a_totals_block(configured):
    payload = json.loads(rendered(labeled_job(), "json"))
    assert [seg["speaker"] for seg in payload["segments"]] == [1, 2, 1]
    assert payload["speakers"] == [
        {"speaker": 1, "label": "Speaker 1", "segments": 2, "seconds": 3.5},
        {"speaker": 2, "label": "Speaker 2", "segments": 1, "seconds": 1.5},
    ]
    # The audit-token boundary is unchanged by any of this.
    assert "prompt" not in payload["options"]
    assert payload["options"]["has_prompt"] is False


def test_a_job_without_labels_exports_exactly_as_before(configured):
    """No speaker key, no totals block, no prefix — the plain path stays plain."""
    payload = json.loads(rendered(plain_job(), "json"))
    assert "speakers" not in payload
    assert all("speaker" not in seg for seg in payload["segments"])

    assert rendered(plain_job(), "txt").splitlines() == ["plain one", "plain two"]
    assert rendered(plain_job(), "timestamped").splitlines() == [
        "[00:00:00.000] plain one",
        "[00:00:01.500] plain two",
    ]
    assert rendered(plain_job(), "vtt") == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nplain one\n\n"
        "00:00:01.500 --> 00:00:03.000\nplain two\n"
    )


def test_a_failed_diarization_still_exports_the_transcript(configured):
    """Alignment is skipped when the pass fails, so the segments keep no
    speaker and the export falls back to the plain shape."""
    job = plain_job()
    assert job["segments"] == s.align_speakers(job["segments"], [])
    assert rendered(job, "txt").splitlines() == ["plain one", "plain two"]


# --------------------------------------------------------------------------- #
# The --preload probe
# --------------------------------------------------------------------------- #


def preload_args(allow_diarize: bool) -> Namespace:
    return Namespace(
        model="base", device="cpu", compute_type="int8", allow_diarize=allow_diarize
    )


def test_preload_proves_diarization_when_it_is_offered(configured, monkeypatch):
    """--preload's contract is that a green run means a job will genuinely work,
    so it has to exercise the models too, not just the Whisper one."""
    calls: list[str] = []
    monkeypatch.setattr(s, "load_model", lambda *a: object())
    monkeypatch.setattr(s, "verify_device", lambda *a: None)
    monkeypatch.setattr(s, "probe_diarization", lambda: calls.append("probe"))

    s.preload_model(preload_args(allow_diarize=True))
    assert calls == ["probe"]

    s.preload_model(preload_args(allow_diarize=False))
    assert calls == ["probe"], "--no-diarize must not fetch 42 MB of models"


def test_preload_refuses_to_offer_a_control_that_cannot_work(
    configured, monkeypatch, capsys
):
    """Same reasoning as the CUDA check: exiting 1 with advice beats serving a
    checkbox that fails on every job."""
    monkeypatch.setattr(s, "load_model", lambda *a: object())
    monkeypatch.setattr(s, "verify_device", lambda *a: None)

    def boom() -> None:
        raise RuntimeError("no route to host")

    monkeypatch.setattr(s, "probe_diarization", boom)

    with pytest.raises(SystemExit) as caught:
        s.preload_model(preload_args(allow_diarize=True))
    assert caught.value.code == 1
    assert "--no-diarize" in capsys.readouterr().out


def test_preload_still_fails_when_the_transcriber_cannot_encode(
    configured, monkeypatch
):
    """The original contract, pinned so the diarization branch cannot weaken it."""
    monkeypatch.setattr(s, "load_model", lambda *a: object())

    def cannot_encode(*_a: Any) -> None:
        raise RuntimeError("Library libcublas.so.12 is not found")

    monkeypatch.setattr(s, "verify_device", cannot_encode)
    monkeypatch.setattr(s, "probe_diarization", lambda: pytest.fail("unreachable"))

    with pytest.raises(SystemExit) as caught:
        s.preload_model(preload_args(allow_diarize=True))
    assert caught.value.code == 1


# --------------------------------------------------------------------------- #
# Against the real model, opt-in only
# --------------------------------------------------------------------------- #

REAL_MODELS = os.environ.get("TRANSCRIBE_DIARIZE_MODELS")

needs_real_models = pytest.mark.skipif(
    not REAL_MODELS or importlib.util.find_spec("sherpa_onnx") is None,
    reason="set TRANSCRIBE_DIARIZE_MODELS to a dir with the two .onnx files",
)


@needs_real_models
def test_the_real_diarizer_finds_the_expected_speakers(tmp_path):
    """The only test that loads a model. Point it at a real recording whose
    speaker count you know: it is the check that the plumbing still feeds the
    model what it expects, which every stub above deliberately cannot see."""
    models = {
        "segmentation": Path(REAL_MODELS) / "segmentation.int8.onnx",  # type: ignore[arg-type]
        "embedding": Path(REAL_MODELS) / "embedding.onnx",  # type: ignore[arg-type]
    }
    if not all(path.is_file() for path in models.values()):
        pytest.skip("the model files are not in TRANSCRIBE_DIARIZE_MODELS")
    audio = os.environ.get("TRANSCRIBE_DIARIZE_AUDIO")
    if not audio:
        pytest.skip("set TRANSCRIBE_DIARIZE_AUDIO to a 16 kHz mono wav")

    expected = int(os.environ.get("TRANSCRIBE_DIARIZE_SPEAKERS", "0"))
    turns = s.run_diarizer(audio, expected, models)

    assert turns, "the diarizer returned no turns for real audio"
    assert all(entry["start"] < entry["end"] for entry in turns)
    if expected:
        assert {entry["speaker"] for entry in turns} == set(range(1, expected + 1))
