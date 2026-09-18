"""Folding minor speakers: the cheap guard against Auto over-counting.

On call audio the diarizer's Auto count splits one person into several: a
two-person Zoom call came back as five speakers, three of them holding a few
seconds each. Those slivers are folded into the voice nearest them in time --
only on Auto, because a count the operator pinned is taken literally.
"""

from __future__ import annotations

from typing import Any

import transcribe_server as s


def turn(start: float, end: float, speaker: int) -> dict[str, Any]:
    return {"start": start, "end": end, "speaker": speaker}


# Two real voices over 100 s, and a third that holds 1 s of it.
CALL = [
    turn(0, 30, 1),
    turn(30, 31, 3),  # a laugh, heard as someone new
    turn(31, 60, 2),
    turn(60, 100, 1),
]


def test_a_sliver_is_folded_into_the_voice_nearest_it():
    folded, count = s.fold_minor_speakers(CALL, 0.03)

    assert count == 1
    assert {t["speaker"] for t in folded} == {1, 2}
    # 30-31 touches both neighbours; the earlier one wins the tie, and either
    # way it is a voice that really spoke rather than a third person.
    assert folded[1]["speaker"] in (1, 2)


def test_the_nearest_voice_in_time_takes_the_sliver():
    turns = [turn(0, 40, 1), turn(50, 51, 3), turn(51.5, 90, 2)]

    folded, _ = s.fold_minor_speakers(turns, 0.03)

    assert folded[1]["speaker"] == folded[2]["speaker"], (
        "0.5 s from speaker 2's turn, 10 s from speaker 1's"
    )


def test_numbering_is_contiguous_after_a_fold():
    turns = [turn(0, 1, 1), turn(1, 50, 2), turn(50, 100, 3)]  # 1 is the sliver

    folded, count = s.fold_minor_speakers(turns, 0.03)

    assert count == 1
    assert sorted({t["speaker"] for t in folded}) == [1, 2]
    assert folded[0]["speaker"] == 1, "whoever speaks first is Speaker 1"


def test_speakers_above_the_share_are_left_alone():
    turns = [turn(0, 50, 1), turn(50, 60, 2), turn(60, 100, 3)]  # 10% is enough

    folded, count = s.fold_minor_speakers(turns, 0.03)

    assert count == 0
    assert folded == s.relabel_speakers(turns)


def test_a_zero_share_turns_folding_off():
    folded, count = s.fold_minor_speakers(CALL, 0.0)

    assert count == 0
    assert {t["speaker"] for t in folded} == {1, 2, 3}


def test_a_single_voice_is_never_folded_away():
    folded, count = s.fold_minor_speakers([turn(0, 5, 1)], 0.5)

    assert count == 0 and folded == [turn(0, 5, 1)]


def test_everyone_below_the_share_keeps_the_loudest():
    """A share so high that every speaker falls under it must still leave the
    transcript with the speaker who held the most of it."""
    turns = [turn(0, 10, 1), turn(10, 30, 2), turn(30, 45, 3)]

    folded, count = s.fold_minor_speakers(turns, 0.9)

    assert count == 2
    assert {t["speaker"] for t in folded} == {1}


# --------------------------------------------------------------------------- #
# Wired into the speaker pass
# --------------------------------------------------------------------------- #

WORDED = [
    {
        "start": 0.0,
        "end": 100.0,
        "text": "x",
        "words": [
            {"start": 0.0, "end": 29.0, "word": " one"},
            {"start": 30.1, "end": 30.9, "word": " ha"},
            {"start": 32.0, "end": 59.0, "word": " two"},
            {"start": 61.0, "end": 99.0, "word": " three"},
        ],
    }
]


def finish(configured, monkeypatch, speakers: int) -> dict[str, Any]:
    job_id = configured.make_job(diarize="true", speakers=speakers)
    monkeypatch.setattr(s, "diarize_job", lambda *_a: [dict(t) for t in CALL])
    s.label_and_finish(
        job_id, s.JOBS[job_id], s.JOBS[job_id]["opts"], WORDED, "en", 100.0
    )
    return s.JOBS[job_id]


def test_auto_folds_and_says_so(configured, monkeypatch):
    job = finish(configured, monkeypatch, speakers=0)

    assert {seg["speaker"] for seg in job["segments"]} == {1, 2}
    assert "1 minor voice folded" in job["message"]
    diarized = [e for e in configured.events() if e["event"] == "job.diarized"]
    assert diarized[-1]["folded"] == 1


def test_a_pinned_count_is_taken_literally(configured, monkeypatch):
    job = finish(configured, monkeypatch, speakers=3)

    assert {seg["speaker"] for seg in job["segments"]} == {1, 2, 3}
    assert "folded" not in job["message"]
