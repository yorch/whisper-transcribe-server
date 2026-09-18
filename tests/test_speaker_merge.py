"""Merging speakers: "Speaker 3 is really Speaker 1", fixed on the card.

A merge only moves numbers -- no name, no audio, nothing sensitive -- so it
edits the finished job in place: its segments, its per-speaker totals and
every export follow. Speakers are renumbered afterwards so the labels stay
1..N with no gap.
"""

from __future__ import annotations

from typing import Any

import transcribe_server as s


def seg(start: float, end: float, speaker: int, text: str = "x") -> dict[str, Any]:
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def labelled_job(configured, segments: list[dict[str, Any]]) -> str:
    job_id = configured.make_job(diarize="true")
    s.patch_job(job_id, state="done", segments=segments)
    return job_id


FIVE = [seg(0, 10, 1), seg(10, 11, 2), seg(11, 20, 3), seg(20, 21, 4), seg(21, 30, 5)]


def merge(client, job_id: str, speaker: int, into: int):
    return client.post(
        f"/api/jobs/{job_id}/speakers/merge",
        data={"speaker": str(speaker), "into": str(into)},
    )


def test_a_merge_moves_every_line_and_renumbers(client, configured):
    job_id = labelled_job(configured, [dict(x) for x in FIVE])

    response = merge(client, job_id, speaker=2, into=1)

    assert response.status_code == 200, response.text
    assert [x["speaker"] for x in s.JOBS[job_id]["segments"]] == [1, 1, 2, 3, 4]
    # What each surviving old number became, so the page can follow it.
    assert response.json()["renumbered"] == {"1": 1, "3": 2, "4": 3, "5": 4}


def test_merges_add_up_to_the_count_you_meant(client, configured):
    """The Zoom case: five labels, two people."""
    job_id = labelled_job(configured, [dict(x) for x in FIVE])

    for speaker, into in ((2, 1), (3, 2), (3, 1)):  # numbers shift after each
        assert merge(client, job_id, speaker, into).status_code == 200

    assert {x["speaker"] for x in s.JOBS[job_id]["segments"]} == {1, 2}


def test_a_merge_changes_what_the_exports_say(client, configured):
    job_id = labelled_job(configured, [seg(0, 1, 1, "hi"), seg(1, 2, 2, "yo")])

    merge(client, job_id, speaker=2, into=1)
    text = client.get(f"/api/jobs/{job_id}/text?format=txt").text

    assert text == "Speaker 1: hi\nSpeaker 1: yo\n"


def test_a_merge_bumps_the_revision_the_page_watches(client, configured):
    """A merge changes no segment count and no progress, so without a counter
    in the list payload an open card would never notice it."""
    job_id = labelled_job(configured, [dict(x) for x in FIVE])
    before = client.get("/api/jobs").json()["jobs"][0].get("labels_rev", 0)

    merge(client, job_id, speaker=5, into=1)

    assert client.get("/api/jobs").json()["jobs"][0]["labels_rev"] == before + 1


def test_the_detail_carries_the_per_speaker_totals(client, configured):
    job_id = labelled_job(configured, [seg(0, 10, 1), seg(10, 12, 2)])

    speakers = client.get(f"/api/jobs/{job_id}?since=2").json()["speakers"]

    assert [(x["speaker"], x["seconds"]) for x in speakers] == [(1, 10.0), (2, 2.0)]


def test_a_merge_is_audited_by_number_only(client, configured):
    job_id = labelled_job(configured, [dict(x) for x in FIVE])

    merge(client, job_id, speaker=4, into=2)

    merged = [e for e in configured.events() if e["event"] == "job.speakers_merged"]
    assert merged and (merged[-1]["speaker"], merged[-1]["into"]) == (4, 2)


def test_nonsense_merges_are_refused(client, configured):
    job_id = labelled_job(configured, [seg(0, 1, 1), seg(1, 2, 2)])

    assert merge(client, job_id, speaker=1, into=1).status_code == 400
    assert merge(client, job_id, speaker=3, into=1).status_code == 400
    running = configured.make_job(diarize="true")
    s.patch_job(running, state="running", segments=[seg(0, 1, 1), seg(1, 2, 2)])
    assert merge(client, running, speaker=2, into=1).status_code == 409
    assert [x["speaker"] for x in s.JOBS[job_id]["segments"]] == [1, 2], (
        "a refused merge changes nothing"
    )
