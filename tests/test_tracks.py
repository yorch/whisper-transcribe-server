"""One call, one track per person: Zoom's "separate audio file per participant".

Each file is one person, so who spoke is known exactly and no diarization is
needed: every track is transcribed with its own speaker number, the results
are merged by time, and the speakers are named from Zoom's file names. The
names are speaker names, so they follow the same rule: the main audit log
records hashes, the sidecar holds the text.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import transcribe_server as s

AUDIO = b"RIFF....WAVEfmt "


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("audioJaneDoe11234567890.m4a", "Jane Doe"),
        ("audioBob21234567890.m4a", "Bob"),
        ("audioMaría José31234567890.m4a", "María José"),
        ("audio_Jane_Doe.m4a", "Jane Doe"),
        ("interview-host.wav", "interview host"),
        ("audio12345.m4a", "Track 2"),
        ("<b>.m4a", "Track 2"),
    ],
)
def test_a_speaker_is_named_after_their_track(filename, expected):
    assert s.track_speaker_name(filename, 2) == expected


def upload_tracks(client, *names: str, **form: str):
    return client.post(
        "/api/jobs/tracks",
        files=[("files", (name, AUDIO, "audio/mp4")) for name in names],
        data={"model": "base", **form},
    )


def test_a_call_is_one_job_with_a_named_speaker_per_track(client, configured):
    response = upload_tracks(
        client, "audioJaneDoe11234.m4a", "audioBobSmith21234.m4a", diarize="true"
    )

    assert response.status_code == 200, response.text
    job = s.JOBS[response.json()["id"]]
    assert [Path(t["path"]).exists() for t in job["tracks"]] == [True, True]
    assert job["speaker_names"] == {"1": "Jane Doe", "2": "Bob Smith"}
    assert job["opts"]["diarize"] is False, "the tracks already say who spoke"
    assert job["filename"] == "Call · 2 tracks"
    assert s.JOB_QUEUE.qsize() == 1


def test_the_names_never_reach_the_main_audit_log(client, configured):
    job_id = upload_tracks(client, "audioJaneDoe1.m4a", "audioBobSmith2.m4a").json()[
        "id"
    ]

    trail = json.dumps(configured.events())
    assert "Jane" not in trail and "Bob" not in trail
    assert configured.audit.read_prompt(job_id)["speaker_names"] == {
        "1": "Jane Doe",
        "2": "Bob Smith",
    }


def test_a_call_needs_two_to_ten_tracks(client, configured):
    assert upload_tracks(client, "a.m4a").status_code == 400
    many = [f"t{i}.m4a" for i in range(s.MAX_TRACKS + 1)]
    assert upload_tracks(client, *many).status_code == 400
    assert not any(s.UPLOAD_DIR.iterdir()), "a refused call leaves no files behind"


def test_the_page_sees_a_track_count_not_server_paths(client, configured):
    upload_tracks(client, "a.m4a", "b.m4a")

    listed = client.get("/api/jobs").json()["jobs"][0]

    assert listed["track_count"] == 2
    assert "tracks" not in listed
    assert str(s.UPLOAD_DIR) not in json.dumps(listed)


class FakeModel:
    """Two tracks, interleaved in time: Jane at 0 and 10 s, Bob at 5 s."""

    SCRIPT: dict[str, list[tuple[float, float, str]]] = {
        "jane": [(0.0, 4.0, "Hi Bob."), (10.0, 12.0, "Great.")],
        "bob": [(5.0, 9.0, "Hi Jane, all good.")],
    }

    def transcribe(self, path: str, **_kw: Any):
        who = "jane" if "Jane" in Path(path).name else "bob"
        segments = [
            SimpleNamespace(start=a, end=b, text=f" {t}", words=None)
            for a, b, t in self.SCRIPT[who]
        ]
        return iter(segments), SimpleNamespace(language="en", duration=12.0)


def test_the_tracks_are_merged_by_time_under_their_speakers(
    client, configured, monkeypatch
):
    monkeypatch.setattr(s, "load_model", lambda *_a: FakeModel())
    monkeypatch.setattr(
        s, "diarize_job", lambda *_a: pytest.fail("a call is never diarized")
    )
    job_id = upload_tracks(client, "audioJane1.m4a", "audioBob2.m4a").json()["id"]

    s.run_job(job_id)

    job = s.JOBS[job_id]
    assert job["state"] == "done", job["message"]
    assert [(x["speaker"], x["text"]) for x in job["segments"]] == [
        (1, "Hi Bob."),
        (2, "Hi Jane, all good."),
        (1, "Great."),
    ]
    assert job["labels_rev"] >= 1, "the final sort must make open cards refetch"
    assert job["duration"] == 12.0 and job["language"] == "en"


def test_a_call_cannot_be_relabelled_but_can_be_retried(client, configured):
    old = upload_tracks(client, "audioJane1.m4a", "audioBob2.m4a").json()["id"]
    s.patch_job(old, state="done", segments=[])

    relabel = client.post(f"/api/jobs/{old}/speakers", data={"speakers": "2"})
    retry = client.post(f"/api/jobs/{old}/retry", data={"model": "base"})

    assert relabel.status_code == 409 and "tracks" in relabel.json()["detail"]
    assert retry.status_code == 200, retry.text
    new = s.JOBS[retry.json()["id"]]
    assert [t["path"] for t in new["tracks"]] == [
        t["path"] for t in s.JOBS[old]["tracks"]
    ]
    assert new["speaker_names"] == s.JOBS[old]["speaker_names"]


def test_retry_needs_every_track(client, configured):
    job_id = upload_tracks(client, "a.m4a", "b.m4a").json()["id"]
    s.patch_job(job_id, state="done", segments=[])
    Path(s.JOBS[job_id]["tracks"][1]["path"]).unlink()

    assert client.get("/api/jobs").json()["jobs"][0]["can_retry"] is False
    assert (
        client.post(f"/api/jobs/{job_id}/retry", data={"model": "base"}).status_code
        == 409
    )


def test_removing_a_call_removes_every_track(client, configured):
    job_id = upload_tracks(client, "a.m4a", "b.m4a").json()["id"]
    paths = [Path(t["path"]) for t in s.JOBS[job_id]["tracks"]]
    s.patch_job(job_id, state="done", segments=[])

    client.delete(f"/api/jobs/{job_id}")

    assert not any(p.exists() for p in paths)
