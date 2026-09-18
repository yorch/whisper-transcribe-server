"""Relabelling: speaker labels again, with another count, without transcribing.

Auto speaker counting is the weak part of diarization, and on call audio it can
turn two people into five. Retry fixes that by running the whole job again; a
relabel runs only the speaker pass over the transcript the job already has, so
fixing a wrong count costs the diarization pass rather than the transcription.

The diarizer is stubbed throughout: these tests pin the job plumbing, not the
model, and must pass without sherpa-onnx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import transcribe_server as s


def words(*triples: tuple[float, float, str]) -> list[dict[str, Any]]:
    return [{"start": a, "end": b, "word": w} for a, b, w in triples]


# As Whisper produced them: one segment spanning both speakers, word-timed.
TRANSCRIBED = [
    {
        "start": 0.0,
        "end": 4.0,
        "text": "Hello there. Hi back.",
        "words": words(
            (0.0, 0.5, " Hello"),
            (0.5, 1.5, " there."),
            (2.5, 3.0, " Hi"),
            (3.0, 4.0, " back."),
        ),
    },
]

# What a two-speaker pass says about that audio.
TWO_TURNS = [
    {"start": 0.0, "end": 2.0, "speaker": 1},
    {"start": 2.0, "end": 4.0, "speaker": 2},
]


def finished_job(configured, **opts: str) -> str:
    """A done job with word timings, as a transcription would leave it."""
    job_id = configured.make_job(word_timestamps="true", **opts)
    s.patch_job(
        job_id,
        state="done",
        language="en",
        duration=4.0,
        segments=[dict(seg) for seg in TRANSCRIBED],
        finished=1.0,
    )
    return job_id


def no_model(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("a relabel must not load the transcription model")


# --------------------------------------------------------------------------- #
# The endpoint
# --------------------------------------------------------------------------- #


def test_relabel_queues_a_new_job_over_the_same_transcript(client, configured):
    old = finished_job(configured)

    response = client.post(f"/api/jobs/{old}/speakers", data={"speakers": "2"})

    assert response.status_code == 200, response.text
    new = response.json()["id"]
    assert new != old
    job = s.JOBS[new]
    assert job["relabel_of"] == old
    assert job["state"] == "queued"
    assert job["opts"]["diarize"] is True and job["opts"]["speakers"] == 2
    assert job["transcribed"] == TRANSCRIBED
    assert job["path"] == s.JOBS[old]["path"], "same audio, no re-upload"
    assert s.JOBS[old]["state"] == "done", "the original is left as it was"
    assert s.JOB_QUEUE.qsize() == 1  # make_job does not queue; this does
    relabelled = [e for e in configured.events() if e["event"] == "job.relabelled"]
    assert relabelled and relabelled[-1]["from_job"] == old
    assert relabelled[-1]["speakers"] == 2


def test_relabel_starts_from_the_transcript_before_the_last_split(client, configured):
    """A labelled job's lines were split where the last guess put speaker
    changes. A new count has to start from Whisper's lines, not those."""
    old = finished_job(configured)
    split = [dict(TRANSCRIBED[0], speaker=1), dict(TRANSCRIBED[0], speaker=2)]
    s.patch_job(old, segments=split, transcribed=[dict(t) for t in TRANSCRIBED])

    new = client.post(f"/api/jobs/{old}/speakers", data={"speakers": "2"}).json()

    assert s.JOBS[new["id"]]["transcribed"] == TRANSCRIBED


def test_only_a_finished_job_can_be_relabelled(client, configured):
    job_id = configured.make_job(word_timestamps="true")  # still queued

    response = client.post(f"/api/jobs/{job_id}/speakers", data={"speakers": "2"})

    assert response.status_code == 409
    assert "finished" in response.json()["detail"]


def test_a_transcript_without_word_timings_is_refused_with_the_way_out(
    client, configured
):
    """Labels are split at word boundaries; without them the only honest
    answer is Retry, which can ask for the timings."""
    job_id = configured.make_job()
    s.patch_job(job_id, state="done", segments=[dict(TRANSCRIBED[0], words=None)])

    response = client.post(f"/api/jobs/{job_id}/speakers", data={"speakers": "2"})

    assert response.status_code == 409
    assert "Retry" in response.json()["detail"]


def test_relabel_needs_the_audio(client, configured):
    job_id = finished_job(configured)
    Path(s.JOBS[job_id]["path"]).unlink()

    response = client.post(f"/api/jobs/{job_id}/speakers", data={"speakers": "2"})

    assert response.status_code == 409
    assert "no longer on disk" in response.json()["detail"]


def test_relabel_is_refused_when_the_server_does_not_offer_speakers(
    client, configured, monkeypatch
):
    job_id = finished_job(configured)
    monkeypatch.setattr(s.ARGS, "allow_diarize", False)

    response = client.post(f"/api/jobs/{job_id}/speakers", data={"speakers": "2"})

    assert response.status_code == 409


def test_the_page_is_told_which_jobs_can_be_relabelled(client, configured):
    ready = finished_job(configured)
    untimed = configured.make_job(name="plain.wav")
    s.patch_job(untimed, state="done", segments=[])

    listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}

    assert listed[ready]["can_relabel"] is True
    assert listed[untimed]["can_relabel"] is False


def test_the_kept_transcript_never_rides_along_in_the_payloads(client, configured):
    """It is a second copy of the transcript: the list is polled every 1.2 s."""
    job_id = finished_job(configured)
    s.patch_job(job_id, transcribed=[dict(t) for t in TRANSCRIBED])

    listed = client.get("/api/jobs").json()["jobs"][0]
    detail = client.get(f"/api/jobs/{job_id}").json()

    assert "transcribed" not in listed and "transcribed" not in detail


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #


def relabel(client, configured, speakers: int = 2) -> str:
    old = finished_job(configured)
    return client.post(
        f"/api/jobs/{old}/speakers", data={"speakers": str(speakers)}
    ).json()["id"]


def test_a_relabel_runs_only_the_speaker_pass(client, configured, monkeypatch):
    monkeypatch.setattr(s, "load_model", no_model)
    asked: list[int] = []

    def diarize(job_id: str, audio: str, opts: dict[str, Any]):
        asked.append(opts["speakers"])
        return TWO_TURNS

    monkeypatch.setattr(s, "diarize_job", diarize)
    job_id = relabel(client, configured)

    s.run_job(job_id)

    job = s.JOBS[job_id]
    assert asked == [2], "the pass is asked for the count the operator gave"
    assert job["state"] == "done"
    assert [(seg["speaker"], seg["text"]) for seg in job["segments"]] == [
        (1, "Hello there."),
        (2, "Hi back."),
    ]
    assert job["language"] == "en" and job["duration"] == 4.0
    assert job["transcribed"] == TRANSCRIBED, "kept, so it can be relabelled again"


def test_a_failed_relabel_still_finishes_with_the_transcript(
    client, configured, monkeypatch
):
    """The invariant from the first pass holds here too: labels are worth less
    than the transcript, so a diarizer failure never fails the job."""
    monkeypatch.setattr(s, "load_model", no_model)

    def broken(*_a: Any) -> None:
        raise RuntimeError("the diarizer exited with 1")

    monkeypatch.setattr(s, "diarize_job", broken)
    job_id = relabel(client, configured)

    s.run_job(job_id)

    job = s.JOBS[job_id]
    assert job["state"] == "done"
    assert "no speaker labels" in job["message"]
    assert [seg["text"] for seg in job["segments"]] == ["Hello there. Hi back."]


def test_a_relabel_cancelled_mid_pass_stays_cancelled(client, configured, monkeypatch):
    monkeypatch.setattr(s, "load_model", no_model)
    job_id = relabel(client, configured)

    def cancel_then_stop(job: str, *_a: Any) -> None:
        s.patch_job(job, state="cancelled", message="Cancelled")
        return None  # what diarize_job returns for a cancelled pass

    monkeypatch.setattr(s, "diarize_job", cancel_then_stop)

    s.run_job(job_id)

    assert s.JOBS[job_id]["state"] == "cancelled"
