"""Naming speakers: "Speaker 1" becomes "Alice" on the card and in exports.

A list of names is a list of who was in the room -- the same category of data
as prompt and hotword text, which the audit trail never logs in clear. So a
name is visible to the app (whoever named the speaker reads the transcript
anyway), recorded in the main log only as a length and a hash, and kept in
clear only in the job's audit sidecar, which the audit token alone can read.
"""

from __future__ import annotations

import json
from typing import Any

import transcribe_server as s

NAME = "Alice Nightjar"


def seg(start: float, end: float, speaker: int, text: str) -> dict[str, Any]:
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def labelled_job(configured, **opts: str) -> str:
    job_id = configured.make_job(diarize="true", **opts)
    s.patch_job(
        job_id,
        state="done",
        language="en",
        duration=2.0,
        segments=[seg(0, 1, 1, "hi"), seg(1, 2, 2, "yo")],
    )
    return job_id


def name(client, job_id: str, speaker: int, value: str):
    return client.post(
        f"/api/jobs/{job_id}/speakers/name",
        data={"speaker": str(speaker), "name": value},
    )


def test_a_name_replaces_the_label_in_every_export(client, configured):
    job_id = labelled_job(configured)

    assert name(client, job_id, 1, NAME).status_code == 200

    txt = client.get(f"/api/jobs/{job_id}/text?format=txt").text
    vtt = client.get(f"/api/jobs/{job_id}/text?format=vtt").text
    data = client.get(f"/api/jobs/{job_id}/text?format=json").json()
    assert txt == f"{NAME}: hi\nSpeaker 2: yo\n"
    assert f"<v {NAME}>hi" in vtt
    assert [x["label"] for x in data["speakers"]] == [NAME, "Speaker 2"]


def test_the_card_gets_the_names(client, configured):
    job_id = labelled_job(configured)
    name(client, job_id, 2, "Bob")

    detail = client.get(f"/api/jobs/{job_id}").json()

    assert detail["speaker_names"] == {"2": "Bob"}
    assert [x["label"] for x in detail["speakers"]] == ["Speaker 1", "Bob"]


def test_the_main_log_records_a_hash_never_the_name(client, configured):
    job_id = labelled_job(configured)

    name(client, job_id, 1, NAME)

    events = configured.events()
    assert NAME not in json.dumps(events), "a name reached the main audit log"
    named = [e for e in events if e["event"] == "job.speaker_named"][-1]
    assert named["name_len"] == len(NAME)
    assert named["name_sha256"] == s.digest(NAME)


def test_the_sidecar_holds_the_names_for_the_audit_token_only(client, configured):
    job_id = labelled_job(configured)
    name(client, job_id, 1, NAME)

    assert configured.audit.read_prompt(job_id)["speaker_names"] == {"1": NAME}
    assert client.get(f"/api/audit/prompts/{job_id}").status_code == 401
    audited = client.get(
        f"/api/audit/prompts/{job_id}",
        headers={"x-audit-token": configured.audit_token},
    )
    assert audited.json()["speaker_names"] == {"1": NAME}


def test_naming_keeps_the_prompt_already_in_the_sidecar(client, configured):
    job_id = labelled_job(configured, prompt="A call about Nightjar.")
    s.store_prompt_sidecar(
        job_id, "meeting.wav", s.JOBS[job_id]["opts"], source="upload"
    )

    name(client, job_id, 1, NAME)

    side = configured.audit.read_prompt(job_id)
    assert side["prompt"] == "A call about Nightjar."
    assert side["speaker_names"] == {"1": NAME}


def test_no_audit_prompts_keeps_names_out_of_the_sidecar_too(
    client, configured, monkeypatch
):
    monkeypatch.setattr(configured.audit, "store_prompts", False)
    job_id = labelled_job(configured)

    assert name(client, job_id, 1, NAME).status_code == 200

    assert configured.audit.read_prompt(job_id) is None
    assert s.JOBS[job_id]["speaker_names"] == {"1": NAME}, "naming still works"


def test_an_empty_name_puts_the_label_back(client, configured):
    job_id = labelled_job(configured)
    name(client, job_id, 1, NAME)

    name(client, job_id, 1, "   ")

    assert s.JOBS[job_id]["speaker_names"] == {}
    assert client.get(f"/api/jobs/{job_id}/text?format=txt").text.startswith(
        "Speaker 1: hi"
    )


def test_a_name_bumps_the_revision_the_page_watches(client, configured):
    job_id = labelled_job(configured)

    name(client, job_id, 1, NAME)

    assert s.JOBS[job_id]["labels_rev"] == 1


def test_names_that_would_break_an_export_are_refused(client, configured):
    job_id = labelled_job(configured)

    for bad in ("<b>Alice</b>", "Alice\nBob", "x" * 61):
        assert name(client, job_id, 1, bad).status_code == 400, bad
    assert name(client, job_id, 9, "Alice").status_code == 400
    assert "speaker_names" not in s.JOBS[job_id] or not s.JOBS[job_id]["speaker_names"]


def test_only_a_finished_job_can_be_named(client, configured):
    job_id = configured.make_job(diarize="true")
    s.patch_job(job_id, state="running", segments=[seg(0, 1, 1, "hi")])

    assert name(client, job_id, 1, "Alice").status_code == 409


def test_a_merge_carries_the_names_across(client, configured):
    job_id = configured.make_job(diarize="true")
    s.patch_job(
        job_id,
        state="done",
        segments=[seg(0, 1, 1, "a"), seg(1, 2, 2, "b"), seg(2, 3, 3, "c")],
    )
    name(client, job_id, 2, "Bob")
    name(client, job_id, 3, "Carol")

    # 2 into 1: Speaker 1 has no name, so it takes Bob's; Carol becomes 2.
    client.post(
        f"/api/jobs/{job_id}/speakers/merge", data={"speaker": "2", "into": "1"}
    )

    assert s.JOBS[job_id]["speaker_names"] == {"1": "Bob", "2": "Carol"}
    assert configured.audit.read_prompt(job_id)["speaker_names"] == {
        "1": "Bob",
        "2": "Carol",
    }
