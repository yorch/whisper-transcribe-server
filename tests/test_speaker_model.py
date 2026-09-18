"""Choosing the speaker (voice) model per job.

The server-wide diarization_embedding is the default; a job may ask for any
model in DIARIZE_EMBEDDINGS, unless the operator pinned it with
embedding_choice = false -- the same shape as --pin-model for Whisper. Each
model has its own distance scale, so a job gets its model's calibrated
threshold; an explicit server threshold applies to the server's default model,
which is the one it was set for.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

import transcribe_server as s


def job_opts(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "model": "base",
        "compute_type": "int8",
        "language": "",
        "vad": "true",
        "quality": "balanced",
        "prompt": "",
        "hotwords": "",
        "translate": "false",
        "condition": "false",
        "word_timestamps": "false",
        "min_silence_ms": 2000,
        "speech_pad_ms": 400,
        "diarize": "false",
        "speakers": 0,
    }
    args.update(overrides)
    return s.build_opts(**args)


def test_a_job_can_ask_for_another_voice_model(configured):
    opts = job_opts(diarize="true", speaker_model="eres2net-en")

    assert opts["speaker_model"] == "eres2net-en"


def test_no_choice_means_the_server_default(configured):
    assert job_opts(diarize="true")["speaker_model"] == "titanet-small"


def test_an_unknown_voice_model_is_refused(configured):
    with pytest.raises(HTTPException) as exc:
        job_opts(diarize="true", speaker_model="cam++")

    assert exc.value.status_code == 400


def test_a_pinned_voice_model_ignores_the_client(configured, monkeypatch):
    monkeypatch.setattr(s.ARGS, "diarization_embedding_choice", False)

    opts = job_opts(diarize="true", speaker_model="eres2net-en")

    assert opts["speaker_model"] == "titanet-small"


def test_no_speaker_labels_means_no_voice_model(configured):
    assert (
        job_opts(diarize="false", speaker_model="eres2net-en")["speaker_model"] is None
    )


def test_each_model_gets_its_own_threshold(configured, monkeypatch):
    """0.8 is titanet-small's scale; eres2net-en's is 0.9. An operator's explicit
    threshold was set for the server's default model, so only that one uses it."""
    monkeypatch.setattr(s.ARGS, "diarization_threshold", 0.75)

    assert s.threshold_for("titanet-small") == 0.75
    assert s.threshold_for("eres2net-en") == 0.9


def test_the_job_fetches_and_uses_its_own_model(configured, monkeypatch):
    fetched: list[str | None] = []
    ran: list[float] = []

    def fetch(embedding: str | None = None) -> dict[str, Any]:
        fetched.append(embedding)
        return {}

    def diarize(*_a: Any, threshold: float, **_k: Any) -> list[dict[str, Any]]:
        ran.append(threshold)
        return []

    monkeypatch.setattr(s, "fetch_diarize_models_or_explain", fetch)
    monkeypatch.setattr(s, "run_diarizer", diarize)
    job_id = configured.make_job(diarize="true", speaker_model="eres2net-en")

    s.diarize_job(job_id, "audio.wav", s.JOBS[job_id]["opts"])

    assert fetched == ["eres2net-en"]
    assert ran == [0.9]


def test_the_status_offers_the_choice(client, configured):
    status = client.get("/api/status").json()

    assert status["speaker_models"] == ["titanet-small", "eres2net-en"]
    assert status["default_speaker_model"] == "titanet-small"
    assert status["allow_speaker_model_choice"] is True


def test_an_upload_carries_the_choice(client, configured):
    response = client.post(
        "/api/jobs",
        files={"file": ("a.wav", b"RIFF....WAVEfmt ", "audio/wav")},
        data={"model": "base", "diarize": "true", "speaker_model": "eres2net-en"},
    )

    assert response.status_code == 200, response.text
    assert s.JOBS[response.json()["id"]]["opts"]["speaker_model"] == "eres2net-en"


def test_a_relabel_can_switch_the_voice_model(client, configured):
    """The cheapest comparison there is: same transcript, the other model."""
    job_id = configured.make_job(word_timestamps="true", diarize="true")
    s.patch_job(
        job_id,
        state="done",
        segments=[
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hi",
                "words": [{"start": 0.0, "end": 1.0, "word": " hi"}],
            }
        ],
    )

    response = client.post(
        f"/api/jobs/{job_id}/speakers",
        data={"speakers": "2", "speaker_model": "eres2net-en"},
    )

    assert response.status_code == 200, response.text
    assert s.JOBS[response.json()["id"]]["opts"]["speaker_model"] == "eres2net-en"
