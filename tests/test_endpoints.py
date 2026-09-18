"""Endpoint-level tests, driven through the real ASGI app.

These exercise the middleware (host check, auth, body-size precheck, hardening
headers) and the handlers, which unit tests on the helpers cannot cover. The
worker thread is never started, so an upload stays queued and no model loads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

import transcribe_server as s

PROMPT_TEXT = "A call about Project Nightjar and the Acme merger."
TERMS_TEXT = "Nightjar,Acme"
AUDIO = b"RIFF....WAVEfmt "


def upload(client, name: str = "meeting.wav", **form):
    data = {"model": "base", **form}
    return client.post(
        "/api/jobs", files={"file": (name, AUDIO, "audio/wav")}, data=data
    )


# --------------------------------------------------------------------------- #
# The prompt-text leak (P1): app token must not reveal prompt/hotwords
# --------------------------------------------------------------------------- #


def test_upload_is_accepted_and_queued(client, configured):
    response = upload(client, prompt=PROMPT_TEXT, hotwords=TERMS_TEXT)
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]

    assert s.JOBS[job_id]["state"] == "queued"
    # The upload landed on disk, and the worker is not running in tests.
    assert s.JOB_QUEUE.qsize() == 1


def test_job_list_and_detail_never_expose_prompt_text(client, configured):
    job_id = upload(client, prompt=PROMPT_TEXT, hotwords=TERMS_TEXT).json()["id"]

    listing = client.get("/api/jobs")
    assert listing.status_code == 200
    assert PROMPT_TEXT not in listing.text
    assert TERMS_TEXT not in listing.text
    assert "Nightjar" not in listing.text

    detail = client.get(f"/api/jobs/{job_id}")
    assert detail.status_code == 200
    assert PROMPT_TEXT not in detail.text
    assert "Nightjar" not in detail.text

    opts = detail.json()["opts"]
    assert opts["has_prompt"] is True and opts["has_hotwords"] is True
    assert opts["prompt_len"] == len(PROMPT_TEXT)
    assert opts["model"] == "base"


def test_json_export_never_exposes_prompt_text(client, configured):
    job_id = upload(client, prompt=PROMPT_TEXT, hotwords=TERMS_TEXT).json()["id"]
    s.patch_job(job_id, state="done", segments=[])

    response = client.get(f"/api/jobs/{job_id}/text?format=json")
    assert response.status_code == 200
    assert PROMPT_TEXT not in response.text
    assert response.json()["options"]["has_prompt"] is True


def test_prompt_text_is_still_recoverable_with_the_audit_token(client, configured):
    """The text is not lost: it lives behind the audit credential."""
    job_id = upload(client, prompt=PROMPT_TEXT, hotwords=TERMS_TEXT).json()["id"]

    assert configured.audit.read_prompt(job_id)["prompt"] == PROMPT_TEXT

    response = client.get(
        "/api/audit/prompts/" + job_id,
        headers={"x-audit-token": configured.audit_token},
    )
    assert response.status_code == 200
    assert response.json()["prompt"] == PROMPT_TEXT


def test_audit_trail_records_only_the_hash(client, configured):
    upload(client, prompt=PROMPT_TEXT, hotwords=TERMS_TEXT)

    created = [e for e in configured.events() if e["event"] == "job.created"]
    assert created, "job.created should be recorded"
    record = created[-1]

    assert PROMPT_TEXT not in json.dumps(record)
    assert record["opts"]["prompt_sha256"] == s.digest(PROMPT_TEXT)
    assert record["opts"]["hotwords_sha256"] == s.digest(TERMS_TEXT)
    assert record["file"] == "meeting.wav"
    assert record["bytes"] == len(AUDIO)


# --------------------------------------------------------------------------- #
# Auth and host gating
# --------------------------------------------------------------------------- #


def test_api_requires_the_app_token(configured):
    with TestClient(s.app) as anon:
        assert anon.get("/api/status").status_code == 401
        assert anon.post("/api/jobs").status_code == 401


def test_audit_api_requires_its_own_token(client, configured):
    # The app token must not open the audit trail.
    assert client.get("/api/audit").status_code == 401
    assert client.get("/api/audit?limit=1").status_code == 401

    response = client.get(
        "/api/audit", headers={"x-audit-token": configured.audit_token}
    )
    assert response.status_code == 200
    assert "events" in response.json() and "dates" in response.json()


def test_audit_api_is_404_when_neither_a_token_nor_open_is_configured(client):
    """'off' is only reachable with ARGS built by hand: main() always leaves
    either a generated token or an explicit --audit-open."""
    saved = s.ARGS.audit_token
    s.ARGS.audit_token = ""
    try:
        assert client.get("/api/audit").status_code == 404
    finally:
        s.ARGS.audit_token = saved


def test_audit_open_serves_the_trail_and_the_prompt_text(client, configured):
    """--audit-open is the one deliberate way past the audit credential."""
    job_id = upload(client, prompt=PROMPT_TEXT, hotwords=TERMS_TEXT).json()["id"]
    s.ARGS.audit_open = True
    try:
        response = client.get("/api/audit", params={"include_prompts": 1})
        assert response.status_code == 200
        revealed = [
            e
            for e in response.json()["events"]
            if e.get("job") == job_id and e.get("prompt_text")
        ]
        # The exposure is the entire point of the flag, so assert it is real
        # rather than trusting that the gate quietly stopped being checked.
        assert revealed and revealed[-1]["prompt_text"] == PROMPT_TEXT
    finally:
        s.ARGS.audit_open = False


def test_audit_open_does_not_open_the_app_api(configured):
    """Opening the trail must not widen what an app client can reach."""
    s.ARGS.audit_open = True
    try:
        with TestClient(s.app) as anon:
            assert anon.get("/api/audit").status_code == 200
            assert anon.get("/api/jobs").status_code == 401
    finally:
        s.ARGS.audit_open = False


def test_unknown_host_is_refused_and_carries_security_headers(configured):
    with TestClient(s.app) as c:
        response = c.get("/api/status", headers={"Host": "evil.example"})

    assert response.status_code == 421
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "Content-Security-Policy" in response.headers


def test_refusals_are_audited_with_a_reason(configured):
    with TestClient(s.app) as c:
        c.get("/api/status", headers={"Host": "evil.example"})
        c.get("/api/status", headers={"x-token": "wrong"})

    events = configured.events()
    kinds = {e["event"] for e in events}
    assert "security.host_rejected" in kinds
    assert "security.auth_failed" in kinds

    auth = next(e for e in events if e["event"] == "security.auth_failed")
    assert auth["reason"] == "bad-token" and auth["status"] == 401


def test_cross_site_requests_are_refused(client, configured):
    response = client.get("/api/status", headers={"Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 403
    assert any(e["event"] == "security.cross_site" for e in configured.events())


# --------------------------------------------------------------------------- #
# Body-size precheck (P1: the cap must not be applied only after spooling)
# --------------------------------------------------------------------------- #


def test_oversized_body_is_refused_before_it_is_read(client, configured):
    s.ARGS.max_upload_mb = 1  # cap becomes 1 MB + 1 MB of multipart slack
    try:
        response = client.post(
            "/api/jobs",
            files={"file": ("big.wav", b"x" * (3 * 1024 * 1024), "audio/wav")},
            data={"model": "base"},
        )
    finally:
        s.ARGS.max_upload_mb = 64

    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()
    rejected = [e for e in configured.events() if e["event"] == "job.rejected"]
    assert rejected and rejected[-1]["reason"] == "body-too-large"


def test_within_limit_upload_still_succeeds(client, configured):
    assert upload(client, "small.wav").status_code == 200


# --------------------------------------------------------------------------- #
# Audit read path
# --------------------------------------------------------------------------- #


def test_audit_search_string_is_not_written_to_the_trail(client, configured):
    """An audit reader must not be able to plant text in the record."""
    marker = "Nightjar confidential phrase"
    response = client.get(
        "/api/audit",
        headers={"x-audit-token": configured.audit_token},
        params={"q": marker},
    )
    assert response.status_code == 200

    events = configured.events()
    accessed = [e for e in events if e["event"] == "audit.accessed"]
    assert accessed, "an audit read should itself be recorded"
    record = accessed[-1]

    assert "search" not in record
    assert record["search_len"] == len(marker)
    assert record["search_sha256"] == s.digest(marker)
    assert marker not in json.dumps(events)


def test_audit_query_validation(client, configured):
    headers = {"x-audit-token": configured.audit_token}

    assert client.get("/api/audit?date=2026-9-1", headers=headers).status_code == 400
    assert client.get("/api/audit?job=../etc", headers=headers).status_code == 400

    ok = client.get("/api/audit?date=2026-01-01", headers=headers)
    assert ok.status_code == 200
    assert ok.json()["events"] == [] and ok.json()["total"] == 0


def test_audit_include_prompts_returns_sidecar_text(client, configured):
    job_id = upload(client, prompt=PROMPT_TEXT).json()["id"]

    response = client.get(
        "/api/audit",
        headers={"x-audit-token": configured.audit_token},
        params={"include_prompts": 1, "job": job_id},
    )
    assert response.status_code == 200
    events = response.json()["events"]
    assert events
    assert events[0]["prompt_text"] == PROMPT_TEXT


def test_audit_prompt_endpoint_validates_the_job_id(client, configured):
    headers = {"x-audit-token": configured.audit_token}
    assert (
        client.get("/api/audit/prompts/deadbeef1234", headers=headers).status_code
        == 404
    )
    assert (
        client.get("/api/audit/prompts/..%2f..%2fetc", headers=headers).status_code
        == 404
    )


# --------------------------------------------------------------------------- #
# Misc surface
# --------------------------------------------------------------------------- #


def test_pages_are_served_and_audit_page_is_data_free(client):
    assert client.get("/").status_code == 200
    page = client.get("/audit")
    assert page.status_code == 200
    assert "Audit trail" in page.text
    # The page itself must not embed any trail data.
    assert "job.created" not in page.text


def test_audit_page_renders_the_mode_the_server_runs(client, configured):
    """The mode is server-rendered: the page must never have to guess from a
    failed request whether the token or the endpoint itself is the problem."""
    page = client.get("/audit").text
    assert 'data-mode="token"' in page
    assert 'class="gate" id="gate"' in page, "token mode needs no script to work"
    assert 'class="gate locked" id="off"' in page

    s.ARGS.audit_open = True
    try:
        opened = client.get("/audit").text
        assert 'data-mode="open"' in opened
        # Starting hidden is what stops a flash of "paste the audit token".
        assert 'class="gate locked" id="gate"' in opened
    finally:
        s.ARGS.audit_open = False

    s.ARGS.audit_token = ""
    try:
        off = client.get("/audit").text
        assert 'data-mode="off"' in off
        assert 'class="gate locked" id="gate"' in off
        assert 'class="gate" id="off"' in off, "the notice replaces the gate"
    finally:
        s.ARGS.audit_token = configured.audit_token


def test_delete_cancels_a_queued_job_and_keeps_the_record(client, configured):
    job_id = upload(client).json()["id"]
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200

    # A job that has not finished is cancelled, not dropped: the worker may
    # still be holding it, and the record is what the audit trail refers to.
    assert s.JOBS[job_id]["state"] == "cancelled"
    cancelled = [e for e in configured.events() if e["event"] == "job.cancelled"]
    assert cancelled and cancelled[-1]["job"] == job_id


def test_a_cancelled_job_keeps_its_audio_until_the_record_goes(client, configured):
    """'job' retention keeps the upload while the record exists, and a cancel
    keeps the record. Deleting the audio anyway made can_retry false for every
    cancelled job, so the page never offered Retry after a Cancel -- the moment
    it is most wanted, when the model or the language was wrong."""
    job_id = upload(client).json()["id"]
    source = Path(s.JOBS[job_id]["path"])
    client.delete(f"/api/jobs/{job_id}")

    assert source.exists()
    assert client.get(f"/api/jobs/{job_id}").json()["can_retry"] is True

    # Removing the record is what lets the audio go.
    client.delete(f"/api/jobs/{job_id}")
    assert job_id not in s.JOBS
    assert not source.exists()


def test_delete_removes_a_finished_job(client, configured):
    job_id = upload(client).json()["id"]
    s.patch_job(job_id, state="done", segments=[])

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert job_id not in s.JOBS

    deleted = [e for e in configured.events() if e["event"] == "job.deleted"]
    assert deleted and deleted[-1]["job"] == job_id


def test_retry_of_a_missing_source_is_refused_and_audited(client, configured):
    response = client.post("/api/jobs/deadbeef1234/retry", data={"model": "base"})
    assert response.status_code == 404
    assert any(e["event"] == "request.rejected" for e in configured.events())


def test_queue_cap_refuses_uploads(client, configured):
    s.ARGS.max_queue = 1
    try:
        assert upload(client, "one.wav").status_code == 200
        second = upload(client, "two.wav")
        assert second.status_code == 429
    finally:
        s.ARGS.max_queue = 20


def test_audit_disabled_still_leaves_one_startup_marker(configured, monkeypatch):
    """--no-audit must be distinguishable from the server being down."""
    disabled = s.AuditLog(
        configured.root / "audit2", enabled=False, retain_days=30, prompts=True
    )
    monkeypatch.setattr(s, "AUDIT", disabled)

    with TestClient(s.app) as c:
        c.get("/api/status")  # generates an api.read candidate

    files = list(disabled.dir.glob("audit-*.jsonl"))
    assert not files, "a disabled trail records no events"

    disabled.emit("server.started", force=True, audit=False)
    files = list(disabled.dir.glob("audit-*.jsonl"))
    assert len(files) == 1
    assert "server.started" in files[0].read_text()


def test_status_endpoint_shape(client):
    body = client.get("/api/status").json()
    for key in (
        "device",
        "models",
        "max_upload_mb",
        "retry_available",
        "active_jobs",
    ):
        assert key in body


def test_day_file_is_named_for_the_utc_day(client, configured):
    upload(client)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    assert (configured.audit.dir / f"audit-{day}.jsonl").exists()
