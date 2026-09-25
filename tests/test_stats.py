"""The stats page, and the aggregates behind it.

The arithmetic is the risky part. A relabel replays a stored transcript through
the speaker pass and reuses the original job's `duration` without transcribing
anything, so a fold that just summed `duration` over `job.done` would count
relabelled audio twice. Telling the passes apart is what `relabel_of`/`retry_of`
on `job.done` exist for, and the first tests here pin exactly that.

The rest pins the plumbing the design review said was wrong in the first draft:
that the audit token gates the endpoint, that the page's own polling does not
land in the numbers it displays, and that the aggregate kept beside the trail is
never trusted further than its provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import transcribe_server as s


def fresh(tmp_path: Path, **kwargs: Any) -> s.AuditLog:
    kwargs.setdefault("retain_days", 0)
    return s.AuditLog(tmp_path / "audit", **kwargs)


def only_day(log: s.AuditLog) -> str:
    files = sorted(log.dir.glob("audit-*.jsonl"))
    assert files, "the log wrote no day file"
    return files[-1].stem[len("audit-") :]


def audit_headers(configured: Any) -> dict[str, str]:
    return {"x-audit-token": configured.audit_token}


def write_day(log: s.AuditLog, day: str, *records: dict[str, Any]) -> Path:
    log.dir.mkdir(parents=True, exist_ok=True)
    path = log.dir / f"audit-{day}.jsonl"
    path.write_text(
        "".join(json.dumps({"ts": "2026-01-01T00:00:00Z", **r}) + "\n" for r in records)
    )
    return path


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #


def test_a_relabel_contributes_no_audio(tmp_path):
    log = fresh(tmp_path)
    log.emit(
        "job.done", job="j1", duration=3600.0, elapsed=60.0, model="base", language="en"
    )
    log.emit("job.done", job="j2", duration=3600.0, elapsed=5.0, relabel_of="j1")

    agg = log.day_summary(only_day(log))

    assert agg["recordings"] == 1
    assert agg["passes"] == 1
    assert agg["relabels"] == 1
    assert agg["recording_seconds"] == 3600.0
    assert agg["audio_seconds"] == 3600.0, "a relabel transcribed nothing"
    assert agg["processing_seconds"] == 60.0, "and took no transcription time"
    assert agg["models"] == {"base": 1}, "a relabel is not a model's transcription"


def test_a_retry_is_a_pass_but_not_another_recording(tmp_path):
    log = fresh(tmp_path)
    log.emit("job.done", job="j1", duration=100.0, elapsed=10.0, model="base")
    log.emit(
        "job.done", job="j2", duration=100.0, elapsed=9.0, model="base", retry_of="j1"
    )

    agg = log.day_summary(only_day(log))

    assert agg["recordings"] == 1, "the same audio, asked for once"
    assert agg["passes"] == 2
    assert agg["recording_seconds"] == 100.0
    assert agg["audio_seconds"] == 200.0, "but the GPU did transcribe it twice"


def test_failures_and_refusals_are_counted_separately(tmp_path):
    log = fresh(tmp_path)
    log.emit("job.done", job="a", duration=1.0)
    log.emit("job.error", job="b", stage="transcribe")
    log.emit("job.diarize_failed", job="c", reason="RuntimeError")
    log.emit("job.rejected", reason="queue-full")
    log.emit("security.auth_failed", scope="app", reason="bad-token")
    log.emit("security.auth_failed", scope="audit", reason="bad-token")
    log.emit("security.rejected_summary", scope="job.rejected", suppressed=7)

    agg = log.day_summary(only_day(log))

    assert agg["failed"] == 1, "a diarize failure still finished the job"
    assert agg["diarize_failed"] == 1
    assert agg["rejected"] == 1
    assert agg["security"]["security.auth_failed"] == 2
    assert agg["suppressed"] == 7, "the collapsed refusals are still accounted for"


def test_the_session_fold_counts_what_the_process_did(tmp_path):
    log = fresh(tmp_path)
    log.emit("job.done", job="a", duration=60.0, elapsed=6.0)
    snap = log.session.snapshot()
    assert snap["recordings"] == 1 and snap["audio_seconds"] == 60.0

    log.emit("job.done", job="b", duration=60.0, elapsed=6.0)
    assert snap["recordings"] == 1, "a snapshot is a copy, not a live view"
    assert log.session.snapshot()["recordings"] == 2


def test_the_session_fold_runs_even_when_recording_is_off(tmp_path):
    """--no-audit has no files, so the counters are the only record there is."""
    log = s.AuditLog(tmp_path / "audit", retain_days=0, enabled=False)
    log.emit("job.done", job="a", duration=60.0, elapsed=6.0)

    assert log.session.snapshot()["recordings"] == 1
    assert list(log.dir.glob("audit-*.jsonl")) == [], "and it wrote nothing"


# --------------------------------------------------------------------------- #
# The stored aggregate is derived data, and is not trusted further than that
# --------------------------------------------------------------------------- #


def test_a_finished_day_is_folded_once_and_reused(tmp_path, monkeypatch):
    log = fresh(tmp_path)
    day = "2026-01-01"
    write_day(log, day, {"event": "job.done", "job": "a", "duration": 120.0})
    assert log.day_summary(day)["recordings"] == 1
    assert log.summary_path(day).exists()

    second = fresh(tmp_path)

    def boom(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("the day was refolded despite a valid summary")

    monkeypatch.setattr(s.AuditLog, "_fold_bytes", staticmethod(boom))
    assert second.day_summary(day)["recordings"] == 1


def test_a_changed_day_file_invalidates_its_summary(tmp_path):
    log = fresh(tmp_path)
    day = "2026-01-01"
    path = write_day(log, day, {"event": "job.done", "job": "a", "duration": 60.0})
    assert log.day_summary(day)["recordings"] == 1

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": "x", "event": "job.done", "duration": 60.0}) + "\n")

    assert fresh(tmp_path).day_summary(day)["recordings"] == 2, (
        "a summary that does not match the file must be discarded"
    )


def test_a_corrupt_summary_is_refolded_not_trusted(tmp_path):
    log = fresh(tmp_path)
    day = "2026-01-01"
    write_day(log, day, {"event": "job.done", "job": "a", "duration": 60.0})
    log.day_summary(day)
    log.summary_path(day).write_text("{ not json at all")

    assert fresh(tmp_path).day_summary(day)["recordings"] == 1


def test_a_summary_from_an_older_schema_is_refolded(tmp_path):
    log = fresh(tmp_path)
    day = "2026-01-01"
    write_day(log, day, {"event": "job.done", "job": "a", "duration": 60.0})
    log.day_summary(day)

    payload = json.loads(log.summary_path(day).read_text(encoding="utf-8"))
    payload["schema"] = s.STATS_SCHEMA + 1
    log.summary_path(day).write_text(json.dumps(payload))

    assert fresh(tmp_path).day_summary(day)["recordings"] == 1


def test_coerce_stats_ignores_junk_and_never_invents_a_key():
    out = s.coerce_stats(
        {
            "events": "lots",
            "audio_seconds": "3.5",
            "models": {"base": 2, "bad": "x"},
            "nonsense": 1,
        }
    )

    assert out["events"] == 0
    assert out["audio_seconds"] == 3.5
    assert out["models"] == {"base": 2}
    assert "nonsense" not in out
    assert set(out) == set(s.blank_stats())


def test_a_malformed_day_is_refused_before_it_reaches_a_filename(tmp_path):
    log = fresh(tmp_path)
    assert log.day_summary("../../etc/passwd") == s.blank_stats()
    assert log.day_summary("") == s.blank_stats()


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #


def test_the_predicate_covers_both_audit_prefixes():
    for path in ("/api/audit", "/api/audit/verify", "/api/stats"):
        assert s.audit_gated(path), path
    for path in ("/api/status", "/api/jobs", "/stats", "/"):
        assert not s.audit_gated(path), path


def test_stats_needs_the_audit_token(configured):
    with TestClient(s.app) as anon:
        assert anon.get("/api/stats").status_code == 401
        assert anon.get("/stats").status_code == 200, "the shell stays public"


def test_the_app_token_does_not_open_the_stats_api(client, configured):
    # `client` sends x-token and nothing else.
    assert client.get("/api/stats").status_code == 401


def test_stats_is_404_when_neither_a_token_nor_open_is_configured(client, configured):
    saved = s.ARGS.audit_token
    s.ARGS.audit_token = ""
    try:
        assert client.get("/api/stats").status_code == 404
    finally:
        s.ARGS.audit_token = saved


# --------------------------------------------------------------------------- #
# The endpoint
# --------------------------------------------------------------------------- #


def test_stats_reports_the_session_the_days_and_the_total(client, configured):
    configured.audit.emit(
        "job.done", job="a", duration=3600.0, elapsed=60.0, model="base"
    )

    body = client.get("/api/stats", headers=audit_headers(configured)).json()

    assert body["session"]["recordings"] == 1
    assert body["retained"]["recordings"] == 1
    assert body["retain_days"] == configured.audit.retain_days
    assert [d["date"] for d in body["days"]] == sorted(
        d["date"] for d in body["days"]
    ), "oldest first, so a table reads left to right"
    assert body["days"][-1]["audio_seconds"] == 3600.0


def test_polling_stats_does_not_log_a_generic_read(client, configured):
    """--audit-reads is on in the fixture; the page must not feed itself."""
    client.get("/api/stats", headers=audit_headers(configured))

    events = configured.events()
    assert any(e["event"] == "audit.stats_read" for e in events), "the read is logged"
    assert not [e for e in events if e["event"] == "api.read"], (
        "the stats poll logged itself as a generic read"
    )


def test_a_failed_stats_call_does_not_double_log(client, configured, monkeypatch):
    """The handler logs before doing any work, so a later 500 is not also a
    request.rejected. Failing after that point is what pins the ordering."""

    def boom(self: s.AuditLog, day: str) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(s.AuditLog, "day_summary", boom)
    configured.audit.emit("job.created", job="a")
    with TestClient(s.app, raise_server_exceptions=False) as c:
        c.get("/api/stats", headers=audit_headers(configured))

    events = configured.events()
    assert any(e["event"] == "audit.stats_read" for e in events), (
        "the read is logged before the work that can fail"
    )
    assert not [e for e in events if e["event"] == "request.rejected"], (
        "audit_handled must stop the middleware logging it twice"
    )


def test_stats_says_so_when_recording_is_off(client, configured):
    """--no-audit still writes one forced marker.

    So the retained view is not empty: it holds exactly that marker, and the
    response has to say recording is off rather than let one row imply history.
    """
    configured.audit.enabled = False
    configured.audit.emit("server.started", force=True, audit=False)

    body = client.get("/api/stats", headers=audit_headers(configured)).json()

    assert body["recording"] is False
    assert len(body["days"]) == 1
    assert body["days"][0]["recordings"] == 0, "a marker is not a recording"
    assert body["retained"]["events"] == 1


def test_the_stats_page_renders_the_mode_the_server_runs(client, configured):
    page = client.get("/stats").text
    assert 'data-mode="token"' in page
    assert '<div class="gate" id="gate">' in page, "token mode shows the gate"
    assert '<div class="gate locked" id="off">' in page, "and not the notice"

    s.ARGS.audit_open = True
    try:
        opened = client.get("/stats").text
    finally:
        s.ARGS.audit_open = False
    assert 'data-mode="open"' in opened
    assert '<div class="gate locked" id="gate">' in opened

    s.ARGS.audit_token = ""
    try:
        off = client.get("/stats").text
    finally:
        s.ARGS.audit_token = configured.audit_token
    assert 'data-mode="off"' in off
    assert '<div class="gate" id="off">' in off, "a switched-off API says so"
    assert '<div class="gate locked" id="gate">' in off, "and asks for no token"


def test_the_stats_page_links_to_the_audit_trail(client):
    assert 'href="/audit"' in client.get("/stats").text
    assert 'href="/stats"' in client.get("/audit").text
