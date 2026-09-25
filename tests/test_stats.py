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
from datetime import datetime, timezone
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

    assert out["events"] == 0, "a string is not a count"
    assert out["audio_seconds"] == 0.0, "nor a number"
    assert out["models"] == {"base": 2}
    assert "nonsense" not in out
    assert set(out) == set(s.blank_stats())


def test_coerce_stats_refuses_non_finite_numbers():
    """json.loads accepts Infinity and NaN, and the page would render them."""
    out = s.coerce_stats(
        {
            "audio_seconds": float("inf"),
            "recording_seconds": float("nan"),
            "processing_seconds": 1e308 * 10,
            "events": True,
        }
    )

    assert out["audio_seconds"] == 0.0
    assert out["recording_seconds"] == 0.0
    assert out["processing_seconds"] == 0.0
    assert out["events"] == 0, "a bool is not a count"


def test_a_summary_that_does_not_round_trip_is_refolded(tmp_path):
    """Coercing to zeros would under-report silently; refold instead."""
    log = fresh(tmp_path)
    day = "2026-01-01"
    write_day(log, day, {"event": "job.done", "job": "a", "duration": 3600.0})
    log.day_summary(day)

    payload = json.loads(log.summary_path(day).read_text(encoding="utf-8"))
    payload["stats"]["audio_seconds"] = "Infinity"
    log.summary_path(day).write_text(json.dumps(payload))

    assert fresh(tmp_path).day_summary(day)["audio_seconds"] == 3600.0


def test_a_malformed_day_is_refused_before_it_reaches_a_filename(tmp_path):
    log = fresh(tmp_path)
    assert log.day_summary("../../etc/passwd") == s.blank_stats()
    assert log.day_summary("") == s.blank_stats()


# --------------------------------------------------------------------------- #
# The marks have to be EMITTED, not just folded
# --------------------------------------------------------------------------- #


def test_a_relabel_records_its_provenance_on_the_job_done(client, configured):
    """Folding is tested above; emitting is a separate step.

    Deleting `relabel_of=job.get("relabel_of")` from label_and_finish would
    leave every other test green while production double counted every relabel.
    """
    job_id = configured.make_job(diarize="false")
    s.patch_job(job_id, relabel_of="origjob1234", state="running")

    s.label_and_finish(job_id, s.JOBS[job_id], s.JOBS[job_id]["opts"], [], "en", 12.0)

    done = [e for e in configured.events() if e["event"] == "job.done"]
    assert done, "label_and_finish should finish the job"
    assert done[-1]["relabel_of"] == "origjob1234"


def test_a_first_pass_carries_neither_mark(client, configured):
    job_id = configured.make_job(diarize="false")
    s.patch_job(job_id, state="running")

    s.label_and_finish(job_id, s.JOBS[job_id], s.JOBS[job_id]["opts"], [], "en", 12.0)

    done = [e for e in configured.events() if e["event"] == "job.done"]
    assert "relabel_of" not in done[-1] and "retry_of" not in done[-1]


def test_a_retry_marks_the_new_job_with_its_predecessor(client, configured):
    job_id = configured.make_job()
    s.patch_job(job_id, state="done", duration=5.0, segments=[])

    response = client.post(
        f"/api/jobs/{job_id}/retry", data={"model": "base", "quality": "fast"}
    )

    assert response.status_code == 200, response.text
    assert s.JOBS[response.json()["id"]]["retry_of"] == job_id


def test_a_lost_attestation_riding_a_relabel_is_counted(tmp_path):
    """The relabel branch must not return before the `lost` tail.

    A caller cannot pass `lost` -- it is reserved, so an event cannot forge an
    attestation -- which means the only way a record carries one is a write that
    failed just before it.
    """
    log = fresh(tmp_path)
    log.emit("a")
    real = log._fh
    assert real is not None

    class Boom:
        def write(self, text: str) -> None:
            raise OSError("disk full")

        def flush(self) -> None:
            pass

        def truncate(self, size: int) -> None:
            pass

    log._fh = Boom()  # pyright: ignore[reportAttributeAccessIssue]
    log.emit("b")
    assert log.lost == 1

    log._fh = real
    log.emit("job.done", job="r", duration=60.0, relabel_of="orig")

    agg = log.day_summary(only_day(log))

    assert agg["relabels"] == 1
    assert agg["audio_seconds"] == 0.0
    assert agg["lost"] == 1, "the relabel record carried the attestation"


# --------------------------------------------------------------------------- #
# The delta cache
# --------------------------------------------------------------------------- #


def test_a_second_read_folds_only_the_appended_bytes(tmp_path, monkeypatch):
    log = fresh(tmp_path)
    log.emit("job.done", job="a", duration=60.0, model="base")
    day = only_day(log)
    path = log.dir / f"audit-{day}.jsonl"
    assert log.day_summary(day)["recordings"] == 1

    starts: list[tuple[int, int]] = []
    real = s.AuditLog._fold_bytes

    def spy(fh: Any, agg: Any, start: int, end: int) -> int:
        starts.append((start, end))
        return real(fh, agg, start, end)

    monkeypatch.setattr(s.AuditLog, "_fold_bytes", staticmethod(spy))
    size_before = path.stat().st_size
    log.emit("job.done", job="b", duration=60.0, model="base")

    assert log.day_summary(day)["recordings"] == 2
    assert starts == [(size_before, path.stat().st_size)], (
        "the second read must resume where the first stopped"
    )


def test_a_record_split_across_the_read_boundary_is_folded_once(tmp_path):
    """A half-written record must not be folded, and must not be lost either."""
    log = fresh(tmp_path)
    day = "2026-01-01"
    log.dir.mkdir(parents=True, exist_ok=True)
    path = log.dir / f"audit-{day}.jsonl"
    record = json.dumps({"ts": "x", "event": "job.done", "job": "a", "duration": 60.0})

    path.write_text(record[: len(record) // 2])
    assert log.day_summary(day)["recordings"] == 0, "a torn line is not a record"

    path.write_text(record + "\n")
    assert log.day_summary(day)["recordings"] == 1, "and it is not skipped once whole"


def test_a_summary_that_covered_part_of_the_file_resumes_from_its_offset(tmp_path):
    """A torn tail must not be remembered as covered, or it is skipped forever."""
    log = fresh(tmp_path)
    day = "2026-01-01"
    log.dir.mkdir(parents=True, exist_ok=True)
    path = log.dir / f"audit-{day}.jsonl"
    whole = json.dumps({"ts": "x", "event": "job.done", "job": "a", "duration": 60.0})
    path.write_text(whole + "\n" + whole[: len(whole) // 2])

    assert log.day_summary(day)["recordings"] == 1
    # A fresh process reads the stored summary, which covers one record, and
    # must resume at that offset rather than at the file's size.
    assert fresh(tmp_path).day_summary(day)["recordings"] == 1

    path.write_text(whole + "\n" + whole + "\n")
    assert fresh(tmp_path).day_summary(day)["recordings"] == 2


def test_a_shrunk_day_file_is_refolded(tmp_path):
    log = fresh(tmp_path)
    day = "2026-01-01"
    path = write_day(
        log,
        day,
        {"event": "job.done", "job": "a", "duration": 60.0},
        {"event": "job.done", "job": "b", "duration": 60.0},
    )
    assert log.day_summary(day)["recordings"] == 2

    path.write_text(
        json.dumps({"ts": "x", "event": "job.done", "duration": 60.0}) + "\n"
    )

    assert log.day_summary(day)["recordings"] == 1, "a shorter file is not a prefix"


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #


def test_prune_removes_a_days_summary_and_sweeps_orphans(tmp_path):
    log = s.AuditLog(tmp_path / "audit", retain_days=1)
    log.dir.mkdir(parents=True, exist_ok=True)
    (log.dir / "audit-2000-01-01.jsonl").write_text(
        json.dumps({"ts": "x", "event": "job.done", "duration": 1.0}) + "\n"
    )
    (log.dir / "stats-2000-01-01.json").write_text("{}")
    (log.dir / "stats-1999-12-31.json").write_text("{}")  # day file already gone
    (log.dir / ".stats-2000-01-01.json.tmp").write_text("{}")

    removed = log.prune()

    assert "audit-2000-01-01.jsonl" in removed
    assert not (log.dir / "stats-2000-01-01.json").exists()
    assert not (log.dir / "stats-1999-12-31.json").exists(), (
        "an orphan summary is unreachable and nothing else would ever sweep it"
    )
    assert not (log.dir / ".stats-2000-01-01.json.tmp").exists()


def test_a_settled_day_is_not_rewritten_on_every_poll(tmp_path, monkeypatch):
    """Saving unconditionally means N temp+fsync+replace cycles per request."""
    log = fresh(tmp_path)
    day = "2026-01-01"
    write_day(log, day, {"event": "job.done", "job": "a", "duration": 60.0})
    log.day_summary(day)  # the first pass writes it

    saves = {"n": 0}
    real = s.AuditLog._save_summary

    def counting(self: Any, *args: Any, **kwargs: Any) -> None:
        saves["n"] += 1
        real(self, *args, **kwargs)

    monkeypatch.setattr(s.AuditLog, "_save_summary", counting)
    for _ in range(5):
        log.day_summary(day)

    assert saves["n"] == 0, "a day that has not changed must not be rewritten"


def test_eviction_never_drops_today(tmp_path, monkeypatch):
    """Today is the one day whose summary is never persisted."""
    monkeypatch.setattr(s.AuditLog, "MAX_DAY_CACHE", 3)
    log = fresh(tmp_path)
    log.dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log.emit("job.done", job="now", duration=60.0)
    assert log.day_summary(today)["recordings"] == 1

    # Fill past the cap, newest-first like the endpoint's loop does.
    for i in range(6):
        other = f"2026-01-{i + 1:02d}"
        (log.dir / f"audit-{other}.jsonl").write_text(
            json.dumps({"ts": "x", "event": "job.done", "duration": 1.0}) + "\n"
        )
        log.day_summary(other)

    assert today in log._day_stats, "evicting today means refolding the whole day"


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #


def test_the_predicate_covers_both_audit_prefixes():
    for path in ("/api/audit", "/api/audit/verify", "/api/stats"):
        assert s.audit_gated(path), path
    for path in ("/api/status", "/api/jobs", "/stats", "/"):
        assert not s.audit_gated(path), path


def test_an_unknown_api_subpath_is_refused_not_served(configured):
    """`audit_gated` over-matches on purpose. /api/statsfoo is not a route, and a
    miss has to fail closed rather than be reachable without the credential."""
    with TestClient(s.app) as anon:
        assert anon.get("/api/statsfoo").status_code == 401


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
