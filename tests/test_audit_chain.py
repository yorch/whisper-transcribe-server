"""The audit trail's integrity, health and growth guard.

Three things are pinned here:

* **the chain** -- every record links to the one before it, a day carries the
  previous day's head forward, and an edit, a deletion or a torn write in a day
  file is reported at the exact line. Records written before the chain existed
  are counted as legacy and never make a day look broken.
* **health** -- a write that fails is counted, attested by the next record that
  lands, and surfaced through /api/audit and /api/status, so a trail that is
  quietly dropping events cannot look like a quiet day.
* **the guard** -- a day file that reaches its byte cap says so once and then
  stops taking events, and a sidecar write is atomic and serialized.

README "Audit trail" is the operator-facing description; this file is about the
promises it makes.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import transcribe_server as s

JOB = "abcdef123456"


def fresh(tmp_path: Path, **kwargs: Any) -> s.AuditLog:
    """A log with retention off, so a test can use whatever dates it likes.

    Retention runs on every rotation, and a fixture dated 2020 is deleted the
    moment _rotate touches it. That is correct behaviour, so tests that pin a
    date turn it off rather than working around it.
    """
    kwargs.setdefault("retain_days", 0)
    return s.AuditLog(tmp_path / "audit", **kwargs)


def day_file(log: s.AuditLog) -> Path:
    files = sorted(log.dir.glob("audit-*.jsonl"))
    assert files, "the log wrote no day file"
    return files[-1]


def day_of(path: Path) -> str:
    return path.stem[len("audit-") :]


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #


def test_records_link_and_a_clean_day_verifies(tmp_path):
    log = fresh(tmp_path)
    for i in range(4):
        log.emit("job.created", job=JOB, n=i)

    path = day_file(log)
    recs = records(path)
    assert [r["seq"] for r in recs] == [1, 2, 3, 4]
    assert recs[0]["prev"] is None
    for earlier, later in zip(recs, recs[1:], strict=False):
        assert later["prev"] == earlier["chain"]

    result = log.verify(day_of(path))
    assert result["ok"] is True
    assert result["checked"] == 4
    assert result["legacy"] == 0
    assert result["reason"] is None
    assert result["head"] == recs[-1]["chain"]


def test_the_head_value_matches_the_last_record(tmp_path):
    log = fresh(tmp_path)
    log.emit("a")
    log.emit("b")
    path = day_file(log)
    head = log.head(day_of(path))
    assert head["seq"] == 2
    assert head["chain"] == records(path)[-1]["chain"]


def test_an_edited_record_is_detected_at_its_line(tmp_path):
    log = fresh(tmp_path)
    for i in range(4):
        log.emit("e", n=i)
    path = day_file(log)
    lines = path.read_text(encoding="utf-8").splitlines()
    edited = json.loads(lines[2])
    edited["n"] = 999
    lines[2] = json.dumps(edited)
    path.write_text("\n".join(lines) + "\n")

    result = log.verify(day_of(path))
    assert result["ok"] is False
    assert result["first_bad_line"] == 3
    assert "hash" in result["reason"]


def test_a_deleted_line_is_detected(tmp_path):
    log = fresh(tmp_path)
    for i in range(4):
        log.emit("e", n=i)
    path = day_file(log)
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]  # the second record simply is not there any more
    path.write_text("\n".join(lines) + "\n")

    result = log.verify(day_of(path))
    assert result["ok"] is False
    assert result["first_bad_line"] == 2
    assert "gap" in result["reason"].lower()


def test_a_truncated_tail_is_detected(tmp_path):
    log = fresh(tmp_path)
    for i in range(3):
        log.emit("e", n=i)
    path = day_file(log)
    path.write_text(path.read_text(encoding="utf-8")[:-12])

    result = log.verify(day_of(path))
    assert result["ok"] is False
    assert "unparsable" in result["reason"].lower()


def test_a_restart_mid_day_continues_the_same_chain(tmp_path):
    first = fresh(tmp_path)
    first.emit("a")
    first.emit("b")

    # A second process opens the same day file: it must recover seq and prev,
    # not start a fresh chain.
    second = fresh(tmp_path)
    second.emit("c")

    path = day_file(second)
    recs = records(path)
    assert [r["seq"] for r in recs] == [1, 2, 3]
    assert "chain_reset" not in recs[-1]
    assert second.verify(day_of(path))["ok"] is True


def test_a_new_day_carries_the_previous_days_head(tmp_path):
    log = fresh(tmp_path)
    log.emit("today")

    yesterday = "2020-01-01"
    (log.dir / f"audit-{yesterday}.jsonl").write_text(
        json.dumps(
            {
                "ts": "2020-01-01T00:00:00.000Z",
                "event": "old",
                "seq": 1,
                "prev": None,
                "chain": "a" * 64,
            }
        )
        + "\n"
    )
    log._rotate("2020-01-02")
    assert log._carry == "a" * 64
    log._append(log._fh, {"ts": "2020-01-02T00:00:00.000Z", "event": "new"})

    carried = records(log.dir / "audit-2020-01-02.jsonl")[0]
    assert carried["carry"] == "a" * 64
    assert carried["seq"] == 1 and carried["prev"] is None
    assert log.verify("2020-01-02")["carry_ok"] is True


def test_records_from_before_the_chain_are_tolerated(tmp_path):
    """A day that predates the feature must not read as tampered."""
    log = fresh(tmp_path)
    log.dir.mkdir(parents=True, exist_ok=True)
    path = log.dir / "audit-2020-02-02.jsonl"
    path.write_text(
        json.dumps({"ts": "2020-02-02T00:00:00.000Z", "event": "legacy1"})
        + "\n"
        + json.dumps({"ts": "2020-02-02T00:00:01.000Z", "event": "legacy2"})
        + "\n"
    )
    log._rotate("2020-02-02")  # the tail has no chain, so the chain restarts
    log._append(log._fh, {"ts": "2020-02-02T00:00:02.000Z", "event": "new"})

    result = log.verify("2020-02-02")
    assert result["ok"] is True
    assert result["checked"] == 1
    assert result["legacy"] == 2
    # The discontinuity is declared rather than implied.
    assert records(path)[-1]["chain_reset"] is True


def test_a_failed_write_is_attested_and_does_not_gap_the_chain(tmp_path):
    """A torn write must neither corrupt the file nor fake a sequence gap."""
    log = fresh(tmp_path)
    log.emit("a")
    real = log._fh
    assert real is not None

    class Torn:
        """Half a line, then the disk is full."""

        def write(self, text: str) -> None:
            real.write(text[: len(text) // 2])
            raise OSError("disk full")

        def flush(self) -> None:
            real.flush()

        def truncate(self, size: int) -> None:
            real.truncate(size)

    log._fh = Torn()  # pyright: ignore[reportAttributeAccessIssue]
    log.emit("b")  # must not raise
    assert log.lost == 1 and log.lost_total == 1
    assert log.last_error is not None and "disk full" in log.last_error

    log._fh = real
    log.emit("c")

    path = day_file(log)
    recs = records(path)
    assert [r["event"] for r in recs] == ["a", "c"], "the torn line should be gone"
    assert [r["seq"] for r in recs] == [1, 2], "no sequence gap from the lost event"
    assert recs[-1]["lost"] == 1
    assert log.verify(day_of(path))["ok"] is True
    assert log.lost == 0, "a landed record clears the pending gap"


# --------------------------------------------------------------------------- #
# The byte cap
# --------------------------------------------------------------------------- #


def test_the_cap_writes_one_marker_then_suppresses(tmp_path):
    log = fresh(tmp_path, max_mb=0)
    log.max_bytes = 400  # small enough to trip in a handful of writes
    for i in range(50):
        log.emit("e", n=i, pad="x" * 40)

    events = [r["event"] for r in records(day_file(log))]
    assert "audit.full" in events, "the file must say why it stopped"
    assert events.count("audit.full") == 1
    assert log.full is True
    assert log.suppressed > 0
    assert log.degraded() is True
    # What did land is still a valid chain.
    assert log.verify(day_of(day_file(log)))["ok"] is True


def test_zero_max_mb_means_unlimited(tmp_path):
    log = fresh(tmp_path, max_mb=0)
    for i in range(200):
        log.emit("e", n=i)
    assert log.full is False
    assert log.suppressed == 0
    assert log.max_mb == 0


def test_the_cap_resets_when_the_day_rolls_over(tmp_path):
    log = fresh(tmp_path, max_mb=0)
    log.max_bytes = 300
    for _ in range(20):
        log.emit("e", pad="x" * 40)
    assert log.full is True

    log._rotate("2099-01-01")
    assert log.full is False
    assert log.suppressed == 0, "a new day starts with its own budget"


# --------------------------------------------------------------------------- #
# Prompt sidecars: atomic and serialized
# --------------------------------------------------------------------------- #


def test_a_sidecar_amend_keeps_the_prompt_and_adds_the_names(tmp_path):
    log = fresh(tmp_path)
    log.write_prompt(JOB, {"job": JOB, "prompt": "secret", "hotwords": "x"})
    log.amend_prompt(JOB, {"job": JOB}, {"speaker_names": {"1": "Alice"}})

    record = log.read_prompt(JOB)
    assert record is not None
    assert record["prompt"] == "secret", "the amend must not drop the prompt"
    assert record["speaker_names"] == {"1": "Alice"}


def test_a_sidecar_write_leaves_no_temp_file_behind(tmp_path):
    log = fresh(tmp_path)
    log.write_prompt(JOB, {"job": JOB, "prompt": "secret"})
    assert not list(log.prompts_dir.glob(".*.json.tmp"))


def test_concurrent_amends_do_not_lose_each_others_field(tmp_path):
    """Two writers for one sidecar: the lock is the whole point.

    Without it, read-modify-write interleaves and one field is silently lost.
    """
    log = fresh(tmp_path)
    log.write_prompt(JOB, {"job": JOB, "prompt": "secret"})

    def bump(key: str, value: Any) -> None:
        for _ in range(200):
            log.amend_prompt(JOB, {"job": JOB}, {key: value})

    threads = [
        threading.Thread(target=bump, args=("speaker_names", {"1": "Alice"})),
        threading.Thread(target=bump, args=("hotwords", "terms")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    record = log.read_prompt(JOB)
    assert record is not None
    assert record["prompt"] == "secret"
    assert record["speaker_names"] == {"1": "Alice"}
    assert record["hotwords"] == "terms"


# --------------------------------------------------------------------------- #
# Health over the API
# --------------------------------------------------------------------------- #


def audit_headers(configured) -> dict[str, str]:
    return {"x-audit-token": configured.audit_token}


def test_audit_query_reports_health_and_cap_fields(client, configured):
    response = client.get("/api/audit", headers=audit_headers(configured))
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is False
    assert body["lost_total"] == 0
    assert body["full"] is False
    assert body["suppressed"] == 0
    assert body["max_mb"] == configured.audit.max_mb


def test_a_degraded_trail_says_so_on_both_surfaces(client, configured):
    # Seed one write first: the first emit rotates, and rotation is what clears
    # the cap flag for a new day. After that the API's own record cannot reset it.
    configured.audit.emit("seed")
    configured.audit.full = True

    audited = client.get("/api/audit", headers=audit_headers(configured)).json()
    assert audited["degraded"] is True

    # The app-token surface gets the boolean only: the failure text and the
    # audit path stay behind the audit credential.
    status = client.get("/api/status").json()
    assert status["audit_degraded"] is True


def test_verify_and_head_endpoints_need_the_audit_token(configured):
    with TestClient(s.app) as anon:
        assert anon.get("/api/audit/verify").status_code == 401
        assert anon.get("/api/audit/head").status_code == 401


def test_verify_and_head_endpoints_report_a_clean_day(client, configured):
    configured.audit.emit("job.created", job=JOB)

    verified = client.get("/api/audit/verify", headers=audit_headers(configured))
    assert verified.status_code == 200
    body = verified.json()
    assert body["ok"] is True and body["checked"] >= 1

    head = client.get("/api/audit/head", headers=audit_headers(configured)).json()
    assert head["chain"] == configured.audit.head(head["date"])["chain"]


def test_verify_endpoint_rejects_a_bad_date_and_an_unknown_day(client, configured):
    headers = audit_headers(configured)
    assert (
        client.get("/api/audit/verify?date=2026-9-1", headers=headers).status_code
        == 400
    )
    assert (
        client.get("/api/audit/verify?date=1999-01-01", headers=headers).status_code
        == 404
    )
    assert (
        client.get("/api/audit/head?date=1999-01-01", headers=headers).status_code
        == 404
    )


def test_a_bulk_prompt_read_names_the_jobs_it_disclosed(client, configured):
    """The text is gated, so a read that hands it over must say whose it was."""
    job_id = configured.make_job(prompt="Project Nightjar", hotwords="Nightjar")
    opts = s.JOBS[job_id]["opts"]
    s.store_prompt_sidecar(job_id, "meeting.wav", opts, source="upload")
    # The sidecar is keyed by job id, so the read needs a record that names it.
    configured.audit.emit("job.created", job=job_id, file="meeting.wav")

    response = client.get(
        "/api/audit", params={"include_prompts": 1}, headers=audit_headers(configured)
    )
    assert response.status_code == 200
    read = [e for e in response.json()["events"] if e["event"] == "job.created"]
    assert read and read[0]["prompt_text"] == "Project Nightjar"

    accessed = [e for e in configured.events() if e["event"] == "audit.accessed"][-1]
    assert accessed["prompts_returned"] == 1
    assert accessed["prompt_jobs"] == [job_id]
    # The text itself is never copied into the authoritative record.
    assert "Project Nightjar" not in json.dumps(accessed)


def test_startup_snapshot_records_where_the_trail_lives_and_whether_the_token_was_new(
    tmp_path, configured, monkeypatch
):
    monkeypatch.delenv("TRANSCRIBE_AUDIT_TOKEN", raising=False)
    cfg = tmp_path / "c.toml"
    cfg.write_text("")
    args, _, _ = s.resolve_args(["--config", str(cfg)])

    snapshot = s.startup_snapshot(args, cfg, True)
    assert snapshot["audit_generated"] is True
    assert snapshot["audit_dir"], "a custom audit dir must be recoverable"
    assert "audit" in str(snapshot["audit_dir"])


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_cap_layers_and_refuses_a_negative(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_AUDIT_MAX_MB", raising=False)
    cfg = tmp_path / "c.toml"
    cfg.write_text("")

    args, _, _ = s.resolve_args(["--config", str(cfg)])
    assert args.audit_max_mb == 1024, "a generous cap is the default"

    cfg.write_text("[audit]\nmax_mb = 512\n")
    args, _, _ = s.resolve_args(["--config", str(cfg)])
    assert args.audit_max_mb == 512

    monkeypatch.setenv("TRANSCRIBE_AUDIT_MAX_MB", "7")
    args, _, _ = s.resolve_args(["--config", str(cfg)])
    assert args.audit_max_mb == 7, "the environment wins over the file"

    monkeypatch.delenv("TRANSCRIBE_AUDIT_MAX_MB")
    cfg.write_text("[audit]\nmax_mb = -1\n")
    with pytest.raises(SystemExit) as exc:
        s.resolve_args(["--config", str(cfg)])
    assert "cannot be negative" in str(exc.value)


def test_the_starter_config_names_the_cap():
    assert "max_mb = 1024" in s.CONFIG_TEMPLATE
