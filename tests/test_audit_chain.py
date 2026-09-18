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
import os
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


def test_a_write_that_cannot_be_rolled_back_keeps_the_next_record_readable(tmp_path):
    """A torn line plus a failed rollback must not swallow the next record.

    The chain stays consistent either way, because _append commits after the
    write. But the `lost` attestation rides the next record, and concatenating
    onto the damage would bury it inside an unparsable line.
    """
    log = fresh(tmp_path)
    log.emit("a")
    real = log._fh
    assert real is not None

    class TornNoTruncate:
        """Die mid-record, and refuse to be cut back."""

        def write(self, text: str) -> None:
            real.write(text[: len(text) // 2])
            raise OSError("disk full")

        def flush(self) -> None:
            real.flush()

        def truncate(self, size: int) -> None:
            raise OSError("cannot truncate")

    log._fh = TornNoTruncate()  # pyright: ignore[reportAttributeAccessIssue]
    log.emit("b")
    log._fh = real
    log.emit("c")

    lines = [x for x in day_file(log).read_text(encoding="utf-8").splitlines() if x]
    assert len(lines) == 3, "the damage must stay on its own line"
    with pytest.raises(ValueError):
        json.loads(lines[1])  # the torn half, honestly unreadable
    good = json.loads(lines[2])
    assert good["event"] == "c"
    assert good["seq"] == 2, "the torn write must not advance the chain"
    assert good["lost"] == 1, "the attestation must survive the damage"


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
    assert body["sidecar_lost_total"] == 0
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
    # Against the file itself, not against AUDIT.head() -- comparing the endpoint
    # to the method it calls only restates the implementation.
    path = configured.audit.dir / f"audit-{head['date']}.jsonl"
    last = records(path)[-1]
    assert last["event"] == "audit.head_read", "the read is logged before the head"
    assert head["chain"] == last["chain"]


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
# Adversarial-review findings
# --------------------------------------------------------------------------- #


def test_rollback_runs_with_the_day_lock_held(tmp_path, monkeypatch):
    """The rollback must not run after the lock is released.

    Released first, a concurrent emit appends onto the torn fragment and the
    rollback's truncate then slices that innocent record in half.
    """
    log = fresh(tmp_path)
    log.emit("a")
    seen: dict[str, bool] = {}
    original = s.AuditLog._rollback

    def spy(self: s.AuditLog) -> None:
        seen["locked"] = self._lock.locked()
        original(self)

    monkeypatch.setattr(s.AuditLog, "_rollback", spy)

    class Boom:
        def write(self, text: str) -> None:
            raise OSError("disk full")

        def flush(self) -> None:
            pass

        def truncate(self, size: int) -> None:
            pass

    log._fh = Boom()  # pyright: ignore[reportAttributeAccessIssue]
    log.emit("b")
    assert seen.get("locked") is True, "the rollback ran outside _lock"


def test_rollback_to_a_torn_tail_does_not_claim_the_file_is_clean(tmp_path):
    """A successful truncate does not make a torn file clean.

    After a crash the file already ends mid-record. Rolling back to that same
    length and declaring it clean is how the next record merges into the torn
    fragment and buries its `lost` attestation.
    """
    log = fresh(tmp_path)
    log.emit("a")
    path = day_file(log)
    path.write_text(path.read_text(encoding="utf-8")[:-10])  # crash mid-record

    resumed = fresh(tmp_path)
    day = day_of(path)
    resumed._rotate(day)
    resumed._day = day  # emit will not re-rotate and re-resume
    assert resumed._clean is False, "a torn tail must resume as dirty"
    real = resumed._fh
    assert real is not None

    class Boom:
        """Fails before writing anything; the rollback truncate succeeds."""

        def write(self, text: str) -> None:
            raise OSError("disk full")

        def flush(self) -> None:
            pass

        def truncate(self, size: int) -> None:
            real.truncate(size)

    resumed._fh = Boom()  # pyright: ignore[reportAttributeAccessIssue]
    resumed.emit("b")
    assert resumed._clean is False, "the torn tail is still torn after rollback"

    resumed._fh = real
    resumed.emit("c")
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x]
    assert json.loads(lines[-1])["event"] == "c", "c must not join the torn line"


def test_a_stripped_chain_field_after_the_chain_began_is_a_break(tmp_path):
    """Removing `chain` is what an editor does to hide a change to the last line."""
    log = fresh(tmp_path)
    for i in range(3):
        log.emit("e", n=i)
    path = day_file(log)
    lines = path.read_text(encoding="utf-8").splitlines()
    tail = json.loads(lines[-1])
    del tail["chain"]
    tail["n"] = 999
    lines[-1] = json.dumps(tail)
    path.write_text("\n".join(lines) + "\n")

    result = log.verify(day_of(path))
    assert result["ok"] is False
    assert result["first_bad_line"] == 3
    assert "chain" in result["reason"]


def test_legacy_records_before_the_chain_are_still_tolerated(tmp_path):
    """The strip check must not turn a pre-chain day into a failure."""
    log = fresh(tmp_path)
    log.dir.mkdir(parents=True, exist_ok=True)
    path = log.dir / "audit-2020-03-03.jsonl"
    path.write_text(
        json.dumps({"ts": "2020-03-03T00:00:00.000Z", "event": "old"}) + "\n"
    )
    log._rotate("2020-03-03")
    log._append(log._fh, {"ts": "2020-03-03T00:00:01.000Z", "event": "new"})

    result = log.verify("2020-03-03")
    assert result["ok"] is True and result["legacy"] == 1 and result["checked"] == 1


def test_carry_ok_is_indeterminate_when_the_previous_day_is_gone(tmp_path):
    """Retention deletes the previous day routinely; that is not a mismatch."""
    log = fresh(tmp_path)
    log.dir.mkdir(parents=True, exist_ok=True)
    previous = log.dir / "audit-2020-01-01.jsonl"
    previous.write_text(
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
    log._append(log._fh, {"ts": "2020-01-02T00:00:00.000Z", "event": "new"})
    assert log.verify("2020-01-02")["carry_ok"] is True

    previous.unlink()  # what retention does to the oldest surviving day
    assert log.verify("2020-01-02")["carry_ok"] is None, "absent is not a mismatch"


def test_a_sidecar_failure_degrades_the_trail(tmp_path, monkeypatch):
    """A prompt that was not stored must not look like a quiet day."""
    log = fresh(tmp_path)
    assert log.degraded() is False

    fail = {"on": True}
    real_replace = s.os.replace

    def flaky(*args: Any, **kwargs: Any) -> None:
        if fail["on"]:
            raise OSError("no space left on device")
        real_replace(*args, **kwargs)

    monkeypatch.setattr(s.os, "replace", flaky)
    log.write_prompt(JOB, {"job": JOB, "prompt": "secret"})
    assert log.sidecar_lost == 1 and log.sidecar_lost_total == 1
    assert log.degraded() is True, "a dropped prompt is a degraded trail"
    assert not list(log.prompts_dir.glob(".*.json.tmp")), "no scratch file left behind"

    fail["on"] = False
    log.write_prompt(JOB, {"job": JOB, "prompt": "secret"})
    assert log.sidecar_lost == 0 and log.degraded() is False
    assert log.read_prompt(JOB) == {"job": JOB, "prompt": "secret"}


def test_naming_a_speaker_goes_through_the_locked_amend(configured, monkeypatch):
    """The production read-modify-write must be the locked one.

    Testing amend_prompt on its own proves nothing about the call site that
    needs it: read_prompt-then-write_prompt is two lock acquisitions and loses
    whichever writer lands first.
    """
    calls: list[str] = []
    original = s.AuditLog.amend_prompt

    def spy(self: s.AuditLog, job_id: str, default: Any, changes: Any) -> None:
        calls.append(job_id)
        original(self, job_id, default, changes)

    monkeypatch.setattr(s.AuditLog, "amend_prompt", spy)
    s.store_speaker_names(JOB, "meeting.wav", {"1": "Alice"})
    assert calls == [JOB]


def test_naming_a_speaker_keeps_a_stored_prompt(configured):
    job_id = configured.make_job(prompt="Project Nightjar")
    opts = s.JOBS[job_id]["opts"]
    s.store_prompt_sidecar(job_id, "meeting.wav", opts, source="upload")
    s.store_speaker_names(job_id, "meeting.wav", {"1": "Alice"})

    record = configured.audit.read_prompt(job_id)
    assert record is not None
    assert record["prompt"] == "Project Nightjar"
    assert record["speaker_names"] == {"1": "Alice"}


def test_a_caller_cannot_forge_chain_fields(tmp_path):
    """Reserved fields are the trail's to assign, not the caller's."""
    log = fresh(tmp_path)
    log.emit(
        "e",
        ts="1970-01-01T00:00:00Z",
        seq=42,
        chain="forged",
        carry="forged",
        lost=99,
        chain_reset=True,
    )
    path = day_file(log)
    rec = records(path)[0]
    assert rec["seq"] == 1, "seq is assigned by the trail"
    assert rec["chain"] != "forged"
    assert rec.get("carry") != "forged"
    assert "lost" not in rec
    assert "chain_reset" not in rec
    assert not str(rec["ts"]).startswith("1970")
    assert log.verify(day_of(path))["ok"] is True


def test_a_record_larger_than_the_tail_window_is_not_read_as_torn(tmp_path):
    """A complete record longer than the 256 KiB window must not force a reset."""
    log = fresh(tmp_path)
    log.emit("small")
    log.emit("huge", blob="x" * 300_000)

    resumed = fresh(tmp_path)
    resumed.emit("next")
    recs = records(day_file(resumed))
    assert [r["event"] for r in recs] == ["small", "huge", "next"]
    assert "chain_reset" not in recs[-1]
    assert resumed.verify(day_of(day_file(resumed)))["ok"] is True


def test_sidecars_are_capped_oldest_first(tmp_path):
    """A bounded day file must not sit beside an unbounded prompts/ directory."""
    log = fresh(tmp_path, max_sidecars=3)
    ids = [f"job{i:09d}" for i in range(5)]
    for i, jid in enumerate(ids):
        log.write_prompt(jid, {"job": jid, "prompt": f"p{i}"})
        # Deterministic ordering: mtime granularity is too coarse for a loop.
        os.utime(log.prompts_dir / f"{jid}.json", (1000 + i, 1000 + i))

    remaining = sorted(p.stem for p in log.prompts_dir.glob("*.json"))
    assert remaining == ids[-3:], "the oldest text goes first"
    newest = log.read_prompt(ids[-1])
    assert newest is not None and newest["prompt"] == "p4"
    assert log.read_prompt(ids[0]) is None
    # The evidence a prompt was used is the day file's, so it survives.
    assert log.verify(day_of(day_file(log)))["ok"] is True


def test_zero_max_sidecars_keeps_everything(tmp_path):
    log = fresh(tmp_path, max_sidecars=0)
    for i in range(10):
        log.write_prompt(f"job{i:09d}", {"job": f"job{i:09d}"})
    assert len(list(log.prompts_dir.glob("*.json"))) == 10


def test_sidecar_capping_is_recorded_in_the_trail(tmp_path):
    log = fresh(tmp_path, max_sidecars=1)
    for i in range(4):
        jid = f"job{i:09d}"
        log.write_prompt(jid, {"job": jid})
        os.utime(log.prompts_dir / f"{jid}.json", (1000 + i, 1000 + i))
    events = [r["event"] for r in records(day_file(log))]
    assert "audit.sidecars_pruned" in events


# --------------------------------------------------------------------------- #
# Cursor paging
# --------------------------------------------------------------------------- #


def test_a_cursor_page_is_stable_when_new_records_land(tmp_path):
    """The whole point of a front-anchored cursor over an offset."""
    log = fresh(tmp_path)
    for i in range(6):
        log.emit("e", n=i)
    day = day_of(day_file(log))

    first = log.read_page(day, limit=3)
    assert [json.loads(x)["n"] for x in first.lines] == [5, 4, 3]
    assert first.has_more is True and isinstance(first.cursor, int)

    log.emit("e", n=6)  # a record lands between pages

    second = log.read_page(day, limit=3, before_line=first.cursor)
    assert [json.loads(x)["n"] for x in second.lines] == [2, 1, 0], (
        "a new record must not duplicate or skip a row"
    )
    assert second.has_more is False


def test_read_still_returns_the_old_shape(tmp_path):
    log = fresh(tmp_path)
    for i in range(3):
        log.emit("e", n=i)
    lines, total = log.read(day_of(day_file(log)), limit=2)
    assert total == 3 and len(lines) == 2


def test_the_audit_api_hands_back_a_usable_cursor(client, configured):
    for i in range(3):
        configured.audit.emit("e", n=i)

    first = client.get(
        "/api/audit", params={"limit": 2}, headers=audit_headers(configured)
    ).json()
    assert first["has_more"] is True
    assert isinstance(first["next_before_line"], int)

    second = client.get(
        "/api/audit",
        params={"limit": 2, "before_line": first["next_before_line"]},
        headers=audit_headers(configured),
    ).json()
    seen = {e.get("n") for e in first["events"]} & {
        e.get("n") for e in second["events"]
    }
    assert not seen, "the two pages must not overlap"
    assert second["has_more"] is False


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
    assert "max_sidecars = 5000" in s.CONFIG_TEMPLATE


def test_the_sidecar_cap_layers_and_refuses_a_negative(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_AUDIT_MAX_SIDECARS", raising=False)
    cfg = tmp_path / "s.toml"
    cfg.write_text("")

    args, _, _ = s.resolve_args(["--config", str(cfg)])
    assert args.audit_max_sidecars == 5000

    cfg.write_text("[audit]\nmax_sidecars = 10\n")
    args, _, _ = s.resolve_args(["--config", str(cfg)])
    assert args.audit_max_sidecars == 10

    monkeypatch.setenv("TRANSCRIBE_AUDIT_MAX_SIDECARS", "3")
    args, _, _ = s.resolve_args(["--config", str(cfg)])
    assert args.audit_max_sidecars == 3

    monkeypatch.delenv("TRANSCRIBE_AUDIT_MAX_SIDECARS")
    cfg.write_text("[audit]\nmax_sidecars = -5\n")
    with pytest.raises(SystemExit) as exc:
        s.resolve_args(["--config", str(cfg)])
    assert "cannot be negative" in str(exc.value)
