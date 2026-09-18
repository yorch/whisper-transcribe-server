"""Regression tests for the defects found in the multi-angle code review.

Each test here corresponds to a finding that was confirmed by reading the code
and, for the P0, reproduced before the fix. They are written against the real
functions, not copies, so they fail if the behaviour regresses.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import transcribe_server as s


# --------------------------------------------------------------------------- #
# P0: self-deadlock on the non-reentrant JOBS_LOCK
# --------------------------------------------------------------------------- #


class FakeSeg:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.words = None


class FakeInfo:
    duration = 10.0
    language = "en"


class FakeModel:
    """Stands in for faster-whisper so run_job can be driven without a GPU."""

    def __init__(self, segments) -> None:
        self._segments = segments

    def transcribe(self, path, **kwargs):  # noqa: ARG002
        return self._segments, FakeInfo()


def test_cancel_mid_transcribe_does_not_deadlock(configured, monkeypatch):
    """The P0: drop_source re-entered JOBS_LOCK from inside `with JOBS_LOCK`.

    A cancel arriving mid-transcribe used to wedge the worker thread forever
    with the lock held, which also froze the event loop (async endpoints take
    the same lock) and so the entire server.
    """
    monkeypatch.setattr(s.ARGS, "source_retention", "run")
    job_id = configured.make_job("long.wav")

    def segments():
        yield FakeSeg(0.0, 1.0, "one")
        # DELETE /api/jobs/{id} lands while we are mid-transcribe.
        s.patch_job(job_id, state="cancelled", message="Cancelled")
        yield FakeSeg(1.0, 2.0, "two")  # the next iteration must notice

    monkeypatch.setattr(s, "load_model", lambda *a, **k: FakeModel(segments()))

    finished = threading.Event()

    def work() -> None:
        s.run_job(job_id)
        finished.set()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()

    assert finished.wait(timeout=10), "run_job deadlocked after a mid-run cancel"
    assert not s.JOBS_LOCK.locked(), "JOBS_LOCK was left held"
    # A cancelled job must not be reported as done.
    assert s.JOBS[job_id]["state"] == "cancelled"
    # ...and with retention=run the source is still cleaned up.
    assert not (s.UPLOAD_DIR / "long.wav").exists()


def test_cancel_during_model_load_is_honoured(configured, monkeypatch):
    """A cancel during a slow load used to be overwritten by state="running"."""
    job_id = configured.make_job("slow.wav")

    def slow_load(*args, **kwargs):  # noqa: ARG001
        # The user hits Cancel while large-v3 is still loading.
        s.patch_job(job_id, state="cancelled", message="Cancelled")
        return FakeModel(iter([FakeSeg(0.0, 1.0, "never")]))

    monkeypatch.setattr(s, "load_model", slow_load)
    s.run_job(job_id)

    assert s.JOBS[job_id]["state"] == "cancelled"


def test_cancelled_job_is_never_moved_to_another_state(configured):
    job_id = configured.make_job()
    s.patch_job(job_id, state="cancelled")

    assert s.patch_job(job_id, state="done", progress=1.0) is False
    assert s.patch_job(job_id, state="error") is False
    assert s.patch_job(job_id, state="running") is False
    assert s.JOBS[job_id]["state"] == "cancelled"

    # A field update that does not change state is still allowed: the guard is
    # about the state machine, not about freezing the record.
    assert s.patch_job(job_id, progress=0.5) is True
    assert s.JOBS[job_id]["state"] == "cancelled"

    # A normal transition still works.
    other = configured.make_job("other.wav")
    assert s.patch_job(other, state="running") is True


def test_patch_job_reports_unknown_jobs(configured):
    assert s.patch_job("deadbeef", state="done") is False


def test_job_cancelled_covers_eviction(configured):
    job_id = configured.make_job()
    assert s.job_cancelled(job_id) is False
    s.patch_job(job_id, state="cancelled")
    assert s.job_cancelled(job_id) is True
    with s.JOBS_LOCK:
        s.JOBS.pop(job_id)
    assert s.job_cancelled(job_id) is True, "an evicted job must stop the worker"


# --------------------------------------------------------------------------- #
# P1: prompt/hotword text must not reach an app-token holder
# --------------------------------------------------------------------------- #

PROMPT_TEXT = "A call about Project Nightjar and the Acme merger."
TERMS_TEXT = "Nightjar,Acme"


def test_public_opts_replaces_prompt_text_with_length_and_flags(configured):
    job_id = configured.make_job(prompt=PROMPT_TEXT, hotwords=TERMS_TEXT)
    opts = s.public_opts(s.JOBS[job_id]["opts"])

    assert "prompt" not in opts and "hotwords" not in opts
    assert opts["has_prompt"] is True and opts["has_hotwords"] is True
    assert opts["prompt_len"] == len(PROMPT_TEXT)
    assert opts["hotwords_len"] == len(TERMS_TEXT)
    assert opts["model"] == "base", "non-sensitive knobs must survive"


def test_job_public_never_serialises_prompt_text(configured):
    job_id = configured.make_job(prompt=PROMPT_TEXT, hotwords=TERMS_TEXT)
    body = json.dumps(s.job_public(s.JOBS[job_id]))

    assert PROMPT_TEXT not in body
    assert TERMS_TEXT not in body
    assert "Nightjar" not in body


def test_json_export_hides_prompt_text(configured):
    job_id = configured.make_job(prompt=PROMPT_TEXT, hotwords=TERMS_TEXT)
    s.patch_job(job_id, state="done", segments=[])
    body, _ = s.render(s.JOBS[job_id], "json")

    assert PROMPT_TEXT not in body
    assert "Nightjar" not in body
    assert json.loads(body)["options"]["has_prompt"] is True


def test_audit_opts_hashes_prompt_text(configured):
    """The main trail carries a length and a hash, never the text itself."""
    job_id = configured.make_job(prompt=PROMPT_TEXT, hotwords=TERMS_TEXT)
    record = s.audit_opts(s.JOBS[job_id]["opts"])

    assert "prompt" not in record and "hotwords" not in record
    assert record["prompt_sha256"] == s.digest(PROMPT_TEXT)
    assert record["hotwords_sha256"] == s.digest(TERMS_TEXT)
    assert record["prompt_len"] == len(PROMPT_TEXT)
    assert PROMPT_TEXT not in json.dumps(record)


def test_sidecar_holds_the_full_text(configured):
    job_id = configured.make_job(prompt=PROMPT_TEXT, hotwords=TERMS_TEXT)
    opts = s.JOBS[job_id]["opts"]
    s.store_prompt_sidecar(job_id, "meeting.wav", opts, source="upload")

    stored = configured.audit.read_prompt(job_id)
    assert stored is not None
    assert stored["prompt"] == PROMPT_TEXT
    assert stored["hotwords"] == TERMS_TEXT
    assert stored["source"] == "upload"


def test_sidecar_is_skipped_when_there_is_nothing_to_store(configured):
    job_id = configured.make_job()
    s.store_prompt_sidecar(job_id, "meeting.wav", s.JOBS[job_id]["opts"], "upload")
    assert configured.audit.read_prompt(job_id) is None


# --------------------------------------------------------------------------- #
# Host allowlist
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "header,expected",
    [
        ("localhost", True),
        ("localhost:8765", True),
        ("127.0.0.1:8765", True),
        ("[::1]:8765", True),
        ("::1", True),
        ("LOCALHOST", True),
        ("localhost.", True),
        ("https://localhost/path", True),
        ("evil.com", False),
        ("", False),
        (None, False),
        ("evil.com:80:trycloudflare.com", False),  # malformed, was accepted
        ("127.0.0.1:8765:evil", False),
    ],
)
def test_host_allowed_exact_names(configured, header, expected):
    assert s.host_allowed(header) is expected


@pytest.mark.parametrize(
    "header,expected",
    [
        ("abc-123.trycloudflare.com", True),
        ("abc-123.trycloudflare.com:443", True),
        ("https://abc-123.trycloudflare.com/x", True),
        ("trycloudflare.com", True),  # the bare suffix itself
        ("eviltrycloudflare.com", False),  # classic missing-dot bypass
        ("trycloudflare.com.evil.dev", False),
        ("evil.dev/?x=.trycloudflare.com", False),
        ("x:1:trycloudflare.com", False),
    ],
)
def test_host_allowed_suffix_matching(configured, monkeypatch, header, expected):
    monkeypatch.setattr(s, "ALLOWED_SUFFIXES", {".trycloudflare.com"})
    assert s.host_allowed(header) is expected


def test_normalize_host():
    assert s.normalize_host("https://Example.COM:8443/a/b") == "example.com"
    assert s.normalize_host("[::1]:8765") == "::1"
    assert s.normalize_host("Host.Trailing.") == "host.trailing"
    assert s.normalize_host("  spaced  ") == "spaced"


# --------------------------------------------------------------------------- #
# Audit log behaviour
# --------------------------------------------------------------------------- #


def test_audit_roundtrip_and_filters(configured):
    # Emit directly: job.created comes from the endpoint, which the endpoint
    # tests cover. This is about AuditLog.read() itself.
    configured.audit.emit("job.created", job="aaa", file="a.wav")
    configured.audit.emit("job.created", job="bbb", file="b.wav")
    day = time.strftime("%Y-%m-%d", time.gmtime())

    lines, total = configured.audit.read(day)
    assert total == 2
    assert [json.loads(line)["job"] for line in lines] == ["bbb", "aaa"]

    only, total_one = configured.audit.read(day, job="aaa")
    assert total_one == 1 and json.loads(only[0])["job"] == "aaa"

    found, total_q = configured.audit.read(day, needle="b.wav")
    assert total_q == 1 and "b.wav" in found[0]


def test_audit_read_holds_only_the_requested_page(configured):
    """read() used to materialise the whole day file before slicing."""
    day = time.strftime("%Y-%m-%d", time.gmtime())
    for i in range(500):
        configured.audit.emit("noise", n=i)

    page, total = configured.audit.read(day, limit=10, offset=0)
    assert total == 500
    assert len(page) == 10
    assert json.loads(page[0])["n"] == 499, "newest first"

    second, _ = configured.audit.read(day, limit=10, offset=10)
    assert json.loads(second[0])["n"] == 489


def test_prune_is_skipped_when_auditing_is_disabled(tmp_path):
    """--no-audit must not delete a trail it is not managing."""
    root = tmp_path / "audit"
    (root / "prompts").mkdir(parents=True)
    stale = root / "audit-2000-01-01.jsonl"
    stale.write_text('{"event":"old"}\n')

    off = s.AuditLog(root, enabled=False, retain_days=1, prompts=True)
    assert off.prune() == []
    assert stale.exists()

    on = s.AuditLog(root, enabled=True, retain_days=1, prompts=True)
    assert "audit-2000-01-01.jsonl" in on.prune()
    assert not stale.exists()


def test_prune_removes_sidecars_past_retention(tmp_path):
    import os

    root = tmp_path / "audit"
    (root / "prompts").mkdir(parents=True)
    sidecar = root / "prompts" / "deadbeef.json"
    sidecar.write_text('{"prompt":"old"}')
    old = time.time() - 90 * 86400
    os.utime(sidecar, (old, old))

    log = s.AuditLog(root, enabled=True, retain_days=30, prompts=True)
    assert "deadbeef.json" in log.prune()
    assert not sidecar.exists()


def test_rotation_never_goes_backwards(configured):
    """A writer that computed its day before midnight must not flip the handle back.

    The real race is two threads straddling the UTC boundary. The older writer
    must append to the newer file rather than rotating the handle backwards.
    """
    log = configured.audit
    log.emit("first")
    current = log._day
    assert current is not None
    handle_before = log._fh

    # Pretend a stale writer computed a day *earlier* than the live one.
    log._day = current
    log._stale_day_for_test = "2000-01-01"
    stale = "2000-01-01"
    assert stale < current

    # Emit while the handle is on `current`; a backwards rotate would create
    # an audit-2000-01-01.jsonl and split the day.
    log.emit("second")
    assert log._day == current, "the live day must not move backwards"
    assert log._fh is handle_before, "the handle must not be re-opened"
    assert not (log.dir / f"audit-{stale}.jsonl").exists()
    assert (log.dir / f"audit-{current}.jsonl").exists()


def test_emit_force_writes_even_when_disabled(tmp_path):
    """A disabled run still leaves one line, so a gap is attributable."""
    log = s.AuditLog(tmp_path / "audit", enabled=False, retain_days=30, prompts=True)
    log.emit("should.not.appear")
    log.emit("server.started", force=True, audit=False)

    files = list((tmp_path / "audit").glob("audit-*.jsonl"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "should.not.appear" not in text
    assert "server.started" in text


def test_sidecar_write_is_refused_when_prompts_disabled(tmp_path):
    log = s.AuditLog(tmp_path / "audit", enabled=True, retain_days=30, prompts=False)
    log.write_prompt("abc123", {"prompt": "secret"})
    assert log.read_prompt("abc123") is None


@pytest.mark.parametrize("bad", ["../../etc/passwd", "..%2f..", "a/b", "", "x" * 200])
def test_sidecar_job_id_validation(configured, bad):
    assert configured.audit.read_prompt(bad) is None
    configured.audit.write_prompt(bad, {"prompt": "nope"})
    assert not any(configured.audit.prompts_dir.glob("*.json"))


def test_audit_failure_is_reported_and_not_fatal(tmp_path, monkeypatch):
    """A broken audit sink must never break a request, but must be visible."""
    log = s.AuditLog(tmp_path / "audit", enabled=True, retain_days=30, prompts=True)
    monkeypatch.setattr(
        s.AuditLog,
        "_rotate",
        lambda self, day: (_ for _ in ()).throw(OSError("disk full")),
    )
    log.emit("job.created", job="x")  # must not raise
    assert log.last_error is not None and "disk full" in log.last_error


# --------------------------------------------------------------------------- #
# Rejection-flood collapsing
# --------------------------------------------------------------------------- #


class FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class FakeRequest:
    def __init__(self, host: str = "203.0.113.9", path: str = "/api/jobs") -> None:
        self.client = FakeClient(host)
        self.headers = {"host": "localhost"}
        self.method = "GET"
        self.url = type("U", (), {"path": path})()
        self.state = type("S", (), {})()


def as_request(fake: FakeRequest) -> Any:
    """These fakes implement only the attributes request_fields touches."""
    return fake


def test_rejection_burst_is_collapsed(configured, monkeypatch):
    """Unauthenticated 401s must not be able to fill the disk line by line."""
    monkeypatch.setattr(s, "_REJECT_STATE", {})
    request = as_request(FakeRequest())

    for _ in range(50):
        s.audit_rejection(
            "security.auth_failed", request, reason="bad-token", status=401
        )

    events = [e for e in configured.events() if e["event"].startswith("security.")]
    logged = [e for e in events if e["event"] == "security.auth_failed"]
    summaries = [e for e in events if e["event"] == "security.rejected_summary"]

    assert len(logged) == s.REJECT_LOG_LIMIT, "only the first few are logged"
    assert not summaries, "the summary is written when the window rolls, not before"
    assert len(events) < 10, f"a 50-request burst produced {len(events)} records"


def test_request_fields_truncates_untrusted_values(configured):
    request = as_request(FakeRequest(host="198.51.100.7", path="/api/" + "x" * 5000))
    fields = s.request_fields(request)

    assert len(fields["path"]) <= 200
    assert fields["client"] == "198.51.100.7"
    assert fields["client_claimed"] is None


def test_client_claimed_is_recorded_but_marked_separate(configured):
    fake = FakeRequest(host="127.0.0.1")
    fake.headers["cf-connecting-ip"] = "203.0.113.55"
    fields = s.request_fields(as_request(fake))

    assert fields["client"] == "127.0.0.1"
    assert fields["client_claimed"] == "203.0.113.55"


# --------------------------------------------------------------------------- #
# Config layering
# --------------------------------------------------------------------------- #


def test_config_precedence_defaults_file_flags_env(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[server]\nport = 1111\nmodel = "medium"\n[limits]\nmax_jobs = 7\n')

    for var in ("TRANSCRIBE_PORT", "TRANSCRIBE_MODEL"):
        monkeypatch.delenv(var, raising=False)

    args, path, _ = s.resolve_args(["--config", str(cfg)])
    assert (args.port, args.model, args.max_jobs) == (1111, "medium", 7)
    assert path == cfg

    args, _, _ = s.resolve_args(["--config", str(cfg), "--port", "2222"])
    assert args.port == 2222 and args.model == "medium"

    monkeypatch.setenv("TRANSCRIBE_PORT", "3333")
    args, _, _ = s.resolve_args(["--config", str(cfg), "--port", "2222"])
    assert args.port == 3333, "environment must win over flags"


def test_allow_host_is_additive_across_layers(tmp_path, monkeypatch):
    """A flag used to silently replace the configured allow-list."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('[server]\nallow_host = [".trycloudflare.com"]\n')
    monkeypatch.delenv("TRANSCRIBE_ALLOW_HOST", raising=False)

    args, _, _ = s.resolve_args(["--config", str(cfg), "--allow-host", "box.local"])
    assert ".trycloudflare.com" in args.allow_host
    assert "box.local" in args.allow_host

    monkeypatch.setenv("TRANSCRIBE_ALLOW_HOST", "env.example.com")
    args, _, _ = s.resolve_args(["--config", str(cfg), "--allow-host", "box.local"])
    assert set(args.allow_host) == {
        ".trycloudflare.com",
        "box.local",
        "env.example.com",
    }


def test_work_dir_flag_moves_the_default_config_path(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_CONFIG", raising=False)
    monkeypatch.delenv("TRANSCRIBE_WORK_DIR", raising=False)
    args, path, _ = s.resolve_args(["--work-dir", str(tmp_path / "state")])
    assert path == tmp_path / "state" / "config.toml"
    assert args.work_dir == str(tmp_path / "state")


def toml_str(value: object) -> str:
    """A value as a valid TOML *basic* string.

    Interpolating a path straight into double quotes breaks on Windows: the
    basic string `"C:\\Users\\x"` reads `\\U` as a unicode escape and the file
    is rejected as invalid TOML. Escaping the backslash (and any quote) is what
    the format actually requires, and it is the difference between these tests
    passing on the platform this project targets and not.
    """
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def default_config(tmp_path, monkeypatch, body: str) -> Path:
    """Write a config at the *default* location, <home>/.transcribe-server.

    HOME is redirected instead of passing --work-dir, because a flag wins over
    the file and would hide the case under test (the file being the only thing
    that sets work_dir).
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("TRANSCRIBE_CONFIG", raising=False)
    monkeypatch.delenv("TRANSCRIBE_WORK_DIR", raising=False)
    cfg = tmp_path / ".transcribe-server" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(body, encoding="utf-8")
    return cfg


def test_work_dir_in_the_default_config_is_a_startup_error(tmp_path, monkeypatch):
    """The file cannot move the state dir it was found in; see the function."""
    cfg = default_config(
        tmp_path, monkeypatch, '[server]\nwork_dir = "/mnt/big/state"\n'
    )

    with pytest.raises(SystemExit) as exc:
        s.resolve_args([])

    message = str(exc.value)
    assert str(cfg) in message, "the error has to name the file to fix"
    assert "--work-dir" in message, "and say what to do instead"


def test_work_dir_equal_to_the_config_dir_is_allowed(tmp_path, monkeypatch):
    """A no-op is not a mistake: the file already sits where it says state goes."""
    home = tmp_path / ".transcribe-server"
    default_config(tmp_path, monkeypatch, f"[server]\nwork_dir = {toml_str(home)}\n")

    args, path, required = s.resolve_args([])
    assert required is False
    assert path == home / "config.toml"
    assert args.work_dir == str(home)


def test_a_windows_path_in_a_config_round_trips(tmp_path, monkeypatch):
    """A path with backslashes has to survive the config file.

    Interpolated into a TOML basic string unescaped it does not: `\\U` reads as a
    unicode escape and the file is rejected before the server starts. This is the
    platform the project targets, so the capability is worth pinning rather than
    leaving to whichever machine runs the suite.
    """
    monkeypatch.delenv("TRANSCRIBE_CONFIG", raising=False)
    win = r"C:\Users\somebody\.transcribe-server"
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"[server]\nwork_dir = {toml_str(win)}\n", encoding="utf-8")

    args, _, _ = s.resolve_args(["--config", str(cfg)])

    assert args.work_dir == win


def test_a_windows_path_in_a_literal_toml_string_also_works(tmp_path, monkeypatch):
    """Single quotes are the friendlier form for a path; they must work too."""
    monkeypatch.delenv("TRANSCRIBE_CONFIG", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[server]\nwork_dir = 'C:\\Users\\somebody\\state'\n", encoding="utf-8")

    args, _, _ = s.resolve_args(["--config", str(cfg)])

    assert args.work_dir == r"C:\Users\somebody\state"


def test_work_dir_from_the_environment_settles_the_default_config(
    tmp_path, monkeypatch
):
    """The check is on the effective value: env wins, so a stale key is harmless."""
    monkeypatch.delenv("TRANSCRIBE_CONFIG", raising=False)
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text('[server]\nwork_dir = "/mnt/big/state"\n')
    monkeypatch.setenv("TRANSCRIBE_WORK_DIR", str(state))

    args, path, _ = s.resolve_args([])
    assert path == state / "config.toml"
    assert args.work_dir == str(state)


def test_work_dir_in_an_explicit_config_is_allowed(tmp_path, monkeypatch):
    """An explicit --config does not live in the state dir, so the key is honest."""
    monkeypatch.delenv("TRANSCRIBE_WORK_DIR", raising=False)
    elsewhere = tmp_path / "elsewhere"
    cfg = tmp_path / "etc.toml"
    cfg.write_text(f"[server]\nwork_dir = {toml_str(elsewhere)}\n")

    args, path, required = s.resolve_args(["--config", str(cfg)])
    assert (path, required) == (cfg, True)
    assert args.work_dir == str(elsewhere)


@pytest.mark.parametrize(
    "body,needle",
    [
        ("[server]\nport = 99999\n", "port"),
        ("[server]\nport = 0\n", "port"),
        ('[model]\nmodel = "gigantic"\n', "not one of"),
        ("[audit]\nnonsense = 1\n", "unknown option"),
        ("[limits]\nmax_upload_mb = 0\n", "at least 1"),
        ("[limits]\nmax_queue = 0\n", "at least 1"),
        ("[limits]\nmax_jobs = -1\n", "negative"),
    ],
)
def test_invalid_config_is_rejected(tmp_path, body, needle):
    cfg = tmp_path / "bad.toml"
    cfg.write_text(body)
    with pytest.raises(SystemExit) as exc:
        s.resolve_args(["--config", str(cfg)])
    assert needle in str(exc.value)


def test_unknown_config_key_is_fatal(tmp_path):
    cfg = tmp_path / "typo.toml"
    cfg.write_text("[server]\nprot = 1234\n")
    with pytest.raises(SystemExit):
        s.resolve_args(["--config", str(cfg)])


def test_missing_explicit_config_is_fatal(tmp_path):
    with pytest.raises(SystemExit):
        s.resolve_args(["--config", str(tmp_path / "nope.toml")])


def test_no_starter_config_flag_and_env(tmp_path, monkeypatch):
    for var in ("TRANSCRIBE_STARTER_CONFIG", "TRANSCRIBE_WORK_DIR"):
        monkeypatch.delenv(var, raising=False)
    args, _, _ = s.resolve_args(["--work-dir", str(tmp_path / "state")])
    assert args.starter_config is True, "creating the starter file is the default"

    args, _, _ = s.resolve_args(
        ["--work-dir", str(tmp_path / "state"), "--no-starter-config"]
    )
    assert args.starter_config is False

    monkeypatch.setenv("TRANSCRIBE_STARTER_CONFIG", "0")
    args, _, _ = s.resolve_args(["--work-dir", str(tmp_path / "state")])
    assert args.starter_config is False, "environment must win over the default"


def test_audit_section_aliases(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_AUDIT_READS", raising=False)
    cfg = tmp_path / "c.toml"
    cfg.write_text("[audit]\nreads = true\nretain_days = 3\nenabled = false\n")
    args, _, _ = s.resolve_args(["--config", str(cfg)])
    assert args.audit_reads is True
    assert args.audit_retain_days == 3
    assert args.audit is False


def test_audit_open_alias_and_mode_rule(tmp_path, monkeypatch):
    """[audit] open, the flag and the env var all reach the same option, and
    the three-way mode rule is what the gate and the page read."""
    monkeypatch.delenv("TRANSCRIBE_AUDIT_OPEN", raising=False)
    monkeypatch.delenv("TRANSCRIBE_AUDIT_TOKEN", raising=False)
    cfg = tmp_path / "c.toml"
    cfg.write_text("[audit]\nopen = true\n")

    args, _, _ = s.resolve_args(["--config", str(cfg)])
    assert args.audit_open is True
    assert s.audit_api_mode(args) == "open"

    # No token and no opt-out is 'off'; a token alone is 'token'.
    args, _, _ = s.resolve_args(["--audit-token", "secret"])
    assert args.audit_open is False
    assert s.audit_api_mode(args) == "token"

    args, _, _ = s.resolve_args([])
    assert s.audit_api_mode(args) == "off"

    monkeypatch.setenv("TRANSCRIBE_AUDIT_OPEN", "1")
    args, _, _ = s.resolve_args([])
    assert args.audit_open is True


def test_audit_open_and_an_audit_token_are_mutually_exclusive(tmp_path, monkeypatch):
    """Refusing beats precedence: the token would otherwise be dropped without
    a word, and the environment layer is the one that wins."""
    monkeypatch.delenv("TRANSCRIBE_AUDIT_OPEN", raising=False)
    monkeypatch.delenv("TRANSCRIBE_AUDIT_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exc:
        s.resolve_args(["--audit-open", "--audit-token", "secret"])
    assert "mutually exclusive" in str(exc.value)

    monkeypatch.setenv("TRANSCRIBE_AUDIT_TOKEN", "from-env")
    with pytest.raises(SystemExit):
        s.resolve_args(["--audit-open"])

    cfg = tmp_path / "c.toml"
    cfg.write_text('[audit]\ntoken = "from-file"\n')
    monkeypatch.delenv("TRANSCRIBE_AUDIT_TOKEN")
    with pytest.raises(SystemExit):
        s.resolve_args(["--config", str(cfg), "--audit-open"])


# --------------------------------------------------------------------------- #
# Starter config
# --------------------------------------------------------------------------- #

KEY_LINE = re.compile(r"^#\s*([a-z_][a-z0-9_]*)\s*=")

# Options the template must not carry as a `key = value` line:
#   config         -- it is this file
#   starter_config -- only matters before the file exists
#   work_dir       -- self-referential here, and a startup error; see
#                     check_default_state_dir
NOT_IN_TEMPLATE = {"config", "starter_config", "work_dir"}

# DEFAULTS stores None for "derive it from the work dir", so the template can
# only show a plausible path. Checked for presence, not for the value.
DERIVED_DEFAULTS = {"audit_dir"}


def materialise(template: str) -> str:
    """The template with every commented `key = value` line uncommented."""
    body = "\n".join(
        KEY_LINE.sub(r"\1 = ", line) if KEY_LINE.match(line) else line
        for line in template.splitlines()
    )
    return body + "\n"


def test_starter_config_is_inert_as_shipped(tmp_path):
    """The point of the whole thing: the file we create pins no values."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(s.CONFIG_TEMPLATE, encoding="utf-8")
    assert s.load_config_file(cfg, required=False) == {}


def test_starter_config_documents_every_option_at_its_default(tmp_path):
    """Uncommenting the whole template must reproduce DEFAULTS exactly.

    That one assertion proves every option is documented, every documented key
    is real (an unknown one is a SystemExit), and no comment quotes a default
    the code no longer uses.
    """
    cfg = tmp_path / "config.toml"
    cfg.write_text(materialise(s.CONFIG_TEMPLATE), encoding="utf-8")
    flat = s.load_config_file(cfg, required=False)

    # `config` is this file, `starter_config` only matters before it exists, and
    # `work_dir` is rejected at this location, so none of them can be set here.
    assert set(flat) == set(s.DEFAULTS) - NOT_IN_TEMPLATE
    for name, value in flat.items():
        if name in DERIVED_DEFAULTS:
            continue
        assert value == s.DEFAULTS[name], f"{name} quotes a stale default"


def test_starter_config_is_created_once_and_never_clobbered(tmp_path):
    cfg = tmp_path / "state" / "config.toml"
    assert s.init_starter_config(cfg, required=False, enabled=True) is True
    assert cfg.read_text(encoding="utf-8") == s.CONFIG_TEMPLATE

    cfg.write_text("[server]\nport = 9000\n", encoding="utf-8")
    assert s.init_starter_config(cfg, required=False, enabled=True) is False
    assert cfg.read_text(encoding="utf-8") == "[server]\nport = 9000\n"


def test_starter_config_skipped_for_explicit_config_and_when_disabled(tmp_path):
    """A typo'd --config must not become a server quietly running on defaults."""
    typo = tmp_path / "config.toml"
    assert s.init_starter_config(typo, required=True, enabled=True) is False
    assert not typo.exists()

    off = tmp_path / "other" / "config.toml"
    assert s.init_starter_config(off, required=False, enabled=False) is False
    assert not off.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_starter_config_is_private(tmp_path):
    """It documents the token keys, so a filled-in copy must not be readable."""
    cfg = tmp_path / "config.toml"
    s.init_starter_config(cfg, required=False, enabled=True)
    mode = stat.S_IMODE(cfg.stat().st_mode)
    assert mode & 0o077 == 0, f"group or other can read it: {mode:04o}"


def test_starter_config_failure_is_not_fatal(tmp_path, capsys):
    """A read-only home must not stop the server from serving."""
    root = tmp_path / "readonly"
    root.mkdir()
    root.chmod(0o500)
    if os.access(root, os.W_OK):  # root ignores the mode bits
        pytest.skip("directory modes are not enforced for this user")

    cfg = root / "config.toml"
    assert s.init_starter_config(cfg, required=False, enabled=True) is False
    assert not cfg.exists()
    assert "Could not create a starter config" in capsys.readouterr().out
    root.chmod(0o700)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [(5, 5), ("7", 7), (None, 3), ("abc", 3), (2.9, 2), ("", 3)],
)
def test_as_int(value, expected):
    assert s.as_int(value, 3) == expected


def test_clamp():
    assert s.clamp(5, 1, 10) == 5
    assert s.clamp(0, 1, 10) == 1
    assert s.clamp(99, 1, 10) == 10


def test_digest():
    assert s.digest("") is None
    d = s.digest("abc")
    assert d is not None and len(d) == 64
    assert s.digest("abc") == s.digest("abc")


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.0, "00:00:00.000"), (1.5, "00:00:01.500"), (3661.25, "01:01:01.250")],
)
def test_stamp(seconds, expected):
    assert s._stamp(seconds) == expected


def test_stamp_survives_a_bad_value():
    assert s._stamp("nonsense") == "00:00:00.000"  # type: ignore[arg-type]


def test_content_disposition_neutralises_header_injection():
    header = s.content_disposition('evil"\r\nX-Injected: 1', "txt")

    # The attack is CRLF/quote injection into the header; the filename text
    # itself is inert once those characters are gone.
    assert "\r" not in header and "\n" not in header
    assert header.count("filename=") == 1
    assert header.startswith('attachment; filename="')
    # Only the RFC 5987 parameter may follow.
    assert header.split('filename="')[1].count('"') == 1


def test_content_disposition_keeps_unicode_via_rfc5987():
    header = s.content_disposition("caf\u00e9 meeting", "txt")
    assert "filename*=UTF-8''" in header
    assert "caf%C3%A9" in header


def test_render_formats_are_numbered_per_export(configured):
    job_id = configured.make_job()
    s.patch_job(
        job_id,
        state="done",
        segments=[
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 1.0, "end": 2.0, "text": "second"},
        ],
    )
    job = s.JOBS[job_id]

    srt, mime = s.render(job, "srt")
    assert mime.startswith("application/x-subrip")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,000\nfirst")

    vtt, _ = s.render(job, "vtt")
    assert vtt.startswith("WEBVTT")

    txt, _ = s.render(job, "txt")
    assert txt.strip() == "first\nsecond"

    with pytest.raises(s.HTTPException):
        s.render(job, "docx")


def test_new_job_survives_a_missing_source(configured):
    """A retry can race an eviction that already unlinked the source."""
    missing = Path(s.UPLOAD_DIR) / "gone.wav"
    opts = s.build_opts(
        None, None, "", "true", "fast", "", "", "false", "false", "false", 2000, 400
    )
    job_id = s.new_job("gone.wav", missing, opts)
    assert s.JOBS[job_id]["size"] == 0


def test_prune_jobs_evicts_and_audits(configured, monkeypatch):
    monkeypatch.setattr(s.ARGS, "max_jobs", 2)
    ids = [configured.make_job(f"f{i}.wav") for i in range(4)]
    for job_id in ids:
        s.patch_job(job_id, state="done")

    s.prune_jobs()

    assert len(s.JOBS) == 2
    evicted = [e for e in configured.events() if e["event"] == "job.evicted"]
    assert len(evicted) == 2
