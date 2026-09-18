"""Shared fixtures.

The tests import transcribe_server directly, so they need its runtime
dependencies (fastapi, uvicorn, python-multipart, faster-whisper). See the
"Tests" section of the README for the one command that builds that venv.

transcribe_server keeps its configuration in module globals that main()
populates. The `configured` fixture fills them with a small, explicit
configuration and tears them down, so tests never depend on a running server.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transcribe_server as s  # noqa: E402


class Configured:
    """A configured module plus helpers for building jobs and reading the trail."""

    def __init__(
        self, root: Path, audit: s.AuditLog, app_token: str, audit_token: str
    ) -> None:
        self.root = root
        self.audit = audit
        self.app_token = app_token
        self.audit_token = audit_token

    def upload(self, name: str = "meeting.wav", data: bytes = b"RIFFfake") -> Path:
        # basename only, so a test can't accidentally write outside the fixture
        path = s.UPLOAD_DIR / Path(name).name
        path.write_bytes(data)
        return path

    def make_job(self, name: str = "meeting.wav", **overrides) -> str:
        path = self.upload(name)
        opts = s.build_opts(
            overrides.get("model"),
            overrides.get("compute_type"),
            overrides.get("language", ""),
            overrides.get("vad", "true"),
            overrides.get("quality", "balanced"),
            overrides.get("prompt", ""),
            overrides.get("hotwords", ""),
            overrides.get("translate", "false"),
            overrides.get("condition", "false"),
            overrides.get("word_timestamps", "false"),
            overrides.get("min_silence_ms", 2000),
            overrides.get("speech_pad_ms", 400),
        )
        return s.new_job(name, path, opts)

    def events(self) -> list[dict]:
        """Every audit record written so far, oldest first."""
        out: list[dict] = []
        for path in sorted(self.audit.dir.glob("audit-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
        return out


@pytest.fixture
def configured(tmp_path, monkeypatch):
    """Point the module at a throwaway work dir with a known configuration."""
    work = tmp_path / "work"
    uploads = work / "uploads"
    uploads.mkdir(parents=True)

    audit = s.AuditLog(work / "audit", enabled=True, retain_days=30, prompts=True)

    # Generated per run: a real credential shape with nothing hardcoded.
    app_token = secrets.token_urlsafe(16)
    audit_token = secrets.token_urlsafe(16)

    monkeypatch.setattr(s, "WORK_DIR", work)
    monkeypatch.setattr(s, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(s, "AUDIT", audit)
    monkeypatch.setattr(s, "READY", True)
    monkeypatch.setattr(
        s, "ALLOWED_HOSTS", {"localhost", "127.0.0.1", "::1", "testserver"}
    )
    monkeypatch.setattr(s, "ALLOWED_SUFFIXES", set())

    args = argparse.Namespace(
        device="cpu",
        model="base",
        compute_type="float32",
        quality="balanced",
        allow_model_choice=True,
        allow_precision_choice=False,
        model_cache=1,
        max_jobs=60,
        max_queue=20,
        max_upload_mb=64,
        source_retention="job",
        audit_reads=True,
        audit=True,
        host="127.0.0.1",
        port=8765,
    )
    args.token = app_token
    args.audit_token = audit_token
    monkeypatch.setattr(s, "ARGS", args)

    with s.JOBS_LOCK:
        s.JOBS.clear()
    yield Configured(work, audit, app_token, audit_token)
    with s.JOBS_LOCK:
        s.JOBS.clear()
    while not s.JOB_QUEUE.empty():
        s.JOB_QUEUE.get_nowait()
        s.JOB_QUEUE.task_done()


@pytest.fixture
def client(configured):
    """The real ASGI app, authenticated with the app token.

    The worker thread is not started, so an upload stays queued and no model is
    ever loaded.
    """
    with TestClient(s.app) as c:
        c.headers.update({"x-token": configured.app_token})
        yield c
