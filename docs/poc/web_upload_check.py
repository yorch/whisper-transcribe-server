"""Does an upload through the web app come back with person tags?

Starts the real server from this checkout, then does what the page does when you
drop a file on it with the controls at their defaults: checks the served page
for the pre-ticked box, POSTs the form body the page would send to /api/jobs,
polls, and asserts the finished job carries a speaker on every segment and a
speakers block in the JSON export.

Nothing is stubbed: real faster-whisper model, real diarization weights, which
the server fetches itself on first use, so this exercises that path too. It is
slow -- a model load, a transcription and a 42 MB download -- and it binds a
port, so it is not part of the test suite.

    .venv/bin/python docs/poc/web_upload_check.py
    TRANSCRIBE_E2E_AUDIO=meeting.wav TRANSCRIBE_E2E_SPEAKERS=3 ...

It is the end-to-end guard for "a web upload is tagged"; the unit test beside it
only pins that the checkbox ships ticked.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

# The repo root: this file lives in docs/poc/. Override to run it elsewhere.
ROOT = Path(
    os.environ.get("TRANSCRIBE_E2E_ROOT") or Path(__file__).resolve().parents[2]
)
WORK = Path(os.environ.get("TRANSCRIBE_E2E_WORK", "/tmp/tags-e2e"))
AUDIO = Path(
    os.environ.get("TRANSCRIBE_E2E_AUDIO", "/tmp/diar-poc/0-four-speakers-zh.wav")
)
EXPECTED_SPEAKERS = int(os.environ.get("TRANSCRIBE_E2E_SPEAKERS", "4"))
TOKEN = "e2e-token-not-a-secret"  # noqa: S105 -- throwaway, localhost only
# Resolved rather than spelled "uv" so a machine without it says so here
# instead of failing inside Popen with a bare FileNotFoundError.
UV = shutil.which("uv")
if not UV:
    raise SystemExit("!  uv is not on PATH; this script starts the server with it")
PORT = int(os.environ.get("TRANSCRIBE_E2E_PORT", "8801"))
BASE = f"http://127.0.0.1:{PORT}"

server_log: list[str] = [""]

if WORK.exists():
    shutil.rmtree(WORK)
if not AUDIO.is_file():
    raise SystemExit(f"!  no sample audio at {AUDIO}; see docs/poc/README.md")
if not (ROOT / "transcribe_server.py").is_file():
    raise SystemExit(f"!  {ROOT} does not look like the repo root")

server = subprocess.Popen(  # noqa: S603 -- a literal list, no untrusted input
    [
        UV,
        "run",
        "transcribe_server.py",
        "--model",
        "base",
        "--device",
        "cpu",
        "--compute-type",
        "int8",
        "--port",
        str(PORT),
        "--token",
        TOKEN,
        "--work-dir",
        str(WORK),
    ],
    cwd=str(ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)


def drain() -> str:
    return "\n".join((server_log[0] or "").splitlines()[-25:])


try:
    client = httpx.Client(headers={"x-token": TOKEN}, timeout=30.0)

    print("== waiting for the server ==")
    deadline = time.time() + 180
    while time.time() < deadline:
        if server.poll() is not None:
            raise SystemExit("!  server exited early:\n" + drain())
        try:
            if client.get(f"{BASE}/api/status").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    else:
        raise SystemExit("!  server never became ready:\n" + drain())
    print("   up")

    print("\n== the page the browser gets ticks 'Identify speakers' ==")
    page = client.get(f"{BASE}/").text
    idx = page.find('id="diarize"')
    snippet = page[idx - 40 : idx + 40]
    assert idx != -1, "no diarize control in the served page"
    assert "checked" in snippet, f"box is not pre-ticked: ...{snippet}..."
    print(f"   ...{snippet.strip()}...")

    print("\n== POST an upload with the page's own default form body ==")
    # currentSettings() with everything at its default: the box is ticked, so
    # diarize=true; the speaker count input reads 0 (= work it out); word
    # timings are ticked and locked by syncDiarize(), so they go too.
    form = {
        "model": "base",
        "compute_type": "int8",
        "language": "zh",
        "quality": "balanced",
        "vad": "true",
        "prompt": "",
        "hotwords": "",
        "translate": "false",
        "condition": "false",
        "word_timestamps": "true",
        "min_silence_ms": "2000",
        "speech_pad_ms": "400",
        "diarize": "true",
        "speakers": "0",
    }
    with AUDIO.open("rb") as fh:
        response = client.post(
            f"{BASE}/api/jobs",
            data=form,
            files={"file": (AUDIO.name, fh, "audio/wav")},
        )
    assert response.status_code == 200, f"{response.status_code}: {response.text}"
    job_id = response.json()["id"]
    print(f"   queued {job_id}")

    print("\n== poll until it finishes ==")
    deadline = time.time() + 900
    job: dict = {}
    while time.time() < deadline:
        job = client.get(f"{BASE}/api/jobs/{job_id}").json()
        state = job["state"]
        if state in ("done", "error", "cancelled"):
            break
        print(
            f"   {state:10} {job.get('phase') or '-':12} "
            f"{round(job.get('progress') or 0, 2):>4}  {job.get('message')}"
        )
        time.sleep(2)
    print(f"   final: {job['state']} — {job.get('message')}")
    if job["state"] != "done":
        raise SystemExit("!  job did not finish:\n" + drain())

    print("\n== did the transcript come back tagged? ==")
    segments = job.get("segments") or []
    assert segments, "no segments at all"
    missing = [s for s in segments if not isinstance(s.get("speaker"), int)]
    assert not missing, f"{len(missing)} segment(s) without a speaker: {missing[:2]}"
    speakers = sorted({s["speaker"] for s in segments})
    print(
        f"   {len(segments)} segments, every one tagged; speakers present: {speakers}"
    )

    print("\n   sample:")
    for seg in segments[:10]:
        print(
            f"     {seg['start']:7.2f} -- {seg['end']:7.2f}  "
            f"Speaker {seg['speaker']}: {seg['text'][:40]}"
        )

    print("\n== the exports carry them too ==")
    txt = client.get(f"{BASE}/api/jobs/{job_id}/text", params={"format": "txt"}).text
    assert "Speaker 1: " in txt, txt[:200]
    print("   .txt   first line: " + txt.splitlines()[0])

    vtt = client.get(f"{BASE}/api/jobs/{job_id}/text", params={"format": "vtt"}).text
    assert "<v Speaker 1>" in vtt, vtt[:200]
    print("   .vtt   contains <v Speaker 1>")

    payload = client.get(
        f"{BASE}/api/jobs/{job_id}/text", params={"format": "json"}
    ).json()
    assert "speakers" in payload, "json export has no speakers block"
    assert all("speaker" in s for s in payload["segments"])
    print("   .json  speakers block:")
    for entry in payload["speakers"]:
        print(
            f"     {entry['label']}: {entry['segments']} segments, {entry['seconds']}s"
        )

    assert payload["options"]["diarize"] is True
    assert payload["options"]["word_timestamps"] is True, "words should be forced on"
    # The trust boundary is unchanged by any of this.
    assert "prompt" not in payload["options"] and "hotwords" not in payload["options"]

    assert len(speakers) == EXPECTED_SPEAKERS, (
        f"expected {EXPECTED_SPEAKERS} speakers on this file, got {len(speakers)}"
    )
    print("\nALL WEB-UPLOAD CHECKS PASSED")
finally:
    server.terminate()
    try:
        out, _ = server.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        server.kill()
        out, _ = server.communicate()
    server_log[0] = out or ""
    print("\n-- server log (last 6 lines) --")
    for line in server_log[0].strip().splitlines()[-6:]:
        print("   " + line)
