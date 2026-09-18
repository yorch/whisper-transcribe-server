#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi>=0.110",
#     "uvicorn[standard]>=0.27",
#     "python-multipart>=0.0.9",
#     "faster-whisper>=1.0.3",
#     "nvidia-cublas-cu12; sys_platform != 'darwin'",
#     "nvidia-cudnn-cu12>=9,<10; sys_platform != 'darwin'",
# ]
# ///
"""
Local transcription server.

Serves a drag-and-drop page over the LAN and transcribes with faster-whisper on
whatever CUDA GPU is present. One job runs at a time so the GPU is never
oversubscribed; everything else waits in a queue.

    uv run transcribe_server.py
    uv run transcribe_server.py --model medium --port 8765

Access control is on by default: if you don't supply --token, one is generated
at startup and printed with the URL. Pass --no-auth to turn it off deliberately.

Dependencies are declared inline (PEP 723), so there is nothing to install
first and no virtualenv to activate.
"""

from __future__ import annotations

import argparse
import gc
import hmac
import json
import os
import queue
import secrets
import shutil
import socket
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

MODELS = ["large-v3", "large-v3-turbo", "medium", "small", "base"]
# float16 needs compute capability >= 7.0 (Turing/Ampere/Ada/Hopper/Blackwell).
# Pascal and older should use int8 or float32.
COMPUTE_TYPES = ["float16", "int8_float16", "int8", "float32"]
QUALITIES = {"fast": 1, "balanced": 5, "thorough": 8}  # -> beam_size
RETENTION = ["run", "job", "forever"]

PROMPT_LIMIT = 1000
HOTWORDS_LIMIT = 400

WORK_DIR = Path(
    os.environ.get("TRANSCRIBE_WORK_DIR", Path.home() / ".transcribe-server")
)
UPLOAD_DIR = WORK_DIR / "uploads"

# Set in main(). Requests are refused until then, so importing this module and
# serving `app` directly from an ASGI server fails closed rather than open.
ARGS: Optional[argparse.Namespace] = None
ALLOWED_HOSTS: Set[str] = set()
ALLOWED_SUFFIXES: Set[str] = set()  # entries like ".trycloudflare.com"

# --------------------------------------------------------------------------- #
# Job store
# --------------------------------------------------------------------------- #

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()

# LRU, capped by --model-cache. Uncapped, a client walking the model and
# precision dropdowns would pin every combination in VRAM at once.
_MODEL_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_MODEL_LOCK = threading.Lock()


def redact(text: str, limit: int = 300) -> str:
    """Strip local filesystem paths out of anything shown to a client."""
    for base in (str(UPLOAD_DIR), str(WORK_DIR), str(Path.home())):
        for variant in (base, base.replace("\\", "/")):
            if variant:
                text = text.replace(variant, "<path>")
    return text[:limit]


def new_job(filename: str, path: Path, opts: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "filename": filename,
            "path": str(path),
            "size": path.stat().st_size,
            "state": "queued",  # queued | loading | running | done | error | cancelled
            "progress": 0.0,
            "message": "Waiting for the GPU",
            "segments": [],
            "language": None,
            "duration": None,
            "created": time.time(),
            "started": None,
            "finished": None,
            "opts": opts,
        }
    return job_id


def patch_job(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(fields)


def get_job(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return job


def source_shared(path: str, exclude_id: str) -> bool:
    """A retried job reuses the original upload; don't delete it out from under."""
    with JOBS_LOCK:
        return any(j["path"] == path and j["id"] != exclude_id for j in JOBS.values())


def drop_source(job: Dict[str, Any]) -> None:
    if ARGS and ARGS.source_retention == "forever":
        return
    if source_shared(job["path"], job["id"]):
        return
    try:
        Path(job["path"]).unlink(missing_ok=True)
    except OSError:
        pass


def prune_jobs() -> None:
    """Keep memory bounded: evict the oldest finished jobs past the cap."""
    cap = ARGS.max_jobs if ARGS else 60
    evicted: List[Dict[str, Any]] = []
    with JOBS_LOCK:
        if len(JOBS) <= cap:
            return
        finished = sorted(
            (j for j in JOBS.values() if j["state"] in ("done", "error", "cancelled")),
            key=lambda j: j["created"],
        )
        while len(JOBS) > cap and finished:
            victim = finished.pop(0)
            JOBS.pop(victim["id"], None)
            evicted.append(victim)
    for job in evicted:
        drop_source(job)


def job_public(job: Dict[str, Any], include_segments: bool = True) -> Dict[str, Any]:
    out = {k: v for k, v in job.items() if k not in ("path", "segments")}
    now = time.time()
    started = job.get("started")
    finished = job.get("finished")
    out["elapsed"] = round((finished or now) - started, 1) if started else 0.0
    out["segment_count"] = len(job["segments"])
    out["can_retry"] = (
        job["state"] in ("done", "error", "cancelled") and Path(job["path"]).exists()
    )
    if include_segments:
        out["segments"] = job["segments"]
    return out


# --------------------------------------------------------------------------- #
# Job options
# --------------------------------------------------------------------------- #


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def build_opts(
    model: Optional[str],
    compute_type: Optional[str],
    language: str,
    vad: str,
    quality: str,
    prompt: str,
    hotwords: str,
    translate: str,
    condition: str,
    word_timestamps: str,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> Dict[str, Any]:
    """Validate everything a client can influence. Pinned knobs ignore the client."""
    truthy = lambda v: str(v).lower() in ("1", "true", "yes", "on")  # noqa: E731

    if ARGS.allow_model_choice:
        model = model or ARGS.model
        if model not in MODELS:
            raise HTTPException(status_code=400, detail="Unknown model")
    else:
        model = ARGS.model

    if ARGS.allow_precision_choice:
        compute_type = compute_type or ARGS.compute_type
        if compute_type not in COMPUTE_TYPES:
            raise HTTPException(status_code=400, detail="Unknown compute type")
    else:
        compute_type = ARGS.compute_type

    language = (language or "").strip().lower()
    if language and (len(language) > 5 or not language.isalpha()):
        raise HTTPException(status_code=400, detail="Unknown language code")

    if quality not in QUALITIES:
        quality = ARGS.quality

    return {
        "model": model,
        "compute_type": compute_type,
        "language": language,
        "vad": truthy(vad),
        "quality": quality,
        "beam_size": QUALITIES[quality],
        "prompt": (prompt or "").strip()[:PROMPT_LIMIT],
        "hotwords": (hotwords or "").strip()[:HOTWORDS_LIMIT],
        "translate": truthy(translate),
        # Whisper's repetition loops on long audio come from carrying a poisoned
        # context forward, so this is off unless asked for.
        "condition": truthy(condition),
        "word_timestamps": truthy(word_timestamps),
        "min_silence_ms": clamp(int(min_silence_ms or 2000), 100, 10000),
        "speech_pad_ms": clamp(int(speech_pad_ms or 400), 0, 2000),
    }


# --------------------------------------------------------------------------- #
# Transcription worker
# --------------------------------------------------------------------------- #


def load_model(name: str, device: str, compute_type: str):
    key = (name, device, compute_type)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(key)
            return _MODEL_CACHE[key]

        from faster_whisper import WhisperModel

        model = WhisperModel(name, device=device, compute_type=compute_type)
        _MODEL_CACHE[key] = model

        cap = max(1, ARGS.model_cache if ARGS else 1)
        while len(_MODEL_CACHE) > cap:
            old_key, old_model = _MODEL_CACHE.popitem(last=False)
            del old_model
            gc.collect()
            print(f"   unloaded {old_key[0]} / {old_key[2]} to free VRAM")
        return model


def transcribe_kwargs(opts: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "beam_size": opts["beam_size"],
        "language": opts["language"] or None,
        "task": "translate" if opts["translate"] else "transcribe",
        "condition_on_previous_text": opts["condition"],
        "word_timestamps": opts["word_timestamps"],
        "vad_filter": opts["vad"],
    }
    if opts["vad"]:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": opts["min_silence_ms"],
            "speech_pad_ms": opts["speech_pad_ms"],
        }
    if opts["prompt"]:
        kwargs["initial_prompt"] = opts["prompt"]
    if opts["hotwords"]:
        kwargs["hotwords"] = opts["hotwords"]
    return kwargs


def run_job(job_id: str) -> None:
    job = get_job(job_id)
    opts = job["opts"]
    patch_job(job_id, state="loading", started=time.time(), message="Loading model")

    try:
        model = load_model(opts["model"], ARGS.device, opts["compute_type"])
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        patch_job(
            job_id,
            state="error",
            finished=time.time(),
            message=f"Could not load {opts['model']}: {redact(str(exc))}",
        )
        prune_jobs()
        return

    patch_job(job_id, state="running", message="Transcribing")

    try:
        kwargs = transcribe_kwargs(opts)
        try:
            segments, info = model.transcribe(job["path"], **kwargs)
        except TypeError as exc:
            # hotwords landed in faster-whisper 1.0.2; degrade rather than fail.
            if "hotwords" not in str(exc):
                raise
            kwargs.pop("hotwords", None)
            segments, info = model.transcribe(job["path"], **kwargs)

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        patch_job(
            job_id,
            language=getattr(info, "language", None),
            duration=round(duration, 2),
        )

        collected: List[Dict[str, Any]] = []
        for seg in segments:
            entry: Dict[str, Any] = {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            if opts["word_timestamps"] and getattr(seg, "words", None):
                entry["words"] = [
                    {"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word}
                    for w in seg.words
                ]
            collected.append(entry)

            progress = min(seg.end / duration, 1.0) if duration else 0.0
            with JOBS_LOCK:
                live = JOBS.get(job_id)
                if live is None or live["state"] == "cancelled":
                    if ARGS.source_retention == "run":
                        drop_source(job)
                    return
                live["segments"] = list(collected)
                live["progress"] = progress

        patch_job(
            job_id,
            state="done",
            progress=1.0,
            finished=time.time(),
            message=f"{len(collected)} segments",
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        patch_job(
            job_id,
            state="error",
            finished=time.time(),
            message=f"{type(exc).__name__}: {redact(str(exc))}",
        )
    finally:
        if ARGS.source_retention == "run":
            drop_source(job)
        prune_jobs()


def worker_loop() -> None:
    while True:
        job_id = JOB_QUEUE.get()
        try:
            with JOBS_LOCK:
                state = JOBS.get(job_id, {}).get("state")
            if state == "cancelled":
                continue
            run_job(job_id)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            patch_job(
                job_id,
                state="error",
                finished=time.time(),
                message=f"{type(exc).__name__}: {redact(str(exc))}",
            )
        finally:
            JOB_QUEUE.task_done()


# --------------------------------------------------------------------------- #
# Transcript formatting
# --------------------------------------------------------------------------- #


def _stamp(seconds: float, comma: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render(job: Dict[str, Any], fmt: str) -> tuple[str, str]:
    """Return (body, mime) for the requested format."""
    segs = job["segments"]

    if fmt == "txt":
        return "\n".join(s["text"] for s in segs) + "\n", "text/plain; charset=utf-8"

    if fmt == "timestamped":
        lines = [f"[{_stamp(s['start'])}] {s['text']}" for s in segs]
        return "\n".join(lines) + "\n", "text/plain; charset=utf-8"

    if fmt == "srt":
        blocks = []
        for i, s in enumerate(segs, 1):
            blocks.append(
                f"{i}\n{_stamp(s['start'], comma=True)} --> "
                f"{_stamp(s['end'], comma=True)}\n{s['text']}\n"
            )
        return "\n".join(blocks), "application/x-subrip; charset=utf-8"

    if fmt == "vtt":
        blocks = ["WEBVTT\n"]
        for s in segs:
            blocks.append(f"{_stamp(s['start'])} --> {_stamp(s['end'])}\n{s['text']}\n")
        return "\n".join(blocks), "text/vtt; charset=utf-8"

    if fmt == "json":
        payload = {
            "filename": job["filename"],
            "language": job["language"],
            "duration": job["duration"],
            "options": job["opts"],
            "segments": segs,
        }
        return (
            json.dumps(payload, indent=2, ensure_ascii=False),
            "application/json; charset=utf-8",
        )

    raise HTTPException(status_code=400, detail="Unknown format")


def content_disposition(stem: str, ext: str) -> str:
    """RFC 5987 disposition header. Never interpolate a raw filename here."""
    ascii_fallback = (
        "".join(c for c in stem if c.isalnum() or c in " ._-").strip() or "transcript"
    )
    utf8 = quote(f"{stem}.{ext}", safe="")
    return f"attachment; filename=\"{ascii_fallback}.{ext}\"; filename*=UTF-8''{utf8}"


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(title="Transcription server", docs_url=None, redoc_url=None)


def normalize_host(raw: str) -> str:
    """Normalize a Host header value or --allow-host entry to a bare hostname."""
    host = raw.strip().lower()
    # Tolerate pasting a full URL: https://example.com:443/path -> example.com
    if "://" in host:
        host = host.split("://", 1)[1]
    host = host.split("/", 1)[0].strip()
    if host.startswith("[") and "]" in host:
        # [::1] or [::1]:8765
        host = host[1 : host.index("]")]
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    return host.rstrip(".").strip()


def host_allowed(header: Optional[str]) -> bool:
    """Reject DNS-rebinding: only hostnames we expect may address this server."""
    if not header:
        return False
    host = normalize_host(header)
    if not host:
        return False
    if host in ALLOWED_HOSTS:
        return True
    return any(host == s.lstrip(".") or host.endswith(s) for s in ALLOWED_SUFFIXES)


@app.middleware("http")
async def guard(request: Request, call_next):
    if ARGS is None:
        return JSONResponse({"detail": "Server not configured"}, status_code=503)

    raw_host = request.headers.get("host")
    if not host_allowed(raw_host):
        seen = (raw_host or "").strip()[:100]
        print(f"!  Rejected Host header {seen!r} (add it with --allow-host)")
        return JSONResponse(
            {
                "detail": f"Unrecognised Host header {seen!r}. "
                "Restart the server with --allow-host for this name."
            },
            status_code=421,
        )

    path = request.url.path
    if path.startswith("/api/"):
        # A custom header forces a CORS preflight, so a hostile page cannot
        # reach these endpoints even with a CORS-safelisted body type.
        site = request.headers.get("sec-fetch-site")
        if site and site not in ("same-origin", "none"):
            return JSONResponse(
                {"detail": "Cross-site request refused"}, status_code=403
            )

        if ARGS.token:
            supplied = request.headers.get("x-token") or ""
            if not hmac.compare_digest(supplied.encode(), ARGS.token.encode()):
                return JSONResponse({"detail": "Bad or missing token"}, status_code=401)

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'"
    )
    return response


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/api/status")
def status() -> Dict[str, Any]:
    gpu = None
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        gpu = f"{count} CUDA device(s)" if count else None
    except Exception:  # noqa: BLE001
        pass

    try:
        import torch  # optional, only for the pretty device name

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass

    with JOBS_LOCK:
        active = sum(
            1 for j in JOBS.values() if j["state"] in ("queued", "loading", "running")
        )
    with _MODEL_LOCK:
        loaded = [f"{k[0]} / {k[2]}" for k in _MODEL_CACHE]

    return {
        "device": ARGS.device,
        "gpu": gpu,
        "compute_type": ARGS.compute_type,
        "default_model": ARGS.model,
        "default_quality": ARGS.quality,
        "models": MODELS,
        "compute_types": COMPUTE_TYPES,
        "qualities": list(QUALITIES),
        "allow_model_choice": ARGS.allow_model_choice,
        "allow_precision_choice": ARGS.allow_precision_choice,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "active_jobs": active,
        "loaded_models": loaded,
        "model_cache": ARGS.model_cache,
        "max_upload_mb": ARGS.max_upload_mb,
        "retry_available": ARGS.source_retention != "run",
        "prompt_limit": PROMPT_LIMIT,
        "hotwords_limit": HOTWORDS_LIMIT,
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form(None),
    compute_type: str = Form(None),
    language: str = Form(""),
    vad: str = Form("true"),
    quality: str = Form("balanced"),
    prompt: str = Form(""),
    hotwords: str = Form(""),
    translate: str = Form("false"),
    condition: str = Form("false"),
    word_timestamps: str = Form("false"),
    min_silence_ms: int = Form(2000),
    speech_pad_ms: int = Form(400),
) -> Dict[str, Any]:
    opts = build_opts(
        model,
        compute_type,
        language,
        vad,
        quality,
        prompt,
        hotwords,
        translate,
        condition,
        word_timestamps,
        min_silence_ms,
        speech_pad_ms,
    )

    with JOBS_LOCK:
        pending = sum(
            1 for j in JOBS.values() if j["state"] in ("queued", "loading", "running")
        )
    if pending >= ARGS.max_queue:
        raise HTTPException(
            status_code=429, detail=f"Queue is full ({ARGS.max_queue} jobs)"
        )

    raw_name = Path(file.filename or "audio").name
    safe = "".join(c for c in raw_name if c.isalnum() or c in " ._-").strip()
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe or 'audio'}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    limit = ARGS.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {ARGS.max_upload_mb} MB limit",
                    )
                fh.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise

    job_id = new_job(filename=raw_name[:200], path=dest, opts=opts)
    JOB_QUEUE.put(job_id)
    return {"id": job_id}


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    model: str = Form(None),
    compute_type: str = Form(None),
    language: str = Form(""),
    vad: str = Form("true"),
    quality: str = Form("balanced"),
    prompt: str = Form(""),
    hotwords: str = Form(""),
    translate: str = Form("false"),
    condition: str = Form("false"),
    word_timestamps: str = Form("false"),
    min_silence_ms: int = Form(2000),
    speech_pad_ms: int = Form(400),
) -> Dict[str, Any]:
    """Re-run the same source audio with different settings, no re-upload."""
    old = get_job(job_id)
    source = Path(old["path"])
    if not source.exists():
        raise HTTPException(
            status_code=409,
            detail="The source audio is no longer on disk; re-upload it",
        )

    opts = build_opts(
        model,
        compute_type,
        language,
        vad,
        quality,
        prompt,
        hotwords,
        translate,
        condition,
        word_timestamps,
        min_silence_ms,
        speech_pad_ms,
    )

    with JOBS_LOCK:
        pending = sum(
            1 for j in JOBS.values() if j["state"] in ("queued", "loading", "running")
        )
    if pending >= ARGS.max_queue:
        raise HTTPException(
            status_code=429, detail=f"Queue is full ({ARGS.max_queue} jobs)"
        )

    new_id = new_job(filename=old["filename"], path=source, opts=opts)
    JOB_QUEUE.put(new_id)
    return {"id": new_id}


@app.get("/api/jobs")
def list_jobs() -> Dict[str, Any]:
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)
        return {"jobs": [job_public(j, include_segments=False) for j in jobs]}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> Dict[str, Any]:
    return job_public(get_job(job_id))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if job["state"] in ("queued", "loading", "running"):
        patch_job(job_id, state="cancelled", message="Cancelled")
    else:
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
    drop_source(job)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/text")
def job_text(job_id: str, format: str = "txt", download: int = 0) -> Response:
    job = get_job(job_id)
    body, mime = render(job, format)
    headers = {}
    if download:
        ext = {"timestamped": "txt"}.get(format, format)
        headers["Content-Disposition"] = content_disposition(
            Path(job["filename"]).stem, ext
        )
    return Response(content=body, media_type=mime, headers=headers)


# --------------------------------------------------------------------------- #
# Front end
# --------------------------------------------------------------------------- #

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Transcription</title>
<style>
  :root{
    --panel:#dcdfe3;
    --panel-hi:#eef0f2;
    --panel-lo:#c3c8cd;
    --ink:#1b2027;
    --muted:#5c646e;
    --amber:#b9610f;
    --amber-dim:#d9b48a;
    --green:#36704a;
    --red:#9d3427;
    --display:"Bahnschrift","DIN Alternate","Roboto Condensed",system-ui,sans-serif;
    --body:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{
    background:var(--panel);
    background-image:linear-gradient(180deg,#e3e6e9 0%,#d3d7dc 100%);
    color:var(--ink);
    font-family:var(--body);
    font-size:15px;
    line-height:1.55;
    min-height:100vh;
    padding:28px 20px 72px;
  }
  .wrap{max-width:660px;margin:0 auto}

  .readout{
    border:1px solid var(--panel-lo);
    border-top-color:var(--panel-hi);
    background:linear-gradient(180deg,#e9ebee,#d6dade);
    border-radius:3px;
    padding:12px 16px;
    display:flex;
    flex-wrap:wrap;
    gap:6px 26px;
    align-items:baseline;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.7);
  }
  .readout dl{margin:0;display:flex;gap:8px;align-items:baseline}
  .readout dt{
    font-family:var(--display);
    font-size:10.5px;
    letter-spacing:.14em;
    text-transform:uppercase;
    color:var(--muted);
  }
  .readout dd{margin:0;font-family:var(--display);font-size:14px;letter-spacing:.01em}
  .lamp{
    width:8px;height:8px;border-radius:50%;
    display:inline-block;margin-right:7px;vertical-align:1px;
    background:var(--amber-dim);
    box-shadow:0 0 0 1px rgba(0,0,0,.18) inset;
  }
  .lamp.on{background:var(--amber);box-shadow:0 0 6px rgba(185,97,15,.55)}
  .lamp.bad{background:var(--red)}

  h1{
    font-family:var(--display);
    font-weight:600;
    font-size:30px;
    letter-spacing:-.01em;
    margin:30px 0 4px;
  }
  .sub{color:var(--muted);margin:0 0 22px;max-width:52ch}

  .gate{
    border:1px solid var(--panel-lo);
    border-left:3px solid var(--amber);
    background:rgba(255,255,255,.5);
    border-radius:3px;
    padding:16px 18px;
    margin-bottom:22px;
  }
  .gate p{margin:0 0 12px}
  .gate .row{display:flex;gap:8px;flex-wrap:wrap}
  input[type=password]{
    flex:1;min-width:200px;
    font-family:var(--body);font-size:15px;
    padding:6px 9px;
    border:1px solid #a9b0b8;border-radius:3px;background:#fff;color:var(--ink);
  }
  input[type=password]:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  .gate-err{color:var(--red);font-size:13.5px;margin:10px 0 0}

  .intake{
    border:2px dashed #a9b0b8;
    border-radius:4px;
    background:rgba(255,255,255,.42);
    padding:40px 24px;
    text-align:center;
    transition:border-color .14s,background .14s;
    cursor:pointer;
  }
  .intake:hover{border-color:#8d959e}
  .intake.hot{border-color:var(--amber);background:rgba(185,97,15,.07)}
  .intake:focus-visible{outline:2px solid var(--amber);outline-offset:3px}
  .intake p{margin:0;font-family:var(--display);font-size:19px}
  .intake small{display:block;margin-top:8px;color:var(--muted);font-size:13px}
  input[type=file]{display:none}

  .controls{
    margin-top:14px;
    display:flex;
    flex-wrap:wrap;
    gap:14px 22px;
    align-items:center;
    padding:12px 2px;
    border-top:1px solid var(--panel-lo);
  }
  label.field{display:flex;align-items:center;gap:8px;font-size:13.5px;color:var(--muted)}
  select{
    font-family:var(--display);
    font-size:14px;
    color:var(--ink);
    background:#fff;
    border:1px solid #a9b0b8;
    border-radius:3px;
    padding:5px 8px;
  }
  select:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  input[type=checkbox]{accent-color:var(--amber);width:15px;height:15px}

  .job{
    margin-top:18px;
    border:1px solid var(--panel-lo);
    border-top-color:var(--panel-hi);
    border-radius:3px;
    background:linear-gradient(180deg,#eceef1,#e1e4e8);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.7);
    overflow:hidden;
  }
  .job-head{display:flex;gap:12px;align-items:baseline;padding:13px 16px 11px}
  .job-name{font-family:var(--display);font-size:17px;flex:1;min-width:0;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .job-meta{font-family:var(--display);font-size:12.5px;color:var(--muted);letter-spacing:.02em;
    white-space:nowrap}

  .meter{display:flex;gap:2px;padding:0 16px 13px}
  .meter i{
    flex:1;height:11px;
    background:#c9ced3;
    box-shadow:inset 0 0 0 1px rgba(0,0,0,.07);
    transition:background .18s;
  }
  .meter i.lit{background:var(--amber)}
  .meter i.done{background:var(--green)}
  .meter i.fail{background:var(--red)}

  .job-body{padding:0 16px 14px}
  .status{font-size:13.5px;color:var(--muted);margin:0 0 10px}
  .status.err{color:var(--red)}
  .transcript{
    background:#fbfbfc;
    border:1px solid #c9ced3;
    border-radius:3px;
    padding:12px 14px;
    max-height:300px;
    overflow:auto;
    font-size:14.5px;
    white-space:pre-wrap;
    word-break:break-word;
  }
  .transcript:empty{display:none}
  .actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
  button{
    font-family:var(--display);
    font-size:13px;
    letter-spacing:.03em;
    color:var(--ink);
    background:linear-gradient(180deg,#f4f5f6,#dfe3e7);
    border:1px solid #a9b0b8;
    border-radius:3px;
    padding:6px 12px;
    cursor:pointer;
  }
  button:hover{background:linear-gradient(180deg,#fff,#e7ebef)}
  button:active{background:#d8dce0}
  button:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  button.ghost{background:none;border-color:transparent;color:var(--muted)}
  button.ghost:hover{background:rgba(0,0,0,.05);color:var(--ink)}

  details.advanced{margin-top:2px;border-top:1px solid var(--panel-lo);padding-top:10px}
  details.advanced summary{
    font-family:var(--display);font-size:13px;letter-spacing:.03em;
    color:var(--muted);cursor:pointer;list-style:none;
  }
  details.advanced summary::-webkit-details-marker{display:none}
  details.advanced summary::before{content:"+ ";font-weight:600}
  details.advanced[open] summary::before{content:"\2212 "}
  details.advanced summary:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
  .adv-grid{display:flex;flex-direction:column;gap:12px;padding:14px 0 4px}
  .adv-grid textarea{
    font-family:var(--body);font-size:14px;line-height:1.5;
    padding:8px 10px;border:1px solid #a9b0b8;border-radius:3px;background:#fff;color:var(--ink);
    resize:vertical;min-height:62px;width:100%;
  }
  .adv-grid textarea:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  .adv-row{display:flex;flex-wrap:wrap;gap:12px 22px;align-items:center}
  .hint{display:block;color:var(--muted);font-size:12.5px;margin-top:4px;max-width:56ch}
  .field-block{display:block;font-size:13.5px;color:var(--muted)}
  .field-block > span{display:block;margin-bottom:5px}
  input[type=number]{
    font-family:var(--display);font-size:14px;width:88px;
    padding:5px 7px;border:1px solid #a9b0b8;border-radius:3px;background:#fff;color:var(--ink);
  }
  input[type=number]:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  .empty{margin-top:26px;color:var(--muted);font-size:14px}
  .locked{display:none}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">

  <div class="readout">
    <dl><dt>Device</dt><dd><span class="lamp" id="lamp"></span><span id="r-device">checking</span></dd></dl>
    <dl><dt>Precision</dt><dd id="r-precision">&mdash;</dd></dl>
    <dl><dt>ffmpeg</dt><dd id="r-ffmpeg">&mdash;</dd></dl>
    <dl><dt>Queue</dt><dd id="r-queue">0</dd></dl>
    <dl><dt>Loaded</dt><dd id="r-loaded">none</dd></dl>
  </div>

  <h1>Transcribe audio on the workstation</h1>
  <p class="sub">Drop a recording here and it runs through Whisper on the GPU in this
  machine. Nothing leaves the network.</p>

  <div class="gate locked" id="gate">
    <p>This server needs an access token. It was printed in the terminal when the
    server started.</p>
    <div class="row">
      <input type="password" id="gate-token" autocomplete="off" spellcheck="false"
             placeholder="Access token" aria-label="Access token">
      <button id="gate-go">Unlock</button>
    </div>
    <p class="gate-err locked" id="gate-err">That token was rejected. Check the terminal output.</p>
  </div>

  <div id="main" class="locked">
    <div class="intake" id="intake" tabindex="0" role="button"
         aria-label="Choose files to transcribe">
      <p>Drop files to transcribe</p>
      <small>or click to browse &middot; audio and video both work</small>
      <input type="file" id="picker" multiple
             accept="audio/*,video/*,.m4a,.mp3,.wav,.mp4,.mov,.mkv,.flac,.ogg,.opus,.webm">
    </div>

    <div class="controls">
      <label class="field">Model
        <select id="model"></select>
      </label>
      <label class="field">Language
        <select id="language">
          <option value="">Detect</option>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="pt">Portuguese</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="he">Hebrew</option>
        </select>
      </label>
      <label class="field">Quality
        <select id="quality"></select>
      </label>
      <label class="field locked" id="compute-field">Precision
        <select id="compute"></select>
      </label>
      <label class="field"><input type="checkbox" id="vad" checked> Skip silence</label>
    </div>

    <details class="advanced">
      <summary>Advanced</summary>
      <div class="adv-grid">
        <label class="field-block">
          <span>Names and terms to expect</span>
          <textarea id="hotwords" rows="2" spellcheck="false"
            placeholder="Kubernetes, CODEOWNERS, MCP gateway, PostgreSQL"></textarea>
          <small class="hint">Comma-separated. Biases the decoder toward words it would
          otherwise mangle &mdash; product names, acronyms, people.</small>
        </label>

        <label class="field-block">
          <span>Context prompt</span>
          <textarea id="prompt" rows="2" spellcheck="false"
            placeholder="A recorded engineering planning call about developer tooling."></textarea>
          <small class="hint">A sentence describing the recording. Nudges style and
          vocabulary more broadly than the term list above.</small>
        </label>

        <div class="adv-row">
          <label class="field"><input type="checkbox" id="translate"> Translate to English</label>
          <label class="field"><input type="checkbox" id="words"> Word-level timings</label>
          <label class="field"><input type="checkbox" id="condition"> Carry context forward</label>
        </div>
        <small class="hint">Carrying context forward is off by default: on long
        recordings it's the usual cause of Whisper repeating a phrase for minutes.
        Word timings cost time, and help if you later run diarization.</small>

        <div class="adv-row" id="vad-tuning">
          <label class="field-block">
            <span>Min silence (ms)</span>
            <input type="number" id="min-silence" value="2000" min="100" max="10000" step="100">
          </label>
          <label class="field-block">
            <span>Speech padding (ms)</span>
            <input type="number" id="speech-pad" value="400" min="0" max="2000" step="50">
          </label>
        </div>
      </div>
    </details>

    <div id="jobs"></div>
    <p class="empty" id="empty">No jobs yet.</p>
  </div>
</div>

<script>
"use strict";
const el = (id) => document.getElementById(id);

/* Every value that reaches innerHTML goes through this. Filenames and error
   strings are attacker-controlled: anyone who can upload can plant markup. */
const ESCAPES = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;","`":"&#96;"};
const esc = (v) => String(v == null ? "" : v).replace(/[&<>"'`]/g, (c) => ESCAPES[c]);

/* ---------- token: accepted from the URL once, then moved out of it ---------- */
let TOKEN = sessionStorage.getItem("tk") || "";
{
  const fromUrl = new URLSearchParams(location.search).get("token");
  if(fromUrl){
    TOKEN = fromUrl;
    sessionStorage.setItem("tk", TOKEN);
    // Keep it out of history, bookmarks and any Referer we might emit.
    history.replaceState(null, "", location.pathname);
  }
}

let unlocked = false;

async function api(path, opts={}){
  const headers = {...(opts.headers||{})};
  if(TOKEN) headers["x-token"] = TOKEN;   // header only: forces a CORS preflight
  const r = await fetch(path, {...opts, headers, credentials:"omit"});
  if(r.status === 401){
    unlocked = false;
    el("gate").classList.remove("locked");
    el("main").classList.add("locked");
    throw new Error("unauthorised");
  }
  return r;
}

el("gate-go").addEventListener("click", submitToken);
el("gate-token").addEventListener("keydown", (e) => { if(e.key === "Enter") submitToken(); });

async function submitToken(){
  const value = el("gate-token").value.trim();
  if(!value) return;
  TOKEN = value;
  try{
    const r = await fetch("/api/status", {headers:{"x-token":TOKEN}, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    sessionStorage.setItem("tk", TOKEN);
    el("gate-err").classList.add("locked");
    el("gate").classList.add("locked");
    el("gate-token").value = "";
    await boot();
  }catch(e){
    el("gate-err").classList.remove("locked");
  }
}

/* ---------- status strip ---------- */
async function refreshStatus(){
  try{
    const s = await (await api("/api/status")).json();
    unlocked = true;
    el("main").classList.remove("locked");
    el("gate").classList.add("locked");

    el("lamp").className = "lamp " + (s.device === "cuda" ? "on" : "bad");
    el("r-device").textContent = s.gpu || s.device;
    el("r-precision").textContent = s.compute_type;
    el("r-ffmpeg").textContent = s.ffmpeg ? "found" : "missing";
    el("r-ffmpeg").style.color = s.ffmpeg ? "" : "var(--red)";
    el("r-queue").textContent = s.active_jobs;
    el("r-loaded").textContent = (s.loaded_models && s.loaded_models.length)
      ? s.loaded_models.join(", ") : "none";
    MAX_MB = s.max_upload_mb;
    RETRY_OK = s.retry_available;

    if(!POPULATED){
      POPULATED = true;
      fill("model", s.models, s.default_model);
      fill("quality", s.qualities, s.default_quality);
      fill("compute", s.compute_types, s.compute_type);
      // Precision is a property of this machine, not of a recording. It only
      // appears if the operator explicitly opened it up.
      if(s.allow_precision_choice) el("compute-field").classList.remove("locked");
      if(!s.allow_model_choice){
        el("model").disabled = true;
        el("model").title = "Pinned by the server";
      }
    }
  }catch(e){
    if(e.message !== "unauthorised"){
      el("lamp").className = "lamp bad";
      el("r-device").textContent = "server unreachable";
    }
  }
}

/* ---------- upload ---------- */
let MAX_MB = 0, RETRY_OK = true, POPULATED = false;

function fill(id, values, selected){
  const node = el(id);
  node.textContent = "";
  for(const v of values){
    const o = document.createElement("option");
    o.value = o.textContent = v;
    if(v === selected) o.selected = true;
    node.append(o);
  }
}

/* Whatever is in the controls right now, as a form body. Used for both a fresh
   upload and a retry, so a retry re-runs the same audio under new settings. */
function currentSettings(){
  const fd = new FormData();
  fd.append("model", el("model").value);
  fd.append("compute_type", el("compute").value);
  fd.append("language", el("language").value);
  fd.append("quality", el("quality").value);
  fd.append("vad", el("vad").checked ? "true" : "false");
  fd.append("prompt", el("prompt").value);
  fd.append("hotwords", el("hotwords").value);
  fd.append("translate", el("translate").checked ? "true" : "false");
  fd.append("condition", el("condition").checked ? "true" : "false");
  fd.append("word_timestamps", el("words").checked ? "true" : "false");
  fd.append("min_silence_ms", el("min-silence").value || "2000");
  fd.append("speech_pad_ms", el("speech-pad").value || "400");
  return fd;
}

const intake = el("intake"), picker = el("picker");
intake.addEventListener("click", () => picker.click());
intake.addEventListener("keydown", (e) => {
  if(e.key === "Enter" || e.key === " "){ e.preventDefault(); picker.click(); }
});
picker.addEventListener("change", () => { send([...picker.files]); picker.value = ""; });

for(const ev of ["dragenter","dragover"])
  intake.addEventListener(ev, (e) => { e.preventDefault(); intake.classList.add("hot"); });
for(const ev of ["dragleave","drop"])
  intake.addEventListener(ev, (e) => { e.preventDefault(); intake.classList.remove("hot"); });
intake.addEventListener("drop", (e) => send([...e.dataTransfer.files]));
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => e.preventDefault());

async function send(files){
  for(const f of files){
    if(MAX_MB && f.size > MAX_MB * 1024 * 1024){
      alert(f.name + " is larger than the " + MAX_MB + " MB limit.");
      continue;
    }
    const fd = currentSettings();
    fd.append("file", f);
    try{
      const r = await api("/api/jobs", {method:"POST", body:fd});
      if(!r.ok){
        let detail = r.statusText;
        try{ detail = (await r.json()).detail || detail; }catch(_){}
        throw new Error(detail);
      }
      await r.json();
      await tick();
    }catch(err){
      if(err.message !== "unauthorised") alert("Upload failed for " + f.name + ": " + err.message);
    }
  }
}

el("vad").addEventListener("change", () => {
  el("vad-tuning").style.display = el("vad").checked ? "" : "none";
});

/* ---------- rendering ---------- */
const TICKS = 40;
const known = new Map();

function meter(job){
  const lit = Math.round((job.progress || 0) * TICKS);
  const cls = job.state === "done" ? "done" : job.state === "error" ? "fail" : "lit";
  let html = '<div class="meter">';
  for(let i=0;i<TICKS;i++) html += '<i class="' + (i<lit?cls:"") + '"></i>';
  return html + "</div>";
}

function fmtTime(s){
  s = Math.round(s);
  const m = Math.floor(s/60);
  return m ? m + "m " + String(s%60).padStart(2,"0") + "s" : s + "s";
}

function statusLine(job){
  if(job.state === "queued")  return "Waiting for the GPU";
  if(job.state === "loading") return "Loading " + esc(job.opts.model);
  if(job.state === "running"){
    const pct = Math.round((job.progress||0)*100);
    const eta = job.progress > 0.02
      ? " \u00b7 about " + fmtTime(job.elapsed/job.progress - job.elapsed) + " left"
      : "";
    return pct + "% \u00b7 " + job.segment_count + " segments" + eta;
  }
  if(job.state === "done"){
    const speed = job.duration && job.elapsed
      ? " \u00b7 " + (job.duration/job.elapsed).toFixed(1) + "\u00d7 realtime" : "";
    return "Finished in " + fmtTime(job.elapsed) + speed
      + " \u00b7 " + esc(job.language || "?") + " \u00b7 " + job.segment_count + " segments";
  }
  if(job.state === "cancelled") return "Cancelled";
  return esc(job.message);
}

/* Only show the knobs that were actually off-default, so the line stays short. */
function jobTags(job){
  const o = job.opts || {};
  const bits = [o.model];
  if(o.quality && o.quality !== "balanced") bits.push(o.quality);
  if(o.translate) bits.push("translated");
  if(o.condition) bits.push("context on");
  if(o.word_timestamps) bits.push("word times");
  if(o.hotwords) bits.push("terms");
  if(job.duration) bits.push(fmtTime(job.duration));
  return bits.join(" \u00b7 ");
}

function render(job){
  const id = "job-" + job.id;
  let node = document.getElementById(id);
  if(!node){
    node = document.createElement("div");
    node.className = "job";
    node.id = id;
    el("jobs").prepend(node);
  }
  const dl = (fmt, label) =>
    '<button data-dl="' + fmt + '" data-id="' + esc(job.id) + '">' + label + "</button>";

  const text = (job.segments || []).map(s => s.text).join(" ");
  const name = esc(job.filename);

  node.innerHTML =
    '<div class="job-head">' +
      '<span class="job-name" title="' + name + '">' + name + "</span>" +
      '<span class="job-meta">' + esc(jobTags(job)) + "</span>" +
    "</div>" +
    meter(job) +
    '<div class="job-body">' +
      '<p class="status' + (job.state === "error" ? " err" : "") + '">' + statusLine(job) + "</p>" +
      '<div class="transcript">' + esc(text) + "</div>" +
      '<div class="actions">' +
        (job.state === "done"
          ? '<button data-copy="' + esc(job.id) + '">Copy text</button>' +
            dl("txt","Save .txt") + dl("timestamped","Save timestamped") +
            dl("srt","Save .srt") + dl("vtt","Save .vtt") + dl("json","Save .json")
          : "") +
        (RETRY_OK && job.can_retry
          ? '<button data-retry="' + esc(job.id) + '">Retry with these settings</button>'
          : "") +
        '<button class="ghost" data-del="' + esc(job.id) + '">' +
          (["queued","loading","running"].includes(job.state) ? "Cancel" : "Remove") +
        "</button>" +
      "</div>" +
    "</div>";
}

/* Downloads go through fetch so the token stays in a header, never a URL. */
async function download(jobId, fmt){
  const r = await api("/api/jobs/" + encodeURIComponent(jobId)
                      + "/text?format=" + encodeURIComponent(fmt) + "&download=1");
  if(!r.ok) return;
  const blob = await r.blob();
  const cd = r.headers.get("content-disposition") || "";
  let name = "transcript." + (fmt === "timestamped" ? "txt" : fmt);
  const star = /filename\*=UTF-8''([^;]+)/i.exec(cd);
  const plain = /filename="([^"]*)"/i.exec(cd);
  if(star) { try { name = decodeURIComponent(star[1]); } catch(_){} }
  else if(plain) name = plain[1];
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

el("jobs").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if(!b) return;
  try{
    if(b.dataset.dl) await download(b.dataset.id, b.dataset.dl);
    if(b.dataset.copy){
      const r = await api("/api/jobs/" + encodeURIComponent(b.dataset.copy) + "/text?format=txt");
      const body = await r.text();
      if(navigator.clipboard && window.isSecureContext){
        await navigator.clipboard.writeText(body);
      }else{
        // Plain HTTP is not a secure context, so the Clipboard API is unavailable.
        const ta = document.createElement("textarea");
        ta.value = body;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.append(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      b.textContent = "Copied";
      setTimeout(() => (b.textContent = "Copy text"), 1400);
    }
    if(b.dataset.retry){
      b.disabled = true;
      const r = await api("/api/jobs/" + encodeURIComponent(b.dataset.retry) + "/retry",
                          {method:"POST", body:currentSettings()});
      if(!r.ok){
        let detail = r.statusText;
        try{ detail = (await r.json()).detail || detail; }catch(_){}
        alert("Retry failed: " + detail);
        b.disabled = false;
      }
      await tick();
    }
    if(b.dataset.del){
      await api("/api/jobs/" + encodeURIComponent(b.dataset.del), {method:"DELETE"});
      await tick();
    }
  }catch(err){ /* 401 already handled by api() */ }
});

/* ---------- poll ---------- */
async function tick(){
  if(!unlocked) return;
  try{
    const {jobs} = await (await api("/api/jobs")).json();
    const live = jobs.map(j => j.id);
    el("empty").style.display = jobs.length ? "none" : "";

    for(const stale of [...known.keys()].filter(id => !live.includes(id))){
      const n = document.getElementById("job-" + stale);
      if(n) n.remove();
      known.delete(stale);
    }

    for(const summary of jobs){
      const sig = summary.state + ":" + summary.progress + ":" + summary.segment_count;
      if(known.get(summary.id) === sig) continue;
      known.set(summary.id, sig);
      const full = await (await api("/api/jobs/" + encodeURIComponent(summary.id))).json();
      render(full);
    }
  }catch(e){ /* server blip or 401; next tick retries */ }
}

async function boot(){
  await refreshStatus();
  await tick();
}

boot();
setInterval(tick, 1200);
setInterval(refreshStatus, 5000);
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def local_names(bind_host: str, extra: List[str]) -> tuple[Set[str], Set[str]]:
    """Hostnames this server will answer to. Anything else is a rebinding attempt.

    Returns (exact_names, suffixes). A --allow-host entry starting with
    "*." or "." becomes a suffix match, so --allow-host .trycloudflare.com
    covers the random hostnames `cloudflared tunnel --url` hands out.
    """
    names = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    suffixes: Set[str] = set()
    try:
        hostname = socket.gethostname()
        names.add(hostname.lower())
        for info in socket.getaddrinfo(hostname, None):
            names.add(str(info[4][0]).lower())
    except OSError:
        pass
    if bind_host not in ("0.0.0.0", "::"):
        names.add(bind_host.lower())
    for entry in extra:
        e = entry.strip().lower()
        if not e:
            continue
        if e.startswith("*."):
            e = e[1:]  # "*.example.com" -> ".example.com"
        if e.startswith("."):
            suffix = normalize_host(e)
            if suffix:
                suffixes.add("." + suffix.lstrip("."))
        else:
            norm = normalize_host(e)
            if norm:
                names.add(norm)
    return names, suffixes


def main() -> None:
    global ARGS, ALLOWED_HOSTS, ALLOWED_SUFFIXES
    p = argparse.ArgumentParser(description="LAN transcription server (faster-whisper)")

    net = p.add_argument_group("network and access")
    net.add_argument(
        "--host", default="0.0.0.0", help="bind address (default: all interfaces)"
    )
    net.add_argument("--port", type=int, default=8765)
    net.add_argument(
        "--token",
        default=os.environ.get("TRANSCRIBE_TOKEN", ""),
        help="access token; one is generated if omitted (prefer TRANSCRIBE_TOKEN "
        "over the flag, which is visible in the process list)",
    )
    net.add_argument(
        "--no-auth",
        action="store_true",
        help="serve without a token (only on a network you control)",
    )
    net.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="extra Host header value to accept; repeatable. "
        'Prefix with "." for a suffix match, e.g. --allow-host '
        ".trycloudflare.com for Cloudflare tunnels.",
    )

    gpu = p.add_argument_group("model and hardware")
    gpu.add_argument(
        "--model", default="large-v3", choices=MODELS, help="default model"
    )
    gpu.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    gpu.add_argument(
        "--compute-type",
        default="float16",
        choices=COMPUTE_TYPES,
        help="float16 needs compute capability 7.0+; use int8 or float32 "
        "on Pascal and older",
    )
    gpu.add_argument(
        "--quality",
        default="balanced",
        choices=list(QUALITIES),
        help="default beam size: fast=1, balanced=5, thorough=8",
    )
    gpu.add_argument(
        "--model-cache",
        type=int,
        default=1,
        metavar="N",
        help="models held in VRAM at once (default 1; raising this lets "
        "several model/precision combinations stay resident)",
    )
    gpu.add_argument(
        "--allow-model-choice",
        dest="allow_model_choice",
        action="store_true",
        default=True,
        help="let clients pick the model (default)",
    )
    gpu.add_argument(
        "--pin-model",
        dest="allow_model_choice",
        action="store_false",
        help="force every job to use --model",
    )
    gpu.add_argument(
        "--allow-precision-choice",
        dest="allow_precision_choice",
        action="store_true",
        default=False,
        help="expose the precision selector in the UI (off by default: "
        "precision is a property of the machine, not the recording)",
    )
    gpu.add_argument(
        "--preload",
        action="store_true",
        help="load the default model at startup instead of on first job",
    )

    lim = p.add_argument_group("limits and storage")
    lim.add_argument(
        "--max-upload-mb", type=int, default=2048, help="per-file upload ceiling"
    )
    lim.add_argument(
        "--max-queue", type=int, default=20, help="max jobs pending at once"
    )
    lim.add_argument(
        "--max-jobs",
        type=int,
        default=60,
        help="finished job records retained before the oldest are evicted",
    )
    lim.add_argument(
        "--source-retention",
        default="job",
        choices=RETENTION,
        help="'run' deletes the upload right after transcription (no retry), "
        "'job' keeps it while the job record exists (default), "
        "'forever' never deletes it",
    )

    args = p.parse_args()

    if args.no_auth:
        args.token = ""
        generated = False
    else:
        generated = not args.token
        if generated:
            args.token = secrets.token_urlsafe(24)

    ARGS = args
    ALLOWED_HOSTS, ALLOWED_SUFFIXES = local_names(args.host, args.allow_host)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if not shutil.which("ffmpeg"):
        print("!  ffmpeg not found on PATH. Most formats will fail to decode.")
        print("   winget install Gyan.FFmpeg   (then reopen the terminal)\n")

    threading.Thread(target=worker_loop, daemon=True, name="transcriber").start()

    if args.preload:
        print(f"Loading {args.model} on {args.device} ({args.compute_type}) ...")
        load_model(args.model, args.device, args.compute_type)
        print("Model ready.")

    print()
    if args.token:
        print(f"Access token: {args.token}")
        if generated:
            print(
                "(generated for this run; pass --token or TRANSCRIBE_TOKEN to pin it)"
            )
        print(f"\nOpen:  http://<this-machine-ip>:{args.port}/?token={args.token}")
        print("       the token moves out of the URL as soon as the page loads")
    else:
        print("!  Running with --no-auth. Anyone who can reach this port can read")
        print("   every transcript and upload files to this machine.")
        print(f"\nOpen:  http://<this-machine-ip>:{args.port}/")

    print(
        f"\nModel      {args.model}"
        f"{'' if args.allow_model_choice else '  (pinned)'}"
    )
    print(
        f"Precision  {args.compute_type}"
        f"{'  (selectable)' if args.allow_precision_choice else '  (pinned)'}"
    )
    print(f"VRAM cache {args.model_cache} model(s)")
    print(
        f"Sources    retention={args.source_retention}"
        f"{'  retry enabled' if args.source_retention != 'run' else '  retry disabled'}"
    )
    print(f"\nAccepting Host: {', '.join(sorted(ALLOWED_HOSTS | ALLOWED_SUFFIXES))}")
    print("(add more with --allow-host)\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
