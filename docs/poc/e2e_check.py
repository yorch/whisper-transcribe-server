#!/usr/bin/env python3
"""End-to-end check: real models, real child worker, real alignment and exports.

The test suite stubs the child on purpose, so it can run in a few seconds and
without 42 MB of weights. This drives the actual shipped path instead, because
stubs assert the protocol and not the answers:

    ensure_diarize_models() -> run_diarizer()   (spawns DIARIZE_WORKER, real
                                                 .onnx, decodes via PyAV)
                            -> align_speakers() -> render()

Run it against the segmentation model's sample recordings. Download them with
the commands in this directory's README, then:

    .venv/bin/python docs/poc/e2e_check.py .                 # cwd = repo root
    TRANSCRIBE_DIARIZE_SPEAKERS=4 .../0-four-speakers-zh.wav

It is not part of the suite and nothing imports it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))

import transcribe_server as s  # noqa: E402

# Where the throwaway work dir, the weights and the sample live. Override with
# the environment to point at a real recording of your own.
WORK = Path(os.environ.get("TRANSCRIBE_DIARIZE_WORK", "/tmp/diar-e2e"))
POOL = Path(os.environ.get("TRANSCRIBE_DIARIZE_POOL", "/tmp/diar-poc"))
AUDIO = Path(
    os.environ.get("TRANSCRIBE_DIARIZE_AUDIO", str(POOL / "0-four-speakers-zh.wav"))
)
EXPECTED = int(os.environ.get("TRANSCRIBE_DIARIZE_SPEAKERS", "4"))

if not AUDIO.is_file():
    raise SystemExit(f"!  no sample audio at {AUDIO}; see docs/poc/README.md")

# Stage the weights where the server looks for them, rather than re-downloading:
# the hash check below is then doing real work against the pinned values.
MODELS = WORK / "diarize-models"
if WORK.exists():
    shutil.rmtree(WORK)
MODELS.mkdir(parents=True)
for source, name in [
    (
        POOL / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx",
        "segmentation.int8.onnx",
    ),
    (POOL / "nemo_en_titanet_small.onnx", "embedding.onnx"),
]:
    if not source.is_file():
        raise SystemExit(f"!  missing {source}; see docs/poc/README.md")
    shutil.copy(source, MODELS / name)

s.WORK_DIR = WORK
s.ARGS = type(
    "A",
    (),
    {
        "allow_diarize": True,
        "allow_model_choice": True,
        "allow_precision_choice": False,
        "model": "base",
        "compute_type": "int8",
        "quality": "balanced",
    },
)()

print("== ensure_diarize_models (verifies the pinned hashes) ==")
t0 = time.perf_counter()
models = s.ensure_diarize_models()
print(
    f"   resolved in {time.perf_counter() - t0:.2f}s: "
    f"{ {k: v.name for k, v in models.items()} }"
)

print("\n== run_diarizer through the real child process ==")
progress: list[int] = []
t0 = time.perf_counter()
turns = s.run_diarizer(str(AUDIO), EXPECTED, models, on_progress=progress.append)
elapsed = time.perf_counter() - t0
if turns is None:
    raise SystemExit("!  the run was cancelled, which nothing should have done")
print(
    f"   {len(turns)} turns in {elapsed:.2f}s, progress: "
    f"{progress[:3]}{'...' if len(progress) > 3 else ''} -> {progress[-1:]}"
)
for t in turns:
    print(f"   {t['start']:7.2f} -- {t['end']:7.2f}  Speaker {t['speaker']}")

found = {t["speaker"] for t in turns}
assert found == set(range(1, EXPECTED + 1)), (
    f"expected {EXPECTED} speakers, got {sorted(found)}"
)
assert next(iter(turns))["speaker"] == 1, "Speaker 1 should speak first"
print(f"   OK: {EXPECTED} speakers, renumbered from first appearance")

print("\n== alignment against hand-written Whisper-style segments ==")
# Deliberately coarse segments straddling real speaker changes, which is the
# case word-level alignment exists for. The word times are chosen to fall inside
# turns of the real output above; adjust them if you point this at other audio.
segments = [
    {
        "start": 0.5,
        "end": 11.0,
        "text": "alpha beta gamma delta epsilon zeta eta theta",
        "words": [
            {"start": 0.7, "end": 1.2, "word": " alpha"},
            {"start": 1.4, "end": 1.9, "word": " beta"},
            {"start": 7.2, "end": 7.7, "word": " gamma"},
            {"start": 8.0, "end": 8.5, "word": " delta"},
            {"start": 11.6, "end": 12.1, "word": " epsilon"},
            {"start": 13.9, "end": 14.4, "word": " zeta"},
            {"start": 22.3, "end": 22.8, "word": " eta"},
            {"start": 27.8, "end": 28.3, "word": " theta"},
        ],
    }
]
aligned = s.align_speakers(segments, turns)
for entry in aligned:
    print(
        f"   {entry['start']:7.2f} -- {entry['end']:7.2f}  "
        f"Speaker {entry['speaker']}: {entry['text']}"
    )

assert len(aligned) > 1, "a segment crossing speaker changes must be split"
assert all(e["speaker"] in found for e in aligned)
# alpha/beta are in Speaker 1's first turn; gamma/delta/epsilon all fall inside
# Speaker 2's two adjacent turns so they group into one run; eta lands back in
# Speaker 1's second turn; theta in Speaker 4's. Word tokens carry their own
# spacing, so a run rejoins into readable text.
assert [e["speaker"] for e in aligned] == [1, 2, 3, 1, 4], [
    e["speaker"] for e in aligned
]
assert aligned[1]["text"] == "gamma delta epsilon", aligned[1]["text"]

print("\n== render() with the real labels ==")
job = {
    "filename": AUDIO.name,
    "language": None,
    "duration": 56.9,
    "opts": s.build_opts(
        "base",
        "int8",
        "",
        "true",
        "balanced",
        "",
        "",
        "false",
        "false",
        "false",
        2000,
        400,
        "true",
        0,
    ),
    "segments": aligned,
}
print(s.render(job, "txt")[0])
vtt = s.render(job, "vtt")[0]
assert "<v Speaker 1>alpha beta" in vtt, vtt
payload = json.loads(s.render(job, "json")[0])
print("json speakers block:", json.dumps(payload["speakers"]))
assert [sp["speaker"] for sp in payload["speakers"]] == [1, 2, 3, 4]

print("\n== probe_diarization (what --preload runs) ==")
t0 = time.perf_counter()
s.probe_diarization()
print(f"   OK in {time.perf_counter() - t0:.2f}s")
assert not (WORK / "diarize-probe.wav").exists(), "the probe left its wav behind"

print("\nALL END-TO-END CHECKS PASSED")
