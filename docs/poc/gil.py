"""Does sherpa_onnx's C++ process() release the GIL?

This decides whether diarization can run on a background thread while the GPU
transcribes. If the C++ call holds the GIL for its whole duration, a background
thread would starve the asyncio event loop and the server would look hung.

Method: a Python ticker thread wakes every 10 ms and records how late it was.
If the GIL is released during process(), the ticker keeps ticking. If it is
held, the ticker is starved and the gaps balloon.
"""

from __future__ import annotations

import statistics
import threading
import time
import wave

import numpy as np
import sherpa_onnx

SEG = "sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
EMB = "nemo_en_titanet_small.onnx"
REPEATS = 10
TICK = 0.01

with wave.open("0-four-speakers-zh.wav") as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    audio = audio.astype(np.float32) / 32768.0

config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=SEG, window_shift_ratio=0.1
        ),
        num_threads=4,
    ),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB, num_threads=4),
    clustering=sherpa_onnx.FastClusteringConfig(num_clusters=4, threshold=0.5),
    min_duration_on=0.3,
    min_duration_off=0.5,
)
sd = sherpa_onnx.OfflineSpeakerDiarization(config)


def run_ticker(stop, gaps, counts):
    """Sample event-loop responsiveness the way an asyncio tick would."""
    last = time.perf_counter()
    n = 0
    while not stop.is_set():
        time.sleep(TICK)
        now = time.perf_counter()
        gaps.append(now - last - TICK)
        last = now
        n += 1
        counts["ticks"] = n


def measure(work, label):
    stop, gaps, counts = threading.Event(), [], {"ticks": 0}
    t = threading.Thread(target=run_ticker, args=(stop, gaps, counts), daemon=True)
    t.start()
    time.sleep(0.2)  # let the ticker settle
    t0 = time.perf_counter()
    work()
    elapsed = time.perf_counter() - t0
    stop.set()
    t.join(timeout=2.0)
    gaps.sort()
    p99 = gaps[int(len(gaps) * 0.99)] if gaps else float("nan")
    print(
        f"{label:28} {elapsed:6.2f}s  ticks {counts['ticks']:5d}  "
        f"tick gap: median {statistics.median(gaps) * 1000:7.2f} ms  "
        f"p99 {p99 * 1000:8.2f} ms  max {max(gaps) * 1000:8.2f} ms"
    )
    return max(gaps)


# Baseline: the main thread does nothing GIL-heavy either.
base = measure(lambda: time.sleep(REPEATS * 2.0), "baseline (sleep)")

# The question: does the event loop keep running during process()?
worst = 0.0
for _ in range(REPEATS):
    worst = max(worst, measure(lambda: sd.process(audio), "diarizing (main thread)"))

print()
print(f"worst tick gap, baseline   : {base * 1000:8.2f} ms")
print(f"worst tick gap, diarizing  : {worst * 1000:8.2f} ms")
verdict = "GIL RELEASED" if worst < 0.5 else "GIL HELD"
print(f"verdict: {verdict}")
