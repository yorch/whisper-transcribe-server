"""Bounded proof of concept: sherpa-onnx diarization with no PyTorch.

Reads a 16 kHz mono 16-bit WAV with the stdlib only (no soundfile/librosa) to
prove the decode side needs nothing new either.
"""

from __future__ import annotations

import sys
import time
import wave

import numpy as np
import sherpa_onnx

print("sherpa_onnx", sherpa_onnx.__version__)

wav = sys.argv[1] if len(sys.argv) > 1 else "0-four-speakers-zh.wav"
seg = "sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
emb = "nemo_en_titanet_small.onnx"

with wave.open(wav) as w:
    rate = w.getframerate()
    channels = w.getnchannels()
    width = w.getsampwidth()
    raw = w.readframes(w.getnframes())
print(
    f"wav: {rate} Hz, {channels} ch, {width * 8} bit, "
    f"{len(raw) / (rate * channels * width):.1f} s"
)

audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
if channels == 2:
    audio = audio.reshape(-1, 2).mean(axis=1)

config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=seg, window_shift_ratio=0.1
        ),
        num_threads=4,
    ),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=4),
    clustering=sherpa_onnx.FastClusteringConfig(num_clusters=4, threshold=0.5),
    min_duration_on=0.3,
    min_duration_off=0.5,
)
assert config.validate(), "config invalid"
assert sherpa_onnx.OfflineSpeakerDiarization(config).sample_rate == rate, (
    "rate mismatch"
)

sd = sherpa_onnx.OfflineSpeakerDiarization(config)
t0 = time.perf_counter()
result = sd.process(audio).sort_by_start_time()
elapsed = time.perf_counter() - t0

duration = len(audio) / rate
for r in result:
    print(f"{r.start:7.3f} -- {r.end:7.3f}  speaker_{r.speaker:02d}")

print(
    f"\nduration {duration:.1f}s  elapsed {elapsed:.2f}s  RTF {elapsed / duration:.3f}"
)

# Does the audio survive a round trip through faster-whisper's decoder?
try:
    from faster_whisper.audio import decode_audio

    decoded = decode_audio(wav, sampling_rate=16000)
    print(
        f"faster_whisper.decode_audio -> {decoded.dtype} {decoded.shape} "
        f"peak {abs(decoded).max():.3f}"
    )
    t0 = time.perf_counter()
    result2 = sd.process(decoded).sort_by_start_time()
    print(
        f"re-run on that array: {time.perf_counter() - t0:.2f}s, "
        f"{len(result2)} turns (same count as {len(result)}: "
        f"{len(result2) == len(result)})"
    )
except ImportError as exc:
    print("faster_whisper not on this path:", exc)
