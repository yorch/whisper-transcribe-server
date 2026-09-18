"""Threshold calibration across all four files sherpa's own CI uses.

Two of the four numbers matter: the speaker count each setting produces, versus
the count the file actually has. Over-segmentation is what 0.5 does; the risk in
raising it is merging two real speakers, and the three English files are
two-speaker recordings, so they are the counter-case.
"""

from __future__ import annotations

import sherpa_onnx
from faster_whisper.audio import decode_audio

SEG = "/tmp/diar-poc/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
EMB = "/tmp/diar-poc/nemo_en_titanet_small.onnx"

FILES = [
    ("0-four-speakers-zh.wav", 4),
    ("1-two-speakers-en.wav", 2),
    ("2-two-speakers-en.wav", 2),
    ("3-two-speakers-en.wav", 2),
]
SETTINGS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def run(audio, clusters, threshold):
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=SEG, window_shift_ratio=0.1
            ),
            num_threads=4,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB, num_threads=4),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=clusters, threshold=threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    turns = diarizer.process(audio).sort_by_start_time()
    return sorted({int(t.speaker) for t in turns}), len(turns)


print(
    f"{'file':26} {'want':>4} | "
    + " | ".join(f"thr {s:.1f}" for s in SETTINGS)
    + " |  pinned"
)
print("-" * 108)

results = {}
for name, want in FILES:
    audio = decode_audio(f"/tmp/diar-poc/{name}", sampling_rate=16000)
    cells = []
    for threshold in SETTINGS:
        found, _ = run(audio, -1, threshold)
        results[(name, threshold)] = len(found)
        mark = "" if len(found) == want else ("+" if len(found) > want else "-")
        cells.append(f"{len(found):>5}{mark} ")
    pinned, _ = run(audio, want, 0.5)
    print(f"{name:26} {want:>4} | " + "|".join(cells) + f"|  {len(pinned)}")

print("\n+ = too many speakers, - = too few, blank = exactly right")
print("\ncorrect across all four files:")
for threshold in SETTINGS:
    hits = sum(1 for name, want in FILES if results[(name, threshold)] == want)
    print(f"  threshold {threshold:.1f}: {hits}/4")
