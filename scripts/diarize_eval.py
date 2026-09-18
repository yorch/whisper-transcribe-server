"""Measure speaker diarization settings against recordings with known answers.

For every segmentation model x embedding model x clustering threshold, run the
diarizer over each recording in one or more manifests and report how often
Auto finds the right speaker count, with and without the server's
minor-speaker fold -- and, where a manifest carries reference turns, the
diarization error rate (missed speech + false alarm + confusion, over reference
speech; 10 ms frames, no collar, greedy speaker mapping). A pinned run (the
true count given) is included per model pair: it scores attribution alone,
which is what matters once the operator sets Speakers.

Runs in the dev environment, where sherpa-onnx and the server are importable:

    uv sync
    uv run python scripts/diarize_eval.py tmp/eval/public/manifest.json \\
        tmp/eval/synthetic/manifest.json --out tmp/eval/results.json

A manifest is a JSON list of {"path": ..., "speakers": N, "turns": [...]},
paths relative to the manifest; "turns" (start, end, speaker) is optional.
Your own recordings go in the same shape -- a count alone is enough.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx
from faster_whisper.audio import decode_audio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import transcribe_server as s  # noqa: E402

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATIONS = {
    "pyannote-3.0": (
        f"{RELEASES}/speaker-segmentation-models/"
        "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx",
    ),
    "reverb-v1": (
        f"{RELEASES}/speaker-segmentation-models/"
        "sherpa-onnx-reverb-diarization-v1.tar.bz2",
        "sherpa-onnx-reverb-diarization-v1/model.int8.onnx",
    ),
}
EMBEDDINGS = [
    "nemo_en_titanet_small.onnx",
    "nemo_en_titanet_large.onnx",
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
    "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "wespeaker_en_voxceleb_CAM++_LM.onnx",
]
FPS = 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  fetching {url.rsplit('/', 1)[-1]}", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs
        tmp.rename(dest)
    return dest


def segmentation_model(name: str, models: Path) -> Path:
    url, member = SEGMENTATIONS[name]
    out = models / f"seg-{name}.onnx"
    if not out.exists():
        tarball = fetch(url, models / url.rsplit("/", 1)[-1])
        with tarfile.open(tarball) as tar:
            data = tar.extractfile(member)
            if data is None:
                raise SystemExit(f"{member} is missing from {tarball.name}")
            out.write_bytes(data.read())
    return out


def diarizer(seg: Path, emb: Path, threads: int) -> Any:
    def config(clusters: int, threshold: float) -> Any:
        return sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg), window_shift_ratio=0.1
                ),
                num_threads=threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb), num_threads=threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=clusters, threshold=threshold
            ),
            min_duration_on=s.DIARIZE_MIN_DURATION_ON,
            min_duration_off=s.DIARIZE_MIN_DURATION_OFF,
        )

    engine = sherpa_onnx.OfflineSpeakerDiarization(config(-1, 0.8))

    def run(audio: np.ndarray, clusters: int, threshold: float) -> list[dict]:
        engine.set_config(config(clusters, threshold))
        result = engine.process(audio).sort_by_start_time()
        return s.relabel_speakers(
            [{"start": r.start, "end": r.end, "speaker": r.speaker} for r in result]
        )

    return run


def frames(turns: list[dict], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int32)
    for t in turns:
        out[int(t["start"] * FPS) : int(t["end"] * FPS)] = int(t["speaker"])
    return out


def der(ref: list[dict], hyp: list[dict], seconds: float) -> float:
    n = int(seconds * FPS) + 1
    r, h = frames(ref, n), frames(hyp, n)
    speech = r > 0
    if not speech.any():
        return float("nan")
    missed = int(np.sum(speech & (h == 0)))
    false_alarm = int(np.sum(~speech & (h > 0)))
    overlap = {
        (a, b): int(np.sum((r == a) & (h == b)))
        for a, b in itertools.product(set(r[speech]), set(h[h > 0]))
    }
    correct, used_r, used_h = 0, set(), set()
    for (a, b), count in sorted(overlap.items(), key=lambda kv: -kv[1]):
        if a not in used_r and b not in used_h:
            correct += count
            used_r.add(a)
            used_h.add(b)
    confusion = int(np.sum(speech & (h > 0))) - correct
    return (missed + false_alarm + confusion) / int(speech.sum())


def count(turns: list[dict]) -> int:
    return len({t["speaker"] for t in turns})


def load(manifests: list[Path]) -> list[dict]:
    items = []
    for manifest in manifests:
        for item in json.loads(manifest.read_text()):
            path = manifest.parent / item["path"]
            audio = decode_audio(str(path), sampling_rate=16000)
            items.append(
                {
                    "name": f"{manifest.parent.name}/{item['path']}",
                    "speakers": int(item["speakers"]),
                    "turns": item.get("turns"),
                    "audio": audio,
                    "seconds": len(audio) / 16000,
                }
            )
    return items


def summarise(rows: list[dict]) -> None:
    def mean(values: list[float]) -> float:
        values = [v for v in values if v == v]
        return sum(values) / len(values) if values else float("nan")

    print(
        f"\n{'segmentation':13} {'embedding':44} {'thr':>6} "
        f"{'exact':>6} {'+fold':>6} {'|err|':>6} {'DER':>6} {'+fold':>6}"
    )
    key = lambda r: (r["segmentation"], r["embedding"], str(r["threshold"]))  # noqa: E731
    for (seg, emb, thr), group in itertools.groupby(sorted(rows, key=key), key=key):
        g = list(group)
        exact = mean([r["found"] == r["speakers"] for r in g])
        exact_fold = mean([r["found_fold"] == r["speakers"] for r in g])
        err = mean([abs(r["found"] - r["speakers"]) for r in g])
        print(
            f"{seg:13} {emb[:44]:44} {thr:>6} {exact:6.0%} {exact_fold:6.0%} "
            f"{err:6.2f} {mean([r['der'] for r in g]):6.1%} "
            f"{mean([r['der_fold'] for r in g]):6.1%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / "tmp/eval/models")
    parser.add_argument("--segmentation", nargs="+", default=list(SEGMENTATIONS))
    parser.add_argument("--embedding", nargs="+", default=EMBEDDINGS)
    parser.add_argument(
        "--threshold", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    parser.add_argument("--fold", type=float, default=s.DIARIZE_FOLD_SHARE)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/eval/results.json")
    args = parser.parse_args()

    args.models.mkdir(parents=True, exist_ok=True)
    items = load(args.manifests)
    print(f"{len(items)} recordings, {sum(i['seconds'] for i in items) / 60:.1f} min")
    rows: list[dict] = []
    for seg_name, emb_name in itertools.product(args.segmentation, args.embedding):
        seg = segmentation_model(seg_name, args.models)
        emb = fetch(f"{RELEASES}/speaker-recongition-models/{emb_name}",
                    args.models / emb_name)  # fmt: skip
        print(f"{seg_name} + {emb_name}  (sha256 {sha256(emb)[:16]}…)", flush=True)
        run = diarizer(seg, emb, args.threads)
        for threshold in [*args.threshold, "pinned"]:
            started = time.time()
            for item in items:
                pinned = threshold == "pinned"
                turns = run(item["audio"], item["speakers"] if pinned else -1,
                            0.8 if pinned else float(threshold))  # fmt: skip
                folded, _ = (
                    (turns, 0) if pinned else s.fold_minor_speakers(turns, args.fold)
                )
                ref = item["turns"]
                rows.append(
                    {
                        "segmentation": seg_name,
                        "embedding": emb_name,
                        "threshold": threshold,
                        "file": item["name"],
                        "speakers": item["speakers"],
                        "found": count(turns),
                        "found_fold": count(folded),
                        "der": der(ref, turns, item["seconds"]) if ref else None,
                        "der_fold": der(ref, folded, item["seconds"]) if ref else None,
                    }
                )
            print(f"  {threshold}: {time.time() - started:.0f}s", flush=True)
            args.out.write_text(json.dumps(rows, indent=1))
    summarise([{**r, "der": r["der"] if r["der"] is not None else float("nan"),
                "der_fold": r["der_fold"] if r["der_fold"] is not None
                else float("nan")} for r in rows])  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
