# Diarization proof of concept

Throwaway scripts backing the measurements in `../speaker-diarization.md`.
They are not part of the server and nothing imports them; they exist so the two
claims the design rests on can be re-run rather than believed.

```bash
# ~42 MB of models, none of it gated
mkdir -p /tmp/diar-poc && cd /tmp/diar-poc
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
curl -sSLO $B/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
curl -sSLO $B/speaker-recongition-models/nemo_en_titanet_small.onnx
curl -sSLO $B/speaker-segmentation-models/0-four-speakers-zh.wav
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

cp ~-/docs/poc/*.py .          # or however you got here
uv run --python 3.12 --with sherpa-onnx --with numpy python pipeline.py
uv run --python 3.12 --with sherpa-onnx --with numpy python gil.py
```

- `pipeline.py` — does diarization work at all, how fast, and does
  `faster_whisper.audio.decode_audio` feed it? Wants `--with faster-whisper` too.
- `gil.py` — does `OfflineSpeakerDiarization.process()` release the GIL? This is
  the one that decides the architecture. A ticker thread records how late it
  wakes; if the worst gap equals the length of the pass, the GIL was held.

Expect `pipeline.py` to identify four speakers and report RTF ≈ 0.04 on a
multi-core desktop, and `gil.py` to print `verdict: GIL HELD`.
