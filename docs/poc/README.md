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
# the four recordings sherpa's own CI uses, with known speaker counts 4, 2, 2, 2
for w in 0-four-speakers-zh.wav 1-two-speakers-en.wav \
         2-two-speakers-en.wav 3-two-speakers-en.wav; do
  curl -sSLO $B/speaker-segmentation-models/$w
done
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

cp ~-/docs/poc/*.py .          # or however you got here
uv run --python 3.12 --with sherpa-onnx --with numpy --with faster-whisper \
  python pipeline.py
uv run --python 3.12 --with sherpa-onnx --with numpy python gil.py
uv run --python 3.12 --with sherpa-onnx --with numpy python calibrate.py
```

- `pipeline.py` — does diarization work at all, how fast, and does
  `faster_whisper.audio.decode_audio` feed it? Wants `--with faster-whisper` too.
- `gil.py` — does `OfflineSpeakerDiarization.process()` release the GIL? This is
  the one that decides the architecture. A ticker thread records how late it
  wakes; if the worst gap equals the length of the pass, the GIL was held.
- `calibrate.py` — sweeps the auto-detect threshold across the four files above
  and prints how many speakers each setting finds against the true count. This
  is what moved the shipped default from 0.5 to 0.8.
- `e2e_check.py` — the real shipped path (child process, real weights,
  alignment, exports, the `--preload` probe) against `0-four-speakers-zh.wav`.
  Run it from the repo root with the worktree venv:

  ```bash
  .venv/bin/python docs/poc/e2e_check.py .
  ```

Expected: `pipeline.py` identifies four speakers at RTF ≈ 0.04 on a multi-core
desktop; `gil.py` prints `verdict: GIL HELD`; `calibrate.py` scores 4/4 at 0.8
and 1/4 at 0.5; `e2e_check.py` ends with ALL END-TO-END CHECKS PASSED.
