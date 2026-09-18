# Speaker diarization — research and design

Question: *can this server tell you which person said what?*

Short answer: **yes, and without PyTorch.** `sherpa-onnx` ships the same
pyannote segmentation model everyone else uses, as ONNX, with a ~14 MB wheel and
models fetched from GitHub releases rather than gated on Hugging Face. It works,
and it is fast (~28× realtime, measured in §5).

The labour is not in the models. It is in two things: the **alignment** between
diarization turns and Whisper segments, which decides whether the output is
readable or quietly wrong; and the fact that the diarizer **holds the GIL for
the whole pass**, which makes the obvious integration — run it on a thread, or
even just run it after transcription — freeze this server's entire HTTP surface
for minutes (§5, §7.3). That second one is the finding that actually shapes the
design, and it is the reason this document recommends a child process.

This document records what was checked, what it costs, what it will get wrong,
and the integration points. **It is built** — the decisions in §8 were taken, and
§9 lists what landed. The measurements and the calibration in §5 are the parts
worth re-reading before changing anything, because two of them contradict what
the upstream documentation implies.

Current state: the README's "Notes and limits" says *"No speaker labels. Whisper
doesn't do diarization."* (README.md:495.) Every line reference below is against
`main@d6d6860`; `transcribe_server.py` is byte-identical to `5912e1b`, which is
what this branch was cut from.

---

## 1. Three different questions hide behind "who is talking"

They cost wildly different amounts and it is worth being explicit about which
one is being asked.

| Level | Question | Output | Feasible here |
| --- | --- | --- | --- |
| **Turn detection** | Is this a new speaker turn? | `[SPEAKER_TURN]` markers | No useful path (see §3) |
| **Diarization** | How many distinct voices, and when does each speak? | `SPEAKER_00`, `SPEAKER_01` — anonymous, per-job | **Yes, this is the recommendation** |
| **Recognition** | *Which* person — Alice, Bob — across recordings | A name, from an enrolled voiceprint | Not from audio alone |

Level 2 is what people usually mean by "identify each person". The labels are
anonymous and stable only within one file: `SPEAKER_00` in today's standup is
not the same human as `SPEAKER_00` in yesterday's. That is a real limitation and
it is the honest one to state up front, because "who said what" reads like it
promises names.

Level 3 (naming) needs enrollment audio per person and a matching step. It is a
different feature with a different privacy story, and it should not be smuggled
into this one. A cheap partial substitute is manual: let the operator rename
`Speaker 1 → Alice` in the UI and carry that rename into the exports. That is a
display mapping, not recognition, and §7.4 explains why it is more delicate than
it looks in this codebase.

## 2. Why not the obvious answer

The standard recipe is `WhisperX` or `pyannote.audio`. Both are correct and both
are the wrong shape for this project.

Two constraints drive it. `transcribe_server.py` must stay a **PEP 723
single-file script** (AGENTS.md), and the documented
setup path is "install uv, run it" with no virtualenv. The existing dependency
set is deliberately PyTorch-free — the README calls this out: *"faster-whisper
does **not** use PyTorch, so those two packages are the real GPU dependency."*

What `pip install pyannote.audio` resolves to, computed here with
`uv pip compile --python-version 3.12`:

| | sherpa-onnx path | + `torch`, `pyannote.audio` |
| --- | --- | --- |
| packages resolved | **28** | **122** |
| new wheels | `sherpa-onnx` + `sherpa-onnx-core` | torch, torchaudio, torchcodec, lightning, pytorch-lightning, torchmetrics, speechbrain-adjacent stack, `pyannote-*` ×5, `pyannoteai-sdk` |
| CUDA stacks | the existing cu12 one | **cu12 *and* a second cu13 stack** (`nvidia-cublas==13.1.1.3`, `nvidia-cudnn-cu13`, `nvidia-cuda-runtime==13`, nccl, cusolver, cusparse, …) |

The second CUDA stack is the part that matters. This server has an entire
hand-rolled bootstrap that locates `site-packages/nvidia/*/{bin,lib}` and
`RTLD_GLOBAL`s the cuBLAS/cuDNN 12 that CTranslate2 loads by soname at encode
time (`enable_cuda_libraries`, line 416; §"Where the libraries come from" in the
README). Adding a package tree that also installs cu13 libraries puts a second,
unrelated set of `libcublas`-ish names on disk. It probably still works — the
loader wiring keys off the `nvidia` package layout — but it converts a solved,
well-understood failure mode into an ambiguous one, and the failure surfaces
minutes into a job rather than at startup. That is the exact failure class
`--preload` was built to eliminate.

Two further practical costs:

- **Hugging Face gating.** `pyannote/speaker-diarization-community-1` and its
  predecessor are gated: you must accept conditions on the model page and set
  `HF_TOKEN`. A refusal arrives as a 403 deep inside the pipeline. WhisperX has
  a running history of exactly this confusing operators (issues #992, #1051,
  #1240). This server's zero-setup promise would acquire an account step.
- **TLS/offline.** Everything else here works on a machine that has never seen
  a Hugging Face token. Diarization shouldn't be the feature that breaks that.

The two things pyannote still wins on are worth stating plainly: its full
pipeline handles **overlapping speech** (two speakers at once) better, and it is
the reference against which everything else is measured. §6 says what we give up.

## 3. Options considered, including the ones that don't work

**tinydiarize (`-tdrz`).** whisper.cpp's turn detector. Not available: it
requires a `tdrz`-fine-tuned model (`small.en` only) and it is compiled into
whisper.cpp's decoder, not transferable to faster-whisper. It also only marks
*turns*, with no notion of which speaker, and no identity — so it does not answer
the question even where it works.

**NeMo / Sortformer.** NVIDIA's streaming diarizer, capped at 4 speakers, needs
`nemo_toolkit` + torch. Same objection as §2. ONNX exports exist in the wild but
are unofficial and incomplete.

**`diart`, `whisper-diarization`, `senko`, `speechbrain`, `resemblyzer`.** All
torch. `senko` is a thin wrapper over pyannote, so it inherits the gating.

**Roll our own: ONNX speaker embeddings + agglomerative clustering.** This is
what `sherpa-onnx` already is — segmentation model, embedding extractor, fast
clustering, plus the windowing and resegmentation around them. Building it by
hand means adding `onnxruntime` (already present, see below) plus scipy or
sklearn for the clustering, and then owning the tuning. No reason.

**Cloud APIs** (Deepgram, AssemblyAI, pyannoteAI precision-2). Best accuracy,
zero local compute, and they ship the audio to a third party — which is
incompatible with the posture this server is built around (no TLS, LAN-only
default, `0700` upload dirs, "meeting audio doesn't sit on disk"). Not a fit.

**`sherpa-onnx`.** Winner. Details next.

## 4. The recommendation: sherpa-onnx

### 4.1 What it is

A C++/ONNX runtime inference framework. For diarization it composes three pieces
that are each independently replaceable:

```
pyannote segmentation-3.0 (ONNX)   →  speech activity per frame, speaker-agnostic
speaker embedding extractor (ONNX) →  a voiceprint per speech region
fast clustering (built in)         →  collapse voiceprints into SPEAKER_00..N
```

The middle line is the part that is easy to get wrong when reading about this:
the segmentation model answers *"is someone talking, and how many overlapping
voices"*, never *"who"*. Its speaker slots are **local to one window** and have
no relationship to the slots in the next window — it has no memory and no
identity. Identities come only from the third line, which is why the clustering
threshold (or the speaker count you supply) is what decides how many people the
transcript ends up with.

Measured geometry, from the model's own metadata:

| | value |
| --- | --- |
| window | `window_size=160000` samples = **10 s** |
| hop | `window_shift_ratio=0.1` = **1 s** |
| output frame | `receptive_field_shift=270` samples ≈ **17 ms** |
| local speaker slots | `num_speakers=3` |
| classes | `num_classes=7` = silence + 3 singles + 3 *pairs* |

The 7 classes are the detail worth noticing: the model is trained on a
**powerset**, so one frame can say "slots 1 and 3 are both active" instead of
forcing an either/or. That is how simultaneous speech is represented at all.

The Python surface is:

```python
config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=".../model.int8.onnx", window_shift_ratio=0.1
        )
    ),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=".../titanet.onnx"),
    # num_clusters=-1 means "decide from the audio", which is what makes the
    # threshold matter. 0.8 and not the 0.5 the upstream examples show: see the
    # calibration in section 5.
    clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.8),
    min_duration_on=0.3,
    min_duration_off=0.5,
)
result = sherpa_onnx.OfflineSpeakerDiarization(config).process(audio_f32_16k)
for r in result.sort_by_start_time():
    ...  # r.start, r.end, r.speaker
```

`num_clusters=-1` means detect; a positive value pins the count when the
operator knows it (a two-person interview). `threshold` is the only real tuning
knob: smaller → more speakers. `process()` takes an optional progress callback,
which maps directly onto this server's existing progress reporting.

### 4.2 Why it fits *this* codebase

**The wheel is small and has no transitive dependencies.** `sherpa-onnx` 1.13.8
requires exactly `sherpa-onnx-core==1.13.8`, whose metadata declares no
dependencies at all — ONNX Runtime is linked statically inside.

| wheel | linux x86_64 | win_amd64 | macOS arm64 |
| --- | --- | --- | --- |
| `sherpa-onnx` | 4.4 MB | 2.3 MB | 2.1 MB |
| `sherpa-onnx-core` | 10.6 MB | 16.9 MB | 9.6 MB |
| **added** | **~15 MB** | **~19 MB** | **~12 MB** |

For scale: the CUDA wheels this project already downloads are ~1.5 GB
(`nvidia-cublas-cu12` alone is 581 MB, `nvidia-cudnn-cu12` 770 MB). The
marginal cost is about **1%** of an existing first run. CPU wheels exist for
linux x64/aarch64/armv7l, macOS x64/arm64 and Windows x64/x86, Python 3.7+
through cp314 — the same platform spread the project already claims.

**`onnxruntime` is already in the dependency tree.** `faster-whisper==1.2.1`
depends on it (its Silero VAD), so nothing new appears there either.

**The decode step already exists.** sherpa wants float32 mono at 16 kHz.
`faster_whisper.audio.decode_audio(path, sampling_rate=16000)` returns exactly
that, and it uses PyAV, which *bundles* FFmpeg — no `soundfile`, no `librosa`, no
system ffmpeg binary. The example script in the sherpa repo uses `soundfile` +
`librosa`; we do not need either.

**No gating, no account.** Models come from
`github.com/k2-fsa/sherpa-onnx/releases`. Plain HTTPS, no token, no license
click-through, and a `.tar.bz2` that stdlib `tarfile` unpacks.

**Model weights are small.** Never mind next to `large-v3`:

| model | size | license |
| --- | --- | --- |
| `sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | 5.7 MB (int8: 1.5 MB) | MIT (converted from `pyannote/segmentation-3.0`) |
| …`sherpa-onnx-reverb-diarization-v1/model.onnx` | 9.1 MB (int8: 2.3 MB) | see LICENSE in the tarball (converted from `Revai/reverb-diarization-v1`) |
| `nemo_en_titanet_small.onnx` | 40.3 MB | NVIDIA NeMo — check before redistribution |
| `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` | ~40 MB | Apache-2.0 |

**Precedent.** whisper.cpp's server has exactly this feature behind `--diarize`,
using these same sherpa-onnx models. A faster-whisper server doing the same is
not exotic.

### 4.3 CUDA: don't

sherpa-onnx publishes CUDA wheels — 190 MB (CUDA 11.8) and 246 MB
(CUDA 12.8+cudnn9), Linux and Windows x64 only, from a Hugging-Face-hosted wheel
index that needs `--no-index -f <url>`. That re-opens the entire CUDA bootstrap
problem for a model small enough to run on the CPU, and the CPU cost (§5) is
already in the same order as transcription itself.

**CPU is the right default.** Note the pleasing property: diarization is
CPU-bound and transcription is GPU-bound, so if the two ever run concurrently
they use different resources — see §7.3.

## 5. Measured on this machine

Not a benchmark — a smoke test to establish that the pipeline works, what it
costs to add, and whether it can run where the design wants to put it. Run on
this workstation: RTX 3060, 20 cores, 15 GB RAM, Python 3.12,
`sherpa-onnx 1.13.8`, against the four-speaker reference clip vendored with the
segmentation model (`0-four-speakers-zh.wav`, 16 kHz, 56.9 s). The scripts behind
these numbers are in `docs/poc/`, so they can be re-run rather than believed.

### Install cost

```
$ uv run --with sherpa-onnx --with numpy python poc.py
Downloading sherpa-onnx (4.2MiB)
Downloading sherpa-onnx-core (10.1MiB)
```

**Two wheels, 14.3 MB, no other package pulled in.** That confirms the metadata
reading in §4.2: the only transitive requirement is `sherpa-onnx-core`, and ONNX
Runtime is linked statically. Nothing in the existing resolution moved.

Model files: segmentation 1.5 MB (int8) / 5.7 MB (fp32), embedding 40.3 MB →
**~42 MB**, less than a `base` Whisper model and 1.4% of `large-v3`.

### Output

Four speakers, ten turns, and it reproduces the vendored example's own published
output to within ~20 ms on the turn ends — so this is the expected behaviour of
the model, not a lucky guess:

```
  0.638 --   6.848  speaker_00
  7.017 --  10.679  speaker_01
 11.472 --  13.548  speaker_01
 13.784 --  16.990  speaker_02
 22.154 --  24.837  speaker_00
 27.655 --  29.461  speaker_03
 30.018 --  31.503  speaker_03
 33.680 --  37.915  speaker_03
 48.040 --  50.487  speaker_02
 52.546 --  54.605  speaker_00
```

### Speed

**RTF 0.036** — 2.03 s to diarize 56.9 s of audio (~28× realtime) with
`num_threads=4`. For comparison the sherpa docs report 0.11 RTF (~9× realtime)
for the same model pair single-threaded on a Mac; §6's caveats about that gap
being about the machine, not the model.

Extrapolated, and this is the number the placement decision turns on:

| recorded length | diarization | next to `base` (~10× realtime) | next to `large-v3` (~1×) |
| --- | --- | --- | --- |
| 1 hour | ~2.2 min | +36% wall clock | +3.7% |
| 3 hours | ~6.5 min | +36% | +3.7% |

So the *compute* is cheap, and the serial-after placement (§7.3) is affordable.
The problem with placing it is not its cost.

### The decode handoff works

`faster_whisper.audio.decode_audio(path, sampling_rate=16000)` produced a
`float32 (909771,)` array, and feeding that to the diarizer gave the same ten
turns in 2.11 s. **No `soundfile`, no `librosa`, no system ffmpeg** — the whole
decode side is already in the dependency tree, exactly as §4.2 predicted.

### The auto-detect threshold had to be calibrated, and the documented value was wrong

`FastClusteringConfig` takes either a speaker count or a threshold. Sherpa's own
examples pass **0.5**, and I copied that. It is wrong, and the reason it looks
fine in their docs is that they always pass `num_clusters` alongside it, which
makes the threshold inert.

Run the auto path (threshold only, no pinned count) against the four recordings
in sherpa's own CI, whose speaker counts are known:

| file | true speakers | 0.4 | 0.5 | 0.6 | 0.7 | **0.8** | 0.9 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0-four-speakers-zh.wav` | 4 | 8 | 7 | 7 | 4 | **4** | 4 | 3 |
| `1-two-speakers-en.wav` | 2 | 2 | 2 | 2 | 2 | **2** | 2 | 2 |
| `2-two-speakers-en.wav` | 2 | 5 | 3 | 3 | 2 | **2** | 2 | 2 |
| `3-two-speakers-en.wav` | 2 | 5 | 4 | 3 | 3 | **2** | 2 | 2 |
| **exact** | | 1/4 | **1/4** | 1/4 | 3/4 | **4/4** | 4/4 | 3/4 |

At 0.5 — the value I had written down — **three of the four files over-segment**,
and a two-person call is reported as four speakers. That is the worst kind of
wrong: plausible, specific, and invisible unless you already know the answer.
0.8 sits at the safe end of the [0.8, 0.9] plateau that is right on all four;
1.0 starts merging the four-speaker file down to three.

This is the strongest argument in the document for the UI's advice to **pin the
speaker count when you know it**. Pinned, all four files come out exact. The
auto path is a guess about a clustering distance, and four files is not a
sample — treat 0.8 as a better default than 0.5 rather than as a solved problem.

### The child-process path works with the real models

Everything above was measured with the diarizer called in-process. The shipped
path is a child process, so it was checked too: with real weights and no
stubbing, `ensure_diarize_models()` → `run_diarizer()` → `align_speakers()` →
`render()` runs clean, and the child reproduces the pinned four-speaker result
exactly (10 turns, same boundaries) in **3.0 s**. `probe_diarization()`, which is
what `--preload` runs, completes in **0.43 s** and cleans up after itself.

The script is `docs/poc/e2e_check.py`. It is also the check that caught the
threshold problem: every stub in `tests/test_diarization.py` passes with 0.5,
because the stubs assert the protocol, not the answers.

### The GIL is held for the entire pass — and this changes the design

A ticker thread waking every 10 ms, measuring how late it ran:

| | ticks in ~20 s | worst gap |
| --- | --- | --- |
| baseline (main thread sleeping) | 2000 | **0.79 ms** |
| main thread diarizing | 21 | **2320 ms** |

The ticker ran 21 times — about 20 of them in the 0.2 s before the pass started
— and then recorded a single gap equal to the *entire* `process()` call. It was
starved, which means no other Python thread ran either.

`OfflineSpeakerDiarization.process()` does not release the GIL, on 1.13.8, with
no progress callback. A search for an upstream fix turned up nothing applicable
(the GIL-release work in sherpa-onnx targets the streaming/websocket APIs). This
is a direct measurement on the version we would ship, not an inference.

Two consequences, and the second one is the surprise:

1. **The concurrent placement is dead.** A background diarization thread would
   hold the GIL and the event loop would stop. Not "slower" — stopped.
2. **The serial placement has the same problem.** `run_job` runs on the worker
   thread, but the GIL is process-wide, so a serial pass freezes the HTTP
   surface for its whole duration too. For an hour of audio that is **~2.2
   minutes of unresponsive server** on this machine, and several times that on
   the single-threaded hardware the RTF figures above suggest.

This is a *new* failure mode for this server, not an existing one. Today a job
never blocks the loop: CTranslate2 releases the GIL while it encodes, which is
precisely why live segment streaming, the 1.2 s status poll, the progress meter
and Cancel all work during a long transcription. Adding diarization naively
would take all four away for the duration of the second pass, and the symptom —
UI frozen, cancel doing nothing — is indistinguishable from the self-deadlock
this repo already carries a regression test for.

§7.3 is rewritten around this.

## 6. What it will get wrong

Setting expectations honestly, because the output *looks* authoritative and that
is the risk.

- **Cross-talk.** When two people talk over each other, the pipeline assigns the
  frame to one of them. pyannote's full pipeline detects overlap; this
  configuration effectively does not. In a meeting with frequent interruptions,
  expect the interrupter's interjection to be attributed to the person holding
  the floor.
- **Similar voices.** Two similar-sounding speakers, or one speaker on two
  different microphones (far end vs near end in a video call), can collapse into
  one cluster or split into two. Video-call audio is the hard case: each remote
  participant has been through a different codec and mixer.
- **Speaker count.** Auto-detection is a clustering threshold, not a classifier,
  and it is the weakest part of this (§5). At the calibrated 0.8 it was right on
  all four recordings sherpa publishes, but that is four recordings. Pinning the
  count when the operator knows it is much more reliable than tuning the
  threshold, which is why the UI offers the number and not the threshold.
- **Short turns.** `min_duration_on=0.3` / `min_duration_off=0.5` exist to stop
  a cough becoming a speaker. Back-channels ("mm-hm", "right") get absorbed into
  the neighbouring turn.
- **Alignment, not diarization, is usually the visible error.** A diarizer that
  is 95% right still produces a transcript where a sentence is split under the
  wrong name if the Whisper segment boundary and the speaker boundary disagree.
  §7.2 is the whole ballgame.
- **The number that matters.** Diarization Error Rate, where ~10% is
  respectable on conversational audio and ~20% is noticeable in a transcript.
  Publications quote DER for their own pipelines; treat any specific figure for
  this configuration as unverified until we measure on our own recordings. The
  four-speaker sample in §5 is a smoke test, not a benchmark.

Practical mitigation, and it is cheap: **labels are a first-class field in the
JSON export**, so a downstream script or a human can fix names without touching
audio or re-running anything. And the numeric labels mean a wrong split is
visible as `Speaker 2` appearing for one line rather than being silently
confident.

## 7. Integration design

### 7.1 Options, audit, and the API

`build_opts` (line 1187) gains three knobs, threaded through `create_job` (1819)
and `retry_job` (1902) as form fields, exactly like `word_timestamps`:

| option | form field | meaning |
| --- | --- | --- |
| `diarize` | `diarize=true` | run the second pass |
| `speakers` | `speakers=0` | 0 = detect, else pin the count (1–10) |
| `speaker_threshold` | `speaker_threshold=0.5` | only when `speakers=0`; advanced, probably hidden |

These are machine-independent (unlike precision), so they belong with quality,
not behind `--allow-precision-choice`. A server flag `--no-diarize` to remove the
control entirely is worth having for a deployment that must never load the
models — mirroring how `--pin-model` disables a selector.

Because `public_opts` (1124) is a filter over `opts`, these three pass through
automatically and appear in the JSON export. **Nothing here is prompt text**, so
the two-token split is untouched — see §7.4 for the one knob that would change
that.

Audit: emit `job.diarized` with `speakers_found`, `num_clusters_requested`,
`threshold`, `backend`, `model`, and `elapsed`. No transcript text, no speaker
names, consistent with `audit_opts` (986).

### 7.2 Alignment: the part that decides whether this is good

Whisper segments and diarization turns do not share boundaries. A 30-second
Whisper segment can contain four speaker turns; a 0.4-second turn can sit
entirely inside one segment.

**With `word_timestamps` on** — the good path:

1. For each word, find the diarization turn with maximum temporal overlap.
2. Group consecutive words sharing a speaker into sub-segments.
3. Re-split the segment's `text` at those boundaries. faster-whisper's word
   tokens carry their leading space, so joining a run of words reproduces the
   original text with the original spacing.

This gives sub-segment boundaries that land on actual speaker changes, which is
what makes a transcript readable. It is the same trick WhisperX uses, minus the
forced alignment (Whisper's own word timings are good enough to bucket words
into turns; they are not good enough to place a boundary to the millisecond, and
they don't need to be).

**Without word timestamps** — the degraded path: assign each segment the speaker
with the greatest overlap, and if a turn boundary falls inside the segment and
both sides exceed some minimum, split the segment at the boundary and split the
text proportionally by character count. Crude, and it will put the boundary in
the wrong place inside a sentence.

**Therefore: requesting diarization should turn on `word_timestamps`.** Not
silently — the UI should show the box ticked and the hint refreshed, because the
cost is real. The existing hint at line ~2428 already says *"Word timings cost
time, and help if you later run diarization"* — that sentence becomes a control.

A consequence worth noting: this makes the quality of the whole feature depend on
a checkbox that is currently off by default and framed as optional.

### 7.3 When the pass runs — and why it can't be a thread

The measurement in §5 dominates this decision: `process()` holds the GIL, so
**any** placement that calls it from a Python thread freezes the event loop for
the duration of the pass.

**(a) After transcription.** `run_job` (1316) finishes the segment loop, then
diarizes, then merges. Straight-line change to one function, no effect on the
worker loop's invariants — but it freezes the HTTP surface for ~2.2 min per hour
of audio (§5), during which the UI's poll hangs, Cancel is queued rather than
honoured, and `/api/status` on *every other job* stops answering.

**(b) Before transcription.** Same freeze, moved earlier. Worse: the operator
watches a dead page instead of a growing transcript.

**(c) Concurrently, on a thread.** Ruled out by measurement. Not a trade-off.

So placement is not really the question — **isolation is.** Three ways out:

**1. Run it in a child process (recommended).** The GIL is per-interpreter, so a
subprocess removes the problem completely rather than shrinking it. Shape: pass
the audio path plus the clustering config, read back JSON turns, and `.terminate()`
the child on cancel — which also makes diarization the first thing in this server
that is genuinely interruptible. Costs: the models load in the child (~1 s for
these two), and a child whose parent dies needs cleaning up.

This runs into the single-file constraint, gently. `multiprocessing` with the
`spawn` start method would re-import `transcribe_server.py` in the child — which
builds the whole FastAPI app as a side effect, and depends on `__main__`
re-import semantics that differ on Windows and under PyInstaller. The tidier
route is `subprocess` with `sys.executable -c <source>`, where `source` is a
small worker embedded as a string constant in `transcribe_server.py`. It keeps
the file self-contained, and under `uv run` `sys.executable` is already the
interpreter that has `sherpa_onnx` installed.

**2. Do the pipeline by hand, in chunks.** `sherpa-onnx` exposes the stages
separately — segmentation, `SpeakerEmbeddingExtractor`, `FastClustering` — and
the release tarball ships `speaker-diarization-onnx.py`, a worked example of the
whole thing. Processing in chunks gives small GIL windows and a natural progress
signal. The cost is honest: that example is **~300 lines of numpy** doing window
striding, powerset mapping, per-chunk embedding extraction and resegmentation,
all of which has to be exactly right, and all of which `process()` already does.
In exchange, cross-chunk speaker identity becomes *our* problem, since clustering
would otherwise happen per chunk and `SPEAKER_00` in minute 5 would not be
`SPEAKER_00` in minute 20. I would not take this on to avoid a subprocess.

**3. Accept it and document it.** Cheapest, and wrong: a tool that streams a live
transcript and offers Cancel cannot then go silent for minutes with no
indication. If this route is ever chosen, the UI needs an explicit "server busy
diarizing" state, which is more work than the subprocess.

Whichever is chosen, the progress reporting has to route through the existing
`patch_job(... message=...)` mechanism ("Diarizing 40%"), and the ETA logic has
to know the job has a second phase — otherwise `progress` sits at 100% while
nothing appears to be happening.

Note what isolation buys beyond the freeze: it is also the only variant where
Cancel actually interrupts a pass in flight, which matters most for exactly the
long recordings where the freeze is worst.

### 7.4 Speaker names and the trust boundary

The tempting next feature is `SPEAKER_00 → "Alice"`. Do not ship it in v1, and
not for effort reasons.

Prompt and hotword text is the *one* field the app token is gated out of
(`public_opts` docstring; AGENTS.md calls it the invariant whose regression
"silently voids the two-token split"). The README's own reasoning for that is:
*"Prompts and hotwords routinely contain real names."* A speaker-name mapping is
the same category of data — a list of who was in the room — and it would need
the same treatment: out of `public_opts`, a length and a hash in the main log,
the text in `prompts/<job_id>.json`, readable only with the audit token. Which
is a lot of machinery for a rename.

v1 therefore ships **numeric labels only**. The UI can offer a local rename that
lives in `sessionStorage` and is applied client-side to the rendered text and to
export requests — nothing that contains a name ever reaches the server or the
audit trail. If a name mapping is ever wanted server-side, it goes through
`store_prompt_sidecar` (996) and gets an audit-token route, like prompts do.

There is a second, smaller leak to keep in mind. Diarization output is derived
from the *voice*, so a job's JSON export now contains a speaker count. That is
not new information of the same kind as transcript text, but it is worth not
logging the count in the audit trail when the operator asked for it to be
sensitive... which it isn't, and the existing trail already logs segment counts.
Fine as designed; noted so it is a decision rather than an oversight.

### 7.5 Rendering and exports

`render` (1493) is the easy half:

- `txt` — `Speaker 1: text` per line when the job is diarized, unchanged
  otherwise. Arguably this should be a separate format so plain copy-paste is
  unaffected; that is a judgement call for the operator.
- `timestamped` — `[00:01:23] Speaker 1: text`.
- `srt` — prefix the text. SRT has no voice construct.
- `vtt` — use the native voice span, `<v Speaker 1>text`, which players
  understand.
- `json` — `speaker` on every segment, plus a top-level `speakers` array with
  total speaking time per speaker. This is the machine-readable contract worth
  getting right, because it is what a downstream fix-up script consumes.

`jobTags` (2662) gets a `"N speakers"` bit.

### 7.6 Models: fetch, cache, and prove

Models should download lazily, on the first diarized job, and cache under the
work dir (`<work dir>/diarize-models/`) — not into the Hugging Face cache, since
they don't come from Hugging Face. Download with `urllib.request`, unpack with
`tarfile`; both stdlib, no new dependency, and this is the only place the
project would fetch something itself rather than letting a library do it.

Three things this project's style demands:

- **Verify a checksum.** A truncated `.onnx` produces a confusing failure deep
  inside ONNX Runtime, exactly the shape of the cuBLAS bug this repo already
  fixed once. Pin SHA-256 per file.
- **Make `--preload` prove it**, per the AGENTS.md invariant ("`--preload` must
  *prove* the device can encode"). If diarization is enabled, `--preload` builds
  a diarization session and runs `process()` on a second of silence, so a broken
  download is reported at startup instead of on the first job. It **warns and
  carries on** rather than exiting: transcription is unaffected, and the failure
  this is most likely to hit in practice — a machine that cannot reach github.com
  for the 42 MB — would otherwise take a working server down with it. That also
  keeps `--preload`'s exit code meaning one thing: "a job will genuinely run". A
  job does run; it just has no labels. See §7.3 for why that is the same choice
  `run_job` makes.
- **Fail loudly and early if the download can't happen.** An on-LAN machine with
  no internet should get "diarization needs a one-time 46 MB download; run X, or
  uncheck the box" — not a traceback from `urllib`.

Note the AGENTS.md constraint interacts with the dependency choice. There are two
ways to get `sherpa-onnx` in:

1. **Add it to the inline `# /// script` block.** +15–19 MB on every first run,
   even for operators who never diarize; the feature always works; the
   `./transcribe_server.py` shebang path keeps working.
2. **Keep it optional, documented as `uv run --with sherpa-onnx transcribe_server.py`.**
   Zero cost for people who don't want it, but the directly-executable path
   silently loses the feature, and the server must feature-detect with
   `importlib.util.find_spec` and hide the checkbox. The README already uses
   `uv run --with torch` for the cosmetic GPU-name feature, so there is
   precedent for both.

**The packaging work decides this, and it decides it for (1).** `packaging/` and
`launcher/` (commit `d6d6860`, landed while this document was being written) ship
a tray launcher, not a frozen server, and it starts the server as
`uv run --no-project transcribe_server.py --port N`. No `--with`. So under option
(2) **diarization would be unreachable in the shipped Windows app** — the model
would download, the CUDA runtime would resolve, and the checkbox would simply
never appear. Making it work would mean teaching the launcher a second command
shape, and AGENTS.md is explicit that this is not to happen: *"The launcher never
parses console output… the server needs no launcher-aware code — no state file,
no protocol. Keep it that way."*

Which leaves (1), and the arithmetic is not close: the packaged first run
resolves **~2.2 GB**, almost all of it CUDA (`nvidia-cudnn-cu12` alone is most of
the 2.2 GB). sherpa-onnx adds **15–19 MB, roughly 0.7%**. The packaging README's
own reasoning for not freezing — that re-shipping 2.2 GB per update is the thing
to avoid — applies with more force to a rounding error than to a third of a
percent.

So: inline dependency. The `--with` route stays viable for a source checkout and
dead for the installer, and a feature that exists only in a source checkout is a
feature with a support burden attached.

### 7.7 Interference with `feat/transcript-preview`

The uncommitted work in the sibling worktree rewrites card rendering to be
**append-only**: `appendSegments` appends only `segments[view.shown..]` and only
re-renders from scratch when `segments.length < view.shown`. It renders one
`.seg` row per segment with a `.ts` gutter and a `.tx` body.

This collides with placement §7.3(a), where labels arrive after the segments are
already on screen: appended rows would never gain a speaker label. Any design
that labels segments after streaming needs either a revision counter on the job
that tells the client to re-render, or a mutate path in `appendSegments` that
updates existing rows. The speaker column also wants to live *between* `.ts` and
`.tx`, which is a small edit to that markup.

Worth deciding the merge order deliberately: landing diarization on `main` first
and the preview branch second (or vice versa) will produce a conflict in
`render`/`appendSegments` either way, and the resolution is easier if the
diarization side knows about the append-only structure.

## 8. Decision points, and how they were settled

1. **Ship at all?** Yes. The README's "No speaker labels" note is now an honest
   description of anonymous, per-recording labels, including what it gets wrong.
2. **Dependency placement** — **inline** (§7.6). `--with` would have made the
   feature unreachable in the packaged tray app, and 15 MB against a 2.2 GB
   first run is not a real cost.
3. **Isolation** — **child process** (§7.3). Not a preference: the GIL
   measurement leaves no thread-based option, and the naive serial version
   freezes the whole server.
4. **`word_timestamps` coupling** — **forced on** when diarizing, with the box
   ticked and locked in the UI so the cost is visible rather than silent.
5. **`txt` export** — labels are prefixed when the job has them. A job without
   them exports byte-identically to before, which is pinned by a test.
6. **Numeric labels only** in v1. A server-side name mapping would need the
   audit-token plumbing; renaming stays client-side.
7. **Segmentation model** — `pyannote-segmentation-3-0`, int8 (MIT), and the
   threshold was **calibrated rather than copied**: 0.8, not the 0.5 the
   upstream examples suggest (§5).

One thing was deliberately narrowed: the plan here originally included an
operator-tunable clustering threshold. It is not exposed. Pinning the speaker
count is both easier to explain and measurably better, so the threshold is a
single calibrated constant with the measurement next to it.

## 9. What was built

- `transcribe_server.py` — inline `sherpa-onnx` dependency; `DIARIZE_MODELS`
  with pinned SHA-256s; the embedded `DIARIZE_WORKER` child; `run_diarizer` with
  a stall guard, a cancel path and Windows `CREATE_NO_WINDOW`; `align_speakers`;
  `render` output for txt/timestamped/srt/vtt/json; `--no-diarize`; the
  `--preload` probe, which reports diarization trouble without refusing to
  start; `job.diarized` / `job.diarize_failed` audit events.
- `tests/test_diarization.py` — the GIL-isolation regression test, the child
  protocol, an alignment case per failure mode, the export shapes, a regression
  for the non-monotonic-cursor mis-tag (§9.1), and an opt-in test against the
  real model.
- `tests/test_ui_preview.py` — the append-only preview now redraws when labels
  arrive, with the gutter present but empty beforehand so rows do not shift.
- `docs/poc/` — the measurements, `e2e_check.py` for the real path, and
  `web_upload_check.py` for the whole application: it starts a server, checks the
  page ticks the box, uploads, and asserts the job comes back tagged.
- `README.md`, `AGENTS.md` — the operator view and the three new invariants.

Two things were settled after the first merge, both from re-reading the code
rather than from a failing test:

### 9.1 The alignment cursor assumed time only moves forward

`speaker_for` walks `segments` and `turns` together with one cursor that never
goers back. A word whose start is *earlier* than a previously seen one therefore
left the cursor stranded past the turn it belonged to, and the word was silently
attributed to a neighbour. Jumping back exactly one turn happened to survive,
because the nearest-turn fallback computes a negative gap and picks the earlier
side; two turns back it picks the wrong one.

Measured exposure before fixing it: **108 real words from faster-whisper, zero
inversions**, and the largest backwards gap between consecutive words was
**0.00 s**. So it was unreachable — but it failed by mis-tagging rather than by
raising, in the one function whose entire job is to get the tags right, so the
fix is cheap: remember the previous word's start and reset the cursor when time
goes backwards. The reset costs one scan of the turns and only fires on input
that does not occur.

### 9.2 The web page now asks for labels, the API still does not

A dropped file should come back labelled, so the *Identify speakers* box ships
ticked. The API keeps `diarize=false` as its default, so a script has to say
what it wants. The consequence to be aware of is that word-level timings are
force-enabled by that box (see §7.2), so a web upload is slower than it used to
be: roughly +36% against `base`, +3.7% against `large-v3`.

Ticking the box in the markup created one interaction worth recording, because
it is the kind of thing a changed default hides: the script that couples the two
boxes ran at page load, before `/api/status` had said whether the server offers
diarization at all. On a `--no-diarize` server the control is hidden — so the
page would have sat there with **word timings ticked and disabled for a feature
that does not exist**, quietly changing what the export contains. The coupling
now consults `DIARIZE_OK` and is applied only once status has answered.

The tagging itself was never at risk from that: `build_opts` forces word timings
server-side whenever diarization is on, so the client-side lock is an affordance
rather than the guarantee.

Not built, and deliberately: server-side speaker names (§7.4), a GPU
`onnxruntime` path (§4.3), and crossing-diarization identities.

## 10. Sources

- sherpa-onnx diarization docs — <https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html>
- Pre-trained models, published RTF figures, model sizes —
  <https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html>
- The four sample recordings with known speaker counts, and the CI job that uses
  them — <https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/speaker-diarization.yaml>
- Python API example — <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py>
- Install / wheel platforms — <https://k2-fsa.github.io/sherpa/onnx/python/install.html>
- `faster_whisper/audio.py` (`decode_audio`, PyAV-bundled FFmpeg) —
  <https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/audio.py>
- pyannote gating on HF —
  <https://huggingface.co/pyannote/speaker-diarization-community-1>
- WhisperX's recurring pyannote-token confusion — <https://github.com/m-bain/whisperX/issues/992>
- tinydiarize, turns not identities — <https://github.com/akashmjn/tinyDiarize>
- whisper.cpp server `--diarize` precedent —
  <https://github.com/ggml-org/whisper.cpp/blob/master/examples/server/README.md>
