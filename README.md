# Transcription server

Drag-and-drop Whisper transcription served over your LAN.

## Setup on the Windows machine

Dependencies are declared inline in the script (PEP 723), so there's no
requirements file, no venv to create, and nothing to install by hand.

```powershell
winget install astral-sh.uv
```

That's it. The first `uv run` resolves and caches everything, including the CUDA
runtime libs CTranslate2 needs (`nvidia-cublas-cu12` and `nvidia-cudnn-cu12`) —
faster-whisper does **not** use PyTorch, so those two packages are the real
GPU dependency.

Optionally, installing PyTorch gives the status strip a proper GPU name
("NVIDIA GeForce RTX 3080") instead of a device count. Not required:

```powershell
uv run --with torch transcribe_server.py
```

ffmpeg is required for decoding anything that isn't plain WAV:

```powershell
winget install Gyan.FFmpeg
```

Reopen the terminal afterwards so PATH picks it up.

## Run it

```powershell
uv run transcribe_server.py --preload
```

Access control is on by default. If you don't pass `--token`, one is generated
for the run and printed with the URL:

```
Access token: 8Kf2pQ...
Open:  http://<this-machine-ip>:8765/?token=8Kf2pQ...
```

Open that link once and the token moves into `sessionStorage` and out of the URL
immediately. Paste it into the unlock prompt if you land on the page without it.
To pin a stable token across restarts, prefer the environment variable over the
flag (flags are visible in the process list):

```powershell
$env:TRANSCRIBE_TOKEN = "something-long"
uv run transcribe_server.py --preload
```

`--no-auth` disables the token entirely. Only on a network you fully control.

On macOS or Linux you can also mark it executable and run it directly, since the
shebang hands off to uv:

```bash
chmod +x transcribe_server.py
./transcribe_server.py --device cpu
```

Useful flags:

| Flag                          | Effect                                                                                                                      |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `--model medium`              | default model in the dropdown (`large-v3`, `large-v3-turbo`, `medium`, `small`, `base`)                                     |
| `--port 8765`                 | listen port                                                                                                                 |
| `--preload`                   | load the model at startup so the first job doesn't stall                                                                    |
| `--token secret123`           | require `?token=secret123` on every API call                                                                                |
| `--device cpu`                | fall back to CPU                                                                                                            |
| `--compute-type int8_float16` | lower VRAM use                                                                                                              |
| `--quality fast`              | default beam size: `fast`=1, `balanced`=5, `thorough`=8                                                                     |
| `--model-cache 2`             | models held in VRAM at once (default 1)                                                                                     |
| `--pin-model`                 | force every job to `--model`, disable the UI selector                                                                       |
| `--allow-precision-choice`    | expose the precision selector (hidden by default)                                                                           |
| `--no-auth`                   | serve without a token                                                                                                       |
| `--allow-host name`           | accept an extra `Host` header value (repeatable)                                                                            |
| `--max-upload-mb 2048`        | per-file upload ceiling                                                                                                     |
| `--max-queue 20`              | max jobs pending before uploads are refused                                                                                 |
| `--max-jobs 60`               | finished job records retained before eviction                                                                               |
| `--source-retention job`      | `run` deletes audio after transcribing (no retry), `job` keeps it while the record lives (default), `forever` never deletes |

`--help` groups these under network/access, model/hardware, and limits/storage.

First run with a given model downloads it from Hugging Face (a few GB for
`large-v3`) into the HuggingFace cache. After that it's local.

## Open it from the Mac

Find the Windows machine's LAN IP:

```powershell
ipconfig | findstr IPv4
```

Then browse to `http://192.168.x.x:8765`.

Windows Firewall will block the port on first run. Either accept the prompt
Windows shows, or add the rule yourself in an admin PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Transcription server" -Direction Inbound `
  -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private
```

Keep it on the `Private` profile so it isn't exposed on untrusted networks.

## Using it

Drop one or more files on the intake panel. Whatever the controls say when you
drop applies to that job.

**Main controls**

- **Model** — `large-v3` for best accuracy, `large-v3-turbo` for roughly 4× the
  speed at a small accuracy cost, `medium` if VRAM is tight.
- **Language** — leave on _Detect_ unless the audio is short or noisy, where
  pinning it to English avoids misdetection.
- **Quality** — beam size. `fast` is a genuine speed lever when you're iterating;
  `thorough` buys a little accuracy for noticeably more time.
- **Skip silence** — VAD filtering. Speeds up meetings with long dead air and
  reduces hallucinated text during silence. Worth leaving on.

**Advanced**

- **Names and terms to expect** — the highest-value field here. Whisper mangles
  product names, acronyms and people it has no context for; a comma-separated
  list fixes most of it. `hotwords` under the hood.
- **Context prompt** — a sentence describing the recording. Biases style and
  vocabulary more broadly than the term list. `initial_prompt` under the hood.
- **Translate to English** — switches the task from transcribe to translate.
- **Word-level timings** — costs time, but makes a later diarization pass
  (WhisperX, pyannote) align much better. Included in the `.json` export.
- **Carry context forward** — off by default, and that's deliberate. Whisper's
  `condition_on_previous_text` is the usual cause of repetition loops on long
  recordings: one bad segment poisons the context and it repeats a phrase for
  minutes. Turn it on only if you want smoother continuity and are watching for
  that failure.
- **Min silence / speech padding** — VAD tuning. Occasionally useful, mostly
  leave alone.

**Precision** is hidden by default. It's a property of the machine, not of a
recording, so it's pinned to `--compute-type` unless you pass
`--allow-precision-choice`.

**Retry** appears on any finished job whose source audio is still on disk. It
re-runs the same file with whatever the controls say _now_ — so you can fix a bad
transcript by adding the term list and retrying, without re-uploading an hour of
audio. Disabled by `--source-retention run`.

Transcripts stream in live as segments finish, with a rough ETA. When a job
completes you can copy the text or save `.txt`, timestamped `.txt`, `.srt`,
`.vtt`, or `.json` (segment timings, word timings if requested, and the exact
options used).

Jobs run one at a time regardless of how many files you drop, so the GPU isn't
fighting itself. Drop five files and walk away.

## A note on VRAM

Models are cached so repeat jobs don't reload, but the cache is capped at
**one model by default** and evicted LRU. Without a cap, switching between
models and precisions would pin every combination in VRAM at once — five models
times four precisions, with `large-v3` alone at ~4.7 GB. Raise it with
`--model-cache 2` if you routinely alternate between two models and have the
headroom. The status strip shows what's currently resident.

## GPU requirements

Nothing here is specific to one card. It asks CTranslate2 for `device="cuda"` and
takes device 0. Two things decide which `--compute-type` to use:

- `float16` needs **compute capability 7.0+** — Turing, Ampere, Ada, Hopper,
  Blackwell (RTX 20xx through 50xx, T4, A-series). On Pascal or older (GTX 10xx,
  P40) fp16 is emulated and slow: use `int8` or `float32`.
- `large-v3` in fp16 wants roughly **4.7 GB of VRAM**. Below ~6 GB, use
  `int8_float16`, or drop to `medium`.

Multi-GPU isn't wired up — it always uses device 0.

## Security posture

What's enforced:

- **Token required by default**, compared with `hmac.compare_digest`. Sent only
  in an `x-token` header, never a query string, which also forces a CORS
  preflight and blocks cross-site uploads.
- **`Host` header allowlist** (localhost, this machine's own names and IPs, plus
  anything you add with `--allow-host`). Blocks DNS rebinding, where a remote
  page re-resolves its own domain to your LAN IP to get same-origin access.
- **All client-supplied strings are escaped before rendering.** Filenames and
  error text are attacker-controlled; unescaped, a crafted filename is stored XSS
  that steals the token.
- Upload paths are UUID-prefixed and stripped to `[A-Za-z0-9 ._-]`, so a hostile
  name can't escape the uploads directory. Downloads use RFC 5987
  `filename*=UTF-8''` rather than interpolating the raw name into a header.
- Model, precision, language and format are all allowlist-validated server-side.
- Error messages are redacted of local paths; full tracebacks go to the console
  only.
- Response headers set `nosniff`, `DENY` framing, `no-referrer`, and a CSP.

What it still doesn't do, by design:

- **No TLS.** Audio, transcripts and the token cross the network in cleartext.
  Fine on wired LAN; on shared wifi, put it behind a reverse proxy with a cert.
- **No multi-user isolation.** One token, one shared view — anyone holding it
  sees every job. It's a single-operator tool.
- **No rate limiting** beyond the queue cap.
- Untrusted media still goes into native decoders (libav), which is real attack
  surface. Keep ffmpeg current.

## Notes and limits

- **No speaker labels.** Whisper doesn't do diarization. If you need "who said
  what", run the audio through WhisperX or pyannote afterwards using the `.json`
  timings, or use Otter/Fireflies instead.
- Uploads land in `%USERPROFILE%\.transcribe-server\uploads`. By default they're
  kept as long as the job record exists, which is what makes Retry work, and
  deleted when the record is removed or evicted. Use `--source-retention run` if
  you'd rather meeting audio not sit on disk at all.
- Transcripts live in memory only — they're gone when you restart the server, so
  save what you want to keep.
- Finished jobs are evicted past `--max-jobs` (60) to keep memory bounded.
- Job state resets on restart. This is a workstation tool, not a service.
