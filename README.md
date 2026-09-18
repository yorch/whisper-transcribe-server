# Transcription server

Drag-and-drop Whisper transcription served over your LAN.

## Setup on the Windows machine

Two ways in: run it from source (below), or install the tray app. The tray app
is a launcher that manages the process, token and port for you, and bundles
ffmpeg — see [packaging/README.md](packaging/README.md).

```powershell
# from source, zero install
winget install astral-sh.uv
uv run transcribe_server.py --preload

# or build the tray app (add -Installer for a setup .exe)
pwsh -File packaging\build-windows.ps1 -Installer
```

Either way the same `transcribe_server.py` does the work; the launcher just
supervises it. Everything below applies to both.

### Running from source

Dependencies are declared inline in the script (PEP 723), so there's no
requirements file, no venv to create, and nothing to install by hand.

```powershell
winget install astral-sh.uv
```

That's it. The first `uv run` resolves and caches everything, including the CUDA
runtime libs CTranslate2 needs (`nvidia-cublas-cu12` and `nvidia-cudnn-cu12`) —
faster-whisper does **not** use PyTorch, so those two packages are the real
GPU dependency. Budget a couple of GB for that first download; cuBLAS alone is
over 500 MB.

You do **not** need the CUDA Toolkit or a system cuDNN. Those wheels carry
cuBLAS and cuDNN 9, and the server wires them up at startup, before it loads a
model, so `uv run` really is the whole install.

That step is necessary because the wheels drop `libcublas`/`libcudnn` into
`site-packages/nvidia/…`, which is not on the loader's search path, and
CTranslate2 only resolves them when it starts encoding. Without it you get a
model that loads fine and then fails every job with `Library libcublas.so.12 is
not found or cannot be loaded` — or the same for `cublas64_12.dll` on Windows.
[When CUDA fails](#when-cuda-fails) is what to do if you ever see that from some
other entry point.

`--preload` therefore **verifies the device can encode**, not just load, and
exits with an explanation if it cannot. A green "Model ready." means a job will
genuinely run.

Optionally, installing PyTorch gives the status strip a proper GPU name
("NVIDIA GeForce RTX 3080") instead of a device count. Not required:

```powershell
uv run --with torch transcribe_server.py
```

ffmpeg is required for decoding anything that isn't plain WAV (the tray app
bundles it):

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

`--no-auth` disables the app token entirely. Only on a network you fully
control.

The **audit trail** has a second credential of its own, generated the same way,
so a default start prints both:

```
Audit token: 3Zx9wT...
             (generated for this run; pass --audit-token or TRANSCRIBE_AUDIT_TOKEN to pin it)
Audit UI:  http://<this-machine-ip>:8765/audit?token=<audit-token>
```

`--audit-token` is what `/audit` and `/api/audit` check, separately from the app
token. `--audit-open` drops it altogether, making the trail — including the
prompt and hotword text it can read — public to anyone who can reach the port.
`--no-auth` does *not* do that: it drops only the app token, so an audit token is
still required, and the startup banner prints the one for this run.

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
| `--token secret123`           | require the `x-token` header on every API call (`?token=` bootstraps the page, then moves to `sessionStorage`)              |
| `--device cpu`                | always CPU                                                                                                                  |
| `--device auto`               | use the GPU only if it actually works, otherwise CPU                                                                        |
| `--compute-type int8_float16` | lower VRAM use                                                                                                              |
| `--quality fast`              | default beam size: `fast`=1, `balanced`=5, `thorough`=8                                                                     |
| `--model-cache 2`             | models held in VRAM at once (default 1)                                                                                     |
| `--pin-model`                 | force every job to `--model`, disable the UI selector                                                                       |
| `--allow-precision-choice`    | expose the precision selector (hidden by default)                                                                           |
| `--no-auth`                   | serve without a token                                                                                                       |
| `--allow-host name`           | accept an extra `Host` header value (repeatable; prefix with `.` for a suffix match, e.g. `.trycloudflare.com`)             |
| `--max-upload-mb 2048`        | per-file upload ceiling                                                                                                     |
| `--max-queue 20`              | max jobs pending before uploads are refused                                                                                 |
| `--max-jobs 60`               | finished job records retained before eviction                                                                               |
| `--source-retention job`      | `run` deletes audio after transcribing (no retry), `job` keeps it while the record lives (default), `forever` never deletes |
| `--work-dir PATH`             | where uploads and the audit trail live (default `~/.transcribe-server`)                                                     |
| `--config PATH`               | TOML config file (default `<work dir>/config.toml`)                                                                         |
| `--no-starter-config`         | never create a starter config at the default location                                                                       |
| `--audit-dir PATH`            | where daily audit files live (default `<work dir>/audit`)                                                                   |
| `--audit-token`               | separate token for `/audit` and `/api/audit`; generated for the run when unset, like the app token                          |
| `--audit-open`                | drop the audit credential: `/audit` and `/api/audit` become public, prompt/hotword text included                            |
| `--audit-reads`               | also log status/list/detail polls (chatty; off by default)                                                                  |
| `--no-audit`                  | stop recording the audit trail (existing files stay readable)                                                               |
| `--no-audit-prompts`          | never write prompt/hotword text to the sidecar files                                                                        |
| `--audit-retain-days 30`      | delete audit files older than N days at startup (`0` keeps everything)                                                      |

`--help` groups these under network/access, model/hardware, limits/storage and
audit trail.

First run with a given model downloads it from Hugging Face (a few GB for
`large-v3`) into the HuggingFace cache. After that it's local.

## Configuration file

The first run writes a starter file to `<work dir>/config.toml`. Every key in
it is commented out and shows its default, so uncommenting is how you change a
setting and a commented line keeps tracking the default in later releases. The
server never rewrites it: your edits stay yours. The file is created readable
only by you, because it documents the token keys.

Two cases where nothing is created: an explicit `--config` (or
`TRANSCRIBE_CONFIG`) points at a file you expect to exist, so a typo stays a
startup error instead of becoming a server quietly running on defaults, and
`--no-starter-config` turns the whole thing off. If the file cannot be written
(a read-only or container home) the server says so and carries on with
defaults.

`work_dir` is the one key that cannot live in the default config file. That
file is found *in* the state dir, so it cannot move the state dir without
leaving itself behind — the state would move, the file would stay, and a config
written next to the new state would be ignored with nothing on screen to say
so. Setting it there is a startup error. Use `--work-dir` or
`TRANSCRIBE_WORK_DIR`, which move both, or an explicit `--config` pointing at a
file that lives elsewhere, where the key is honest and still works.

Anything settable by a flag can also live in a TOML file, which is what you
want on a machine you don't want to re-type a long command line for:

```toml
# %USERPROFILE%\.transcribe-server\config.toml
[server]
host = "0.0.0.0"
port = 8765
allow_host = [".trycloudflare.com"]

[model]
model = "large-v3-turbo"
device = "cuda"
compute_type = "float16"
quality = "balanced"
preload = true

[limits]
max_upload_mb = 2048
max_queue = 20
source_retention = "job"

[audit]
enabled = true
reads = false
prompts = true
retain_days = 30
open = false    # true serves the trail with no credential at all
```

Precedence, lowest to highest: **defaults → config file → flags → environment
variables.** Environment names are the option uppercased with a `TRANSCRIBE_`
prefix (`TRANSCRIBE_PORT`, `TRANSCRIBE_AUDIT_TOKEN`, `TRANSCRIBE_ALLOW_HOST` as
a comma-separated list, and so on). An unknown or misspelled key is a startup
error rather than a silent no-op, so a typo can't quietly leave auth off.

## Audit trail

The server records what the web app did, one JSON object per line, in daily
files under `<work dir>/audit/`:

```
audit-2026-09-18.jsonl      # the events
prompts/<job_id>.json       # full prompt/hotword text, 0600, separate
```

Events cover uploads, retries, deletions, transcript exports, worker
start/finish/error, VRAM evictions, model loads, the starter config being
created, every refused request (`421` host, `403` cross-site, `401` auth) and a
`server.started` snapshot of how the process was configured. Tokens never
appear, and transcript text never appears.

```json
{"ts":"2026-09-18T02:01:57.677Z","event":"job.created","client":"127.0.0.1",
 "host":"127.0.0.1","method":"POST","path":"/api/jobs","job":"24abc1713890",
 "file":"standup.m4a","bytes":4194304,
 "opts":{"model":"base","quality":"fast","prompt_len":53,
         "prompt_sha256":"c89cb56d74…","hotwords_len":21,"hotwords_sha256":"8319d23d9f…"}}
```

Prompts and hotwords routinely contain real names, so the main log carries only
their length and SHA-256. The text itself goes to `prompts/<job_id>.json`, which
can be permissioned and expired independently. Correlate the two with the job
id or the hash. Set `--no-audit-prompts` to never write that text at all.

Reads (`GET /api/status`, `/api/jobs`, `/api/jobs/{id}`) are skipped by default
because the UI polls them every few seconds; `--audit-reads` turns them on.
Exports are always logged, because that is the moment a transcript leaves the
machine.

### Reading it

Two ways in, deliberately independent:

```powershell
# 1. the files
Get-Content $env:USERPROFILE\.transcribe-server\audit\audit-2026-09-18.jsonl | Select-String "exported"

# 2. the API + UI, with its own token
$env:TRANSCRIBE_AUDIT_TOKEN = "some-other-long-secret"
uv run transcribe_server.py --preload
```

Then open `http://<host>:8765/audit?token=<audit-token>` for a filterable view
(day, job id, substring search, prompts on/off), or query directly:

```http
GET /api/audit?date=2026-09-18&limit=200&offset=0&job=<id>&q=exported&include_prompts=1
GET /api/audit/prompts/<job_id>
```

Both need `x-audit-token`, which is checked separately from the app token —
you can hand out one without the other. Leave it unset and the server mints one
for the run and prints it, so the endpoint works with no configuration at all;
pin it with `TRANSCRIBE_AUDIT_TOKEN` to keep bookmarks and scripts working
across restarts. `--audit-open` is the only way to serve the trail with no
credential at all.

Files older than `audit_retain_days` (30 by default) are deleted at startup
**and** on each daily rollover, sidecars included; `0` keeps everything. With
`--no-audit` nothing is pruned at all, so turning auditing off can never delete
a trail it is not managing.

Two details worth knowing:

- **Prompt sidecars outlive the job.** Removing a job deletes its audio but
  keeps `prompts/<job_id>.json` until retention expires. Deleting it with the
  job would let anyone erase the evidence that a prompt was used, which is
  exactly what the trail exists to prevent. If you would rather the text never
  be stored, use `--no-audit-prompts`.
- **A disabled trail still writes one line.** With `--no-audit`, startup emits a
  single `server.started` record so that a gap in the trail can be told apart
  from the server simply having been down.

Refused requests are logged **before** any credential is checked, so a burst
from one source is collapsed: the first five per source per minute are recorded
individually and the rest are summarised in one `security.rejected_summary`
record. Without that, an unauthenticated client could fill the disk by looping
on a bad token.

### Origin attribution behind a tunnel

Behind `cloudflared`, every request arrives from localhost, so `client` is the
TCP peer and the proxy's claim is recorded next to it, unverified:

```json
{"client":"127.0.0.1","client_claimed":"203.0.113.7"}
```

Treat `client_claimed` as a hint, not as fact: on direct LAN access a client can
send that header itself. `client` is always trustworthy.

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

## Exposing it via Cloudflare Tunnel

The server allowlists the `Host` header, so a tunnel domain is rejected with
`421 Unrecognised Host header` unless you allow it. The 421 response and the
server console both name the rejected host.

```powershell
# terminal 1: start a tunnel to the server
cloudflared tunnel --url http://localhost:8765
# note the URL it prints, e.g. https://abc-123.trycloudflare.com

# terminal 2: allow that exact hostname
uv run transcribe_server.py --preload --allow-host abc-123.trycloudflare.com
```

The quick-tunnel hostname changes on every restart. To accept any of them:

```powershell
uv run transcribe_server.py --preload --allow-host .trycloudflare.com
```

For a stable name, use a named tunnel or custom domain and allow that host
instead. The token is still required — open
`https://<your-tunnel-host>/?token=<token>`. As a bonus, Cloudflare provides
the TLS this server lacks on the LAN.

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
- **Follow** — keeps the newest line of every live preview in view. Remembered
  for the browser session. Scrolling up pauses it (the preview border goes
  amber) and scrolling back to the bottom resumes; the switch is the off switch.

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

Transcripts stream in live as segments finish, with a rough ETA. Each segment
lands in the preview with its start time in a quiet left gutter — the same clock
as the timestamped export, without the milliseconds — so you can read along and
note where something was said. When a job completes you can copy the text or
save `.txt`, timestamped `.txt`, `.srt`, `.vtt`, or `.json` (segment timings,
word timings if requested, and the exact options used).

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

The precision list the UI offers is whatever CTranslate2 reports for the
device that was resolved, so a CPU fallback offers `int8` and `float32` only,
and a client asking for something that device cannot run is refused rather than
queued up to fail. If `--compute-type` names something the device cannot run —
`float16` on CPU is the usual case, since CPU has no float16 at all — the
server runs the best available option instead (`int8`, unless you asked for
`float32` or `int8_float32`, which a CPU can run as-is) and says so at startup.

Multi-GPU isn't wired up — it always uses device 0.

### Where the libraries come from, and when they are missing

CTranslate2 doesn't link against cuBLAS and cuDNN. It loads cuBLAS by name at
runtime (`cublas64_12.dll` on Windows, `libcublas.so.12` elsewhere), and the
`cudnn64_9.dll` it bundles is a shim that loads the cuDNN 9 parts the same way.
A bare `LoadLibrary` searches the executable's directory, the system
directories and `PATH` — and nothing about `uv run` puts
`site-packages\nvidia\...\bin` on `PATH`. Left alone, that produces a very
misleading failure: the server starts, `--preload` loads the model, and the
first job dies minutes later inside the first matrix multiply.

So before it loads anything, the server:

1. Finds the CUDA directories the wheels brought: it asks `importlib` where the
   `nvidia` packages actually live, scans `site-packages/nvidia/*/{bin,lib}`,
   and falls back to a real CUDA 12 toolkit (`CUDA_PATH`) if there is one.
2. Makes them reachable. On Windows it prepends them to `PATH`, which is what a
   bare `LoadLibrary` consults, and registers them with the OS loader as well;
   on macOS and Linux, where the loader read its search path before Python
   started, it loads the libraries by absolute path with `RTLD_GLOBAL`, so a
   later `dlopen` of the same soname resolves against the copy already in
   memory.
3. Verifies the result, then acts. A cheap probe loads cuBLAS and the cuDNN
   parts and asks the driver for a device: `--device auto` becomes `cuda` only
   if that passed, and an explicit `--device cuda` that fails stops startup with
   exit code 1 rather than serving jobs that will die. `--preload` goes one step
   further and runs a real one-second encode, because a GPU can load a model and
   still be unable to encode. If a service wrapper restarts on failure, capture
   the output or treat exit 1 as fatal, or it will restart-loop without ever
   showing you the diagnosis.

`--device cpu` skips all of it.

The probe cannot catch every way a GPU can misbehave — only a real transcription
can, which is what `--preload` is for. Hover the device name in the status strip
for the reason behind the device it settled on; `server.started` in the audit
trail records the same thing.

## When CUDA fails

The failure this section exists for looks like this — usually minutes into a
job, usually with a healthy-looking server in front of it:

```
RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
```

or the same for `cudnn_ops64_9.dll` or `cudnn_cnn64_9.dll`. It means the loader
could not find a library that is sitting on disk in `site-packages\nvidia\...`.
The server now checks for exactly this before it starts serving, and prints the
directories it searched, so the terminal says which case you are in:

| What it says | What to do |
| --- | --- |
| No `nvidia-*` wheel directory exists for this interpreter | `uv run transcribe_server.py`, to re-resolve the inline dependencies. If you started it some other way (`python transcribe_server.py` in a hand-made venv), install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` there, or just use `uv run`. |
| The libraries were found but the loader refused them | `uv cache clean && uv run transcribe_server.py`. Almost always a truncated or wrong-architecture wheel. |
| The libraries load but no CUDA device is visible | A driver problem: install or update the NVIDIA driver (CUDA 12 wants >= 525) and reopen the terminal. |
| `--preload` reports the model *cannot run* | The libraries loaded but a real encode failed. Install the runtime by running through `uv`, or fall back to `--device cpu --compute-type int8`. |
| `--device auto` fell back to CPU | One of the above; run with `--device cuda` for the detail. Everything works, it is just slower. |

`--device cpu` always works and needs none of this: it skips the GPU entirely
and switches the precision to `int8`.

Two warnings on every Windows run are noise rather than problems: Hugging Face
reporting that it cannot use symlinks in `%USERPROFILE%\.cache\huggingface`
(setting `HF_HUB_DISABLE_SYMLINKS_WARNING=1` silences it; Developer Mode fixes
the cause), and a note that you are downloading anonymously
(`$env:HF_TOKEN = "hf_..."` raises the rate limit).

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
- **The audit trail has its own token**, so read access to "who did what" is
  separable from the ability to transcribe. Job ids are validated before they
  reach a filename, and audit files are written `0600` (best effort: on Windows
  `chmod` does not set ACLs, so treat the claim as POSIX-only). It is generated
  per run when unset; `--audit-open` is the single explicit way to drop it, and
  a server running that way says so on startup and records `audit_api: "open"`
  in its `server.started` snapshot.
- **Prompt and hotword text is never returned to an app-token holder** — not in
  the job list, not in the job detail, not in the JSON export. Only the audit
  token can read it, or anyone at all if you deliberately chose `--audit-open`.
  The trail itself carries a length and a SHA-256.
- **Uploads and audit directories are created `0700`**, and uploaded audio
  `0600`, so other local accounts cannot read meeting audio.
- **Every response carries the hardening headers**, including `421`/`403`/`401`
  refusals, which previously returned before the headers were attached.

What it still doesn't do, by design:

- **No TLS.** Audio, transcripts and the token cross the network in cleartext.
  Fine on wired LAN; on shared wifi, put it behind a reverse proxy with a cert.
- **No multi-user isolation.** One token, one shared view — anyone holding it
  sees every job. It's a single-operator tool.
- **No rate limiting** beyond the queue cap.
- Untrusted media still goes into native decoders (libav), which is real attack
  surface. Keep ffmpeg current.
- **The audit trail is not tamper-proof.** It is an append-only file on the same
  machine, written by the same process it describes. Anyone with filesystem
  access can edit or delete it. It answers "what did the web app do", not "prove
  it to a third party".

## Notes and limits

- **No speaker labels.** Whisper doesn't do diarization. If you need "who said
  what", run the audio through WhisperX or pyannote afterwards using the `.json`
  timings, or use Otter/Fireflies instead.
- Uploads land in `%USERPROFILE%\.transcribe-server\uploads`. By default they're
  kept as long as the job record exists, which is what makes Retry work, and
  deleted when the record is removed or evicted. Use `--source-retention run` if
  you'd rather meeting audio not sit on disk at all.
- Transcripts live in memory only — they're gone when you restart the server, so
  save what you want to keep. The audit trail does not change that: it records
  that an export happened, never the text.
- Finished jobs are evicted past `--max-jobs` (60) to keep memory bounded.
- Job state resets on restart. This is a workstation tool, not a service. Because
  of that, every file in `uploads/` is unreferenced after a restart, so the
  server sweeps that directory at startup and logs `uploads.swept`. Use
  `--source-retention forever` if you want the audio kept regardless.
- **The two HTML pages duplicate ~1,000 lines** of CSS/JS (escaping helper, token
  bootstrap, gate, polling). Extracting them into `static/` is the highest-value
  refactor and has not been done; until then, a change to `esc()` must be made in
  both pages.
- `create_job` streams the upload through a threadpool, but a very large upload
  still occupies the threadpool for its duration.
- **Verified on CPU and on a single RTX 3060** (`base`, fp16, ~10x realtime).
  `large-v3`, multi-GPU and Windows are untested by the author.
- **The Windows tray app has never been built on Windows** — the launcher's
  logic is tested, its build scripts are not. See
  [packaging/README.md](packaging/README.md#known-gaps).

## Tests

The suite covers the audit trail, config layering, the starter config, host
allowlist, job state machine and the HTTP surface, including regressions for
the two most serious bugs found in review (a self-deadlock on cancel, and
prompt text leaking to an app-token holder).

It needs the runtime dependencies plus `pytest` and `httpx`, which is what the
`.venv` is for:
```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
  fastapi "uvicorn[standard]" python-multipart faster-whisper pytest httpx
.venv/bin/pytest -q
```

`ruff.toml` and `pyrightconfig.json` point the linters at that same venv:

```bash
uvx ruff check .
uvx pyright --project pyrightconfig.json
```

Tests never start the worker thread, so they queue uploads without loading a
model or touching a GPU.

`tests/test_cuda_bootstrap.py` gives you a one-command check that needs neither
that `.venv` nor faster-whisper: it declares its own dependencies inline
(pytest, fastapi, httpx, numpy) and runs the whole suite, because the
model-loading paths are monkeypatched and the CUDA probes stub the loader and
the driver.

```bash
uv run tests/test_cuda_bootstrap.py
```

The `.venv` above is still what `pyright` type-checks against, and it is the
only environment that exercises a real model load.

The launcher has its own self-test, which needs no display and no tray backend:

```bash
launcher/transcribe_tray.py --self-test                # logic only, fast
launcher/transcribe_tray.py --self-test --with-server   # also probes a real server
```
