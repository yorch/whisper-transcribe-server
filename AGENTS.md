# Project instructions

A single-file Whisper transcription server. Read `README.md` for the operator
view; this file is about working on the code.

## Git hygiene — do this before ending any turn that changed files

Run `scripts/git-sync.sh` and act on the exit code:

```bash
scripts/git-sync.sh -q          # 0 clean+synced, 1 dirty, 2 unpushed, 3 remote unreachable
scripts/git-sync.sh -c "msg"    # commit anything pending, push, verify
```

**Why this is not optional:** an automated formatter rewrites files *after* a
commit completes. A commit that looks successful can still leave a real,
uncommitted change behind — this has happened here. "I committed it" is not
evidence; `git status` is.

Do not report work as committed until the script exits 0, and never claim the
remote matches based on a local ref alone: verify with the script or
`gh api repos/{owner}/{repo}/commits/main --jq .sha`.

The ssh-agent is sometimes empty in this environment, so a plain `git push` can
fail with "Permission denied (publickey)". The script falls back to the `gh`
credential helper over HTTPS; do the same manually rather than reporting a
push failure as a blocker.

## Tests

```bash
uv sync                                # the locked dev env in .venv (pyproject.toml + uv.lock)
uv run pytest -q                       # whole suite; no GPU or model needed
uv run ruff check .                    # configured in pyproject.toml
uv run ruff format --check .           # the formatter that rewrites files post-commit
uv run pyright                         # resolves imports against .venv
```

The test count is deliberately not quoted: it went 170 → 212 in a single day of
parallel sessions, and a stale number is worse than none. `uv run ruff format
--check .` is the same formatter that rewrites files *after* a commit (see Git
hygiene above); a clean `ruff check` does not imply a formatted file.

On macOS `tests/test_cuda_bootstrap.py::test_preload_loads_by_absolute_path`
fails: it asserts Linux `libcublas.so` loading. It is not a regression.

`uv run tests/test_cuda_bootstrap.py` runs the same suite in a throwaway env
with no `.venv` and no CUDA wheels; the two `tests/test_gpu.py` tests that need
the real wheels skip there. Use it for a quick check, and the `.venv` when you
want the wheel-dependent tests to actually execute.

**That throwaway env has no `sherpa-onnx` either, and the suite must still
pass there.** The `.venv` does have it, so a test that quietly depends on it is
green locally and red in the clean env — which has happened twice. If a test
exercises anything past the diarization preflight — `fetch_diarize_models_or_explain`,
and therefore `diarize_job` and `probe_diarization` — it has to say so with the
`sherpa_present` helper in `tests/test_diarization.py`, rather than inheriting
whatever happens to be installed.

`tests/test_ui_preview.py` covers the transcript preview. Most of it needs
`node` (the embedded page script is parsed and its preview functions are run
against a stub DOM) and skips without it; a browser is still the only way to
check how the preview looks and scrolls for real.

`.venv` is built by `uv sync` from the committed `uv.lock`, and is gitignored.
Tests never start the worker thread, so they queue uploads without loading a
model or touching a GPU.

Note: pi-lens's own pyright runner does not use the project venv, so it reports
`fastapi`/`uvicorn`/`pystray`/`PIL` as unresolved imports. `uv run pyright` is
the authoritative check. `.pi-lens.json` disables one rule,
`unchecked-throwing-call-python`, whose ast-grep pattern fires on every bare
`int()`/`float()` in the file — 26 pre-existing hits and no true positives,
because this codebase handles those at the worker boundary. Its scope is a
non-`include`d directory and its `scripts/` findings are for the same reason
ignorable: `transcribe_server.py`, `tests` and `launcher` are what pyright
checks.

## Packaging

`launcher/` and `packaging/` ship a Windows **tray launcher**, not a frozen
server. The launcher owns the process, the access token and the port, and hands
them to the server via `TRANSCRIBE_TOKEN` and `--port`, so the server needs no
launcher-aware code. Keep it that way — no state file, no protocol.

The CUDA runtime is deliberately not bundled (2.2 GB; `uv` resolves it on first
run). `packaging/README.md` records what is and is not verified: the launcher
logic is tested, the Windows build is not (no Windows machine was available).

```bash
launcher/transcribe_tray.py --self-test                # fast, headless
launcher/transcribe_tray.py --self-test --with-server   # also probes a real server
```

## Constraints

- **`transcribe_server.py` stays a PEP 723 script.** Its inline `# /// script`
  block is what `uv run transcribe_server.py`, the launcher and the Windows
  build resolve. `pyproject.toml` is the *dev* environment only: uv ignores the
  enclosing project for a script with inline metadata (verified on uv 0.12), so
  the zero-setup path is unaffected. Its `[project] dependencies` are a copy of
  the inline block; **change both**, then `uv lock`.
  `tests/test_project_metadata.py` fails if they disagree. Do not turn the
  project into a package or drop the inline block: the launcher and the
  PyInstaller build ship the bare script.
- **The pages live in `static/`**, not in the module. They were extracted from
  embedded string literals to remove ~1,100 lines from the Python file and, more
  importantly, to stop `esc()` — the only thing between a filename and stored
  XSS — existing in two copies. `STATIC_DIR` is anchored to the script, so
  `static/` must sit next to `transcribe_server.py`; a missing directory is a 500
  that says so rather than a blank page. Anything that ships the script (the
  PyInstaller spec, the launcher) has to ship `static/` too.
- Prefer stdlib. New runtime dependencies go in the inline metadata *and*
  `pyproject.toml` (see above).

## Invariants worth not breaking

- **Diarization runs in a child process, and must keep doing so.**
  `OfflineSpeakerDiarization.process()` holds the GIL for its entire run
  (measured; `docs/speaker-diarization.md` section 5). Called from the worker
  thread it freezes the event loop for minutes — the status poll, the live
  transcript and Cancel all stop dead — which looks exactly like the cancel
  deadlock above. `run_diarizer` therefore spawns `DIARIZE_WORKER` as a
  subprocess. Do not "simplify" it into a thread;
  `tests/test_diarization.py` fails if you do.
- **A failed diarization pass never fails the job.** Labels are worth less than
  the transcript: `run_job` catches it, records `job.diarize_failed`, and
  finishes the job unlabelled with the reason in its message. A model download
  or a child process going wrong must not cost an hour of transcription.
- **`--preload` warns about diarization, and does not exit.** Same reasoning one
  level up: the most likely failure is a machine that cannot reach github.com
  for the weights, where transcription works fine. `--preload`'s exit code means
  "a job will genuinely run", and a job does run — it just has no labels. Exit 1
  here would contradict the invariant above, and under a service wrapper it buys
  a restart loop instead of a diagnosis.
- **Relabelling is a new job, never an edit of a finished one.**
  `POST /api/jobs/{id}/speakers` queues a job with `relabel_of` and a copy of
  the transcript as Whisper produced it (`transcribed`); the worker runs only
  `label_and_finish` over it, the same tail a full transcription runs. That
  keeps "done never leaves done" and "cancelled never leaves cancelled" true
  without a new state. `transcribed` is a second copy of the transcript:
  `job_public` must keep excluding it, or every 1.2 s poll carries it.
  `relabel_refusal` is the single rule behind both the 409 and `can_relabel`.
  A *merge* (`POST /api/jobs/{id}/speakers/merge`) is the one edit of a
  finished job: it moves speaker numbers only, reads and writes the segments
  under a single hold of `JOBS_LOCK` (no `get_job`/`patch_job` inside), and
  bumps `labels_rev` -- the page's poll signature includes it, because a merge
  changes nothing else a poll can see.
- **Speaker names never reach the main audit log.** A name is visible to the
  app (it renders the transcript), but `job.speaker_named` records only
  `name_len` and `name_sha256`; the text lives in the job's sidecar via
  `store_speaker_names`, which reads and amends the record so a stored prompt
  survives, and which `--no-audit-prompts` silences like prompts.
  `tests/test_speaker_names.py` scans the whole trail for the name.
- **Alignment splits on speaker change, so `word_timestamps` is not optional.**
  Asking for diarization forces it on. Without word timings a whole segment can
  only go to its dominant speaker, which is the version of the feature that is
  confidently wrong.
- **Prompt/hotword text never reaches an app-token holder.** It is the one field
  the audit credential gates. `job_public()` and `render(..., "json")` must emit
  `public_opts()`, which returns a flag and a length, never the text. Only
  `AuditLog.read_prompt` (audit-token routes) may return it, and the only thing
  allowed to drop that gate is the explicit `--audit-open` opt-in. A regression
  here silently voids the two-token split.
- **The audit API is reachable without configuration.** `main()` mints an audit
  token when none is set and the banner prints it; `--audit-open` is the only
  way to serve the trail without one. Do not reintroduce a silent "no token means
  404": that is what made `/audit` ask for a token that did not exist.
- **The audit token and `--audit-open` are mutually exclusive**, refused in
  `resolve_args` rather than resolved by precedence, because the environment
  wins over flags here and would otherwise discard a token without a word.
- **`JOBS_LOCK` is a non-reentrant `threading.Lock`.** Never call `drop_source`,
  `source_shared`, `patch_job` or `get_job` while holding it — that deadlocked
  the whole server once, because the async endpoints take the same lock on the
  event loop. Do work outside the `with` block.
- **A cancelled job never leaves that state.** `patch_job` refuses the
  transition and returns `False`; callers must respect the return value.
- **The audit trail records hashes, not content.** Never write prompt text,
  transcript text, tokens, or a caller-supplied search string into the main log.
- **The audit chain commits only after the write returns.** `_append` sets
  `_seq`/`_chain` *after* `fh.write`, and `emit` truncates back to the last
  known-good length on failure. Moving that commit earlier would make a dropped
  event look like a sequence gap, and `verify` would report a lost write as
  tampering — the exact false positive the `lost: N` field exists to avoid.
  `tests/test_audit_chain.py::test_a_failed_write_is_attested_and_does_not_gap_the_chain`
  pins it.
- **Never return an audit token, and never let the app token reach the trail's
  detail.** `audit_degraded` on `/api/status` is a boolean on purpose: the
  `last_error` text and the audit dir stay behind the audit credential.
- **Reading the head is itself an audit event**, so `/api/audit/head` writes the
  read *before* computing the head; otherwise the value it returns is already
  one write stale and cannot serve as an anchor.
- **`audit_max_mb` is a storage bound, not a rate limit.** Hitting it writes one
  `audit.full` marker and then stops recording for that day, which is only
  acceptable because `degraded`/`full` surface it. Do not make the stop silent.
- **Refusals are logged before authentication**, so anything logged on that path
  goes through `audit_rejection` (burst-collapsed), never `audit` directly.
- `--preload` must *prove* the device can encode, not merely load a model. A
  model that loads and then fails every job is the failure mode this guards.

## Style

- Comments explain *why*, not what. Match the surrounding voice: direct, no
  filler.
- Errors the operator can act on should say what to do, not just what broke.
