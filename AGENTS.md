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
.venv/bin/pytest -q                    # 170 tests
uvx ruff check .                       # configured by ruff.toml
uvx pyright --project pyrightconfig.json
```

`uv run tests/test_cuda_bootstrap.py` runs the same suite in a throwaway env
with no `.venv` and no CUDA wheels; the two `tests/test_gpu.py` tests that need
the real wheels skip there. Use it for a quick check, and the `.venv` when you
want the wheel-dependent tests to actually execute.

`.venv` is created by hand (see the README's Tests section) and is gitignored.
Tests never start the worker thread, so they queue uploads without loading a
model or touching a GPU.

Note: pi-lens's own pyright runner does not use the project venv, so it reports
`fastapi`/`uvicorn` as unresolved imports. `uvx pyright --project
pyrightconfig.json` is the authoritative check.

## Constraints

- **`transcribe_server.py` must stay a PEP 723 single-file script.** Its
  dependencies live in the inline `# /// script` block. Do **not** add a
  `pyproject.toml`: it would change how `uv run transcribe_server.py` resolves
  dependencies and break the documented zero-setup path. `ruff.toml` and
  `pyrightconfig.json` are fine because they only configure tooling.
- Prefer stdlib. New runtime dependencies must go in the inline metadata.

## Invariants worth not breaking

- **Prompt/hotword text never reaches an app-token holder.** It is the one field
  the audit token gates. `job_public()` and `render(..., "json")` must emit
  `public_opts()`, which returns a flag and a length, never the text. Only
  `AuditLog.read_prompt` (audit-token routes) may return it. A regression here
  silently voids the two-token split.
- **`JOBS_LOCK` is a non-reentrant `threading.Lock`.** Never call `drop_source`,
  `source_shared`, `patch_job` or `get_job` while holding it — that deadlocked
  the whole server once, because the async endpoints take the same lock on the
  event loop. Do work outside the `with` block.
- **A cancelled job never leaves that state.** `patch_job` refuses the
  transition and returns `False`; callers must respect the return value.
- **The audit trail records hashes, not content.** Never write prompt text,
  transcript text, tokens, or a caller-supplied search string into the main log.
- **Refusals are logged before authentication**, so anything logged on that path
  goes through `audit_rejection` (burst-collapsed), never `audit` directly.
- `--preload` must *prove* the device can encode, not merely load a model. A
  model that loads and then fails every job is the failure mode this guards.

## Style

- Comments explain *why*, not what. Match the surrounding voice: direct, no
  filler.
- Errors the operator can act on should say what to do, not just what broke.
