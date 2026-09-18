# Packaging for Windows

Ships a **tray launcher**, not a frozen server. The launcher owns the process,
the access token and the port; `transcribe_server.py` stays a PEP 723 script and
is executed by `uv`, which resolves its dependencies on first run.

## Why not freeze the server

Freezing would mean bundling the CUDA runtime, which is **2.2 GB** of the 2.6 GB
dependency set (`nvidia-cudnn-cu12` is most of it). It also means teaching
PyInstaller where ctranslate2's lazily-`dlopen`ed libraries ended up, and
re-shipping all of it on every update. Launcher + `uv` avoids both.

## Layout

| File | Purpose |
| --- | --- |
| `../launcher/transcribe_tray.py` | The launcher. Also runnable from source. |
| `transcribe-launcher.spec` | PyInstaller spec (launcher + vendored extras only) |
| `build-windows.ps1` | Vendors `uv.exe`/`ffmpeg.exe`, builds, smoke-tests, optionally makes the installer |
| `transcribe-server.iss` | Inno Setup script |
| `vendor/` | Populated at build time; gitignored |

## Build

```powershell
winget install astral-sh.uv
pwsh -File packaging\build-windows.ps1 -Installer
```

Output:

- `dist\TranscriptionServer\` — the app folder (~50 MB + ~80 MB for ffmpeg)
- `packaging\output\TranscriptionServer-1.0.0-setup.exe` — the installer

Must run on Windows; PyInstaller cannot cross-compile.

## What the installer does

- Installs to `%ProgramFiles%\Transcription Server`
- Start-menu shortcut, optional desktop shortcut
- Optional **start when I sign in** (a Startup shortcut, not a service — the
  tray app owns the process so it can show the token and stop cleanly)
- Optional **firewall rule** for TCP 8765, scoped to the `private` profile
- Removed again on uninstall

It deliberately **leaves your data behind** on uninstall: the audit trail in
`%USERPROFILE%\.transcribe-server\audit`, the config file next to it, and any
retained audio. Deleting a user's transcripts as a side effect of an uninstall
is a nasty surprise.

## How the launcher and server divide responsibility

The launcher never parses console output. It:

1. generates a stable token (`%LOCALAPPDATA%\TranscriptionServer\token`, `0600`)
2. picks the documented port, or any free one if 8765 is taken
3. starts `uv run --no-project transcribe_server.py --port N` with
   `TRANSCRIBE_TOKEN` set, and prepends the vendored ffmpeg to `PATH`
4. polls `GET /api/status` with that token until it answers, then opens the
   browser
5. keeps the server's stdout in `%LOCALAPPDATA%\TranscriptionServer\server.log`

Because the token and port are chosen by the launcher and passed in, the server
needs **no launcher-aware code** — no state file, no protocol, no changes. The
config file stays the server's too: it writes a commented starter to
`%USERPROFILE%\.transcribe-server\config.toml` on first run, and the launcher
neither reads nor writes it.

## First run

Nothing is preinstalled except `uv.exe` and `ffmpeg.exe`. The first launch
resolves the dependency set (~2.2 GB, mostly CUDA) and downloads a model on the
first job (`base` ≈ 145 MB, `large-v3` ≈ 3 GB). Expect several minutes; the
tray shows "Starting (first run can take minutes)..." and the log has progress.

`--preload` is not used by default, so startup is fast and the model download
happens on the first transcription instead. Pass it through if you prefer to pay
that cost up front:

```powershell
& "Transcription Server.exe" --preload
```

## Verifying a build

`build-windows.ps1` runs the launcher's self-test against the bundle, which
covers token persistence, port selection, uv discovery and command assembly.
For a fuller check, including starting a real server and probing it:

```powershell
& "dist\TranscriptionServer\Transcription Server.exe" --self-test --with-server
```

## Known gaps

Being explicit about what has and has not been exercised:

- **The launcher logic and the supervisor path are tested** (self-test, and
  `--self-test --with-server` starts a real server and probes it). That was run
  on Linux; the code is platform-neutral apart from the Windows-only branches.
- **The Windows build itself is untested by the author** — no Windows machine
  was available. The spec, the PowerShell script and the Inno Setup script are
  written but unverified. Expect to iterate on the first build.
- **Untested Windows-specific paths:** `CREATE_NO_WINDOW`, `os.startfile` for
  the log, the `clip` clipboard call, `os.add_dll_directory` for CUDA, and
  `netsh` firewall rules.
- **No code signing.** Windows SmartScreen will warn on first run, and some
  antivirus products flag PyInstaller output. Signing needs a certificate.
- **No auto-update.** Rebuild and reinstall to update, which also refreshes the
  vendored ffmpeg.
