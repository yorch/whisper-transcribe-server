# Packaging for Windows

Ships a **tray launcher**, not a frozen server. The launcher owns the process,
the credentials and the port; `transcribe_server.py` stays a PEP 723 script and
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

1. generates a stable token (`%LOCALAPPDATA%\TranscriptionServer\token`)
   and a second one for the audit trail (`...\audit-token`), because the server
   would otherwise mint a fresh audit token per run and print it to a log file
   the tray user never opens, leaving `/audit` unreachable. Both are created
   `0600`, which on POSIX is what keeps them private; Windows has no mode bits,
   so there the files rely on the ACL they inherit from `%LOCALAPPDATA%`
2. picks the documented port, or any free one if 8765 is taken
3. starts `uv run --no-project transcribe_server.py --port N` with
   `TRANSCRIBE_TOKEN` and `TRANSCRIBE_AUDIT_TOKEN` set, and prepends the
   vendored ffmpeg to `PATH`. An `--audit-open` in the passthrough arguments (or
   an inherited `TRANSCRIBE_AUDIT_OPEN`) suppresses the audit token instead,
   since the server refuses a token it has been told to ignore
4. polls `GET /api/status` with that token until it answers, then opens the
   browser
5. keeps the server's stdout in `%LOCALAPPDATA%\TranscriptionServer\server.log`

Because the tokens and port are chosen by the launcher and passed in, the server
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
- **The Windows build is verified in CI, not on a desktop.** The `windows`
  workflow runs the PyInstaller spec, the PowerShell script and the Inno Setup
  script on `windows-latest`, then starts the bundled launcher with
  `--self-test --with-server`: it launches a real server through the vendored
  uv (on CPU — the runner has no GPU), probes it, stops it, and fails if the
  server outlives the launcher. What CI cannot do is click the tray icon or
  run on a machine with a GPU; nobody has installed it by hand yet.
- **Windows-specific paths CI does run:** `CREATE_NO_WINDOW`, stopping the
  server's whole process tree with `taskkill /T`, the `clip` clipboard
  round-trip (including non-ASCII), that `.log` has a handler for "Open log",
  and the installer itself — a silent install with the firewall task, the
  `netsh` rule it adds (inbound TCP 8765, private profile), and an uninstall
  that removes both the program and the rule.
- **Still untested:** loading the CUDA DLLs on a machine with a GPU, and
  anything that needs a person at the screen — the tray menu, the log
  actually opening in an editor, and the installer's interactive pages.
- **No code signing.** Windows SmartScreen will warn on first run, and some
  antivirus products flag PyInstaller output. Signing needs a certificate.
- **No auto-update.** Rebuild and reinstall to update, which also refreshes the
  vendored ffmpeg.

## Installer trade-off: per-machine vs per-user

The default install is machine-wide because the firewall rule needs elevation.
That has one consequence worth knowing: the optional "start when I sign in"
shortcut goes to `{userstartup}`, which under an admin install belongs to the
account that approved the elevation — not necessarily the account that will use
the app.

`PrivilegesRequiredOverridesAllowed=dialog` is set, so you can pick a per-user
install instead. Then the shortcut lands in your own profile and the firewall
rule is skipped; add it yourself if other devices need to reach the server:

```powershell
New-NetFirewallRule -DisplayName "Transcription Server" -Direction Inbound `
  -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private
```
