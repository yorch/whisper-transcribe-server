#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pystray>=0.19",
#     "pillow>=10",
# ]
# ///
"""Windows tray launcher for the transcription server.

This is a supervisor, not a second implementation. It owns three things the
server deliberately does not: process lifetime, the credentials, and the port.

The app token, the audit token and the port are chosen here and handed to the
server through TRANSCRIBE_TOKEN, TRANSCRIBE_AUDIT_TOKEN and --port, so the
launcher never has to parse console output to discover them — and
transcribe_server.py needs no launcher-aware code.

    uv run launcher/transcribe_tray.py            # tray
    uv run launcher/transcribe_tray.py --no-tray  # supervisor only (headless)
    uv run launcher/transcribe_tray.py --self-test

Packaged for Windows by packaging/build-windows.ps1; see packaging/README.md.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

APP_NAME = "Transcription Server"
UV_RELEASES = "https://github.com/astral-sh/uv/releases/latest/download"
UV_ASSET = "uv-x86_64-pc-windows-msvc.zip"
DEFAULT_PORT = 8765


# Where the launcher keeps its own state (token, logs, downloaded uv). Separate
# from the server's work dir, which the server owns.
def launcher_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "TranscriptionServer"
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "transcription-server"


def resource_dir() -> Path:
    """Where bundled extras (ffmpeg.exe) live, frozen or from source."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


# --------------------------------------------------------------------------- #
# Token
# --------------------------------------------------------------------------- #


def token_path() -> Path:
    return launcher_dir() / "token"


def audit_token_path() -> Path:
    return launcher_dir() / "audit-token"


def _stable_token(path: Path) -> str:
    """A stable per-install secret, so a bookmarked URL keeps working.

    Generated once with 192 bits of entropy and stored 0600. The server compares
    it with hmac.compare_digest; this only has to be unguessable and stable.

    0600 is a POSIX claim: on Windows chmod only sets the read-only flag, so the
    file there is protected by the ACL it inherits from %LOCALAPPDATA% instead.
    """
    with contextlib.suppress(OSError, ValueError):
        existing = path.read_text(encoding="utf-8").strip()
        if len(existing) >= 16:
            return existing

    token = secrets.token_urlsafe(24)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return token


def ensure_token() -> str:
    """The app credential, handed over as TRANSCRIBE_TOKEN."""
    return _stable_token(token_path())


def ensure_audit_token() -> str:
    """The audit credential, handed over as TRANSCRIBE_AUDIT_TOKEN.

    The server generates one per run when this is missing, which would leave
    /audit unreachable: the value would only ever appear in a log file the tray
    user never opens. Opening the endpoint instead is not an option either, since
    the server binds every interface by default.
    """
    return _stable_token(audit_token_path())


# Mode bits are a POSIX idea. On Windows os.chmod only sets the file's read-only
# attribute and st_mode reports 0666 for any writable file, so a mode-bit check
# there fails on a machine with no problem -- and Windows is the platform the
# launcher ships to. Such a check is reported as skipped, never as passing: a
# PASS would claim evidence this platform cannot provide.
POSIX_MODE_BITS = sys.platform != "win32"


def mode_bit_skip() -> str:
    """Why a mode-bit check cannot be made here, or "" when it can."""
    if POSIX_MODE_BITS:
        return ""
    return (
        "Windows has no mode bits (chmod sets only the read-only flag); "
        f"verify the ACL on {launcher_dir()} with icacls"
    )


# --------------------------------------------------------------------------- #
# uv
# --------------------------------------------------------------------------- #


def find_uv() -> Path | None:
    """Locate uv: bundled, then alongside the app, then on PATH."""
    exe = "uv.exe" if sys.platform == "win32" else "uv"
    for candidate in (
        launcher_dir() / "bin" / exe,
        resource_dir() / "vendor" / exe,
    ):
        if candidate.is_file():
            return candidate

    found = shutil.which("uv")
    return Path(found) if found else None


def download_uv() -> Path:
    """Fetch uv into the launcher dir. Windows-only; returns the exe path."""
    if sys.platform != "win32":
        raise RuntimeError("automatic uv download is only implemented for Windows")

    import io
    import zipfile

    dest = launcher_dir() / "bin"
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{UV_RELEASES}/{UV_ASSET}"
    print(f"Downloading uv from {url} ...")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.namelist():
            if member.endswith("uv.exe"):
                target = dest / "uv.exe"
                target.write_bytes(archive.read(member))
                return target
    raise RuntimeError(f"uv.exe was not found inside {UV_ASSET}")


def ensure_uv(progress=None) -> Path:
    uv = find_uv()
    if uv:
        return uv
    if progress:
        progress("Downloading uv (one time)...")
    return download_uv()


# --------------------------------------------------------------------------- #
# Port
# --------------------------------------------------------------------------- #


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(preferred: int = DEFAULT_PORT) -> int:
    """Prefer the documented port, but never fight another process for it.

    A preferred value of 0 means "any free port". Note that binding port 0
    succeeds, so treating 0 as a normal preference would hand back 0 and leave
    the caller unable to find the server.
    """
    if preferred and port_is_free(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return as_int(sock.getsockname()[1], DEFAULT_PORT)


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    # Matches transcribe_server.as_bool, so a value one layer accepts is not
    # silently ignored by the other.
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Server process
# --------------------------------------------------------------------------- #


def server_script() -> Path:
    """The server script, frozen or from a source checkout."""
    if is_frozen():
        bundled = resource_dir() / "transcribe_server.py"
        if bundled.is_file():
            return bundled
    return Path(__file__).resolve().parent.parent / "transcribe_server.py"


def build_command(
    uv: Path, port: int, extra: list[str], script: Path | None = None
) -> list[str]:
    """The uv invocation that runs the server.

    --no-project keeps uv from adopting a pyproject.toml if one ever appears
    next to the script; the script's own PEP 723 metadata is the dependency
    source of truth.
    """
    target = script or server_script()
    return [
        str(uv),
        "run",
        "--no-project",
        "--python",
        "3.12",
        str(target),
        "--port",
        str(port),
        *extra,
    ]


def child_env(ffmpeg_dir: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    if ffmpeg_dir and ffmpeg_dir.is_dir():
        # The server finds ffmpeg with shutil.which, so PATH is the interface.
        env["PATH"] = str(ffmpeg_dir) + os.pathsep + env.get("PATH", "")
    return env


def creation_flags() -> int:
    """Keep a console from flashing up on Windows."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


class ServerProcess:
    """Owns the child process and its log."""

    def __init__(self, command: list[str], env: dict[str, str], log_path: Path) -> None:
        self.command = command
        self.env = env
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._log = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a", encoding="utf-8", errors="replace")
        self._log.write(
            f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(self.command)} ===\n"
        )
        self._log.flush()
        self.proc = subprocess.Popen(  # noqa: S603
            self.command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=self.env,
            creationflags=creation_flags(),
        )

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 15.0) -> None:
        """Ask nicely, then insist. The server has no shutdown endpoint."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            with contextlib.suppress(OSError):
                self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    self.proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=5)
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
            self._log = None


def wait_for_ready(
    port: int, token: str, proc: ServerProcess, timeout: float = 900.0
) -> bool:
    """Poll /api/status until it answers.

    The first run resolves dependencies and downloads a model, which can take
    many minutes, so the timeout is generous and a dead child aborts early.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/api/status"
    while time.monotonic() < deadline:
        if not proc.alive():
            return False
        request = urllib.request.Request(url, headers={"x-token": token})  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=3) as response:  # noqa: S310
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(1.0)
    return False


def server_url(port: int, token: str) -> str:
    return f"http://127.0.0.1:{port}/?token={token}"


def audit_url(port: int, token: str) -> str:
    return f"http://127.0.0.1:{port}/audit?token={token}"


def audit_open_requested(server_args: list[str], env: dict[str, str]) -> bool:
    """Whether the child was told to serve without an audit credential.

    Both layers count: the passthrough flag, and an inherited
    TRANSCRIBE_AUDIT_OPEN. Injecting a token on top of either would turn the
    operator's opt-out into the server's mutual-exclusion error.
    """
    return "--audit-open" in server_args or as_bool(
        env.get("TRANSCRIBE_AUDIT_OPEN", "")
    )


def open_browser(url: str) -> None:
    import webbrowser

    with contextlib.suppress(Exception):
        webbrowser.open(url)


def copy_to_clipboard(text: str) -> bool:
    """Windows clipboard via the shell; no extra dependency."""
    if sys.platform != "win32":
        return False
    # Resolve the full path rather than invoking a bare "clip": a partial path
    # is resolved against PATH and the cwd, so it can be hijacked.
    clip = shutil.which("clip")
    if clip is None:
        return False
    try:
        subprocess.run(  # noqa: S603
            [clip],
            input=text.encode("utf-16le"),
            check=True,
            creationflags=creation_flags(),
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def open_in_editor(path: Path) -> None:
    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            os.startfile(path)  # noqa: S606
    else:
        with contextlib.suppress(Exception):
            subprocess.Popen(["xdg-open", str(path)])  # noqa: S603, S607


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #


class Supervisor:
    """Holds the running server and the state the tray renders."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.port = args.port or pick_port()
        self.token = ensure_token()
        self.audit_token = ensure_audit_token()
        self.log_path = launcher_dir() / "server.log"
        self.ffmpeg_dir = self._ffmpeg_dir()
        self.server: ServerProcess | None = None
        self.ready = threading.Event()
        self.failed: str | None = None
        self._thread: threading.Thread | None = None

    def _ffmpeg_dir(self) -> Path | None:
        """Prefer a bundled ffmpeg so the user does not have to install one."""
        vendor = resource_dir() / "vendor"
        exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        if (vendor / exe).is_file():
            return vendor
        return None

    def url(self) -> str:
        return server_url(self.port, self.token)

    def audit_url(self) -> str:
        return audit_url(self.port, self.audit_token)

    def start(self) -> None:
        uv = ensure_uv()
        command = build_command(uv, self.port, self.args.server_args)
        env = child_env(self.ffmpeg_dir)
        env["TRANSCRIBE_TOKEN"] = self.token  # the launcher owns the credentials
        if not audit_open_requested(self.args.server_args, env):
            env["TRANSCRIBE_AUDIT_TOKEN"] = self.audit_token
        self.server = ServerProcess(command, env, self.log_path)
        self.server.start()
        server = self.server

        def watch() -> None:
            if wait_for_ready(self.port, self.token, server, self.args.timeout):
                self.ready.set()
                if not self.args.no_browser:
                    open_browser(self.url())
            else:
                code = server.proc.poll() if server.proc else None
                self.failed = (
                    f"server exited with code {code}"
                    if code is not None
                    else "timed out waiting for the server"
                )

        self._thread = threading.Thread(target=watch, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.server:
            self.server.stop()

    def restart(self) -> None:
        self.stop()
        self.ready.clear()
        self.failed = None
        self.start()


# --------------------------------------------------------------------------- #
# Tray
# --------------------------------------------------------------------------- #


def tray_backend():
    """Import the tray stack on demand, with an actionable message if absent.

    Imported dynamically rather than with a top-level `import pystray`: these
    are optional extras of the launcher, not requirements. Only the tray needs
    them, so --self-test and --no-tray must keep working on a machine (or CI
    runner) with no tray backend and no display.
    """
    try:
        return (
            importlib.import_module("pystray"),
            importlib.import_module("PIL.Image"),
            importlib.import_module("PIL.ImageDraw"),
        )
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SystemExit(
            "The tray UI needs pystray and pillow. Either install them:\n"
            "  uv run --with pystray --with pillow launcher/transcribe_tray.py\n"
            "or run without the tray:\n"
            "  launcher/transcribe_tray.py --no-tray"
        ) from exc


def make_icon_image(colour: tuple[int, int, int] = (185, 97, 15)):
    """A filled dot, drawn rather than shipped as an asset."""
    _, Image, ImageDraw = tray_backend()

    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((6, 6, size - 6, size - 6), fill=(*colour, 255))
    return image


def run_tray(supervisor: Supervisor) -> int:
    pystray, _, _ = tray_backend()

    status: dict[str, Any] = {"text": "Starting..."}

    def refresh() -> None:
        if supervisor.failed:
            status["text"] = "Failed"
        elif supervisor.server and not supervisor.server.alive():
            status["text"] = "Stopped"
        elif supervisor.ready.is_set():
            status["text"] = f"Running on :{supervisor.port}"
        else:
            status["text"] = "Starting (first run can take minutes)..."

    def on_open(icon, item):  # noqa: ARG001
        open_browser(supervisor.url())

    def on_copy(icon, item):  # noqa: ARG001
        copy_to_clipboard(supervisor.token)

    def on_copy_audit(icon, item):  # noqa: ARG001
        copy_to_clipboard(supervisor.audit_token)

    def on_log(icon, item):  # noqa: ARG001
        open_in_editor(supervisor.log_path)

    def on_restart(icon, item):  # noqa: ARG001
        threading.Thread(target=supervisor.restart, daemon=True).start()

    def on_quit(icon, item):  # noqa: ARG001
        icon.visible = False
        supervisor.stop()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem(lambda item: status["text"], None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open in browser", on_open),
        pystray.MenuItem("Copy access token", on_copy),
        pystray.MenuItem("Copy audit token", on_copy_audit),
        pystray.MenuItem("Show log", on_log),
        pystray.MenuItem("Restart", on_restart),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon(APP_NAME, make_icon_image(), APP_NAME, menu)

    def tick() -> None:
        while True:
            refresh()
            icon.title = f"{APP_NAME} - {status['text']}"
            time.sleep(2)

    threading.Thread(target=tick, daemon=True).start()
    icon.run()
    return 0


# --------------------------------------------------------------------------- #
# Headless / self-test
# --------------------------------------------------------------------------- #


def run_headless(supervisor: Supervisor) -> int:
    supervisor.start()
    print(f"token: {supervisor.token}")
    print(f"url:   {supervisor.url()}")
    print(f"audit: {supervisor.audit_url()}")
    print(f"log:   {supervisor.log_path}")
    if not supervisor.ready.wait(timeout=supervisor.args.timeout):
        print(f"!  {supervisor.failed or 'timed out'}", file=sys.stderr)
        supervisor.stop()
        return 1
    print(f"ready on port {supervisor.port}")
    try:
        while supervisor.server and supervisor.server.alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.stop()
    return 0


# The three outcomes a self-test line can have. "skip" is first class: see
# report_check().
Check = Literal["skip", "pass", "fail"]


def report_check(
    name: str, ok: bool, detail: str = "", skip: str = "", sink: list[str] | None = None
) -> Check:
    """Print one self-test line, and say which kind it was.

    A skipped check is neither a pass nor a failure: PASS would claim evidence
    the platform cannot provide, and FAIL would report a problem that is not
    there. Saying so is the only honest option, and the summary counts them.

    `sink` collects the same line for the report file. The bundle is a windowed
    executable, so its stdout is not reliably captured when it is launched from
    a console — an exit code would say "passed" without showing which checks
    ran.
    """
    if skip:
        line = f"  SKIP  {name}  {skip}"
    else:
        line = f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}"
    print(line)
    if sink is not None:
        sink.append(line)
    return "skip" if skip else ("pass" if ok else "fail")


def self_test(with_server: bool = False) -> int:
    """Exercise the logic that can be checked without a display.

    With --with-server this also starts a real server, waits for readiness the
    same way the tray does, and stops it — the only way to prove the readiness
    probe and the audit credential match what the server actually serves.

    The report is written to a file as well as stdout, so a windowed bundle
    launched from a console still leaves evidence of which checks ran.
    """
    failures: list[str] = []
    skipped: list[str] = []
    lines: list[str] = []

    def check(name: str, ok: bool, detail: str = "", skip: str = "") -> None:
        outcome = report_check(name, ok, detail, skip, sink=lines)
        if outcome == "skip":
            skipped.append(name)
        elif outcome == "fail":
            failures.append(name)

    print("launcher self-test")
    lines.append("launcher self-test")
    check("resource_dir exists", resource_dir().is_dir(), str(resource_dir()))
    check("server script found", server_script().is_file(), str(server_script()))

    token = ensure_token()
    check("token is stable", ensure_token() == token)
    check("token is long enough", len(token) >= 16, f"{len(token)} chars")
    check(
        "token is not world readable",
        not (token_path().stat().st_mode & 0o077),
        skip=mode_bit_skip(),
    )

    audit_token = ensure_audit_token()
    check("audit token is stable", ensure_audit_token() == audit_token)
    check(
        "audit token is long enough",
        len(audit_token) >= 16,
        f"{len(audit_token)} chars",
    )
    check(
        "audit token is not world readable",
        not (audit_token_path().stat().st_mode & 0o077),
        skip=mode_bit_skip(),
    )
    check("the two tokens are distinct", audit_token != token)
    check(
        "--audit-open suppresses the audit token",
        audit_open_requested(["--audit-open"], {}),
    )
    check(
        "an inherited TRANSCRIBE_AUDIT_OPEN suppresses the audit token",
        audit_open_requested([], {"TRANSCRIBE_AUDIT_OPEN": "true"}),
    )
    check(
        "an ordinary passthrough still gets the audit token",
        not audit_open_requested(["--model", "base"], {}),
    )

    free = pick_port(0)
    check("pick_port returns a usable port", 1024 < free < 65536, str(free))
    check("a free port is reused", pick_port(free) == free)
    check("a busy port is avoided", pick_port(busy_port()) != busy_port())

    uv = find_uv()
    check("uv located", uv is not None, str(uv))
    if uv:
        command = build_command(uv, 8765, ["--model", "base"])
        check("command includes --no-project", "--no-project" in command)
        check("command passes the port", "8765" in command)
        check("command passes extra args", "--model" in command)

    check("child env keeps PATH", "PATH" in child_env(Path("/nonexistent")))
    check("icon renders", make_icon_image().size == (64, 64))

    if with_server:
        # A crash here must be a recorded failure, not a lost traceback. The
        # bundle is a windowed executable: an unhandled exception prints to a
        # stdout nobody captures, the report never gets written, and the run
        # fails with no explanation of why.
        try:
            ready, audit_ready = end_to_end_probe()
        except Exception as exc:  # noqa: BLE001
            note = f"  probe raised: {type(exc).__name__}: {exc}"
            print(note)
            lines.append(note)
            ready = audit_ready = False
        check("readiness probe against a real server", ready)
        check("audit API accepts the launcher's token", audit_ready)

    # A skip is part of the answer, not noise: it names what went unverified.
    summary = "FAILURES: " + ", ".join(failures) if failures else "all checks passed"
    if skipped:
        summary += f" ({len(skipped)} skipped: {', '.join(skipped)})"
    print(f"\n{summary}")
    lines.append(f"\n{summary}")

    # Durable evidence, not just an exit code: CI reads this back, and "Show
    # log" in the tray can point at it when something looks wrong. Written on
    # every path, including the failing ones, so the report can be trusted to
    # describe this run rather than an earlier one.
    report = launcher_dir() / "self-test.log"
    with contextlib.suppress(OSError):
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}  with_server={with_server}\n"
            + "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )
    return 1 if failures else 0


# Held open for the lifetime of the process: if these were collected the port
# would be released and the negative test would flake.
_BUSY_SOCKETS: list[socket.socket] = []


def busy_port() -> int:
    """A port that is definitely occupied, for the negative test."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    _BUSY_SOCKETS.append(sock)
    return as_int(sock.getsockname()[1], 0)


def audit_probe(port: int, token: str, timeout: float = 5.0) -> bool:
    """Prove the audit credential opens /api/audit."""
    request = urllib.request.Request(  # noqa: S310
        f"http://127.0.0.1:{port}/api/audit?limit=1",
        headers={"x-audit-token": token},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def end_to_end_probe(timeout: float = 900.0) -> tuple[bool, bool]:
    """Start the server exactly as the tray would, and probe it.

    Returns (ready, audit_api). The second value is what proves the launcher's
    audit credential is the one the server accepts: it is checked by request,
    not by reading startup output the launcher deliberately never parses.
    """
    token = ensure_token()
    audit_token = ensure_audit_token()
    port = pick_port(0)
    uv = find_uv()
    if uv is None:
        return False, False
    script = server_script()
    command = build_command(uv, port, [], script)
    env = child_env(None)
    env["TRANSCRIBE_TOKEN"] = token
    if not audit_open_requested([], env):
        env["TRANSCRIBE_AUDIT_TOKEN"] = audit_token
    # Deliberately not self-test.log: that is the report's file, and the
    # server's banner would overwrite the very evidence this probe exists to
    # produce.
    server = ServerProcess(command, env, launcher_dir() / "self-test-server.log")
    server.start()
    try:
        ready = wait_for_ready(port, token, server, timeout)
        return ready, ready and audit_probe(port, audit_token)
    finally:
        server.stop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} launcher")
    parser.add_argument("--port", type=int, default=None, help="preferred port")
    parser.add_argument("--no-tray", action="store_true", help="supervise headless")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser"
    )
    parser.add_argument("--self-test", action="store_true", help="check launcher logic")
    parser.add_argument(
        "--with-server",
        action="store_true",
        help="with --self-test, also start a real server and probe it",
    )
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="seconds to wait for readiness"
    )
    parser.add_argument(
        "server_args",
        nargs="*",
        default=[],
        help="extra arguments passed through to transcribe_server.py",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test(with_server=args.with_server)

    supervisor = Supervisor(args)
    if args.no_tray or sys.platform != "win32":
        # The tray is the Windows experience; elsewhere stay a plain supervisor.
        return run_headless(supervisor)
    try:
        return run_tray(supervisor)
    finally:
        supervisor.stop()


if __name__ == "__main__":
    raise SystemExit(main())
