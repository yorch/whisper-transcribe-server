"""The launcher's Windows-only paths, exercised where they can be: Windows CI.

The tray's "Copy token" and "Open log" items call `clip` and os.startfile,
which only exist on Windows and were never run anywhere. The runner has no
one to look at a window, so these check what can be checked headless: the
clipboard round-trip, and that a .log file has a handler, which is what
decides whether "Open log" opens anything or fails in silence.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

import launcher.transcribe_tray as launcher

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


@windows_only
def test_copy_to_clipboard_round_trips_through_the_windows_clipboard():
    # Non-ASCII on purpose: clip reads UTF-16, which is why the launcher
    # encodes it that way, and a token-only test would never notice.
    text = "token-é·日本"

    assert launcher.copy_to_clipboard(text) is True

    # A full path, as the launcher resolves clip: a bare name is looked up
    # through PATH and the cwd.
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell is missing from PATH"
    back = subprocess.run(  # noqa: S603 - fixed argv
        [
            powershell,
            "-NoProfile",
            "-Command",
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard",
        ],
        capture_output=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()
    assert back == text


def test_a_log_file_has_a_handler_for_open_log():
    """os.startfile on a file with no handler raises OSError, which the
    launcher swallows: "Open log" would do nothing, and say nothing."""
    # A platform check rather than the marker, so the type checker (which
    # runs on the developer's OS) sees winreg only where it exists.
    if sys.platform != "win32":
        pytest.skip("Windows only")
    import winreg

    handler = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, ".log")

    assert handler, "no program is registered for .log files"


def test_the_clipboard_is_not_attempted_off_windows(monkeypatch):
    monkeypatch.setattr(launcher.sys, "platform", "linux")

    assert launcher.copy_to_clipboard("x") is False
