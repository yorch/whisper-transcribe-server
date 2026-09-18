"""Tests for the launcher's self-test reporting.

Windows is the platform the launcher ships to, and it is the one where the two
"not world readable" checks cannot be made at all: chmod there only sets the
file's read-only attribute, and st_mode reports 0666 for any writable file, so
the mode-bit expression fails on a machine with no problem.

These pin the way out of that: the check is reported as skipped, which is
neither a pass it cannot support nor a failure that is not real, and the
summary says how many were skipped rather than claiming everything passed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import launcher.transcribe_tray as launcher


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """A self-test that touches neither the real install nor a display."""
    monkeypatch.setattr(launcher, "launcher_dir", lambda: tmp_path)
    monkeypatch.setattr(launcher, "find_uv", lambda: tmp_path / "uv.exe")
    monkeypatch.setattr(
        launcher, "make_icon_image", lambda: SimpleNamespace(size=(64, 64))
    )
    return tmp_path


def test_a_skipped_check_is_neither_pass_nor_fail(capsys):
    outcome = launcher.report_check("something", False, skip="not here it isn't")

    assert outcome == "skip"
    out = capsys.readouterr().out
    assert "SKIP  something" in out
    assert "not here it isn't" in out
    assert "FAIL" not in out, "a skipped check must not read as a failure"


def test_pass_and_fail_still_report_as_before(capsys):
    assert launcher.report_check("good", True) == "pass"
    assert launcher.report_check("bad", False) == "fail"

    out = capsys.readouterr().out
    assert "PASS  good" in out
    assert "FAIL  bad" in out


def test_mode_bits_are_reported_as_uncheckable_on_windows(monkeypatch):
    """The Windows branch, forced here: this machine is not Windows."""
    monkeypatch.setattr(launcher, "POSIX_MODE_BITS", False)

    message = launcher.mode_bit_skip()
    assert "Windows" in message
    assert "icacls" in message, "say what to check instead, not just that it can't"


@pytest.mark.skipif(sys.platform == "win32", reason="asserts the POSIX branch")
def test_mode_bits_are_checkable_on_posix():
    assert launcher.mode_bit_skip() == ""


@pytest.mark.skipif(sys.platform == "win32", reason="asserts the POSIX branch")
def test_the_self_test_checks_mode_bits_where_they_mean_something(stubbed, capsys):
    assert launcher.self_test() == 0

    out = capsys.readouterr().out
    assert "PASS  token is not world readable" in out
    assert "PASS  audit token is not world readable" in out
    assert "SKIP" not in out
    assert "all checks passed" in out


def test_the_self_test_skips_both_credential_checks_on_windows(
    stubbed, monkeypatch, capsys
):
    """The whole run, not just the helper: the wiring is what a later edit drops."""
    monkeypatch.setattr(launcher, "POSIX_MODE_BITS", False)

    assert launcher.self_test() == 0, "a skipped check must not fail the run"

    out = capsys.readouterr().out
    assert "SKIP  token is not world readable" in out
    assert "SKIP  audit token is not world readable" in out
    assert "FAIL" not in out
    assert "all checks passed" in out
    assert "2 skipped" in out, "the summary has to say what went unverified"
