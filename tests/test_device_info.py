"""Tests for the device readout.

Two things worth pinning here beyond the formatting:

1. **The parsing is done against canned nvidia-smi output**, because the real
   thing is not available in CI and the fields that differ between driver
   versions are exactly the ones that break a naive split — "N/A" for a value
   the driver declines to report, and whitespace after every comma.
2. **Refreshing is rate-limited.** The UI polls status every ~1.2s and a refresh
   spawns a process, so "how often does this actually run" is behaviour, not an
   implementation detail. A machine with no nvidia-smi must spawn nothing.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

import transcribe_server as s

# What `--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap
# --format=csv,noheader,nounits` prints on a healthy machine.
GPU_ROW = [["0", " NVIDIA GeForce RTX 3060", " 12288", " 1713", " 610.88", " 8.6"]]
PASCAL_ROW = [["0", " NVIDIA GeForce GTX 1050 Ti", " 4096", " 120", " 535.104", " 6.1"]]
TWO_GPUS = [
    ["0", " NVIDIA GeForce RTX 3060", " 12288", " 1713", " 610.88", " 8.6"],
    ["1", " NVIDIA GeForce RTX 4090", " 24564", " 100", " 610.88", " 8.9"],
]


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the probe cache, which is deliberately module-global."""
    monkeypatch.setattr(s, "DEVICE_INFO", {})
    monkeypatch.setattr(s, "_DEVICE_AT", 0.0)
    monkeypatch.setattr(s, "_NVIDIA_SMI", None)


def fake_driver(monkeypatch: pytest.MonkeyPatch, gpu_rows=None, apps: str = "1700"):
    """Make nvidia-smi exist and answer, counting how often it is asked."""
    calls: list[str] = []
    rows = GPU_ROW if gpu_rows is None else gpu_rows

    monkeypatch.setattr(s, "find_nvidia_smi", lambda: "/fake/bin/nvidia-smi")

    def nvidia_smi_rows(exe: str, domain: str, fields: str):
        calls.append(domain)
        if domain == "gpu":
            return rows
        return [[str(os.getpid()), f" {apps}"]] if apps is not None else []

    monkeypatch.setattr(s, "nvidia_smi_rows", nvidia_smi_rows)
    return calls


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_leading_int_reads_nvidia_smis_cells():
    assert s.leading_int(" 12288") == 12288
    assert s.leading_int("12288 MiB") == 12288
    # The driver says "N/A" rather than omitting a field it will not report.
    assert s.leading_int("N/A") is None
    assert s.leading_int("") is None


def test_a_healthy_gpu_row_becomes_the_details():
    info = s.parse_gpu_rows(GPU_ROW)
    assert info["name"] == "NVIDIA GeForce RTX 3060"
    assert info["driver"] == "610.88"
    assert info["compute_cap"] == "8.6"
    assert info["vram_total_mb"] == 12288
    assert info["vram_used_mb"] == 1713
    assert info["vram_free_mb"] == 12288 - 1713


def test_the_first_card_is_the_one_reported():
    """CTranslate2 always takes device 0, so that is the memory that matters."""
    assert s.parse_gpu_rows(TWO_GPUS)["name"] == "NVIDIA GeForce RTX 3060"
    assert len(TWO_GPUS) == 2, "the count comes from the row count, not the parse"


def test_a_driver_that_will_not_report_memory_yields_no_free_figure():
    info = s.parse_gpu_rows([["0", " Some Card", " N/A", " N/A", " 610.88", " 8.6"]])
    assert info["name"] == "Some Card"
    assert info["vram_total_mb"] is None
    # Nothing to subtract, so no free figure rather than a wrong one.
    assert "vram_free_mb" not in info


def test_a_truncated_row_is_ignored_rather_than_half_read():
    assert s.parse_gpu_rows([["0", " Card", " 12288"]]) == {}
    assert s.parse_gpu_rows([]) == {}


def test_app_rows_are_matched_by_this_process():
    rows = [["1234", " 500"], [str(os.getpid()), " 1713"]]
    assert s.parse_app_rows(rows, os.getpid()) == 1713
    assert s.parse_app_rows([["1234", " 500"]], os.getpid()) is None


def test_an_unreported_process_figure_is_none_not_zero():
    """0 would be the claim "this server holds no VRAM", which is a different
    and wrong statement."""
    assert s.parse_app_rows([[str(os.getpid()), " N/A"]], os.getpid()) is None


# --------------------------------------------------------------------------- #
# Finding the binary
# --------------------------------------------------------------------------- #


def test_nvidia_smi_is_preferred_from_path(monkeypatch):
    monkeypatch.setattr(s.shutil, "which", lambda name: "/on/path/nvidia-smi")
    assert s.find_nvidia_smi() == "/on/path/nvidia-smi"


def test_a_wsl_style_install_is_found_without_path(monkeypatch):
    """Where this was noticed: WSL keeps the driver tools in /usr/lib/wsl/lib
    and they are not on PATH, so a which()-only lookup finds nothing."""
    monkeypatch.setattr(s.shutil, "which", lambda name: None)
    wanted = "/usr/lib/wsl/lib/nvidia-smi"
    # as_posix(), not str(): on Windows a WindowsPath renders the separators as
    # backslashes, so str(self) never matches a POSIX candidate and this test
    # failed there for a reason that had nothing to do with the lookup.
    monkeypatch.setattr(s.Path, "is_file", lambda self: self.as_posix() == wanted)
    assert s.find_nvidia_smi() == wanted


def test_no_nvidia_smi_anywhere_is_not_an_error(monkeypatch):
    monkeypatch.setattr(s.shutil, "which", lambda name: None)
    monkeypatch.setattr(s.Path, "is_file", lambda self: False)
    assert s.find_nvidia_smi() is None


# --------------------------------------------------------------------------- #
# The cached snapshot
# --------------------------------------------------------------------------- #


def test_a_cpu_only_machine_reports_the_interpreter_and_nothing_else(
    device, monkeypatch
):
    monkeypatch.setattr(s, "find_nvidia_smi", lambda: None)
    monkeypatch.setattr(
        s, "nvidia_smi_rows", lambda *a: pytest.fail("spawned a process")
    )

    info = s.device_details()

    assert info["python"].count(".") == 2
    assert info["platform"] in (
        "Linux",
        "macOS",
        "Windows",
        "freebsd",
        "linux",
        "darwin",
    )
    assert "name" not in info and "vram_total_mb" not in info


def test_the_platform_label_is_readable(device, monkeypatch):
    """platform_family() normalises to posix, which is right for branching on
    and wrong for a tooltip."""
    monkeypatch.setattr(s, "find_nvidia_smi", lambda: None)
    monkeypatch.setattr(s.sys, "platform", "win32")
    assert s.probe_device_details()["platform"] == "Windows"
    monkeypatch.setattr(s.sys, "platform", "linux")
    assert s.probe_device_details()["platform"] == "Linux"
    # Anything unrecognised is passed through rather than guessed at.
    monkeypatch.setattr(s.sys, "platform", "freebsd12")
    assert s.probe_device_details()["platform"] == "freebsd12"


def test_memory_is_refreshed_on_a_timer_rather_than_every_poll(device, monkeypatch):
    calls = fake_driver(monkeypatch)

    s.device_details()
    assert calls == ["gpu", "compute-apps"], "the first call does the whole probe"

    s.device_details()
    assert calls == ["gpu", "compute-apps"], "a second poll inside the window is free"

    s.device_details(refresh_after=0)
    assert calls == ["gpu", "compute-apps", "gpu", "compute-apps"], (
        "once the window has passed, only the moving numbers are re-read"
    )


def test_a_refresh_updates_memory_without_losing_the_rest(device, monkeypatch):
    calls = fake_driver(monkeypatch)
    first = s.device_details()
    assert first["driver"] == "610.88"

    # The card is busier by the time the next refresh comes round.
    monkeypatch.setattr(
        s,
        "nvidia_smi_rows",
        lambda exe, domain, fields: (
            [["0", " NVIDIA GeForce RTX 3060", " 12288", " 9000", " 610.88", " 8.6"]]
            if domain == "gpu"
            else []
        ),
    )
    second = s.device_details(refresh_after=0)

    assert second["vram_used_mb"] == 9000
    assert second["driver"] == "610.88", "the static fields survive a refresh"
    assert second["process_mb"] is None, "no processes listed means exactly that"
    assert calls[:2] == ["gpu", "compute-apps"]


def test_the_path_to_nvidia_smi_is_never_published(device, monkeypatch):
    monkeypatch.setattr(s, "find_nvidia_smi", lambda: "/opt/sekrit/bin/nvidia-smi")
    monkeypatch.setattr(s, "nvidia_smi_rows", lambda exe, domain, fields: GPU_ROW)

    info = s.device_details()

    assert "/opt/sekrit" not in json.dumps(info)
    # Kept internally, where the refresh needs it.
    assert s._NVIDIA_SMI == "/opt/sekrit/bin/nvidia-smi"


# --------------------------------------------------------------------------- #
# What the readout shows
# --------------------------------------------------------------------------- #


def test_the_cell_names_the_card_and_falls_back_to_a_count(configured, monkeypatch):
    monkeypatch.setattr(s, "CUDA", {"usable": True, "device_count": 2})
    assert s.gpu_cell({"name": "RTX 3060"}) == "RTX 3060"
    # Unnamed, but the probe still knows how many: better than nothing.
    assert s.gpu_cell({}) == "2 CUDA device(s)"
    assert s.gpu_cell({"count": 1}) == "1 CUDA device(s)"


def test_the_cell_is_empty_when_there_is_no_gpu_at_all(configured, monkeypatch):
    monkeypatch.setattr(s, "CUDA", None)
    assert s.gpu_cell({}) is None
    monkeypatch.setattr(s, "CUDA", {"usable": False, "device_count": 0})
    assert s.gpu_cell({}) is None


def test_the_banner_line_reads_like_a_bug_report_wants():
    line = s.device_banner(
        {
            "name": "NVIDIA GeForce RTX 3060",
            "vram_total_mb": 12288,
            "vram_used_mb": 1713,
            "driver": "610.88",
            "compute_cap": "8.6",
        }
    )
    assert line == (
        "NVIDIA GeForce RTX 3060, 12.0 GB, 1.7 GB used, driver 610.88, compute 8.6"
    )


def test_the_banner_line_says_nothing_when_there_is_no_card():
    assert s.device_banner({"python": "3.12.2", "platform": "linux"}) is None
    assert s.device_banner({"name": None}) is None


def test_status_serves_the_device_from_the_cache(configured, client, monkeypatch):
    """And does it without asking CTranslate2 or attempting an import torch on
    every poll, which is what this replaced."""
    monkeypatch.setattr(
        s,
        "device_details",
        lambda *a, **k: {
            "name": "Test Card",
            "vram_total_mb": 8192,
            "vram_used_mb": 2048,
            "python": "3.12.2",
            "platform": "linux",
        },
    )

    body = client.get("/api/status").json()

    assert body["gpu"] == "Test Card"
    assert body["device_info"]["vram_used_mb"] == 2048
    assert body["device_info"]["python"] == "3.12.2"


def test_a_broken_driver_does_not_break_the_status_endpoint(configured, client):
    """nvidia-smi missing, hanging or failing is the normal case on a CPU box."""

    def explode(*_a: Any, **_k: Any):
        raise AssertionError("no process should be spawned here")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(s, "find_nvidia_smi", lambda: None)
    monkey.setattr(s, "nvidia_smi_rows", explode)
    monkey.setattr(s, "DEVICE_INFO", {})
    monkey.setattr(s, "_DEVICE_AT", 0.0)
    monkey.setattr(s, "CUDA", None)
    try:
        body = client.get("/api/status").json()
        assert body["device_info"]["python"]
        assert body["gpu"] is None
    finally:
        monkey.undo()
