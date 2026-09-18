# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest>=8",
#     "fastapi>=0.110",
#     "uvicorn[standard]>=0.27",
#     "python-multipart>=0.0.9",
#     "httpx>=0.27",
#     "numpy>=1.24",
#     "tomli>=2; python_version < '3.11'",
# ]
# ///
"""Tests for the CUDA library bootstrap in transcribe_server.py.

    uv run tests/test_cuda_bootstrap.py

or, with the dependencies already installed:

    pytest tests/
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transcribe_server as s  # noqa: E402  (needs the sys.path fix above)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def wheel_tree(root: Path, packages: tuple[str, ...] = ("cublas", "cudnn")) -> Path:
    """A fake <site-packages> populated the way the nvidia-* wheels do."""
    for name in packages:
        (root / "nvidia" / name / "bin").mkdir(parents=True)
    return root


def probe(usable: bool, **overrides) -> dict:
    report = {
        "usable": usable,
        "kind": None if usable else "missing-libs",
        "reason": None if usable else "cublas64_12.dll could not be loaded",
        "missing": [] if usable else ["cublas64_12.dll"],
        "device_count": 1 if usable else 0,
    }
    report.update(overrides)
    return report


def fake_loader(failing: tuple[str, ...] = ()):
    """Stands in for load_library_probe: returns an error string or None."""

    def load(name: str, platform: str | None = None) -> str | None:
        return f"could not load {name}" if name in failing else None

    return load


@pytest.fixture(autouse=True)
def clean_module_state(monkeypatch):
    """The handles and caches are process-wide; keep tests independent."""
    monkeypatch.setattr(s, "_DLL_DIR_HANDLES", [], raising=False)
    monkeypatch.setattr(s, "_REGISTERED_DLL_DIRS", set(), raising=False)
    monkeypatch.setattr(s, "_PRELOADED_LIBS", [], raising=False)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_site_package_dirs_finds_the_running_environment():
    dirs = s.site_package_dirs()
    assert dirs, "expected at least one site-packages for this interpreter"
    assert all(d.is_dir() for d in dirs)

    # The property that matters is functional: wherever this interpreter's
    # installed packages actually live has to be in the list. Asserting the leaf
    # name is not enough — uv's ephemeral environments are not laid out like a
    # normal prefix, which is exactly what the Windows runner showed.
    import fastapi

    installed = Path(fastapi.__file__).resolve().parent.parent
    assert installed in [d.resolve() for d in dirs], f"{installed} not in {dirs}"


def test_nvidia_lib_dirs_finds_the_wheel_layout(tmp_path):
    root = wheel_tree(tmp_path)
    assert s.nvidia_lib_dirs([root]) == [
        root / "nvidia" / "cublas" / "bin",
        root / "nvidia" / "cudnn" / "bin",
    ]


def test_nvidia_lib_dirs_is_empty_without_wheels(tmp_path):
    assert s.nvidia_lib_dirs([tmp_path]) == []


def test_nvidia_lib_dirs_ignores_loose_files(tmp_path):
    root = tmp_path
    (root / "nvidia").mkdir()
    (root / "nvidia" / "cublas.py").write_text("")
    assert s.nvidia_lib_dirs([root]) == []


def test_cuda_search_dirs_ranks_wheels_above_a_toolkit(tmp_path, monkeypatch):
    root = wheel_tree(tmp_path / "site-packages")
    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    monkeypatch.setattr(s, "spec_lib_dirs", lambda *a, **k: [])
    monkeypatch.setattr(s, "site_package_dirs", lambda: [root])
    monkeypatch.setenv("CUDA_PATH", str(toolkit))
    monkeypatch.delenv("CUDA_HOME", raising=False)

    dirs = s.cuda_search_dirs(platform="win32")
    assert dirs[0] == root / "nvidia" / "cublas" / "bin"
    assert dirs[-1] == toolkit / "bin"


def test_cuda_search_dirs_prefers_the_importable_wheel_location(tmp_path, monkeypatch):
    """find_spec knows where the interpreter would import from; the scan guesses."""
    spec_dir = tmp_path / "real" / "nvidia" / "cublas" / "lib"
    scan_dir = tmp_path / "stale" / "nvidia" / "cublas" / "lib"
    spec_dir.mkdir(parents=True)
    scan_dir.mkdir(parents=True)
    monkeypatch.setattr(s, "spec_lib_dirs", lambda *a, **k: [spec_dir])
    monkeypatch.setattr(s, "site_package_dirs", lambda: [tmp_path / "stale"])

    dirs = s.cuda_search_dirs(platform="linux")

    assert dirs == [spec_dir, scan_dir]


def test_cuda_search_dirs_skips_missing_and_duplicate_entries(tmp_path, monkeypatch):
    root = wheel_tree(tmp_path / "site-packages")
    monkeypatch.setattr(s, "spec_lib_dirs", lambda *a, **k: [])
    monkeypatch.setattr(s, "site_package_dirs", lambda: [root])
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "nowhere"))
    monkeypatch.delenv("CUDA_HOME", raising=False)

    dirs = s.cuda_search_dirs(
        extra=[root / "nvidia" / "cublas" / "bin"], platform="win32"
    )
    assert dirs == [
        root / "nvidia" / "cublas" / "bin",
        root / "nvidia" / "cudnn" / "bin",
    ]


def test_cuda_search_dirs_uses_lib64_for_posix_toolkits(tmp_path, monkeypatch):
    toolkit = tmp_path / "cuda"
    (toolkit / "lib64").mkdir(parents=True)
    monkeypatch.setattr(s, "spec_lib_dirs", lambda *a, **k: [])
    monkeypatch.setattr(s, "site_package_dirs", lambda: [])
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    assert s.cuda_search_dirs(platform="linux") == [toolkit / "lib64"]


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_cuda_library_wiring_prepends_path_and_keeps_the_original(
    tmp_path, monkeypatch
):
    root = wheel_tree(tmp_path)
    monkeypatch.setattr(s, "spec_lib_dirs", lambda *a, **k: [])
    monkeypatch.setattr(s, "site_package_dirs", lambda: [root])
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))

    wired = s.cuda_library_wiring(platform="win32")

    entries = os.environ["PATH"].split(os.pathsep)
    assert entries[:2] == [
        str(root / "nvidia" / "cublas" / "bin"),
        str(root / "nvidia" / "cudnn" / "bin"),
    ]
    assert entries[2:] == ["/usr/bin", "/bin"]
    assert wired["added_to_path"] == entries[:2]


def test_cuda_library_wiring_is_idempotent(tmp_path, monkeypatch):
    root = wheel_tree(tmp_path)
    monkeypatch.setattr(s, "site_package_dirs", lambda: [root])
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")

    s.cuda_library_wiring(platform="win32")
    after_first = os.environ["PATH"]
    second = s.cuda_library_wiring(platform="win32")

    assert os.environ["PATH"] == after_first
    assert second["added_to_path"] == []


def test_cuda_library_wiring_registers_each_directory_once(tmp_path, monkeypatch):
    root = wheel_tree(tmp_path)
    monkeypatch.setattr(s, "spec_lib_dirs", lambda *a, **k: [])
    monkeypatch.setattr(s, "site_package_dirs", lambda: [root])
    monkeypatch.setenv("PATH", "/usr/bin")
    calls: list[str] = []
    monkeypatch.setattr(s.os, "add_dll_directory", calls.append, raising=False)

    s.cuda_library_wiring(platform="win32")
    s.cuda_library_wiring(platform="win32")

    assert calls == [
        str(root / "nvidia" / "cublas" / "bin"),
        str(root / "nvidia" / "cudnn" / "bin"),
    ]


def test_cuda_library_wiring_survives_a_refusing_add_dll_directory(
    tmp_path, monkeypatch
):
    root = wheel_tree(tmp_path)
    monkeypatch.setattr(s, "site_package_dirs", lambda: [root])
    monkeypatch.setenv("PATH", "/usr/bin")

    def refuse(_path: str):
        raise OSError("nope")

    monkeypatch.setattr(s.os, "add_dll_directory", refuse, raising=False)
    wired = s.cuda_library_wiring(platform="win32")

    assert wired["added_to_path"], "PATH must still be wired"
    assert str(root / "nvidia" / "cublas" / "bin") in os.environ["PATH"]


def test_cuda_library_wiring_leaves_posix_alone(tmp_path, monkeypatch):
    root = wheel_tree(tmp_path)
    monkeypatch.setattr(s, "site_package_dirs", lambda: [root])
    monkeypatch.setenv("PATH", "/usr/bin")

    wired = s.cuda_library_wiring(platform="linux")

    assert os.environ["PATH"] == "/usr/bin"
    assert wired["added_to_path"] == []
    assert wired["path_error"] is None
    assert wired["dirs"], "the directories are still reported for the probe"


def test_cuda_library_wiring_reports_a_path_it_could_not_set(tmp_path, monkeypatch):
    """A PATH that cannot take the new value must not crash startup."""
    root = wheel_tree(tmp_path)
    monkeypatch.setattr(s, "site_package_dirs", lambda: [root])
    real_environ = os.environ
    monkeypatch.setenv("PATH", "/usr/bin")

    class Refusing:
        """Reads normally, refuses writes: a full or locked environment block."""

        def get(self, key, default=None):
            return real_environ.get(key, default)

        def __setitem__(self, key, value):
            raise OSError("environment block is full")

    monkeypatch.setattr(s, "_environ", Refusing)

    wired = s.cuda_library_wiring(platform="win32")

    assert wired["added_to_path"] == []
    assert wired["path_error"] == "environment block is full"
    assert real_environ["PATH"] == "/usr/bin"


# --------------------------------------------------------------------------- #
# Probing and the device decision
# --------------------------------------------------------------------------- #


def test_probe_passes_when_everything_loads(monkeypatch):
    monkeypatch.setattr(s, "load_library_probe", fake_loader())
    monkeypatch.setattr(s, "cuda_device_count", lambda: 2)

    report = s.probe_cuda(platform="win32")

    assert report["usable"] is True
    assert report["kind"] is None
    assert report["device_count"] == 2


def test_probe_names_the_library_it_could_not_load(monkeypatch):
    monkeypatch.setattr(s, "load_library_probe", fake_loader(("cublas64_12.dll",)))

    def never_called():  # pragma: no cover - the driver is not asked
        raise AssertionError(
            "device count must not be queried when a library is missing"
        )

    monkeypatch.setattr(s, "cuda_device_count", never_called)
    report = s.probe_cuda(platform="win32")

    assert report["usable"] is False
    assert report["kind"] == "missing-libs"
    assert "cublas64_12.dll" in report["reason"]
    assert report["missing"] == ["cublas64_12.dll"]


def test_probe_checks_the_posix_names(monkeypatch):
    seen: list[str] = []

    def loader(name: str, platform: str | None = None) -> str | None:
        seen.append(name)
        return None

    monkeypatch.setattr(s, "load_library_probe", loader)
    monkeypatch.setattr(s, "cuda_device_count", lambda: 1)
    s.probe_cuda(platform="linux")

    assert seen == [
        "libcublas.so.12",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn.so.9",
    ]


def test_probe_reports_a_missing_device(monkeypatch):
    monkeypatch.setattr(s, "load_library_probe", fake_loader())
    monkeypatch.setattr(s, "cuda_device_count", lambda: 0)

    report = s.probe_cuda(platform="win32")

    assert report["usable"] is False
    assert report["kind"] == "no-device"
    assert "device" in report["reason"]


def test_probe_reports_a_broken_ctranslate2(monkeypatch):
    monkeypatch.setattr(s, "load_library_probe", fake_loader())

    def explode() -> int:
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    monkeypatch.setattr(s, "cuda_device_count", explode)
    report = s.probe_cuda(platform="win32")

    assert report["usable"] is False
    assert report["kind"] == "no-ctranslate2"
    assert "CTranslate2 could not be loaded" in report["reason"]


@pytest.mark.parametrize(
    ("requested", "usable", "expected"),
    [
        ("cpu", True, ("cpu", False)),
        ("cpu", False, ("cpu", False)),
        ("cuda", True, ("cuda", False)),
        ("cuda", False, ("cuda", True)),
        ("auto", True, ("cuda", False)),
        ("auto", False, ("cpu", False)),
    ],
)
def test_decide_device(requested, usable, expected):
    assert s.decide_device(requested, probe(usable)) == expected


CPU_TYPES = {"float32", "int16", "int8", "int8_float32"}
CUDA_TYPES = {"bfloat16", "float16", "float32", "int8", "int8_float16", "int8_float32"}


def test_device_precisions_follow_the_backend(monkeypatch):
    monkeypatch.setattr(s, "supported_compute_types", lambda device: CPU_TYPES)
    assert s.device_precisions("cpu") == ["int8", "float32"]

    monkeypatch.setattr(s, "supported_compute_types", lambda device: CUDA_TYPES)
    assert s.device_precisions("cuda") == s.COMPUTE_TYPES


def test_device_precisions_survives_a_broken_ctranslate2(monkeypatch):
    def explode(device: str) -> set[str]:
        raise ImportError("no ctranslate2 here")

    monkeypatch.setattr(s, "supported_compute_types", explode)
    assert s.device_precisions("cpu") == s.COMPUTE_TYPES


def test_resolve_precision_keeps_a_supported_request(monkeypatch):
    monkeypatch.setattr(s, "supported_compute_types", lambda device: CUDA_TYPES)
    assert s.resolve_precision("cuda", "float16") == (
        "float16",
        None,
        s.COMPUTE_TYPES,
    )


def test_resolve_precision_swaps_float16_on_cpu(monkeypatch):
    monkeypatch.setattr(s, "supported_compute_types", lambda device: CPU_TYPES)
    precision, notice, usable = s.resolve_precision("cpu", "float16")

    assert precision == "int8"
    assert usable == ["int8", "float32"]
    assert notice and "float16" in notice and "cpu" in notice


def test_resolve_precision_takes_the_first_option_without_int8(monkeypatch):
    monkeypatch.setattr(s, "supported_compute_types", lambda device: {"float32"})
    precision, notice, usable = s.resolve_precision("cpu", "float16")

    assert (precision, usable) == ("float32", ["float32"])
    assert notice


# --------------------------------------------------------------------------- #
# Startup orchestration and messages
# --------------------------------------------------------------------------- #


def test_startup_is_fatal_for_an_explicit_cuda_request(monkeypatch, tmp_path):
    monkeypatch.setattr(s, "platform_family", lambda *a, **k: "win32")
    monkeypatch.setattr(
        s,
        "cuda_library_wiring",
        lambda *a, **k: {
            "dirs": [tmp_path],
            "added_to_path": [],
            "path_error": None,
            "preloaded": [],
        },
    )
    monkeypatch.setattr(s, "probe_cuda", lambda *a, **k: probe(False))

    decision = s.cuda_startup("cuda")

    assert decision["fatal"] is True
    assert decision["device"] == "cuda"
    assert decision["lines"], "a fatal GPU must explain itself"


def test_startup_falls_back_to_cpu_for_auto(monkeypatch, tmp_path):
    monkeypatch.setattr(s, "platform_family", lambda *a, **k: "win32")
    monkeypatch.setattr(
        s,
        "cuda_library_wiring",
        lambda *a, **k: {
            "dirs": [tmp_path],
            "added_to_path": [],
            "path_error": None,
            "preloaded": [],
        },
    )
    monkeypatch.setattr(s, "probe_cuda", lambda *a, **k: probe(False))

    decision = s.cuda_startup("auto")

    assert decision["fatal"] is False
    assert decision["device"] == "cpu"
    assert any("auto" in line for line in decision["lines"])


def test_startup_resolves_auto_to_cuda_when_it_works(monkeypatch, tmp_path):
    monkeypatch.setattr(s, "platform_family", lambda *a, **k: "win32")
    monkeypatch.setattr(
        s,
        "cuda_library_wiring",
        lambda *a, **k: {
            "dirs": [tmp_path],
            "added_to_path": [],
            "path_error": None,
            "preloaded": [],
        },
    )
    monkeypatch.setattr(s, "probe_cuda", lambda *a, **k: probe(True))

    decision = s.cuda_startup("auto")

    assert decision["device"] == "cuda"
    assert decision["fatal"] is False
    assert decision["lines"] == []


def test_startup_announces_what_it_preloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(
        s,
        "cuda_library_wiring",
        lambda *a, **k: {
            "dirs": [tmp_path],
            "added_to_path": [],
            "path_error": None,
            "preloaded": ["libcublas.so.12", "libcudnn.so.9"],
        },
    )
    monkeypatch.setattr(s, "probe_cuda", lambda *a, **k: probe(True))

    decision = s.cuda_startup("cuda")

    assert decision["device"] == "cuda"
    assert any("preloaded 2" in line for line in decision["lines"])


def test_cuda_library_wiring_preloads_the_discovered_dirs_on_posix(
    tmp_path, monkeypatch
):
    root = wheel_tree(tmp_path)
    monkeypatch.setattr(s, "cuda_search_dirs", lambda *a, **k: [root])
    calls: list[list[Path]] = []
    monkeypatch.setattr(
        s,
        "preload_cuda_libraries",
        lambda dirs: calls.append(dirs) or ["libcublas.so.12"],
    )

    wired = s.cuda_library_wiring(platform="linux")

    assert calls == [[root]], "the dirs it discovered, not a second discovery"
    assert wired["preloaded"] == ["libcublas.so.12"]
    assert s.enable_cuda_libraries(platform="linux") == ["libcublas.so.12"]


def test_startup_keeps_the_exception_text_out_of_the_report(monkeypatch, tmp_path):
    """The raw loader error goes to the console, never to a client or the trail."""
    leaky = RuntimeError(r"could not load C:\Program Files\NVIDIA\cublas64_12.dll")
    monkeypatch.setattr(s, "platform_family", lambda *a, **k: "win32")
    monkeypatch.setattr(
        s,
        "cuda_library_wiring",
        lambda *a, **k: {
            "dirs": [tmp_path],
            "added_to_path": [],
            "path_error": None,
            "preloaded": [],
        },
    )
    monkeypatch.setattr(s, "load_library_probe", fake_loader())

    def explode() -> int:
        raise leaky

    monkeypatch.setattr(s, "cuda_device_count", explode)

    decision = s.cuda_startup("cuda")

    assert decision["fatal"] is True
    assert decision["report"]["reason"] == "CTranslate2 could not be loaded"
    assert "detail" not in decision["report"]
    assert "Program Files" not in json.dumps(decision["report"])
    assert "Program Files" in "\n".join(decision["lines"])


def test_main_stops_before_publishing_state_when_cuda_is_fatal(
    monkeypatch, tmp_path, capsys
):
    """The check has to run before ARGS/READY and before any state is written."""
    args = argparse.Namespace(
        **{**s.DEFAULTS, "work_dir": str(tmp_path / "work"), "no_auth": True}
    )
    monkeypatch.setattr(
        s, "resolve_args", lambda: (args, tmp_path / "config.toml", False)
    )
    monkeypatch.setattr(
        s,
        "cuda_startup",
        lambda requested: {
            "device": requested,
            "report": {
                "usable": False,
                "reason": "cublas64_12.dll could not be loaded",
            },
            "fatal": True,
            "lines": ["!  CUDA is not usable on this machine."],
        },
    )
    monkeypatch.setattr(s.uvicorn, "run", lambda *a, **k: pytest.fail("must not serve"))
    sentinel = argparse.Namespace()
    monkeypatch.setattr(s, "ARGS", sentinel)
    monkeypatch.setattr(s, "READY", False)
    # main() assigns these globals before the CUDA check; pin them so teardown
    # leaves the module the way the rest of the suite expects it.
    for name in ("WORK_DIR", "UPLOAD_DIR", "AUDIT", "CUDA", "PRECISIONS"):
        monkeypatch.setattr(s, name, getattr(s, name))

    with pytest.raises(SystemExit) as stopped:
        s.main()

    assert stopped.value.code == 1
    assert s.ARGS is sentinel
    assert s.READY is False
    assert not (tmp_path / "work" / "uploads").exists()
    assert not list(tmp_path.rglob("audit-*.jsonl"))
    assert "not usable" in capsys.readouterr().out


def test_report_carries_no_machine_paths():
    wired = {
        "dirs": [
            Path(
                "/home/somebody/.local/share/uv/env/lib/site-packages/nvidia/cublas/bin"
            )
        ],
        "added_to_path": [],
        "preloaded": [
            "libcublas.so.12",
            "/home/somebody/site-packages/nvidia/cublas/bin",
        ],
    }
    report = s.cuda_report("cuda", "cuda", wired, probe(False))

    assert report["lib_dirs"] == ["nvidia/cublas/bin"]
    assert report["preloaded"] == ["libcublas.so.12", "nvidia/cublas/bin"]
    assert "/home/somebody" not in json.dumps(report)
    assert report["missing"] == ["cublas64_12.dll"]


def test_diagnosis_points_at_the_missing_wheels(tmp_path):
    report = s.cuda_report(
        "cuda", "cuda", {"dirs": [], "added_to_path": [], "preloaded": []}, probe(False)
    )
    text = "\n".join(
        s.cuda_diagnosis(report, {"dirs": [], "added_to_path": [], "preloaded": []})
    )

    assert "CUDA is not usable" in text
    assert "cublas64_12.dll" in text
    assert "No nvidia-* wheel directory" in text
    assert "--device cpu" in text


def test_diagnosis_shows_where_it_looked_when_the_wheels_are_there(tmp_path):
    wired = {
        "dirs": [tmp_path / "nvidia" / "cublas" / "bin"],
        "added_to_path": [],
        "preloaded": [],
    }
    report = s.cuda_report("cuda", "cuda", wired, probe(False))
    text = "\n".join(s.cuda_diagnosis(report, wired))

    assert str(tmp_path / "nvidia" / "cublas" / "bin") in text
    assert "foreign-architecture" in text
    assert "--device cpu" in text


def test_diagnosis_blames_the_driver_when_no_device_is_visible():
    report = s.cuda_report(
        "cuda",
        "cuda",
        {"dirs": [], "added_to_path": [], "preloaded": []},
        probe(False, kind="no-device", missing=[]),
    )
    text = "\n".join(
        s.cuda_diagnosis(report, {"dirs": [], "added_to_path": [], "preloaded": []})
    )

    assert "driver" in text
    assert "--device cpu" in text


def test_fallback_notice_says_it_will_be_slow():
    text = "\n".join(
        s.cuda_fallback_notice(
            s.cuda_report(
                "auto",
                "cpu",
                {"dirs": [], "added_to_path": [], "preloaded": []},
                probe(False),
            )
        )
    )

    assert "auto" in text
    assert "CPU" in text
    assert "slower" in text


# --------------------------------------------------------------------------- #
# Job errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message",
    [
        "Library cublas64_12.dll is not found or cannot be loaded",
        "Could not locate cudnn_ops64_9.dll",
        "DLL load failed while importing _ext",
    ],
)
def test_friendly_error_adds_the_cuda_hint(message):
    text = s.friendly_error(RuntimeError(message))
    assert message in text
    assert "README.md" in text


def test_friendly_error_leaves_unrelated_errors_alone():
    assert s.friendly_error(ValueError("bad beam size")) == "bad beam size"


def test_friendly_error_still_redacts_paths():
    text = s.friendly_error(RuntimeError(f"cannot read {s.WORK_DIR}/uploads/x.wav"))
    assert str(s.WORK_DIR) not in text


def test_redact_strips_absolute_windows_paths():
    text = s.redact(r"could not load C:\Users\joe\cuda\bin\cublas64_12.dll")

    assert "joe" not in text
    assert "<path>" in text


def test_redact_strips_unc_paths():
    assert "share" not in s.redact(r"failed: \\server\share\models\x.dll")


def test_redact_leaves_urls_and_plain_text_alone():
    text = "see https://huggingface.co/x and 2024/01/02 for float16/int8"
    assert s.redact(text) == text


def test_build_opts_refuses_a_precision_the_device_cannot_run(configured, monkeypatch):
    """The advertised list and the accepted list must be the same list."""
    monkeypatch.setattr(s, "PRECISIONS", ["int8", "float32"])
    s.ARGS.allow_precision_choice = True

    with pytest.raises(HTTPException) as caught:
        s.build_opts(
            "base",
            "float16",
            "",
            "true",
            "balanced",
            "",
            "",
            "false",
            "false",
            "false",
            2000,
            400,
        )

    assert caught.value.status_code == 400


# --------------------------------------------------------------------------- #
# POSIX preload
# --------------------------------------------------------------------------- #


def system_library() -> Path | None:
    """A real .so to stand in for libcublas, so the cdll call is exercised."""
    found = ctypes.util.find_library("c") or ctypes.util.find_library("m")
    if not found:
        return None
    if "/" in found:
        return Path(found)
    for directory in (
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/lib64",
        "/usr/lib64",
    ):
        candidate = Path(directory) / found
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only path")
def test_preload_loads_by_absolute_path(tmp_path):
    real = system_library()
    if real is None:  # pragma: no cover - unusual container
        pytest.skip("no system library to stand in for libcublas")

    (tmp_path / "libcublas.so.12").symlink_to(real)

    assert s.preload_cuda_libraries([tmp_path]) == ["libcublas.so.12"]
    assert s._PRELOADED_LIBS, "the handle must be kept alive"


def build_library(directory: Path, name: str) -> bool:
    """Compile a tiny shared object with a real SONAME. False without a compiler."""
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        return False
    source = directory / "soname_probe.c"
    source.write_text("int soname_probe_marker(void) { return 0; }\n", encoding="utf-8")
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            compiler,
            "-shared",
            "-fPIC",
            f"-Wl,-soname,{name}",
            "-o",
            str(directory / name),
            str(source),
        ],
        capture_output=True,
    )
    return result.returncode == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only path")
def test_preload_makes_a_later_dlopen_by_soname_succeed(tmp_path, monkeypatch):
    """The whole POSIX design rests on this, so prove it rather than assume it.

    Nothing puts tmp_path on the loader's search path, so a by-name dlopen can
    only succeed by matching the copy already loaded from an absolute path.
    """
    name = "libfakecuda_soname_probe.so.9"
    if not build_library(tmp_path, name):  # pragma: no cover - no compiler
        pytest.skip("no C compiler to build a probe library")
    monkeypatch.setattr(s, "POSIX_PRELOAD_ORDER", (name,))

    assert s.preload_cuda_libraries([tmp_path]) == [name]

    ctypes.CDLL(name)  # raises OSError if the soname match does not work


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only path")
def test_preload_skips_what_it_cannot_load(tmp_path):
    (tmp_path / "libcublas.so.12").write_text("not a shared object")

    assert s.preload_cuda_libraries([tmp_path]) == []


def test_preload_is_a_noop_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(s, "platform_family", lambda *a, **k: "win32")
    (tmp_path / "libcublas.so.12").write_text("ignored")

    assert s.preload_cuda_libraries([tmp_path]) == []


if __name__ == "__main__":
    # A subprocess so pytest sees a clean process: this module already imported
    # transcribe_server (and through it fastapi/anyio) by the time we get here.
    raise SystemExit(
        subprocess.call(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "pytest", "-q", str(Path(__file__).parent)]
        )
    )
