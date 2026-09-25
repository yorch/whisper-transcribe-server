#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi>=0.110",
#     "uvicorn[standard]>=0.27",
#     "python-multipart>=0.0.9",
#     "faster-whisper>=1.0.3",
#     "numpy>=1.24",
#     # Speaker diarization. The whole reason "who said what" is possible without
#     # pulling PyTorch in: ~15 MB, no transitive dependencies, and its models
#     # come from GitHub releases rather than a gated Hugging Face repo.
#     "sherpa-onnx>=1.13.8",
#     "tomli>=2; python_version < '3.11'",
#     "nvidia-cublas-cu12; sys_platform != 'darwin'",
#     "nvidia-cudnn-cu12>=9,<10; sys_platform != 'darwin'",
# ]
# ///
"""
Local transcription server.

Serves a drag-and-drop page over the LAN and transcribes with faster-whisper on
whatever CUDA GPU is present. One job runs at a time so the GPU is never
oversubscribed; everything else waits in a queue.

    uv run transcribe_server.py
    uv run transcribe_server.py --model medium --port 8765

Access control is on by default: if you don't supply --token, one is generated
at startup and printed with the URL. Pass --no-auth to turn it off deliberately.

Options live in three layers, later wins: defaults, a TOML config file
(~/.transcribe-server/config.toml), command-line flags, then environment
variables. --config points somewhere else.

The server keeps an append-only audit trail of what the web app did (uploads,
exports, deletions, worker lifecycle, refused requests) as daily JSONL files.

Dependencies are declared inline (PEP 723), so there is nothing to install
first and no virtualenv to activate.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import hmac
import importlib.util
import json
import os
import queue
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
import uuid
import wave
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator, MutableMapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple, TypeVar
from urllib.parse import quote

import uvicorn

try:  # tomllib is 3.11+; the inline dependency covers older interpreters
    import tomllib

    def load_toml(text: str) -> dict[str, Any]:
        return tomllib.loads(text)

except ModuleNotFoundError:  # pragma: no cover - 3.10 only
    import tomli  # pyright: ignore[reportMissingImports]

    def load_toml(text: str) -> dict[str, Any]:
        return tomli.loads(text)


from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Binding every interface is the documented default; the Host allowlist and the
# access token are what make that safe, not the bind address itself.
ANY_INTERFACE = "0.0.0.0"  # noqa: S104

MODELS = ["large-v3", "large-v3-turbo", "medium", "small", "base"]
# float16 needs compute capability >= 7.0 (Turing/Ampere/Ada/Hopper/Blackwell).
# Pascal and older should use int8 or float32.
COMPUTE_TYPES = ["float16", "int8_float16", "int8", "float32"]
QUALITIES = {"fast": 1, "balanced": 5, "thorough": 8}  # -> beam_size
RETENTION = ["run", "job", "forever"]

PROMPT_LIMIT = 1000
HOTWORDS_LIMIT = 400

# Speaker count a client may ask for. 0 means "decide from the audio".
DIARIZE_MAX_SPEAKERS = 10

WORK_DIR = Path(
    os.environ.get("TRANSCRIBE_WORK_DIR", Path.home() / ".transcribe-server")
)
UPLOAD_DIR = WORK_DIR / "uploads"

# Set in main(). Requests are refused until then, so importing this module and
# serving `app` directly from an ASGI server fails closed rather than open.
# Set in main(). Requests are refused until then, so importing this module and
# serving `app` directly from an ASGI server fails closed rather than open.
# ARGS is an empty Namespace (not None) so its attributes type-check; READY is
# the authoritative "main() has populated it" flag that guards every use.
ARGS: argparse.Namespace = argparse.Namespace()
READY = False
ALLOWED_HOSTS: set[str] = set()
ALLOWED_SUFFIXES: set[str] = set()  # entries like ".trycloudflare.com"

SAFE_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

# --------------------------------------------------------------------------- #
# Job store
# --------------------------------------------------------------------------- #

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
JOB_QUEUE: queue.Queue[str] = queue.Queue()

# LRU, capped by --model-cache. Uncapped, a client walking the model and
# precision dropdowns would pin every combination in VRAM at once.
_MODEL_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_MODEL_LOCK = threading.Lock()


# Windows drive-absolute and UNC paths. The directory prefixes above only cover
# the home and work directories; anything else (C:\Program Files\...,
# \\share\...) can still reach a client, so catch absolute Windows paths by
# shape. The lookbehind keeps "https://host/x" from looking like a drive. A path
# containing spaces redacts only up to the first space: swallowing to the end of
# the line would eat the rest of the message instead.
_ABSOLUTE_PATH = re.compile(r"""(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\s"']*""")


def redact(text: str, limit: int = 300) -> str:
    """Strip local filesystem paths out of anything shown to a client."""
    for base in (str(UPLOAD_DIR), str(WORK_DIR), str(Path.home())):
        for variant in (base, base.replace("\\", "/")):
            if variant:
                text = text.replace(variant, "<path>")
    return _ABSOLUTE_PATH.sub("<path>", text)[:limit]


# --------------------------------------------------------------------------- #
# CUDA libraries
# --------------------------------------------------------------------------- #
#
# CTranslate2's wheels ship neither cuBLAS nor the real cuDNN 9 libraries. The
# cuBLAS stub inside libctranslate2 calls LoadLibraryA("cublas64_12.dll"), and
# the cudnn64_9.dll it bundles is itself a shim that does the same for
# cudnn_*64_9.dll. A bare LoadLibrary searches only the application directory,
# the system directories and PATH, so the copies the nvidia-* wheels install
# into site-packages stay invisible: nothing about `uv run` puts them on PATH,
# and os.add_dll_directory does not help those calls either (it only applies to
# LoadLibraryEx calls that pass LOAD_LIBRARY_SEARCH_USER_DIRS).
#
# That is why the failure is so late and so opaque: loading a model never calls
# a cuBLAS kernel, so the server starts, the model loads, and the job dies
# minutes later inside the first matmul with "Library cublas64_12.dll is not
# found or cannot be loaded".
#
# Windows: prepend the wheel directories to PATH, which is what the loader
# really consults, and also register them with os.add_dll_directory for
# anything resolved through CPython's extension loader.
# POSIX: the dynamic loader has already read its search path by the time Python
# starts, so PATH is inert. Instead we dlopen the libraries by absolute path
# with RTLD_GLOBAL, after which a dlopen of the same soname resolves to the
# copy already in memory. Only tried when loading by name failed, so a working
# system CUDA install is never disturbed.

# What CTranslate2 asks the loader for by name, so what it can fail on. The
# cuDNN parts cover conv1d, which the Whisper encoder needs; the rest of cuDNN
# ships with them or is pulled in transitively.
CUDA_PROBE_LIBS: dict[str, tuple[str, ...]] = {
    "win32": ("cublas64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"),
    "posix": (
        "libcublas.so.12",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn.so.9",
    ),
}

# POSIX preload order: dependencies before the libraries that need them.
POSIX_PRELOAD_ORDER = (
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn.so.9",
)

CUDA_ERROR_HINT = (
    " The CUDA libraries CTranslate2 needs are not on the loader search path; "
    "see 'When CUDA fails' in README.md."
)

# os.add_dll_directory() hands back a handle that must outlive the call, and
# ctypes.CDLL() does not unload anything either.
_DLL_DIR_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRS: set[str] = set()
_PRELOADED_LIBS: list[Any] = []

# Set in main(). None when the operator asked for CPU, so nothing was probed.
CUDA: dict[str, Any] | None = None

# Also set in main(): the subset of COMPUTE_TYPES the resolved device can run.
# CTranslate2 rejects the rest at model load, so shipping float16 to a CPU box
# would mean a server that looks healthy and fails every single job.
PRECISIONS: list[str] = list(COMPUTE_TYPES)


def platform_family(platform: str | None = None) -> str:
    """'win32' or 'posix' -- coarse enough for every branch below."""
    return "win32" if (platform or sys.platform) == "win32" else "posix"


def short_path(path: Path | str, parts: int = 3) -> str:
    """Trailing path components, for records that must not carry machine paths.

    Backslashes are normalised first so a Windows path is shortened correctly
    even when the check runs on POSIX (and vice versa), which matters because
    this is a privacy guard, not a display nicety.
    """
    chunks = Path(str(path).replace("\\", "/")).parts
    return "/".join(chunks[-parts:])


def safe_lib_name(entry: str) -> str:
    """A soname as it is; a directory as its last few components.

    Deliberately not `Path(entry).is_absolute()`: on Windows a rooted but
    driveless path such as `/home/somebody/site-packages/...` is *not* absolute,
    so a POSIX path reaching a Windows process slipped through this guard
    unreduced. Anything containing a separator is treated as a path.
    """
    if "/" in entry or "\\" in entry:
        return short_path(entry)
    return entry


def _environ() -> MutableMapping[str, str]:
    """The process environment, behind a seam a test can stand a wall in front of."""
    return os.environ


def site_package_dirs() -> list[Path]:
    """Every site-packages-style directory this interpreter could import from.

    sys.path is the ground truth once the wheels are importable, but the
    prefix-derived layouts are checked too so discovery still works before the
    first import and under interpreters that report site-packages oddly.
    """
    candidates: list[Path] = []
    with contextlib.suppress(Exception):  # embedded or exotic interpreters
        import site

        candidates += [Path(p) for p in site.getsitepackages()]
        candidates.append(Path(site.getusersitepackages()))
    candidates += [
        Path(entry)
        for entry in sys.path
        if entry.endswith(("site-packages", "dist-packages"))
    ]
    version = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    for prefix in (Path(sys.prefix), Path(sys.base_prefix)):
        candidates += [
            prefix / "Lib" / "site-packages",  # Windows layout
            prefix / "lib" / version / "site-packages",  # POSIX layout
        ]
    seen: set[str] = set()
    dirs: list[Path] = []
    for path in candidates:
        key = str(path).replace("\\", "/").casefold()
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        dirs.append(path)
    return dirs


def nvidia_lib_dirs(roots: list[Path] | None = None) -> list[Path]:
    """<site-packages>/nvidia/<package>/bin (Windows) or /lib (POSIX)."""
    dirs: list[Path] = []
    for root in site_package_dirs() if roots is None else roots:
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for package in sorted(nvidia.iterdir()):
            if not package.is_dir():
                continue
            for leaf in ("bin", "lib"):
                candidate = package / leaf
                if candidate.is_dir() and candidate not in dirs:
                    dirs.append(candidate)
    return dirs


def cuda_toolkit_dirs(platform: str | None = None) -> list[Path]:
    """A real CUDA 12 toolkit, if one is installed. Ranks below the wheels."""
    dirs: list[Path] = []
    leaf = "bin" if platform_family(platform) == "win32" else "lib64"
    for name in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(name)
        if root:
            dirs.append(Path(root) / leaf)
    if platform_family(platform) == "win32":
        toolkit = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if toolkit.is_dir():
            dirs += sorted(
                entry / "bin" for entry in toolkit.glob("v12*") if entry.is_dir()
            )
    return dirs


def spec_lib_dirs() -> list[Path]:
    """Where the nvidia-* wheels say they are: the authoritative import location.

    find_spec resolves the package the interpreter would actually import, which
    the directory scan below can only guess at, so it is checked first and
    cannot pick up a copy that belongs to some other environment.
    """
    dirs: list[Path] = []
    for module in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            continue
        locations = getattr(spec, "submodule_search_locations", None)
        if not locations:
            continue
        directory = Path(next(iter(locations)))
        if directory.is_dir() and directory not in dirs:
            dirs.append(directory)
    return dirs


def cuda_search_dirs(
    extra: list[Path] | None = None, platform: str | None = None
) -> list[Path]:
    """Where the CUDA runtime libraries may live, best candidate first.

    The wheels rank above a system toolkit: they are what the inline metadata
    pins, so a machine that has both should run the versions this script
    declares.
    """
    candidates = (
        spec_lib_dirs()
        + nvidia_lib_dirs()
        + cuda_toolkit_dirs(platform)
        + list(extra or [])
    )
    present: list[Path] = []
    for path in candidates:
        if path.is_dir() and path not in present:
            present.append(path)
    return present


def cuda_library_wiring(
    extra: list[Path] | None = None, platform: str | None = None
) -> dict[str, Any]:
    """Make the CUDA libraries the wheels bring visible to the loader, and say
    what it did.

    Windows: prepend the directories to PATH, which is what the loader consults
    for a bare LoadLibrary, and register them with os.add_dll_directory as
    well. POSIX: preload them by absolute path with RTLD_GLOBAL, after which a
    dlopen by soname resolves against the copy already in memory, because the
    dynamic loader read its search path before Python started.

    Idempotent: PATH keeps its original entries behind the new ones, and a
    directory is never registered twice.
    """
    dirs = cuda_search_dirs(extra, platform)
    added_to_path: list[str] = []
    path_error: str | None = None
    if platform_family(platform) == "win32":
        env = _environ()
        current = env.get("PATH", "")
        have = {
            entry.rstrip("\\/").casefold()
            for entry in current.split(os.pathsep)
            if entry
        }
        missing = [d for d in dirs if str(d).rstrip("\\/").casefold() not in have]
        if missing:
            parts = [str(d) for d in missing] + ([current] if current else [])
            try:
                env["PATH"] = os.pathsep.join(parts)
            except OSError as exc:  # a PATH at the 32k limit, or a locked block
                path_error = str(exc)
            else:
                added_to_path = [str(d) for d in missing]
        registered: list[str] = []
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            for directory in dirs:
                key = str(directory).casefold()
                if key in _REGISTERED_DLL_DIRS:
                    registered.append(str(directory))
                    continue
                try:
                    _DLL_DIR_HANDLES.append(add_dll_directory(str(directory)))
                except OSError:
                    continue
                _REGISTERED_DLL_DIRS.add(key)
                registered.append(str(directory))
        preloaded = list(dict.fromkeys(added_to_path + registered))
    else:
        preloaded = preload_cuda_libraries(dirs)
    return {
        "dirs": dirs,
        "added_to_path": added_to_path,
        "path_error": path_error,
        "preloaded": preloaded,
    }


def enable_cuda_libraries(
    extra: list[Path] | None = None, platform: str | None = None
) -> list[str]:
    """Make the CUDA runtime that ships in the nvidia-* wheels loadable.

    ctranslate2 resolves libcublas/libcudnn lazily, when it starts encoding, and
    the wheels install them somewhere the loader does not search. Without this,
    the model loads happily and then every job fails with "Library
    libcublas.so.12 is not found or cannot be loaded" — the failure mode that
    makes a broken GPU look healthy at startup.

    Returns what it wired: sonames on POSIX, directories on Windows, empty when
    no CUDA runtime was found at all.
    """
    return cuda_library_wiring(extra, platform)["preloaded"]


def preload_cuda_libraries(dirs: list[Path]) -> list[str]:
    """POSIX only: absolute-path load so a later dlopen by soname hits memory.

    Dependency order matters and is fixed by POSIX_PRELOAD_ORDER: a bare dlopen
    of an absolute path resolves that object's own dependencies through the
    loader's search path, which does not include the directory it was loaded
    from. Best effort per library; the probe afterwards decides.
    """
    loaded: list[str] = []
    if platform_family() == "win32":
        return loaded
    for name in POSIX_PRELOAD_ORDER:
        path = next((d / name for d in dirs if (d / name).is_file()), None)
        if path is None:
            continue
        try:
            _PRELOADED_LIBS.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
        except OSError:
            continue
        loaded.append(name)
    return loaded


def _load_library_a(name: str) -> str | None:
    """The exact call CTranslate2 makes, so a green probe means what it says.

    ctypes' own loader is not the same search: WinDLL/CDLL pass their own flags
    to LoadLibraryEx, while the fix depends specifically on the bare
    LoadLibraryA standard search order -- the one that includes PATH, which is
    where this module puts the wheel directories.
    """
    # WinDLL/get_last_error only exist in the Windows build of ctypes, and this
    # is type-checked on Linux as well.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # pyright: ignore[reportAttributeAccessIssue]
    kernel32.LoadLibraryA.argtypes = [ctypes.c_char_p]
    kernel32.LoadLibraryA.restype = ctypes.c_void_p
    handle = kernel32.LoadLibraryA(name.encode("ascii"))
    if handle:
        return None
    error = ctypes.get_last_error()  # pyright: ignore[reportAttributeAccessIssue]
    return f"LoadLibraryA({name}) failed (error {error})"


def load_library_probe(name: str, platform: str | None = None) -> str | None:
    """None if the loader can find `name`, else the loader's complaint."""
    if platform_family(platform) == "win32":
        return _load_library_a(name)
    try:
        ctypes.CDLL(name)
    except OSError as exc:
        return str(exc)
    return None


def cuda_device_count() -> int:
    import ctranslate2

    return int(ctranslate2.get_cuda_device_count())


def supported_compute_types(device: str) -> set[str]:
    import ctranslate2

    return set(ctranslate2.get_supported_compute_types(device))


def device_precisions(device: str) -> list[str]:
    """The precisions this device can run, in the UI's order.

    Asked of CTranslate2 rather than hardcoded: CPU has no float16, Pascal has
    no fast one, and the library is the authority on both.
    """
    try:
        supported = supported_compute_types(device)
    except Exception:  # noqa: BLE001 - never fail startup over a dropdown
        return list(COMPUTE_TYPES)
    usable = [kind for kind in COMPUTE_TYPES if kind in supported]
    return usable or list(COMPUTE_TYPES)


def resolve_precision(device: str, requested: str) -> tuple[str, str | None, list[str]]:
    """(precision to run on, a notice if it had to change, device's choices)."""
    usable = device_precisions(device)
    if requested in usable:
        return requested, None, usable
    fallback = "int8" if "int8" in usable else usable[0]
    notice = f"!  {requested} is not available on {device}; using {fallback}."
    return fallback, notice, usable


def probe_cuda(platform: str | None = None) -> dict[str, Any]:
    """Can this machine actually run CTranslate2 on the GPU?

    Cheap on purpose: it forces the libraries that failures come from and asks
    the driver for a device. It cannot catch every way a GPU can misbehave --
    only a real transcription can -- but it catches the whole "library not
    found" family, which is the one that otherwise shows up mid-job.
    """
    missing = [
        name
        for name in CUDA_PROBE_LIBS[platform_family(platform)]
        if load_library_probe(name, platform)
    ]
    report: dict[str, Any] = {
        "usable": False,
        "kind": None,
        "reason": None,
        "missing": missing,
        "device_count": 0,
    }
    if missing:
        report["kind"] = "missing-libs"
        report["reason"] = f"{missing[0]} could not be loaded"
        return report
    try:
        count = cuda_device_count()
    except Exception as exc:  # noqa: BLE001 - any import or driver failure counts
        report["kind"] = "no-ctranslate2"
        # The reason is a fixed sentence because this one is published: the
        # exception text can name a path outside the home directory, which
        # redact() covers only by shape. cuda_diagnosis() prints the detail,
        # which never leaves the console.
        report["reason"] = "CTranslate2 could not be loaded"
        report["detail"] = str(exc)[:400]
        return report
    report["device_count"] = count
    if count < 1:
        report["kind"] = "no-device"
        report["reason"] = "no CUDA device is visible to the driver"
        return report
    report["usable"] = True
    return report


def decide_device(requested: str, probe: dict[str, Any]) -> tuple[str, bool]:
    """(device to run on, whether an unusable GPU should stop startup).

    `auto` means "use the GPU only if it demonstrably works": CTranslate2's own
    auto only counts devices, and a machine with a driver and a missing cuBLAS
    counts them happily.
    """
    if requested == "cpu":
        return "cpu", False
    if probe["usable"]:
        return "cuda", False
    if requested == "auto":
        return "cpu", False
    return requested, True


def cuda_report(
    requested: str, device: str, wired: dict[str, Any], probe: dict[str, Any]
) -> dict[str, Any]:
    """Client- and audit-safe view: short paths in, machine paths out."""
    return {
        "requested_device": requested,
        "device": device,
        "usable": bool(probe["usable"]),
        "kind": probe["kind"],
        "reason": probe["reason"],
        "missing": list(probe["missing"]),
        "device_count": probe["device_count"],
        "lib_dirs": [short_path(d) for d in wired["dirs"]],
        "preloaded": [safe_lib_name(name) for name in wired["preloaded"]],
    }


# --------------------------------------------------------------------------- #
# Device details for the readout
# --------------------------------------------------------------------------- #
#
# nvidia-smi is the only route to a readable GPU name. CTranslate2 exposes a
# device count and the compute types it supports, and nothing else; the pretty
# name used to come from torch, which is the dependency this project does not
# have. nvidia-smi ships with the driver, so it is present wherever CUDA really
# works, but it is not always on PATH -- WSL keeps it in /usr/lib/wsl/lib, which
# is where this was noticed -- hence the candidates.

GPU_QUERY = "index,name,memory.total,memory.used,driver_version,compute_cap"
APPS_QUERY = "pid,used_memory"

NVIDIA_SMI_CANDIDATES = (
    "/usr/lib/wsl/lib/nvidia-smi",
    "/usr/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
    "C:\\Windows\\System32\\nvidia-smi.exe",
    "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
)

# platform_family() normalises to posix/darwin/win32, which is right for
# branching on and reads badly in a tooltip. "Python 3.12.2 on Linux" is the
# sentence this is for.
PLATFORM_LABELS = {"win32": "Windows", "darwin": "macOS", "linux": "Linux"}

# Refreshing the memory numbers means spawning a process (~50 ms) and the UI
# polls status every ~1.2 s, so it is rate-limited to this. A machine with no
# nvidia-smi never spawns anything at all.
DEVICE_REFRESH_SECONDS = 10

# Public fields only. The path to nvidia-smi lives in its own global so that it
# cannot leak into a response by being in here.
DEVICE_INFO: dict[str, Any] = {}
_NVIDIA_SMI: str | None = None
_DEVICE_LOCK = threading.Lock()
_DEVICE_AT = 0.0


def find_nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in NVIDIA_SMI_CANDIDATES:
        with contextlib.suppress(OSError):
            if Path(candidate).is_file():
                return candidate
    return None


def nvidia_smi_rows(exe: str, domain: str, fields: str) -> list[list[str]] | None:
    """One `nvidia-smi --query-*`, split into CSV rows.

    Every failure returns None rather than raising: no driver, an nvidia-smi too
    old to know a field, a wedged driver. None of that is worth failing a status
    poll over, and the caller already has a degraded view to fall back on.
    """
    try:
        result = subprocess.run(  # noqa: S603 - literal argv, path from our own list
            [exe, f"--query-{domain}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return [
        line.split(",") for line in result.stdout.strip().splitlines() if line.strip()
    ]


def leading_int(value: str) -> int | None:
    """The number at the front of an nvidia-smi cell, or None for "N/A"."""
    match = re.match(r"\s*(\d+)", value or "")
    return int(match.group(1)) if match else None


def parse_gpu_rows(rows: list[list[str]]) -> dict[str, Any]:
    """Device 0's name, memory and driver, from nvidia-smi CSV.

    Pure, so it can be tested against canned output -- including the parts that
    differ between driver versions: "N/A" for a field the driver declines to
    report, and whitespace around the commas.

    Device 0 on purpose: CTranslate2 always takes device 0, so the first card is
    the only one whose memory means anything here.
    """
    for row in rows:
        if len(row) < 6 or row[0].strip() not in ("0", ""):
            continue
        total = leading_int(row[2])
        used = leading_int(row[3])
        info: dict[str, Any] = {
            "name": row[1].strip() or None,
            "driver": row[4].strip() or None,
            "compute_cap": row[5].strip() or None,
            "vram_total_mb": total,
            "vram_used_mb": used,
        }
        if total is not None and used is not None:
            info["vram_free_mb"] = total - used
        return info
    return {}


def parse_app_rows(rows: list[list[str]], pid: int) -> int | None:
    """How much VRAM `pid` holds, or None when the driver will not say.

    Used memory reads "N/A" on some drivers even when the process is listed,
    which is why this is None rather than 0: zero would be the claim "this
    server is using no VRAM", which is a different and wrong statement.
    """
    for row in rows:
        if len(row) < 2 or leading_int(row[0]) != pid:
            continue
        return leading_int(row[1])
    return None


def probe_device_details() -> dict[str, Any]:
    """The interpreter, plus whatever the driver will say about device 0."""
    global _NVIDIA_SMI
    info: dict[str, Any] = {
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}"
        ),
        "platform": PLATFORM_LABELS.get(sys.platform, sys.platform),
    }
    _NVIDIA_SMI = find_nvidia_smi()
    if not _NVIDIA_SMI:
        return info
    rows = nvidia_smi_rows(_NVIDIA_SMI, "gpu", GPU_QUERY)
    if rows:
        info["count"] = len(rows)
        info.update(parse_gpu_rows(rows))
    apps = nvidia_smi_rows(_NVIDIA_SMI, "compute-apps", APPS_QUERY)
    if apps is not None:
        info["process_mb"] = parse_app_rows(apps, os.getpid())
    return info


def refresh_device_memory() -> None:
    """Re-read the numbers that move, leaving the rest of DEVICE_INFO alone."""
    if not _NVIDIA_SMI:
        return
    rows = nvidia_smi_rows(_NVIDIA_SMI, "gpu", GPU_QUERY)
    if rows:
        DEVICE_INFO.update(parse_gpu_rows(rows))
        DEVICE_INFO["count"] = len(rows)
    apps = nvidia_smi_rows(_NVIDIA_SMI, "compute-apps", APPS_QUERY)
    if apps is not None:
        DEVICE_INFO["process_mb"] = parse_app_rows(apps, os.getpid())


def device_details(refresh_after: float = DEVICE_REFRESH_SECONDS) -> dict[str, Any]:
    """DEVICE_INFO, with the memory fields re-read at most every `refresh_after`.

    Rate-limited because a refresh spawns a process: doing it per poll would be
    ~50 ms of work every 1.2 s to redraw a number that barely moves. The first
    call does the whole probe, so the startup banner and the first status poll
    share one answer.
    """
    global _DEVICE_AT
    if DEVICE_INFO and time.time() - _DEVICE_AT < refresh_after:
        return DEVICE_INFO
    with _DEVICE_LOCK:
        # Re-check inside the lock: concurrent polls must not each spawn a
        # process because they all saw a stale timestamp.
        if DEVICE_INFO and time.time() - _DEVICE_AT < refresh_after:
            return DEVICE_INFO
        if DEVICE_INFO:
            refresh_device_memory()
        else:
            DEVICE_INFO.update(probe_device_details())
        _DEVICE_AT = time.time()
    return DEVICE_INFO


def gpu_cell(info: dict[str, Any]) -> str | None:
    """What the Device cell shows: a name if we have one, else a count."""
    if info.get("name"):
        return str(info["name"])
    count = info.get("count")
    if count is None and CUDA:
        count = CUDA.get("device_count")
    return f"{count} CUDA device(s)" if count else None


def device_banner(info: dict[str, Any]) -> str | None:
    """The console's one-line version, or None when there is nothing to add."""
    if not info.get("name"):
        return None
    parts = [str(info["name"])]
    if info.get("vram_total_mb"):
        parts.append(f"{info['vram_total_mb'] / 1024:.1f} GB")
    if info.get("vram_used_mb") is not None:
        parts.append(f"{info['vram_used_mb'] / 1024:.1f} GB used")
    if info.get("driver"):
        parts.append(f"driver {info['driver']}")
    if info.get("compute_cap"):
        parts.append(f"compute {info['compute_cap']}")
    return ", ".join(parts)


def cuda_diagnosis(report: dict[str, Any], wired: dict[str, Any]) -> list[str]:
    """Console-only explanation of an unusable GPU, with real paths."""
    lines = ["!  CUDA is not usable on this machine.", f"   {report['reason']}."]
    if report.get("detail"):
        lines.append(f"   {report['detail']}")
    dirs = [str(d) for d in wired["dirs"]]
    if dirs:
        lines.append("   CUDA libraries found in:")
        lines += [f"     {d}" for d in dirs]
    if wired.get("added_to_path"):
        lines.append("   These were prepended to PATH for this process.")
    if wired.get("path_error"):
        lines.append(f"   PATH could not be extended: {wired['path_error']}")
    if report["missing"]:
        lines.append("   Missing: " + ", ".join(report["missing"]))
    lines.append("")
    if report["kind"] == "no-device":
        lines.append("   The libraries load, but the driver reports no GPU. Check the")
        lines.append("   NVIDIA driver is installed and supports CUDA 12 (>= 525).")
        lines.append("   Fix:  uv run transcribe_server.py --device cpu")
        return lines
    if dirs:
        lines.append(
            "   The libraries are there but the loader refused them, which usually"
        )
        lines.append("   means a truncated or foreign-architecture wheel.")
        lines.append("   Fix:  uv cache clean && uv run transcribe_server.py")
    else:
        lines.append(
            "   No nvidia-* wheel directory exists for this interpreter, so there"
        )
        lines.append(
            "   was nothing to wire up. This script declares those wheels inline,"
        )
        lines.append("   so re-resolving them is normally all it takes.")
        lines.append("   Fix:  uv run transcribe_server.py")
    lines.append("   Or:   uv run transcribe_server.py --device cpu")
    return lines


def cuda_fallback_notice(report: dict[str, Any]) -> list[str]:
    return [
        f"!  CUDA is not usable ({report['reason']}); --device auto runs on CPU.",
        "   Transcription will be much slower. Run with --device cuda for the details.",
    ]


def cuda_startup(requested: str) -> dict[str, Any]:
    """Wire the loader, verify the GPU, then decide. Printing stays in main()."""
    wiring = cuda_library_wiring()
    probe = probe_cuda()
    device, fatal = decide_device(requested, probe)
    report = cuda_report(requested, device, wiring, probe)
    if fatal:
        # Console view: the safe report plus the raw exception text, which is
        # deliberately not part of the report itself.
        lines = cuda_diagnosis({**report, "detail": probe.get("detail")}, wiring)
    elif not probe["usable"]:
        lines = cuda_fallback_notice(report)
    else:
        lines = []
    if wiring["preloaded"]:
        lines.append(
            f"CUDA libs  preloaded {len(wiring['preloaded'])} librar(y/ies) from the wheels"
        )
    return {"device": device, "report": report, "fatal": fatal, "lines": lines}


def friendly_error(exc: Exception) -> str:
    """A job error a client can see: redacted, plus a hint when CUDA is to blame."""
    text = redact(str(exc))
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "is not found or cannot be loaded",
            "cublas",
            "cudnn",
            "dll load failed",
        )
    ):
        return text + CUDA_ERROR_HINT
    return text


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #


def digest(value: str) -> str | None:
    """Hash for correlating repeated prompts without storing them in the main log."""
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# Fields the trail assigns itself. They are dropped from caller fields rather
# than merged: `ts`/`event` would be overwritten, and a caller passing `carry`
# or `lost` could poison the chain or forge an attestation.
RESERVED_FIELDS = frozenset(
    {"ts", "event", "seq", "prev", "chain", "carry", "chain_reset", "lost"}
)


class AuditPage(NamedTuple):
    """One page of a day file, with a front-anchored cursor.

    `cursor` is the line number (1-based, counting non-empty lines) of the
    oldest record in the page. Line numbers do not move when new records are
    appended, so handing it back as `before_line` cannot duplicate or skip a
    record the way an offset into the newest-first window does.
    """

    lines: list[str]
    total: int
    cursor: int | None
    has_more: bool


class _DayLines(NamedTuple):
    """Identity, size, mtime and non-empty line count of one day file.

    Identity, size and mtime are what make a cached count safe to trust: a
    replaced, truncated, externally grown or rewritten-in-place file no longer
    matches and is recounted rather than served stale.
    """

    dev: int
    ino: int
    size: int
    mtime_ns: int
    lines: int


class _DayStats:
    """A folded aggregate for one day, and how far into the file it reaches.

    Mutable because the current day grows: `size` is the offset just past the
    last complete line folded, so the next request reads only what was appended.
    """

    __slots__ = ("dev", "ino", "size", "stats")

    def __init__(self, dev: int, ino: int, size: int, stats: dict[str, Any]) -> None:
        self.dev = dev
        self.ino = ino
        self.size = size
        self.stats = stats


class AuditLog:
    """Append-only JSONL trail, one file per UTC day, newest events last.

    Every write is best-effort: a full disk or a locked file must never turn
    into a failed transcription, so I/O errors degrade to a single warning on
    stderr and the request carries on.

    Prompt and hotword text is deliberately kept out of the main log (only
    lengths and hashes go in). The full text lands in prompts/<job_id>.json,
    which can be expired and permissioned separately.
    """

    def __init__(
        self,
        directory: Path,
        enabled: bool = True,
        retain_days: int = 30,
        prompts: bool = True,
        max_mb: int = 0,
        max_sidecars: int = 0,
    ) -> None:
        self.dir = Path(directory)
        self.prompts_dir = self.dir / "prompts"
        self.enabled = enabled
        self.retain_days = max(0, retain_days)
        self.store_prompts = prompts
        self.max_mb = max(0, max_mb)
        self.max_bytes = self.max_mb * 1024 * 1024
        self.max_sidecars = max(0, max_sidecars)
        self._lock = threading.Lock()
        # Sidecars get their own lock: a read-modify-write (a rename) must not
        # interleave with the plain write of a prompt, and the day-file lock is
        # held across a rotation plus a retention sweep.
        self._sidecar_lock = threading.Lock()
        self._fh: Any | None = None
        self._day: str | None = None
        self._warned: set[tuple[str, str]] = set()
        self.last_error: str | None = None
        # Events dropped since the last successful write, and for the life of
        # the process. A failed write is otherwise invisible: the trail simply
        # has less in it, and an empty-looking day is indistinguishable from a
        # broken sink. The next record that does land attests to the gap.
        self.lost = 0
        self.lost_total = 0
        # False while the file may end mid-line, after a write died and the
        # rollback could not cut it back. The next append closes the line first.
        self._clean = True
        # Set by the byte cap (see emit): the day file hit its ceiling and is
        # deliberately no longer taking events.
        self.full = False
        # Chain state for the day the handle is open on. `_seq` numbers records
        # within the day; `_chain` is the previous record's hash; `_carry` is
        # the previous day's final hash, stamped on the first record of a day so
        # the two days can still be tied together; `_chain_reset` says the
        # resume could not link to what was already on disk.
        self._seq = 0
        self._chain: str | None = None
        self._carry: str | None = None
        self._chain_reset = False
        self._bytes = 0
        self.suppressed = 0
        # Sidecar write failures. They are a separate counter because they are a
        # separate store: a lost prompt is not a lost day-file event, but it is
        # still text silently dropped, so it must set `degraded` too.
        self.sidecar_lost = 0
        self.sidecar_lost_total = 0
        # This run's totals, folded as events are emitted. See Stats.
        self.session = Stats()
        # Non-empty line counts per day, so an unfiltered page is a tail read
        # rather than a full scan. The writer keeps the open day's entry in step
        # with _bytes, so the UI's 5 s poll is always a cache hit.
        self._day_lines: dict[str, _DayLines] = {}
        # Folded aggregates per day. A past day is also written beside the trail
        # (stats-<day>.json) so it is folded once ever rather than once per
        # restart; today never is, because it changes with every append.
        self._day_stats: dict[str, _DayStats] = {}
        self._stats_lock = threading.Lock()

    def degraded(self) -> bool:
        """True while the trail is not keeping up: day-file events are being
        dropped or capped, or sidecar writes are failing. A past failure that a
        later record already attested to is not degraded -- the gap is in the
        file."""
        return bool(self.lost) or self.full or bool(self.sidecar_lost)

    # -- writing ---------------------------------------------------------- #

    # Bytes per backwards read. Bounds memory for ordinary records; a single
    # record longer than this is held whole in `carry`, so the real bound is a
    # chunk plus the largest record.
    REVERSE_CHUNK = 64 * 1024
    # Day files whose line count is remembered. Entries are tiny and one day is
    # added per day, but `retain_days = 0` never prunes the files, so this map
    # must not grow with them.
    MAX_DAY_CACHE = 256

    @staticmethod
    def _chain_hash(record: dict[str, Any]) -> str:
        """One record's link value: a hash of it, minus its own hash.

        Canonical form (sorted keys, compact separators) so a verifier can
        rebuild the same bytes from a parsed line. `prev` is inside the hashed
        body, which is what makes this a chain and not a pile of checksums.
        """
        body = {k: v for k, v in record.items() if k != "chain"}
        return hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def _rotate(self, day: str) -> Any:
        if self._fh is not None:
            self._fh.close()
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"audit-{day}.jsonl"
        self._fh = path.open("a", encoding="utf-8", buffering=1)
        for target in (self.dir, path):
            # A no-op for ACLs on Windows; best effort elsewhere.
            with contextlib.suppress(OSError):
                os.chmod(target, 0o700 if target.is_dir() else 0o600)
        self._resume(day, path)
        self.full = False
        self.suppressed = 0
        try:
            self._bytes = path.stat().st_size
        except OSError:
            self._bytes = 0
        # Retention also runs on rotation: a process that stays up for weeks
        # would otherwise never prune again after startup.
        self.prune()
        return self._fh

    def _resume(self, day: str, path: Path) -> None:
        """Recover chain state for a day that may already hold records.

        A restart mid-day must continue the day's chain, not begin a second one.
        A genuinely new day carries the previous day's final hash forward as
        `carry`. A file whose tail cannot be read -- a torn write, or records
        from before the chain existed -- starts fresh and marks its first record
        so the discontinuity is stated rather than implied.
        """
        tail = self._tail_chain(path)
        self._carry = None
        self._chain_reset = False
        if tail is not None:
            self._seq, self._chain = tail
            self._clean = True
            return
        self._seq = 0
        self._chain = None
        try:
            existing = path.stat().st_size > 0
        except OSError:
            existing = False
        # A non-empty file whose tail did not parse may end mid-record (a crash,
        # or a rollback that failed). The next append must not concatenate onto
        # it, or the new record becomes unreadable too -- including the `lost`
        # count it carries.
        self._clean = not existing
        if existing:
            self._chain_reset = True
        else:
            self._carry = self._previous_head(day)[1]

    @staticmethod
    def _tail_chain(path: Path) -> tuple[int, str] | None:
        """(seq, chain) of a file's last record, or None if it has none.

        Reads only the tail: the file can be large, and this runs once per day
        and once per restart. A torn final line fails to parse, which is exactly
        the signal _resume needs to reset the chain.
        """
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size <= 0:
            return None
        window = min(size, 262144)
        lines: list[str] = []
        while True:
            try:
                with path.open("rb") as fh:
                    fh.seek(size - window)
                    chunk = fh.read(window)
            except OSError:
                return None
            lines = chunk.decode("utf-8", errors="replace").splitlines()
            if size <= window:
                break  # the whole file is in view
            if len(lines) > 1:
                lines = lines[1:]  # the first is a partial line
                break
            # One line and the file is bigger than the window: the last record
            # may be longer than what we read. Widen rather than mistake a
            # complete record for a torn tail.
            wider = min(size, window * 4)
            if wider == window:
                break
            window = wider
        for raw in reversed(lines):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except ValueError:
                return None
            seq, chain = rec.get("seq"), rec.get("chain")
            if isinstance(seq, int) and isinstance(chain, str):
                return seq, chain
            return None
        return None

    def _previous_head(self, day: str) -> tuple[bool, str | None]:
        """(an earlier day file exists, its final hash if it could be read).

        The two are separate because they mean different things: no earlier file
        (nothing to compare against) and an earlier file whose tail will not
        parse (cannot compare) are both indeterminate, while a readable head
        that disagrees with this day's `carry` is a real break. Only the nearest
        earlier day is considered -- skipping past an unreadable one would carry
        a link that never existed.
        """
        for path in sorted(
            self.dir.glob("audit-*.jsonl"), key=lambda p: p.stem, reverse=True
        ):
            other = path.stem[len("audit-") :]
            if other >= day:
                continue
            tail = self._tail_chain(path)
            return True, (tail[1] if tail is not None else None)
        return False, None

    def _append(self, fh: Any, record: dict[str, Any]) -> None:
        """Number, link and write one record. The caller holds `_lock`.

        Nothing is committed until the write returns: a failed append must
        leave `_seq` and `_chain` exactly where the last good record left them,
        or the next record would carry a sequence gap its own verifier reads as
        tampering.
        """
        seq = self._seq + 1
        record["seq"] = seq
        record["prev"] = self._chain
        if seq == 1 and self._carry is not None:
            record["carry"] = self._carry
        if seq == 1 and self._chain_reset:
            record["chain_reset"] = True
        record["chain"] = self._chain_hash(record)
        line = json.dumps(record, ensure_ascii=False, default=str)
        # Terminate a line the previous write left open, so the damage stays on
        # its own line and this record -- with whatever `lost` it attests -- is
        # still readable. Usually `_clean` is true and this is nothing.
        blob = ("" if self._clean else "\n") + line + "\n"
        fh.write(blob)
        fh.flush()
        self._seq = seq
        self._chain = record["chain"]
        self._chain_reset = False
        self._clean = True
        self._bytes += len(blob.encode("utf-8"))
        # Keep the open day's line count in step, so a read never rescans it.
        # Only after the write returns, like every other commit here.
        if self._day is not None:
            entry = self._day_lines.get(self._day)
            if entry is not None:
                # Refresh from the handle: our own append moved the mtime, and a
                # stale mtime would make the next read recount the whole day.
                try:
                    st = os.fstat(fh.fileno())
                except OSError:
                    self._day_lines.pop(self._day, None)  # cannot vouch for it
                else:
                    self._day_lines[self._day] = _DayLines(
                        st.st_dev,
                        st.st_ino,
                        st.st_size,
                        st.st_mtime_ns,
                        entry.lines + 1,
                    )

    def _rollback(self) -> None:
        """Cut a failed append back to the last good length.

        On success `_clean` is left exactly as it was before the append, because
        truncating to `_bytes` restores that state byte for byte -- including a
        file that already ended mid-record from an earlier failure. Declaring it
        clean there is how a torn fragment gets inherited and the next record's
        `lost` attestation gets buried.

        The caller holds `_lock`, so no other thread can append onto the torn
        bytes between the failed write and this cut.
        """
        fh = self._fh
        if fh is None:
            self._clean = False
            return
        # A closed or stub handle raises ValueError/AttributeError, and a full
        # disk raises OSError: none of them may escape a best-effort write path.
        try:
            fh.truncate(self._bytes)
            fh.flush()
        except (OSError, ValueError, AttributeError):
            # Could not cut it back at all: the torn bytes are still there, so
            # the next append must close the line off first.
            self._clean = False

    def _warn_once(self, what: str, exc: BaseException) -> None:
        """Warn once per (path, exception type), not once per process.

        A single latch would let the first failure silence every later and
        different failure, which is exactly when an operator needs to hear
        about it.
        """
        self.last_error = f"{what}: {type(exc).__name__}: {exc}"[:300]
        key = (what, type(exc).__name__)
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"!  Audit {what} failed ({exc}); continuing without it")

    def emit(self, event: str, force: bool = False, **fields: Any) -> None:
        # Before the `enabled` check on purpose: the session view counts what
        # this process did, which under --no-audit is the only record there is.
        self.session.record(event, fields)
        if not self.enabled and not force:
            return
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "event": event,
        }
        record.update(
            {
                k: v
                for k, v in fields.items()
                if v is not None and k not in RESERVED_FIELDS
            }
        )
        try:
            with self._lock:
                try:
                    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if self._day is not None and day < self._day:
                        day = self._day  # never rotate backwards on a boundary race
                    if self._day != day or self._fh is None:
                        fh = self._rotate(day)
                        self._day = day
                    else:
                        fh = self._fh
                    # Read under the lock: a concurrent emit must not change the
                    # count between reading it and writing what attests to it.
                    if self.lost:
                        record["lost"] = self.lost
                    if self.max_bytes and self._bytes >= self.max_bytes:
                        # The day file is full. Say so once, in the file, then
                        # stop taking events: a bounded trail that states its own
                        # gap beats a full disk, and the cap resets at the next
                        # day.
                        if self.full:
                            self.suppressed += 1
                            return
                        self.full = True
                        marker: dict[str, Any] = {
                            "ts": record["ts"],
                            "event": "audit.full",
                            "max_mb": self.max_mb,
                            "bytes": self._bytes,
                        }
                        if self.lost:
                            marker["lost"] = self.lost
                        self._append(fh, marker)
                        # The marker is written through _append, not emit, so it
                        # is folded here; `lost` is left out because the event
                        # above already carried it.
                        self.session.record(
                            "audit.full",
                            {"max_mb": self.max_mb, "bytes": marker["bytes"]},
                        )
                        self.suppressed += 1
                        self.lost = 0
                        return
                    self._append(fh, record)
                    self.lost = 0
                except Exception:
                    # Roll back while the lock is still held. Released first, a
                    # concurrent emit would append onto the torn fragment and
                    # this cut would then slice that record in half.
                    self.lost += 1
                    self.lost_total += 1
                    self._rollback()
                    raise
        except Exception as exc:  # noqa: BLE001
            self._warn_once("write", exc)

    def write_prompt(self, job_id: str, payload: dict[str, Any]) -> None:
        if not (self.enabled and self.store_prompts):
            return
        if not SAFE_JOB_ID.fullmatch(job_id):
            return
        with self._sidecar_lock:
            self._write_prompt_locked(job_id, payload)
        self._cap_sidecars()

    def amend_prompt(
        self,
        job_id: str,
        default: dict[str, Any],
        changes: dict[str, Any],
    ) -> None:
        """Read-modify-write a sidecar under one lock.

        A rename and a merge both touch the same file, and a plain read
        followed by a plain write loses whichever landed first. The lock is
        what makes the pair look atomic to the other writer.
        """
        if not (self.enabled and self.store_prompts):
            return
        if not SAFE_JOB_ID.fullmatch(job_id):
            return
        with self._sidecar_lock:
            record = self._read_prompt_locked(job_id) or dict(default)
            record.update(changes)
            self._write_prompt_locked(job_id, record)
        self._cap_sidecars()

    def _write_prompt_locked(self, job_id: str, payload: dict[str, Any]) -> None:
        """Write the sidecar by atomic replace, never by truncate-in-place.

        A reader that catches a half-written JSON file would see the prompt as
        missing, which is the one thing a trail must not do. A crash between
        the write and the replace leaves the previous copy intact.
        """
        tmp: Path | None = None
        try:
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(self.prompts_dir, 0o700)
            path = self.prompts_dir / f"{job_id}.json"
            tmp = self.prompts_dir / f".{job_id}.json.tmp"
            # The mode is set at creation, so there is no window where the text
            # is readable by another local account.
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, indent=2))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            tmp = None
            self.sidecar_lost = 0
        except Exception as exc:  # noqa: BLE001
            self.sidecar_lost += 1
            self.sidecar_lost_total += 1
            self._warn_once("sidecar write", exc)
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink()

    def read_prompt(self, job_id: str) -> dict[str, Any] | None:
        if not SAFE_JOB_ID.fullmatch(job_id):
            return None
        return self._read_prompt_locked(job_id)

    def _cap_sidecars(self) -> None:
        """Keep the sidecar directory under `max_sidecars`, oldest text first.

        Dropping the oldest *text* does not drop the evidence that a prompt was
        used: the day file keeps the length and hash for every job. The cap is a
        count, not bytes, because prompt and hotword length are already bounded
        (PROMPT_LIMIT / HOTWORDS_LIMIT), so a count bounds the directory. 0 keeps
        everything. Called outside `_sidecar_lock`, because it emits.
        """
        if not self.max_sidecars:
            return
        try:
            files = list(self.prompts_dir.glob("*.json"))
        except OSError:
            return
        excess = len(files) - self.max_sidecars
        if excess <= 0:
            return  # the common case: no stat() calls at all
        try:
            files.sort(key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        removed = 0
        for path in files[:excess]:
            if self._unlink(path):
                removed += 1
        if removed:
            self.emit(
                "audit.sidecars_pruned",
                removed=removed,
                max_sidecars=self.max_sidecars,
            )

    def _read_prompt_locked(self, job_id: str) -> dict[str, Any] | None:
        try:
            return json.loads(
                (self.prompts_dir / f"{job_id}.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    # -- reading ---------------------------------------------------------- #

    def dates(self) -> list[str]:
        try:
            return sorted(
                (p.stem[len("audit-") :] for p in self.dir.glob("audit-*.jsonl")),
                reverse=True,
            )
        except OSError:
            return []

    def head(self, day: str) -> dict[str, Any]:
        """The last link of a day, for keeping an anchor off this machine."""
        tail = self._tail_chain(self.dir / f"audit-{day}.jsonl")
        if tail is None:
            return {"seq": None, "chain": None}
        return {"seq": tail[0], "chain": tail[1]}

    def verify(self, day: str) -> dict[str, Any]:
        """Recompute a day's chain and report the first place it breaks.

        Detects an edited record, a deleted or reordered line, and truncation.
        It cannot stop someone with write access from rewriting the tail and
        recomputing every hash after it; an anchor kept off this machine is what
        closes that, which is why the response carries the head value.
        """
        out: dict[str, Any] = {
            "date": day,
            "ok": False,
            "checked": 0,
            "legacy": 0,
            "first_bad_line": None,
            "first_bad_seq": None,
            "reason": None,
            "head": None,
            "carry": None,
            "carry_ok": None,
        }
        try:
            fh = (self.dir / f"audit-{day}.jsonl").open(
                "r", encoding="utf-8", errors="replace"
            )
        except OSError:
            out["reason"] = "No such day"
            return out
        expected_seq = 1
        expected_prev: str | None = None
        first_reset = False
        line_no = 0
        with fh:
            for raw in fh:
                line_no += 1
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except ValueError:
                    out["first_bad_line"] = line_no
                    out["reason"] = "Unparsable record (torn write or edited line)"
                    return out
                if "chain" not in rec:
                    # Records written before the chain existed have no hash to
                    # check -- but only at the *start* of a day. Once a chained
                    # record has been seen, a chain-less one means the field was
                    # removed, which is what an editor would do to hide a change
                    # to the last record of the day.
                    if out["checked"]:
                        out["first_bad_line"] = line_no
                        out["first_bad_seq"] = expected_seq
                        out["reason"] = "Record has no chain hash after the chain began"
                        return out
                    out["legacy"] += 1
                    continue
                seq, prev, chain = rec.get("seq"), rec.get("prev"), rec.get("chain")
                if rec.get("chain_reset"):
                    # A declared discontinuity: the writer could not link to what
                    # was on disk. Accept it, because it says so here.
                    if isinstance(seq, int):
                        expected_seq = seq
                    expected_prev = prev
                    if out["checked"] == 0:
                        first_reset = True
                if seq != expected_seq:
                    out["first_bad_line"] = line_no
                    out["first_bad_seq"] = expected_seq
                    out["reason"] = "Sequence gap"
                    return out
                if prev != expected_prev:
                    out["first_bad_line"] = line_no
                    out["first_bad_seq"] = expected_seq
                    out["reason"] = "Broken link to the previous record"
                    return out
                if self._chain_hash(rec) != chain:
                    out["first_bad_line"] = line_no
                    out["first_bad_seq"] = expected_seq
                    out["reason"] = "Record contents do not match their hash"
                    return out
                out["checked"] += 1
                out["head"] = chain
                if out["checked"] == 1:
                    out["carry"] = rec.get("carry")
                expected_seq += 1
                expected_prev = chain
        out["ok"] = out["first_bad_line"] is None
        # Tie the day to the one before it. A missing or unreadable earlier day
        # is indeterminate, NOT a mismatch: retention deletes the previous day
        # routinely, and reporting that as "carry does NOT match" would cry wolf
        # on a pristine trail every prune cycle.
        found, previous = self._previous_head(day)
        if not found or previous is None:
            out["carry_ok"] = None
        elif out["carry"] is not None:
            out["carry_ok"] = previous == out["carry"]
        elif out["checked"] and not first_reset:
            out["carry_ok"] = False  # a readable earlier day exists but was not carried
        return out

    def _scan_page(
        self,
        path: Path,
        limit: int = 200,
        offset: int = 0,
        job: str | None = None,
        needle: str | None = None,
        before_line: int | None = None,
    ) -> AuditPage:
        """Forward scan of the whole file.

        The only correct reader for a filtered query -- a `needle` substring
        cannot be answered from an index without indexing every substring -- and
        the fallback when the file moves under the tail reader. `before_line`
        takes precedence over `offset`.
        """
        try:
            fh = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            return AuditPage([], 0, None, False)
        span = (
            max(1, limit) if before_line is not None else max(1, limit) + max(0, offset)
        )
        window: deque[tuple[int, str]] = deque(maxlen=span)
        total = 0
        dropped = 0
        line_no = 0
        low = needle.lower() if needle else None
        with fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                line_no += 1
                if low and low not in raw.lower():
                    continue
                if job:
                    try:
                        rec = json.loads(raw)
                    except ValueError:
                        continue
                    if rec.get("job") != job and rec.get("from_job") != job:
                        continue
                # `total` is the whole day's matching count, not this page's and
                # not the cursor's: the tail reader answers from a line count and
                # the two have to agree. Only collection respects before_line.
                total += 1
                if before_line is not None and line_no >= before_line:
                    continue
                if window.maxlen is not None and len(window) == window.maxlen:
                    dropped += 1  # a match older than this page
                window.append((line_no, raw))
        pairs = list(window)[::-1]  # newest first
        if before_line is None:
            pairs = pairs[offset : offset + max(1, limit)]
        else:
            pairs = pairs[: max(1, limit)]
        cursor = pairs[-1][0] if pairs else None
        return AuditPage([raw for _, raw in pairs], total, cursor, dropped > 0)

    @staticmethod
    def _normalize(raw: bytes) -> str:
        """One raw line as the text reader hands it over: decoded, stripped.

        The forward reader strips a *str*, which also drops the `\r` of a CRLF
        line and Unicode spaces like NBSP. Stripping the bytes keeps both, and
        then the two readers disagree about the same record -- and about how
        many records there are, which is the line numbering and `total`.
        """
        return raw.decode("utf-8", errors="replace").strip()

    def _count_lines(self, path: Path, size: int) -> int:
        """Non-empty raw lines that start before byte `size`.

        A line starting inside the snapshot counts even if it ends past it, so
        a count taken mid-write numbers the lines the way the forward reader
        does -- including a torn final fragment. Blankness is judged the way
        `_normalize` judges it, or a whitespace-only line shifts every number
        after it.
        """
        count = 0
        seen = 0
        with path.open("rb") as fh:
            for raw in fh:
                if seen >= size:
                    break
                seen += len(raw)
                if self._normalize(raw):
                    count += 1
        return count

    def _day_line_count(self, day: str, path: Path, st: os.stat_result) -> int:
        """Non-empty lines in a day file, without rescanning on every poll.

        Identity, size and mtime all come from the one `stat` the caller took,
        so a cached count can never describe a different state from the one the
        page is about to be read at.
        """
        size = st.st_size
        cached = self._day_lines.get(day)
        if (
            cached is not None
            and cached.size == size
            and cached.mtime_ns == st.st_mtime_ns
            and (cached.dev, cached.ino) == (st.st_dev, st.st_ino)
        ):
            return cached.lines
        count = self._count_lines(path, size)
        self._day_lines[day] = _DayLines(
            st.st_dev, st.st_ino, size, st.st_mtime_ns, count
        )
        while len(self._day_lines) > self.MAX_DAY_CACHE:
            victim = next((d for d in self._day_lines if d != self._day), None)
            if victim is None:
                break
            self._day_lines.pop(victim)
        return count

    def _reverse_lines(self, path: Path, size: int) -> Iterator[str]:
        """Non-empty lines of the first `size` bytes, newest first, normalized.

        Splits on bytes and only decodes whole assembled lines, so a chunk
        boundary cannot cut a character in half, and every line comes back
        exactly as the forward reader would hand it over.
        """
        carry = b""
        end = size
        with path.open("rb") as fh:
            while end > 0:
                start = max(0, end - self.REVERSE_CHUNK)
                fh.seek(start)
                block = fh.read(end - start)
                if len(block) != end - start:
                    raise OSError("audit file shrank while it was being read")
                parts = block.split(b"\n")
                if len(parts) == 1:
                    # No newline in this chunk: all of it is a fragment of one
                    # line that continues to the left, so it cannot be emitted
                    # yet -- it joins the carry and is finished by an older
                    # chunk (or is the file's first line, at byte 0).
                    carry = block + carry
                    if start == 0:
                        line = self._normalize(carry)
                        if line:
                            yield line
                        carry = b""
                    end = start
                    continue
                # There is a newline: parts[-1] completes the carried fragment.
                joined = self._normalize(parts[-1] + carry)
                if joined:
                    yield joined
                for part in reversed(parts[1:-1]):
                    line = self._normalize(part)
                    if line:
                        yield line
                if start > 0:
                    carry = parts[0]
                else:
                    # No older chunk to finish it, so the file's first line is
                    # whole: yielding it here is what stops it being emitted
                    # twice, once as `carry` after the loop.
                    first = self._normalize(parts[0])
                    if first:
                        yield first
                    carry = b""
                end = start

    def _tail_page(
        self,
        path: Path,
        total: int,
        size: int,
        limit: int,
        offset: int,
        before_line: int | None,
    ) -> AuditPage:
        """The newest page, read from the tail: O(page), not O(file)."""
        want = max(1, limit)
        skipped = 0 if before_line is not None else max(0, offset)
        collected: list[tuple[int, str]] = []
        line_no = total
        for raw in self._reverse_lines(path, size):
            if line_no < 1:
                break
            if before_line is not None and line_no >= before_line:
                line_no -= 1
                continue
            if skipped > 0:
                skipped -= 1
                line_no -= 1
                continue
            collected.append((line_no, raw))
            line_no -= 1
            if len(collected) >= want:
                break
        cursor = collected[-1][0] if collected else None
        # Unfiltered, every non-empty line matches and the oldest line in the
        # day is number 1, so "more" is exactly "the page does not reach it".
        return AuditPage(
            [raw for _, raw in collected],
            total,
            cursor,
            cursor is not None and cursor > 1,
        )

    @staticmethod
    def _snapshot_held(path: Path, before: os.stat_result) -> bool:
        """Whether the bytes read at the snapshot size are still that file.

        An append is fine -- the snapshot prefix is immutable, so the page is
        still a true prefix of the file. A different inode, a shorter file, or a
        same-size file whose mtime moved means the bytes that were read may not
        be the bytes that were counted.
        """
        try:
            after = path.stat()
        except OSError:
            return False
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            return False
        if after.st_size < before.st_size:
            return False
        return not (
            after.st_size == before.st_size and after.st_mtime_ns != before.st_mtime_ns
        )

    def read_page_mode(
        self,
        day: str,
        limit: int = 200,
        offset: int = 0,
        job: str | None = None,
        needle: str | None = None,
        before_line: int | None = None,
    ) -> tuple[AuditPage, str]:
        """(page, which reader answered: "indexed" or "full").

        The mode is returned rather than inferred by the caller, because the
        tail reader can fail and fall back -- and reporting "indexed" after a
        scan would be the one thing worse than being slow.
        """
        path = self.dir / f"audit-{day}.jsonl"
        if job is None and needle is None:
            # A deleted, shrunk or replaced file falls through to the scan:
            # slower, but always right, and the mode says so.
            with contextlib.suppress(OSError):
                st = path.stat()
                total = self._day_line_count(day, path, st)
                page = self._tail_page(
                    path, total, st.st_size, limit, offset, before_line
                )
                if self._snapshot_held(path, st):
                    return page, "indexed"
        return self._scan_page(path, limit, offset, job, needle, before_line), "full"

    def read_page(
        self,
        day: str,
        limit: int = 200,
        offset: int = 0,
        job: str | None = None,
        needle: str | None = None,
        before_line: int | None = None,
    ) -> AuditPage:
        """One page of a day, newest first, plus a cursor for the next one.

        Unfiltered -- the UI's poll -- is served from the tail, so its cost is
        the page rather than the file. A `job` or `needle` query still scans.
        """
        return self.read_page_mode(day, limit, offset, job, needle, before_line)[0]

    def read(
        self,
        day: str,
        limit: int = 200,
        offset: int = 0,
        job: str | None = None,
        needle: str | None = None,
    ) -> tuple[list[str], int]:
        """The (lines newest-first, total matching) pair callers had before
        cursors existed. New callers want read_page()."""
        page = self.read_page(day, limit=limit, offset=offset, job=job, needle=needle)
        return page.lines, page.total

    # -- stats ------------------------------------------------------------ #

    def summary_path(self, day: str) -> Path:
        """Where a day's folded aggregate is kept, beside the trail itself."""
        return self.dir / f"stats-{day}.json"

    @staticmethod
    def _fold_bytes(fh: Any, agg: dict[str, Any], start: int, end: int) -> int:
        """Fold whole records in [start, end); return the offset of the next one.

        Stops at the last newline, so a record still being written is left for
        the next call rather than folded half-formed. Unparsable lines are
        skipped: the trail already tolerates a torn line, and a summary must not
        be the thing that raises because of one.
        """
        fh.seek(start)
        block = fh.read(end - start)
        cut = block.rfind(b"\n")
        if cut < 0:
            return start
        for raw in block[:cut].split(b"\n"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if isinstance(rec, dict):
                name = rec.get("event")
                if isinstance(name, str):
                    fold(agg, name, rec)
        return start + cut + 1

    def _load_summary(self, day: str, st: os.stat_result) -> dict[str, Any] | None:
        """A stored summary for exactly this file, or None to refold."""
        try:
            payload = json.loads(self.summary_path(day).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("schema") != STATS_SCHEMA:
            return None
        if (
            payload.get("dev"),
            payload.get("ino"),
            payload.get("size"),
            payload.get("mtime_ns"),
        ) != (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns):
            return None
        return coerce_stats(payload.get("stats"))

    def _save_summary(self, day: str, st: os.stat_result, agg: dict[str, Any]) -> None:
        """Write a day's aggregate atomically, stamped to the file it describes."""
        tmp: Path | None = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            target = self.summary_path(day)
            tmp = target.with_name(f".{target.name}.tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "schema": STATS_SCHEMA,
                        "day": day,
                        "dev": st.st_dev,
                        "ino": st.st_ino,
                        "size": st.st_size,
                        "mtime_ns": st.st_mtime_ns,
                        "stats": agg,
                    },
                    fh,
                )
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            tmp = None
        except Exception as exc:  # noqa: BLE001
            self._warn_once("summary write", exc)
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink()

    def day_summary(self, day: str) -> dict[str, Any]:
        """Fold one day, from a stored summary or from the trail.

        The trail stays the only source of truth: anything that does not parse,
        does not match the day file's identity/size/mtime, or was written by an
        older schema is discarded and refolded. Today is folded from an in-memory
        running total and never stored, because it changes with every append.
        """
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            # The day reaches a filename below, so it is checked here too rather
            # than trusting every caller (and every future one) to have done it.
            return blank_stats()
        path = self.dir / f"audit-{day}.jsonl"
        try:
            st = path.stat()
        except OSError:
            return blank_stats()
        current = day == datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._stats_lock:
            cached = self._day_stats.get(day)
            if cached is not None and (cached.dev, cached.ino) == (
                st.st_dev,
                st.st_ino,
            ):
                if cached.size <= st.st_size:
                    # Appended since last time: fold only the new bytes. A read
                    # that fails leaves the running total where it was, which is
                    # stale but honest, and never raises into the endpoint.
                    with contextlib.suppress(OSError):
                        with path.open("rb") as fh:
                            cached.size = self._fold_bytes(
                                fh, cached.stats, cached.size, st.st_size
                            )
                    if not current:
                        self._save_summary(day, st, cached.stats)
                    return cached.stats
                # The file is shorter than what we folded: refold from scratch.
            stored = None if current else self._load_summary(day, st)
            if stored is not None:
                self._day_stats[day] = _DayStats(
                    st.st_dev, st.st_ino, st.st_size, stored
                )
            else:
                agg = blank_stats()
                try:
                    with path.open("rb") as fh:
                        end = self._fold_bytes(fh, agg, 0, st.st_size)
                except OSError:
                    return agg
                self._day_stats[day] = _DayStats(st.st_dev, st.st_ino, end, agg)
                stored = agg
                if not current:
                    self._save_summary(day, st, agg)
            while len(self._day_stats) > self.MAX_DAY_CACHE:
                victim = next(iter(self._day_stats))
                if victim == day:
                    break
                self._day_stats.pop(victim)
            return stored

    # -- retention -------------------------------------------------------- #

    def prune(self) -> list[str]:
        """Delete day files and sidecars past retain_days. 0 keeps everything.

        Skipped entirely when auditing is off: `--no-audit` means "stop
        recording", and it must not quietly delete a trail it is not managing.
        """
        if not self.enabled or self.retain_days <= 0:
            return []
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self.retain_days)
        ).strftime("%Y-%m-%d")
        removed: list[str] = []
        for path in self.dir.glob("audit-*.jsonl"):
            if path.stem[len("audit-") :] < cutoff:
                name = self._unlink(path)
                if name:
                    removed.append(name)
        for path in self.prompts_dir.glob("*.json"):
            try:
                stamp = datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).strftime("%Y-%m-%d")
            except OSError:
                continue
            if stamp < cutoff:
                name = self._unlink(path)
                if name:
                    removed.append(name)
        # A temp file left by a crash mid-write. Age-gated so an in-flight
        # write can never have its scratch file swept out from under it.
        for path in self.prompts_dir.glob(".*.json.tmp"):
            try:
                age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                )
            except OSError:
                continue
            if age > timedelta(days=1):
                self._unlink(path)
        # Drop counts for day files that retention just removed, so the cache
        # cannot outlive the files it describes.
        for old in removed:
            if old.startswith("audit-") and old.endswith(".jsonl"):
                day_name = old[len("audit-") : -len(".jsonl")]
                self._day_lines.pop(day_name, None)
                self._day_stats.pop(day_name, None)
                # The summary is derived, so it goes when its day does.
                self._unlink(self.summary_path(day_name))
        for path in self.dir.glob(".stats-*.json.tmp"):
            self._unlink(path)
        return removed

    @staticmethod
    def _unlink(path: Path) -> str | None:
        try:
            path.unlink()
            return path.name
        except OSError:
            return None


AUDIT: AuditLog | None = None


def request_fields(request: Request) -> dict[str, Any]:
    """Who and from where, without treating proxy headers as fact.

    Behind a tunnel the TCP peer is always localhost, so the address the proxy
    claims (CF-Connecting-IP / X-Forwarded-For) is recorded next to it and
    clearly marked unverified rather than silently trusted.

    Values are truncated: these come from an unauthenticated client and end up
    in a file on disk.
    """
    direct = request.client.host if request.client else None
    claimed = (request.headers.get("cf-connecting-ip") or "").strip()
    if not claimed:
        forwarded = request.headers.get("x-forwarded-for") or ""
        claimed = forwarded.split(",")[0].strip()
    return {
        "client": (direct or "")[:64] or None,
        "client_claimed": (claimed[:64] if claimed and claimed != direct else None),
        "host": (normalize_host(request.headers.get("host") or "") or None),
        "method": request.method[:16],
        "path": request.url.path[:200],
    }


# Refusals are written before any credential is checked, so an unauthenticated
# client could otherwise fill the disk by looping on a bad token. The first few
# per source per window are logged individually; the rest are summarised once.
# Actions an *authenticated* client can repeat -- an export in a loop -- share
# the window with a higher budget: the byte cap is the hard guarantee, this just
# keeps a loop from reaching it.
REJECT_LOG_LIMIT = 5
REPEAT_LOG_LIMIT = 20
REJECT_WINDOW = 60.0
REJECT_MAX_SOURCES = 1000
_REJECT_STATE: dict[tuple[str, str], tuple[float, int, int]] = {}
_REJECT_LOCK = threading.Lock()


def _collapse_logged(key: tuple[str, str], limit: int) -> tuple[bool, int]:
    """Windowed per-source counter: (log this one, suppressed since last window).

    Bounded in memory by clearing under a flood of distinct sources. The point
    is to keep a loop from becoming unbounded lines on disk, not to be an exact
    rate limiter; the summary is reported at the start of the next window.
    """
    now = time.monotonic()
    with _REJECT_LOCK:
        if len(_REJECT_STATE) > REJECT_MAX_SOURCES:
            _REJECT_STATE.clear()  # bounded memory under a spoofed-source flood
        window_start, count, logged = _REJECT_STATE.get(key, (now, 0, 0))
        suppressed = 0
        if now - window_start > REJECT_WINDOW:
            suppressed = count - logged
            window_start, count, logged = now, 0, 0
        count += 1
        log_now = logged < limit
        if log_now:
            logged += 1
        _REJECT_STATE[key] = (window_start, count, logged)
    return log_now, suppressed


def audit_rejection(event: str, request: Request, **fields: Any) -> None:
    """Record a refused request, collapsing a burst from one source.

    The suppressed count is reconciled when the window rolls over, in one
    summary per source per window -- see audit_repeated for why that bound is
    chosen over an exact per-call count.
    """
    client = (request.client.host if request.client else "-")[:64]
    log_now, suppressed = _collapse_logged((event, client), REJECT_LOG_LIMIT)
    if log_now:
        audit(event, request, **fields)
    if suppressed:
        audit(
            "security.rejected_summary",
            request,
            scope=event,
            suppressed=suppressed,
        )


def audit_repeated(event: str, request: Request, **fields: Any) -> None:
    """Record an action a client can repeat, collapsing a burst from one source.

    Unlike a refusal this loses a real action when it collapses, so the count is
    reconciled in an `audit.repeated_summary` record when the window rolls over.
    That is deliberately bounded -- one summary per source per window -- which
    means a burst that simply stops, with no further call from that source, has
    its tail count unreconciled until the next call. Exact counts and bounded
    disk are in tension, and the byte cap already fails closed on the other
    side; this keeps the trail from growing with the attack.
    """
    client = (request.client.host if request.client else "-")[:64]
    log_now, suppressed = _collapse_logged(
        (f"repeat:{event}", client), REPEAT_LOG_LIMIT
    )
    if log_now:
        audit(event, request, **fields)
    if suppressed:
        audit(
            "audit.repeated_summary",
            request,
            scope=event,
            suppressed=suppressed,
        )


def audit(event: str, request: Request | None = None, **fields: Any) -> None:
    """Record one event. Never raises, never blocks a response."""
    if AUDIT is None:
        return
    if request is not None:
        fields = {**request_fields(request), **fields}
        with contextlib.suppress(AttributeError):
            request.state.audit_handled = True  # middleware need not log again
    AUDIT.emit(event, **fields)


def audit_opts(opts: dict[str, Any]) -> dict[str, Any]:
    """Job options for the main log: knobs verbatim, prompt text only hashed."""
    out = {k: v for k, v in opts.items() if k not in ("prompt", "hotwords")}
    out["prompt_len"] = len(opts["prompt"])
    out["prompt_sha256"] = digest(opts["prompt"])
    out["hotwords_len"] = len(opts["hotwords"])
    out["hotwords_sha256"] = digest(opts["hotwords"])
    return out


def store_prompt_sidecar(
    job_id: str,
    filename: str,
    opts: dict[str, Any],
    source: str,
    from_job: str | None = None,
) -> None:
    if AUDIT is None or not (opts["prompt"] or opts["hotwords"]):
        return
    AUDIT.write_prompt(
        job_id,
        {
            "job": job_id,
            "ts": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "filename": filename,
            "source": source,
            "from_job": from_job,
            "prompt": opts["prompt"],
            "hotwords": opts["hotwords"],
        },
    )


def store_speaker_names(job_id: str, filename: str, names: dict[str, str]) -> None:
    """Keep a job's speaker names in its audit sidecar, beside any prompt.

    The sidecar is the one place the audit trail holds sensitive text in clear,
    readable only with the audit token; the main log has a hash. The amend is a
    single locked read-modify-write, so a prompt stored for the job survives and
    a concurrent writer cannot drop it. --no-audit-prompts turns this off along
    with the prompts.
    """
    if AUDIT is None:
        return
    AUDIT.amend_prompt(
        job_id,
        default={
            "job": job_id,
            "filename": filename,
            "source": "names",
            "from_job": None,
            "prompt": "",
            "hotwords": "",
        },
        changes={
            "speaker_names": names,
            "ts": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        },
    )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

# Bumped when the aggregate's shape changes, so a summary persisted by an older
# build is refolded rather than misread.
STATS_SCHEMA = 1


def blank_stats() -> dict[str, Any]:
    """One empty aggregate.

    Counts, durations and byte totals only -- no filenames, no prompt text, no
    speaker names -- so a summary can never become a second place that text
    lives.
    """
    return {
        "events": 0,
        "recordings": 0,  # job.done, first pass: a recording we were asked to do
        "passes": 0,  # job.done that transcribed, first pass or retry
        "relabels": 0,  # job.done for a speaker pass over a stored transcript
        "failed": 0,
        "diarize_failed": 0,
        "cancelled": 0,
        "rejected": 0,
        "exports": 0,
        "diarized": 0,
        "audio_seconds": 0.0,  # duration over every pass that transcribed
        "recording_seconds": 0.0,  # duration over first passes only
        "processing_seconds": 0.0,  # elapsed over passes that transcribed
        "uploaded_bytes": 0,
        "segments": 0,
        "lost": 0,  # events the trail itself dropped
        "cap_hits": 0,
        "sidecars_pruned": 0,
        "suppressed": 0,  # refusals and repeats collapsed into a summary
        "models": {},
        "languages": {},
        "security": {},
    }


def fold(agg: dict[str, Any], event: str, record: dict[str, Any]) -> None:
    """Fold one audit record into an aggregate, in place.

    One function for both views -- the live session counters and a day read back
    from the trail -- so the two can never disagree about what a record means.
    Deliberately tolerant: an unknown event still counts, and a malformed field
    falls back rather than raising, because this also runs on the write path.
    """
    agg["events"] += 1
    if event == "job.created":
        agg["uploaded_bytes"] += as_int(record.get("bytes"), 0)
    elif event == "job.done":
        # A relabel replays a stored transcript through the speaker pass: it
        # transcribes nothing, so its duration must never be added to an audio
        # total. A retry does transcribe, so it is a pass but not a new
        # recording. Both marks ride the record (relabel_of/retry_of); without
        # them this one fold would count relabelled audio twice.
        relabel = bool(record.get("relabel_of"))
        retry = bool(record.get("retry_of"))
        seconds = as_float(record.get("duration"), 0.0)
        agg["segments"] += as_int(record.get("segments"), 0)
        if relabel:
            agg["relabels"] += 1
            return
        agg["passes"] += 1
        agg["audio_seconds"] += seconds
        agg["processing_seconds"] += as_float(record.get("elapsed"), 0.0)
        if not retry:
            agg["recordings"] += 1
            agg["recording_seconds"] += seconds
        model = record.get("model")
        if isinstance(model, str) and model:
            agg["models"][model] = agg["models"].get(model, 0) + 1
        language = record.get("language")
        if isinstance(language, str) and language:
            agg["languages"][language] = agg["languages"].get(language, 0) + 1
    elif event == "job.error":
        agg["failed"] += 1
    elif event == "job.diarize_failed":
        agg["diarize_failed"] += 1
    elif event == "job.cancelled":
        agg["cancelled"] += 1
    elif event in ("job.rejected", "job.retry_rejected", "job.relabel_rejected"):
        agg["rejected"] += 1
    elif event == "transcript.exported":
        agg["exports"] += 1
    elif event == "job.diarized":
        agg["diarized"] += 1
    elif event == "audit.full":
        agg["cap_hits"] += 1
    elif event == "audit.sidecars_pruned":
        agg["sidecars_pruned"] += as_int(record.get("removed"), 0)
    elif event.startswith("security."):
        agg["security"][event] = agg["security"].get(event, 0) + 1
        agg["suppressed"] += as_int(record.get("suppressed"), 0)
    if record.get("lost"):
        agg["lost"] += as_int(record["lost"], 0)


def coerce_stats(raw: Any) -> dict[str, Any]:
    """A validated aggregate built from an untrusted dict (a stored summary).

    The persisted summary is derived data on a disk anyone can edit, so nothing
    in it is trusted: an unknown or wrongly-typed key falls back to the blank
    value, and a size mismatch upstream means it is refolded from the trail
    anyway. Worst case a bad file misreports until the next fold; it can never
    raise, and it can never add a key the aggregate does not have.
    """
    out = blank_stats()
    if not isinstance(raw, dict):
        return out
    for key, blank in out.items():
        value = raw.get(key)
        if isinstance(blank, float):
            out[key] = as_float(value, 0.0)
        elif isinstance(blank, int):
            out[key] = as_int(value, 0)
        elif isinstance(blank, dict) and isinstance(value, dict):
            out[key] = {
                str(k): as_int(v, 0)
                for k, v in value.items()
                if isinstance(v, (int, float))
            }
    return out


def merge_stats(into: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Add one aggregate into another, in place.

    Type-guarded on both sides, so a key of the wrong shape can neither raise
    nor invent a field the aggregate does not have.
    """
    for key, value in other.items():
        target = into.get(key)
        if isinstance(target, dict) and isinstance(value, dict):
            for name, count in value.items():
                target[name] = target.get(name, 0) + count
        elif isinstance(target, (int, float)) and isinstance(value, (int, float)):
            into[key] = target + value
    return into


class Stats:
    """Aggregate for one process run, fed from AuditLog.emit.

    The fold runs before emit's `enabled` check, so it counts what the process
    did even under --no-audit, where nothing reaches a file. That is what makes
    "this session" and "the retained days" genuinely different views rather
    than two readings of the same file.
    """

    def __init__(self) -> None:
        self.since = time.time()
        self.total = blank_stats()
        self._lock = threading.Lock()

    def record(self, event: str, fields: dict[str, Any]) -> None:
        """Fold one event. Never raises: the write path must not fail here."""
        with contextlib.suppress(Exception):
            with self._lock:
                fold(self.total, event, fields)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self.total)
            for key in ("models", "languages", "security"):
                out[key] = dict(out[key])
            return out


def new_job(filename: str, path: Path, opts: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    try:
        size = path.stat().st_size
    except OSError:
        # A retry can race an eviction that already unlinked the source.
        size = 0
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "filename": filename,
            "path": str(path),
            "size": size,
            "state": "queued",  # queued | loading | running | done | error | cancelled
            # Which pass the worker is in, so the client can say "identifying
            # speakers" instead of inventing an ETA from a stalled meter.
            "phase": None,  # None | transcribing | diarizing
            "progress": 0.0,
            "message": "Waiting for the GPU",
            "segments": [],
            "language": None,
            "duration": None,
            "created": time.time(),
            "started": None,
            "finished": None,
            "opts": opts,
        }
    return job_id


def patch_job(job_id: str, **fields: Any) -> bool:
    """Apply fields to a job, refusing to move a cancelled job anywhere else.

    Returns False when the write was refused, so a worker that was cancelled
    mid-flight can stop instead of overwriting the user's decision.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return False
        if job["state"] == "cancelled" and fields.get("state") not in (
            None,
            "cancelled",
        ):
            return False
        job.update(fields)
        return True


def job_cancelled(job_id: str) -> bool:
    """True when the job was cancelled, or evicted out from under the worker."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return job is None or job["state"] == "cancelled"


def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return job


def source_shared(path: str, exclude_id: str) -> bool:
    """A retried job reuses the original upload; don't delete it out from under."""
    with JOBS_LOCK:
        return any(j["path"] == path and j["id"] != exclude_id for j in JOBS.values())


def drop_source(job: dict[str, Any]) -> None:
    if ARGS.source_retention == "forever":
        return
    if source_shared(job["path"], job["id"]):
        return
    with contextlib.suppress(OSError):
        Path(job["path"]).unlink(missing_ok=True)


def prune_jobs() -> None:
    """Keep memory bounded: evict the oldest finished jobs past the cap."""
    cap = ARGS.max_jobs if READY else 60
    evicted: list[dict[str, Any]] = []
    with JOBS_LOCK:
        if len(JOBS) <= cap:
            return
        finished = sorted(
            (j for j in JOBS.values() if j["state"] in ("done", "error", "cancelled")),
            key=lambda j: j["created"],
        )
        while len(JOBS) > cap and finished:
            victim = finished.pop(0)
            JOBS.pop(victim["id"], None)
            evicted.append(victim)
    for job in evicted:
        drop_source(job)
        audit(
            "job.evicted",
            job=job["id"],
            file=job["filename"],
            state=job["state"],
            reason="retention-cap",
            cap=cap,
        )


def public_opts(opts: dict[str, Any]) -> dict[str, Any]:
    """Job options as a client may see them: knobs verbatim, prompt text never.

    Prompt and hotword text is the one field the audit token is meant to gate,
    so it is not exposed here even to a holder of the app token. Callers get a
    flag plus a length, which is all the UI ever used.
    """
    out = {k: v for k, v in opts.items() if k not in ("prompt", "hotwords")}
    out["has_prompt"] = bool(opts["prompt"])
    out["has_hotwords"] = bool(opts["hotwords"])
    out["prompt_len"] = len(opts["prompt"])
    out["hotwords_len"] = len(opts["hotwords"])
    return out


def relabel_refusal(job: dict[str, Any]) -> str | None:
    """Why this job's speakers cannot be labelled again, or None if they can.

    One function for both the endpoint's refusal and the page's can_relabel,
    so the button is never offered for a request the server would turn down.
    """
    if not ARGS.allow_diarize:
        return "Speaker identification is turned off on this server"
    if job["state"] != "done":
        return "Only a finished job can have its speakers identified again"
    # Labels are split at word boundaries; a transcript without word timings
    # could only hand each line to its dominant speaker. Retry can ask for them.
    if not job["opts"].get("word_timestamps") or not any(
        seg.get("words") for seg in job.get("transcribed") or job["segments"]
    ):
        return (
            "This transcript has no word timings, which speaker labels need; "
            "use Retry with Identify speakers ticked"
        )
    if not Path(job["path"]).exists():
        return "The source audio is no longer on disk; re-upload it"
    return None


def job_public(job: dict[str, Any], include_segments: bool = True) -> dict[str, Any]:
    # "transcribed" is a second copy of the transcript, kept for relabelling;
    # the list is polled every 1.2 s and must not carry it.
    out = {
        k: v
        for k, v in job.items()
        if k not in ("path", "segments", "opts", "transcribed")
    }
    out["opts"] = public_opts(job["opts"])
    now = time.time()
    started = job.get("started")
    finished = job.get("finished")
    out["elapsed"] = round((finished or now) - started, 1) if started else 0.0
    out["segment_count"] = len(job["segments"])
    out["can_retry"] = (
        job["state"] in ("done", "error", "cancelled") and Path(job["path"]).exists()
    )
    out["can_relabel"] = relabel_refusal(job) is None
    if include_segments:
        out["segments"] = job["segments"]
    return out


# --------------------------------------------------------------------------- #
# Job options
# --------------------------------------------------------------------------- #


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def as_int(value: Any, default: int) -> int:
    """Coerce a caller-supplied value to int, falling back to a default.

    FastAPI already coerces declared int params, but build_opts and the audit
    query are also reachable directly, and a malformed value should clamp to a
    sane default rather than raise a 500.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    """Coerce to float, falling back to a default. Never raises."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pick_speaker_model(requested: str | None) -> str:
    """The job's voice model: the client's pick, if the server lets it choose.

    Pinned (embedding_choice = false) behaves like --pin-model: the request is
    ignored rather than refused, since the client never sees the control.
    """
    if not ARGS.diarization_embedding_choice or not requested:
        return ARGS.diarization_embedding
    if requested not in DIARIZE_EMBEDDINGS:
        raise HTTPException(status_code=400, detail="Unknown voice model")
    return requested


def threshold_for(speaker_model: str | None) -> float:
    """The clustering threshold for a job's voice model.

    Each model has its own distance scale. The server's diarization_threshold
    was resolved for the server's default model -- set explicitly, or that
    model's calibrated value -- so it applies to that model only; any other
    model runs at its own calibrated threshold.
    """
    if not speaker_model or speaker_model == ARGS.diarization_embedding:
        return ARGS.diarization_threshold
    return float(DIARIZE_EMBEDDINGS[speaker_model]["threshold"])


def build_opts(
    model: str | None,
    compute_type: str | None,
    language: str,
    vad: str,
    quality: str,
    prompt: str,
    hotwords: str,
    translate: str,
    condition: str,
    word_timestamps: str,
    min_silence_ms: int,
    speech_pad_ms: int,
    diarize: str = "false",
    speakers: int = 0,
    speaker_model: str | None = None,
) -> dict[str, Any]:
    """Validate everything a client can influence. Pinned knobs ignore the client."""
    truthy = lambda v: str(v).lower() in ("1", "true", "yes", "on")  # noqa: E731

    if ARGS.allow_model_choice:
        model = model or ARGS.model
        if model not in MODELS:
            raise HTTPException(status_code=400, detail="Unknown model")
    else:
        model = ARGS.model

    if ARGS.allow_precision_choice:
        compute_type = compute_type or ARGS.compute_type
        if compute_type not in PRECISIONS:
            raise HTTPException(status_code=400, detail="Unknown compute type")
    else:
        compute_type = ARGS.compute_type

    language = (language or "").strip().lower()
    if language and (len(language) > 5 or not language.isalpha()):
        raise HTTPException(status_code=400, detail="Unknown language code")

    if quality not in QUALITIES:
        quality = ARGS.quality

    # A server started with --no-diarize pins this off rather than refusing, the
    # same way --pin-model pins the model: the client never sees the control.
    if ARGS.allow_diarize and truthy(diarize):
        diarize_on = True
        wanted = clamp(as_int(speakers, 0), 0, DIARIZE_MAX_SPEAKERS)
        # Diarization is only as good as the word boundaries it splits on. With
        # no word timings a whole segment can only be handed to its dominant
        # speaker, which is the version of this feature that is confidently
        # wrong, so asking for speakers asks for the timings too.
        word_timestamps_on = True
    else:
        diarize_on = False
        wanted = 0
        word_timestamps_on = truthy(word_timestamps)

    return {
        "model": model,
        "compute_type": compute_type,
        "language": language,
        "vad": truthy(vad),
        "quality": quality,
        "beam_size": QUALITIES[quality],
        "prompt": (prompt or "").strip()[:PROMPT_LIMIT],
        "hotwords": (hotwords or "").strip()[:HOTWORDS_LIMIT],
        "translate": truthy(translate),
        # Whisper's repetition loops on long audio come from carrying a poisoned
        # context forward, so this is off unless asked for.
        "condition": truthy(condition),
        "word_timestamps": word_timestamps_on,
        "diarize": diarize_on,
        # 0 means "work it out from the audio".
        "speakers": wanted,
        # Which voice model tells the speakers apart; None when not labelling.
        "speaker_model": pick_speaker_model(speaker_model) if diarize_on else None,
        "min_silence_ms": clamp(as_int(min_silence_ms, 2000), 100, 10000),
        "speech_pad_ms": clamp(as_int(speech_pad_ms, 400), 0, 2000),
    }


# --------------------------------------------------------------------------- #
# Speaker diarization
# --------------------------------------------------------------------------- #
#
# Whisper does not do diarization, and the reference implementation
# (pyannote.audio) wants PyTorch — which this server deliberately does not have.
# sherpa-onnx runs the same pyannote segmentation model as ONNX instead: two
# small wheels with no transitive dependencies, and weights fetched from GitHub
# releases rather than gated behind a Hugging Face account.
#
# The work happens in a *child process*, and that is a measurement rather than a
# style choice. OfflineSpeakerDiarization.process() holds the GIL for its whole
# duration (docs/speaker-diarization.md, section 5), so calling it from the
# worker thread — which is a thread, not a process — stops the event loop dead
# for the entire pass, and takes the status poll, the live transcript and Cancel
# with it. A separate interpreter is the only placement where the rest of the
# server keeps working.

diarize_threads = lambda: clamp(os.cpu_count() or 1, 1, 4)  # noqa: E731

# Only the count is exposed; the threshold is what auto-detection diverges on,
# and asking the operator how many people were in the room is both easier and
# more reliable than asking them to tune a clustering distance.
#
# 0.8 is measured, not copied. Sherpa's own examples pass 0.5, but they pin
# num_clusters alongside it, which makes the threshold inert. Against the four
# files in sherpa's own CI, whose speaker counts are known (4, 2, 2, 2), the
# auto path scores 1/4 at 0.5 -- it splits a two-person call into four speakers
# -- 4/4 at 0.8 and 0.9, and 3/4 at 1.0, which merges the four-speaker file
# down to three. 0.8 is the safe end of the plateau that is right on all four.
DIARIZE_THRESHOLD = 0.8

# On Auto, a speaker holding less than this share of the talk time is folded
# into the voice nearest it in time. Call audio is where Auto over-counts: a
# two-person Zoom call came back as five speakers, the extra three a laugh, a
# cough and a raised voice holding seconds each. A pinned count is never
# folded -- the operator said how many there were. The price is a real person
# who says one line in a long meeting; pinning the count is the way out.
DIARIZE_FOLD_SHARE = 0.03
DIARIZE_MIN_DURATION_ON = 0.3
DIARIZE_MIN_DURATION_OFF = 0.5

# A healthy child emits progress at least once per percent, so this much silence
# means it is wedged rather than busy -- with one deliberate exception. The
# callback reports segmentation progress, and then clustering runs to completion
# emitting nothing, so on a long recording the quiet stretch at the end is the
# second-slowest part of the job rather than a hang. The guard therefore has to
# clear the whole clustering phase for a multi-hour file on a slow CPU, which is
# minutes; 120s was not enough, and killed healthy runs.
DIARIZE_STALL_SECONDS = 600
DIARIZE_MARK = "@@DIARIZE@@"

# Pinned by SHA-256 on purpose. A truncated .onnx fails deep inside ONNX Runtime
# with nothing useful to say, which is the same shape as the cuBLAS bug this
# repo already fixed once; better to catch it at download time. Note that the
# upstream release tag really is spelled "recongition".
DIARIZE_MODELS: dict[str, dict[str, Any]] = {
    "segmentation": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-segmentation-models/"
        "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        # The tarball also carries fp32 weights, 4x the size for no difference
        # worth paying for here.
        "member": "sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx",
        "file": "segmentation.int8.onnx",
        "bytes": 1_540_506,
        "sha256": "d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d",
    },
    # Set from DIARIZE_EMBEDDINGS by main(), per diarization_embedding.
    "embedding": {},
}

# The speaker-embedding models an operator can choose between -- the two that
# earned a place in scripts/diarize_eval.py (docs/speaker-diarization.md,
# section 10). Each carries the clustering threshold it was calibrated at,
# because the distance scale differs by model: titanet-small is best at 0.8,
# eres2net-en at 0.9, and a threshold left unset follows the model.
# titanet-small keeps its original file name so an existing install is not
# made to download it again.
DIARIZE_EMBEDDINGS: dict[str, dict[str, Any]] = {
    "titanet-small": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/nemo_en_titanet_small.onnx",
        "member": None,
        "file": "embedding.onnx",
        "bytes": 40_257_283,
        "sha256": "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
        "threshold": 0.8,
    },
    # 3D-Speaker ERes2Net trained on VoxCeleb (English). Ties titanet-small on
    # Auto, is somewhat better with the count pinned, is smaller, and is
    # Apache-2.0 where NeMo's weights need a licence check to redistribute.
    "eres2net-en": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/"
        "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        "member": None,
        "file": "embedding-eres2net-en.onnx",
        "bytes": 26_485_263,
        "sha256": "c59158379255ad66e161679cca6af8d52d51e389e3224ab7d7a7baae295c2db5",
        "threshold": 0.9,
    },
}
DIARIZE_MODELS["embedding"] = DIARIZE_EMBEDDINGS["titanet-small"]

# The child. Kept as a string so transcribe_server.py stays one file: this runs
# as `sys.executable -c`, and under `uv run` that interpreter already has the
# inline dependencies, sherpa-onnx among them. It prints one JSON object per
# line, each prefixed with a marker handed to it as argv[1] -- passing it rather
# than repeating the literal means the two sides cannot drift apart. The prefix
# is what stops anything ONNX Runtime decides to log being read as a result.
DIARIZE_WORKER = r"""
import json, sys

import sherpa_onnx
from faster_whisper.audio import decode_audio

MARK = ""


def emit(payload):
    sys.stdout.write(MARK + json.dumps(payload) + "\n")
    sys.stdout.flush()


def fail(message):
    emit({"t": "error", "error": message})
    return 1


def main():
    global MARK
    MARK = sys.argv[1]
    (audio_path, seg_model, emb_model, num_speakers, threshold,
     min_on, min_off, threads) = sys.argv[2:10]

    # 0 from the UI means "work it out from the audio". The sentinel sherpa
    # documents for that is -1; passing 0 happens to behave the same way, and
    # relying on an undocumented coincidence is not worth it.
    clusters = int(num_speakers) if int(num_speakers) > 0 else -1

    # PyAV through faster-whisper: bundled FFmpeg, no system binary, and the
    # same decode path the transcription used.
    audio = decode_audio(audio_path, sampling_rate=16000)
    if audio.size == 0:
        return fail("no audio could be decoded from the file")

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_model, window_shift_ratio=0.1
            ),
            num_threads=int(threads),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=emb_model, num_threads=int(threads)
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=clusters, threshold=float(threshold)
        ),
        min_duration_on=float(min_on),
        min_duration_off=float(min_off),
    )
    if not config.validate():
        return fail("the diarization models are present but were rejected; "
                    "delete them and let the server fetch them again")

    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    if diarizer.sample_rate != 16000:
        return fail("the decoder produced %d Hz, expected 16000"
                    % diarizer.sample_rate)

    last = [-1]

    def progress(done, total):
        percent = int(done * 100 / total) if total else 100
        if percent != last[0]:
            last[0] = percent
            emit({"t": "progress", "v": percent})
        return 0

    result = diarizer.process(audio, callback=progress).sort_by_start_time()
    emit({"t": "turns", "turns": [[float(r.start), float(r.end), int(r.speaker)]
                                   for r in result]})
    return 0


sys.exit(main())
"""


def diarize_model_dir() -> Path:
    return WORK_DIR / "diarize-models"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    # Every URL here is a module constant, but a wrong scheme would be a quiet
    # local-file read, so say no rather than trusting the table. The noqas are
    # for the same rule on the two lines that actually open it.
    if not url.startswith("https://"):
        raise RuntimeError(f"refusing to fetch model weights from {url!r}")
    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "transcribe-server"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        with dest.open("wb") as fh:
            shutil.copyfileobj(response, fh)


def ensure_diarize_models(embedding: str | None = None) -> dict[str, Path]:
    """Fetch the diarization weights once, into the work dir.

    They do not come from Hugging Face, so they do not belong in its cache, and
    they are small enough that re-verifying the hash on every job is cheaper
    than remembering that we checked.
    """
    directory = diarize_model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(directory, 0o700)

    resolved: dict[str, Path] = {}
    specs = dict(DIARIZE_MODELS)
    if embedding:
        # A job's own voice model; each has its own file, so switching does not
        # evict the other.
        specs["embedding"] = DIARIZE_EMBEDDINGS[embedding]
    for key, spec in specs.items():
        dest = directory / str(spec["file"])
        if dest.is_file() and _sha256_file(dest) == spec["sha256"]:
            resolved[key] = dest
            continue

        # Staged next to the target and renamed only once it verifies, so an
        # interrupted download can never be mistaken for a usable model.
        stage = dest.with_name(dest.name + ".tmp")
        try:
            _download(str(spec["url"]), stage)
            member = spec["member"]
            if member:
                with tarfile.open(stage, "r:bz2") as archive:
                    source = archive.extractfile(str(member))
                    if source is None:
                        raise RuntimeError(f"{member} is not in the archive")
                    with dest.open("wb") as fh:
                        shutil.copyfileobj(source, fh)
                stage.unlink()
                staged = dest
            else:
                staged = stage
            actual = _sha256_file(staged)
            if actual != spec["sha256"]:
                raise RuntimeError(
                    f"checksum mismatch for {spec['file']} "
                    f"(got {actual[:16]}..., expected {str(spec['sha256'])[:16]}...)"
                )
            if staged is not dest:
                staged.replace(dest)
        except Exception:
            with contextlib.suppress(OSError):
                stage.unlink()
            with contextlib.suppress(OSError):
                if dest.is_file():
                    if _sha256_file(dest) != spec["sha256"]:
                        dest.unlink()
            raise
        resolved[key] = dest
    return resolved


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def relabel_speakers(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renumber so Speaker 1 is whoever speaks first.

    Cluster ids come out of the diarizer in arbitrary order, and a transcript
    that opens with "Speaker 3" reads like a bug even when it is correct.
    """
    order: dict[int, int] = {}
    out = []
    for turn in sorted(turns, key=lambda t: (t["start"], t["end"])):
        raw = int(turn["speaker"])
        if raw not in order:
            order[raw] = len(order) + 1
        out.append(
            {
                "start": round(float(turn["start"]), 2),
                "end": round(float(turn["end"]), 2),
                "speaker": order[raw],
            }
        )
    return out


def fold_minor_speakers(
    turns: list[dict[str, Any]], min_share: float
) -> tuple[list[dict[str, Any]], int]:
    """Fold speakers under `min_share` of the talk time into their neighbours.

    Each turn of a minor speaker goes to the kept speaker whose turn is nearest
    it in time, and the result is renumbered by first appearance. Returns the
    turns and how many speakers were folded. The loudest speaker is always
    kept, so the transcript never ends up with nobody in it.
    """
    talk: dict[int, float] = {}
    for t in turns:
        speaker = int(t["speaker"])
        talk[speaker] = talk.get(speaker, 0.0) + max(0.0, t["end"] - t["start"])
    total = sum(talk.values())
    if min_share <= 0 or len(talk) < 2 or total <= 0:
        return relabel_speakers(turns), 0
    minor = {sp for sp, seconds in talk.items() if seconds / total < min_share}
    minor.discard(max(talk, key=lambda sp: talk[sp]))
    if not minor:
        return relabel_speakers(turns), 0

    kept = [t for t in turns if int(t["speaker"]) not in minor]

    def gap(a: dict[str, Any], b: dict[str, Any]) -> float:
        return max(0.0, b["start"] - a["end"], a["start"] - b["end"])

    out = []
    for t in turns:
        if int(t["speaker"]) in minor:
            nearest = min(kept, key=lambda k: (gap(t, k), k["start"]))
            t = {**t, "speaker": nearest["speaker"]}
        out.append(t)
    return relabel_speakers(out), len(minor)


def renumber_segments(
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Renumber speakers 1..N by first appearance, as relabel_speakers does for
    turns. Returns the segments and the old-to-new mapping."""
    order: dict[int, int] = {}
    out = []
    for segment in segments:
        speaker = segment_speaker(segment)
        if speaker is not None:
            if speaker not in order:
                order[speaker] = len(order) + 1
            segment = {**segment, "speaker": order[speaker]}
        out.append(segment)
    return out, order


def speaker_label(speaker: int, names: dict[str, str] | None = None) -> str:
    """The operator's name for a speaker if they gave one, else "Speaker N"."""
    return (names or {}).get(str(speaker)) or f"Speaker {speaker}"


def segment_speaker(segment: dict[str, Any]) -> int | None:
    """The speaker on a segment, or None when the job was not diarized."""
    value = segment.get("speaker")
    return value if isinstance(value, int) else None


def align_speakers(
    segments: list[dict[str, Any]], turns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Label every segment, splitting one where the speaker changes mid-sentence.

    This is the part that decides whether the output is readable. A Whisper
    segment can span four speaker turns, and giving the whole thing to its
    dominant speaker produces a transcript that is confidently wrong in a way
    nobody can see.

    So the split is done at word granularity: each word is bucketed into the
    turn it overlaps most, runs of one speaker become segments, and the text is
    rebuilt from the word tokens — which carry their own leading whitespace, so
    the original spacing survives the surgery.

    Both lists are time-ordered, so one cursor walks them together: everything
    before it has already ended, and the only turns that can overlap a word are
    the ones from the cursor forward that start before the word ends.
    """
    if not turns:
        return segments

    ordered = sorted(turns, key=lambda t: (t["start"], t["end"]))
    cursor = 0
    previous_start: float | None = None

    def speaker_for(start: float, end: float) -> int | None:
        nonlocal cursor, previous_start
        # The cursor is only valid while time moves forward. Word timings are
        # monotonic in practice -- measured over a real transcript: 108 words,
        # zero inversions -- but a backwards jump leaves the cursor past the
        # turn the word belongs to, and that mis-tags silently rather than
        # failing. Rewind and let the loop below walk forward again; the reset
        # costs one scan of the turns and only happens on input that does not
        # occur.
        if previous_start is not None and start < previous_start:
            cursor = 0
        previous_start = start
        while cursor < len(ordered) and ordered[cursor]["end"] <= start:
            cursor += 1
        best: int | None = None
        best_overlap = 0.0
        index = cursor
        while index < len(ordered) and ordered[index]["start"] < end:
            amount = _overlap(
                start, end, ordered[index]["start"], ordered[index]["end"]
            )
            if amount > best_overlap:
                best = int(ordered[index]["speaker"])
                best_overlap = amount
            index += 1
        if best is not None:
            return best
        # No overlap anywhere: the word sits in a pause the diarizer dropped.
        # Take the nearest turn rather than inventing a fourth speaker — and
        # note that nearest is not always the *next* one, since a word trailing
        # a long silence usually belongs to the turn that just ended.
        before = ordered[cursor - 1] if cursor > 0 else None
        after = ordered[cursor] if cursor < len(ordered) else None
        if before is None:
            return int(after["speaker"]) if after is not None else None
        if after is None:
            return int(before["speaker"])
        gap_before = start - float(before["end"])
        gap_after = float(after["start"]) - end
        nearest = before if gap_before <= gap_after else after
        return int(nearest["speaker"])

    out: list[dict[str, Any]] = []
    for segment in segments:
        words = segment.get("words")
        if not words:
            entry = dict(segment)
            entry["speaker"] = speaker_for(
                float(segment["start"]), float(segment["end"])
            )
            out.append(entry)
            continue

        runs: list[tuple[int | None, list[dict[str, Any]]]] = []
        for word in words:
            speaker = speaker_for(float(word["start"]), float(word["end"]))
            if runs and runs[-1][0] == speaker:
                runs[-1][1].append(word)
            else:
                runs.append((speaker, [word]))

        split: list[dict[str, Any]] = []
        for speaker, group in runs:
            text = "".join(str(w["word"]) for w in group).strip()
            if not text:
                continue
            split.append(
                {
                    "start": round(float(group[0]["start"]), 2),
                    "end": round(float(group[-1]["end"]), 2),
                    "text": text,
                    "speaker": speaker,
                    "words": group,
                }
            )
        if not split:
            # Every word stripped to nothing, which should not happen; keep the
            # text rather than lose it.
            entry = dict(segment)
            entry["speaker"] = speaker_for(
                float(segment["start"]), float(segment["end"])
            )
            out.append(entry)
        else:
            out.extend(split)
    return out


def fetch_diarize_models_or_explain(embedding: str | None = None) -> dict[str, Path]:
    """Everything that has to be true before a child can run, or a clear reason."""
    # Say this in the parent's words rather than letting the child die with a
    # ModuleNotFoundError several frames deep: the fix is a different command,
    # not a different file. Deliberately here and not in run_diarizer, which
    # stays a dumb "spawn this worker, parse its output" so it can be tested
    # with a stub worker on an interpreter that has no sherpa-onnx at all.
    if importlib.util.find_spec("sherpa_onnx") is None:
        raise RuntimeError(
            "sherpa-onnx is not installed for this interpreter. It is declared "
            "in the script's inline dependencies, so run it through uv "
            "(uv run transcribe_server.py); otherwise pip install sherpa-onnx"
        )
    try:
        return ensure_diarize_models(embedding)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"could not fetch the diarization models ({redact(str(exc), 200)}). "
            "They are a one-time ~42 MB download; check this machine can reach "
            "github.com, or start the server with --no-diarize"
        ) from exc


def run_diarizer(
    audio_path: str,
    speakers: int,
    models: dict[str, Path],
    on_progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    threshold: float = DIARIZE_THRESHOLD,
) -> list[dict[str, Any]] | None:
    """Diarize one file in a child process and return its turns.

    The two callbacks are optional so that --preload can exercise this exact
    path with no job to report against: the probe has to prove the real thing
    works, not a parallel path that only the probe ever uses.

    Returns None when `cancelled` asked us to stop.
    """

    def stopped() -> bool:
        return bool(cancelled and cancelled())

    if stopped():
        return None

    command = [
        sys.executable,
        "-c",
        DIARIZE_WORKER,
        DIARIZE_MARK,
        audio_path,
        str(models["segmentation"]),
        str(models["embedding"]),
        str(speakers),
        f"{threshold:g}",
        f"{DIARIZE_MIN_DURATION_ON:g}",
        f"{DIARIZE_MIN_DURATION_OFF:g}",
        str(diarize_threads()),
    ]

    creation = 0
    if sys.platform == "win32":
        # Otherwise a console window flashes up on every diarized job.
        creation = subprocess.CREATE_NO_WINDOW  # pyright: ignore[reportAttributeAccessIssue]

    # Not untrusted input: sys.executable, a source string compiled into this
    # file, and paths this server created.
    child = subprocess.Popen(  # noqa: S603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        creationflags=creation,
    )

    # A reader thread rather than reading in the worker thread: it lets us wait
    # with a timeout (so a wedged child is caught) and notice a cancel between
    # lines. It only blocks on a pipe, which releases the GIL, so it cannot
    # recreate the problem this whole design exists to avoid.
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        stream = child.stdout
        try:
            if stream is not None:
                for line in stream:
                    lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    turns: list[dict[str, Any]] | None = None
    error: str | None = None
    noise: list[str] = []
    try:
        while True:
            if stopped():
                return None
            try:
                line = lines.get(timeout=DIARIZE_STALL_SECONDS)
            except queue.Empty:
                raise RuntimeError(
                    f"the diarizer stopped responding for {DIARIZE_STALL_SECONDS}s "
                    "and was killed"
                ) from None
            if line is None:
                break
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith(DIARIZE_MARK):
                # ONNX Runtime warnings and the like. Kept for the error text.
                noise.append(stripped)
                del noise[:-4]
                continue
            try:
                payload = json.loads(stripped[len(DIARIZE_MARK) :])
            except ValueError:
                continue
            kind = payload.get("t")
            if kind == "progress":
                percent = as_int(payload.get("v"), 0)
                # Fill the last tenth of the meter, which the transcription
                # loop deliberately left free when labels were requested.
                if on_progress is not None:
                    on_progress(percent)
            elif kind == "turns":
                turns = [
                    {"start": float(t[0]), "end": float(t[1]), "speaker": int(t[2])}
                    for t in payload.get("turns", [])
                ]
            elif kind == "error":
                error = str(payload.get("error") or "the diarizer failed")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        if child.stdout is not None:
            child.stdout.close()

    if error:
        raise RuntimeError(error)
    if turns is None:
        detail = f" Last output: {' '.join(noise)}" if noise else ""
        raise RuntimeError(f"the diarizer produced no result.{detail}")
    if child.returncode not in (0, None) and not turns:
        detail = f" Last output: {' '.join(noise)}" if noise else ""
        raise RuntimeError(f"the diarizer exited with {child.returncode}.{detail}")
    return relabel_speakers(turns)


def diarize_job(
    job_id: str, audio_path: str, opts: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """The job-aware wrapper: fetch, report progress, honour Cancel."""
    patch_job(job_id, message="Fetching diarization models", phase="diarizing")
    models = fetch_diarize_models_or_explain(opts.get("speaker_model"))
    if job_cancelled(job_id):
        return None

    def note(percent: int) -> None:
        if not job_cancelled(job_id):
            patch_job(
                job_id,
                progress=min(0.9 + percent / 1000.0, 0.99),
                message=f"Identifying speakers {percent}%",
            )

    patch_job(job_id, message="Identifying speakers", phase="diarizing")
    return run_diarizer(
        audio_path,
        as_int(opts.get("speakers"), 0),
        models,
        on_progress=note,
        cancelled=lambda: job_cancelled(job_id),
        threshold=threshold_for(opts.get("speaker_model")),
    )


# --------------------------------------------------------------------------- #
# Transcription worker
# --------------------------------------------------------------------------- #


def verify_device(model: Any, name: str) -> None:
    """Prove the model can actually encode, not merely load.

    A model can load and still fail on the first encode (a missing cuBLAS, for
    instance), so --preload would otherwise report a healthy device that cannot
    transcribe anything. One second of silence is enough to exercise the path.
    """
    import numpy as np

    silence = np.zeros(16000, dtype="float32")
    list(model.transcribe(silence, beam_size=1, vad_filter=False)[0])


def load_model(name: str, device: str, compute_type: str):
    key = (name, device, compute_type)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(key)
            return _MODEL_CACHE[key]

        from faster_whisper import WhisperModel

        cap = max(1, ARGS.model_cache if READY else 1)

        # Evict *before* loading: inserting first would hold the outgoing and
        # incoming models in VRAM at the same time, so a cap of 1 would
        # transiently need room for 2 and could OOM on a tight GPU.
        while len(_MODEL_CACHE) >= cap:
            old_key, old_model = _MODEL_CACHE.popitem(last=False)
            del old_model
            gc.collect()
            print(f"   unloaded {old_key[0]} / {old_key[2]} to free VRAM")
            audit(
                "model.unloaded",
                model=old_key[0],
                device=old_key[1],
                compute_type=old_key[2],
                reason="vram-cache",
            )

        model = WhisperModel(name, device=device, compute_type=compute_type)
        _MODEL_CACHE[key] = model
        audit("model.loaded", model=name, device=device, compute_type=compute_type)
        return model


def transcribe_kwargs(opts: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "beam_size": opts["beam_size"],
        "language": opts["language"] or None,
        "task": "translate" if opts["translate"] else "transcribe",
        "condition_on_previous_text": opts["condition"],
        "word_timestamps": opts["word_timestamps"],
        "vad_filter": opts["vad"],
    }
    if opts["vad"]:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": opts["min_silence_ms"],
            "speech_pad_ms": opts["speech_pad_ms"],
        }
    if opts["prompt"]:
        kwargs["initial_prompt"] = opts["prompt"]
    if opts["hotwords"]:
        kwargs["hotwords"] = opts["hotwords"]
    return kwargs


def label_and_finish(
    job_id: str,
    job: dict[str, Any],
    opts: dict[str, Any],
    collected: list[dict[str, Any]],
    language: str | None,
    duration: float,
) -> None:
    """The speaker pass, if asked for, then the job marked done.

    Shared by a fresh transcription and a relabel, which differ only in where
    `collected` came from. Returns quietly when the job was cancelled.
    """
    # Speaker labels are a second pass over the audio, and they are worth
    # less than the transcript itself: if anything goes wrong in here the
    # job still finishes, unlabelled, and says why. Losing an hour of
    # transcription because a 42 MB download failed would be absurd.
    diarize_note: str | None = None
    folded = 0
    if opts["diarize"] and collected:
        diarize_started = time.time()
        try:
            turns = diarize_job(job_id, job["path"], opts)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            diarize_note = friendly_error(exc)
            audit(
                "job.diarize_failed",
                job=job_id,
                file=job["filename"],
                reason=type(exc).__name__,
                message=redact(str(exc)),
            )
        else:
            if turns is None:
                return  # cancelled while the child was running
            if not turns:
                diarize_note = "the diarizer found no speech"
            else:
                if not opts["speakers"]:
                    turns, folded = fold_minor_speakers(
                        turns, ARGS.diarization_fold_share
                    )
                before = len(collected)
                # Kept as Whisper produced it, before alignment split lines at
                # speaker changes: relabelling with another count starts from
                # this, not from boundaries the last guess drew.
                transcribed = list(collected)
                collected = align_speakers(collected, turns)
                # patch_job is what refuses a cancelled job, not us:
                # overwriting one with a finished result is the exact bug
                # that return value exists to prevent.
                if not patch_job(
                    job_id, segments=list(collected), transcribed=transcribed
                ):
                    return
                audit(
                    "job.diarized",
                    job=job_id,
                    file=job["filename"],
                    backend="sherpa-onnx",
                    segmentation=DIARIZE_MODELS["segmentation"]["file"],
                    embedding=opts.get("speaker_model") or ARGS.diarization_embedding,
                    requested=opts["speakers"],
                    threshold=threshold_for(opts.get("speaker_model")),
                    fold_share=ARGS.diarization_fold_share,
                    folded=folded,
                    speakers=len(
                        {
                            speaker
                            for speaker in map(segment_speaker, collected)
                            if speaker is not None
                        }
                    ),
                    segments_before=before,
                    segments_after=len(collected),
                    elapsed=round(time.time() - diarize_started, 1),
                )

    message = f"{len(collected)} segments"
    if folded:
        message += f" \u00b7 {folded} minor voice{'s' if folded > 1 else ''} folded"
    if diarize_note:
        message += f" \u00b7 no speaker labels ({diarize_note})"

    if not patch_job(
        job_id,
        state="done",
        progress=1.0,
        finished=time.time(),
        message=message,
    ):
        return  # cancelled while finishing; don't claim success
    audit(
        "job.done",
        job=job_id,
        file=job["filename"],
        model=opts["model"],
        compute_type=opts["compute_type"],
        language=language,
        duration=round(duration, 2),
        segments=len(collected),
        elapsed=round(time.time() - (job.get("started") or time.time()), 1),
        # Self-describing pass: a relabel reuses the original duration but does
        # not transcribe, and a retry transcribes the same source again. Without
        # these the record cannot be classified on its own, and any aggregate of
        # `duration` counts relabelled audio twice.
        relabel_of=job.get("relabel_of"),
        retry_of=job.get("retry_of"),
    )


def relabel_job(job_id: str, job: dict[str, Any]) -> None:
    """Only the speaker pass, over a transcript the job was created with.

    No model is loaded and nothing is transcribed: the text is shown at once,
    and the labels land when the pass finishes, exactly as they do at the end of
    a full job. Everything after that -- alignment, the failure note, done --
    is label_and_finish, the same code a fresh transcription runs.
    """
    opts = job["opts"]
    collected = [dict(seg) for seg in job["transcribed"]]
    # patch_job refuses to move a cancelled job, and a relabel cancelled while
    # it was queued must stay that way.
    if not patch_job(
        job_id,
        state="running",
        phase="diarizing",
        started=time.time(),
        progress=0.9,
        segments=list(collected),
        message="Identifying speakers",
    ):
        return
    audit(
        "job.started",
        job=job_id,
        file=job["filename"],
        relabel_of=job["relabel_of"],
        speakers=opts["speakers"],
    )
    try:
        label_and_finish(
            job_id,
            job,
            opts,
            collected,
            job.get("language"),
            float(job.get("duration") or 0.0),
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        if patch_job(
            job_id,
            state="error",
            finished=time.time(),
            message=f"{type(exc).__name__}: {friendly_error(exc)}",
        ):
            audit(
                "job.error",
                job=job_id,
                file=job["filename"],
                stage="relabel",
                reason=type(exc).__name__,
                message=redact(str(exc)),
            )
    finally:
        if ARGS.source_retention == "run":
            drop_source(job)
        prune_jobs()


def run_job(job_id: str) -> None:
    job = get_job(job_id)
    if job.get("relabel_of"):
        relabel_job(job_id, job)
        return
    opts = job["opts"]
    patch_job(job_id, state="loading", started=time.time(), message="Loading model")

    try:
        model = load_model(opts["model"], ARGS.device, opts["compute_type"])
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        audit(
            "job.error",
            job=job_id,
            file=job["filename"],
            stage="load-model",
            reason=type(exc).__name__,
            message=redact(str(exc)),
            opts=audit_opts(opts),
        )
        patch_job(
            job_id,
            state="error",
            finished=time.time(),
            message=f"Could not load {opts['model']}: {friendly_error(exc)}",
        )
        prune_jobs()
        return

    # A cancel during a long model load must not be overwritten by "running".
    if job_cancelled(job_id):
        prune_jobs()
        return

    patch_job(job_id, state="running", message="Transcribing", phase="transcribing")
    audit(
        "job.started",
        job=job_id,
        file=job["filename"],
        model=opts["model"],
        compute_type=opts["compute_type"],
    )

    try:
        kwargs = transcribe_kwargs(opts)
        try:
            segments, info = model.transcribe(job["path"], **kwargs)
        except TypeError as exc:
            # hotwords landed in faster-whisper 1.0.2; degrade rather than fail.
            if "hotwords" not in str(exc):
                raise
            kwargs.pop("hotwords", None)
            segments, info = model.transcribe(job["path"], **kwargs)

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        patch_job(
            job_id,
            language=getattr(info, "language", None),
            duration=round(duration, 2),
        )

        collected: list[dict[str, Any]] = []
        for seg in segments:
            entry: dict[str, Any] = {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            if opts["word_timestamps"]:
                words = getattr(seg, "words", None)
                if words:
                    entry["words"] = [
                        {
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "word": w.word,
                        }
                        for w in words
                    ]
            collected.append(entry)

            # When labels are coming, the meter deliberately stops at 90% here
            # so the diarization pass has somewhere to go. Otherwise it would
            # sit at 100% while the second pass ran, looking stuck.
            ceiling = 0.9 if opts["diarize"] else 1.0
            progress = min(seg.end / duration, 1.0) * ceiling if duration else 0.0
            stop = False
            with JOBS_LOCK:
                live = JOBS.get(job_id)
                if live is None or live["state"] == "cancelled":
                    stop = True
                else:
                    live["segments"] = list(collected)
                    live["progress"] = progress
            if stop:
                # Outside the lock on purpose: drop_source re-acquires JOBS_LOCK,
                # and threading.Lock is not reentrant.
                return

        label_and_finish(
            job_id,
            job,
            opts,
            collected,
            getattr(info, "language", None),
            duration,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        if patch_job(
            job_id,
            state="error",
            finished=time.time(),
            message=f"{type(exc).__name__}: {friendly_error(exc)}",
        ):
            audit(
                "job.error",
                job=job_id,
                file=job["filename"],
                stage="transcribe",
                reason=type(exc).__name__,
                message=redact(str(exc)),
                opts=audit_opts(opts),
            )
    finally:
        if ARGS.source_retention == "run":
            drop_source(job)
        prune_jobs()


def worker_loop() -> None:
    while True:
        job_id = JOB_QUEUE.get()
        try:
            with JOBS_LOCK:
                state = JOBS.get(job_id, {}).get("state")
            if state == "cancelled":
                continue
            run_job(job_id)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            audit(
                "job.error",
                job=job_id,
                stage="worker",
                reason=type(exc).__name__,
                message=redact(str(exc)),
            )
            patch_job(
                job_id,
                state="error",
                finished=time.time(),
                message=f"{type(exc).__name__}: {friendly_error(exc)}",
            )
        finally:
            JOB_QUEUE.task_done()


# --------------------------------------------------------------------------- #
# Transcript formatting
# --------------------------------------------------------------------------- #


def _stamp(seconds: float, comma: bool = False) -> str:
    ms = as_int(as_float(seconds, 0.0) * 1000, 0)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def speaker_summary(
    segs: list[dict[str, Any]], names: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Per-speaker totals, or an empty list when the job was not diarized."""
    totals: dict[int, dict[str, Any]] = {}
    for segment in segs:
        speaker = segment_speaker(segment)
        if speaker is None:
            continue
        entry = totals.setdefault(
            speaker,
            {
                "speaker": speaker,
                "label": speaker_label(speaker, names),
                "segments": 0,
                "seconds": 0.0,
            },
        )
        entry["segments"] += 1
        entry["seconds"] += max(0.0, float(segment["end"]) - float(segment["start"]))
    for entry in totals.values():
        entry["seconds"] = round(entry["seconds"], 2)
    return [totals[key] for key in sorted(totals)]


def render(job: dict[str, Any], fmt: str) -> tuple[str, str]:
    """Return (body, mime) for the requested format.

    A job that was not diarized produces byte-identical output to before: the
    speaker prefix is driven by the presence of a label on the segment, not by
    the option, so an old job and a failing diarization pass both fall through
    to the plain transcript.
    """
    segs = job["segments"]
    names = job.get("speaker_names")

    def spoken(segment: dict[str, Any]) -> str:
        speaker = segment_speaker(segment)
        if speaker is None:
            return str(segment["text"])
        return f"{speaker_label(speaker, names)}: {segment['text']}"

    if fmt == "txt":
        return "\n".join(spoken(s) for s in segs) + "\n", "text/plain; charset=utf-8"

    if fmt == "timestamped":
        lines = [f"[{_stamp(s['start'])}] {spoken(s)}" for s in segs]
        return "\n".join(lines) + "\n", "text/plain; charset=utf-8"

    if fmt == "srt":
        # SRT has no voice construct, so the label is part of the cue text.
        blocks = []
        for i, s in enumerate(segs, 1):
            blocks.append(
                f"{i}\n{_stamp(s['start'], comma=True)} --> "
                f"{_stamp(s['end'], comma=True)}\n{spoken(s)}\n"
            )
        return "\n".join(blocks), "application/x-subrip; charset=utf-8"

    if fmt == "vtt":
        # WebVTT does have one, and players use it to style the speaker.
        blocks = ["WEBVTT\n"]
        for s in segs:
            speaker = segment_speaker(s)
            text = (
                s["text"]
                if speaker is None
                else f"<v {speaker_label(speaker, names)}>{s['text']}"
            )
            blocks.append(f"{_stamp(s['start'])} --> {_stamp(s['end'])}\n{text}\n")
        return "\n".join(blocks), "text/vtt; charset=utf-8"

    if fmt == "json":
        payload: dict[str, Any] = {
            "filename": job["filename"],
            "language": job["language"],
            "duration": job["duration"],
            # public_opts, not job["opts"]: the export must not carry prompt
            # or hotword text out to an app-token holder.
            "options": public_opts(job["opts"]),
            "segments": segs,
        }
        speakers = speaker_summary(segs, names)
        if speakers:
            # Only present when there is something to say, so the shape of a
            # plain export does not change.
            payload["speakers"] = speakers
        return (
            json.dumps(payload, indent=2, ensure_ascii=False),
            "application/json; charset=utf-8",
        )

    raise HTTPException(status_code=400, detail="Unknown format")


def content_disposition(stem: str, ext: str) -> str:
    """RFC 5987 disposition header. Never interpolate a raw filename here."""
    ascii_fallback = (
        "".join(c for c in stem if c.isalnum() or c in " ._-").strip() or "transcript"
    )
    utf8 = quote(f"{stem}.{ext}", safe="")
    return f"attachment; filename=\"{ascii_fallback}.{ext}\"; filename*=UTF-8''{utf8}"


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(title="Transcription server", docs_url=None, redoc_url=None)


STATIC_DIR = Path(__file__).resolve().parent / "static"


def static_asset(name: str) -> str:
    """A file in static/, or a clear 500 if the install is incomplete.

    The pages used to be embedded in this file, so a missing directory was
    impossible. Now it is a real failure mode (a copied .py without static/),
    and it should say so rather than serve a blank page.
    """
    path = STATIC_DIR / name
    if not path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Missing static asset {name!r}; static/ must sit next to "
            "transcribe_server.py",
        )
    return path.read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def normalize_host(raw: str) -> str:
    """Normalize a Host header value or --allow-host entry to a bare hostname."""
    host = raw.strip().lower()
    # Tolerate pasting a full URL: https://example.com:443/path -> example.com
    if "://" in host:
        host = host.split("://", 1)[1]
    host = host.split("/", 1)[0].strip()
    if host.startswith("[") and "]" in host:
        # [::1] or [::1]:8765
        host = host[1 : host.index("]")]
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    return host.rstrip(".").strip()


# A well-formed DNS name or IPv4 literal. Anything else (extra colons, control
# characters, spaces) must not reach suffix matching.
HOSTNAME_RE = re.compile(r"[a-z0-9]([a-z0-9._-]{0,251}[a-z0-9])?")


def host_allowed(header: str | None) -> bool:
    """Reject DNS-rebinding: only hostnames we expect may address this server."""
    if not header:
        return False
    host = normalize_host(header)
    if not host:
        return False
    if host in ALLOWED_HOSTS:
        return True
    # Suffix matching only for well-formed names: a malformed Host such as
    # "x:1:trycloudflare.com" must not slip through on endswith().
    if not HOSTNAME_RE.fullmatch(host):
        return False
    return any(host == s.lstrip(".") or host.endswith(s) for s in ALLOWED_SUFFIXES)


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        # No 'unsafe-inline' for either: the pages carry no inline <style>,
        # <script> or style attributes, so injected markup cannot execute or
        # restyle. That is what makes esc() defence in depth rather than the
        # only line of defence.
        "default-src 'none'; style-src 'self'; script-src 'self'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'"
    ),
}


T = TypeVar("T", bound=Response)


def hardened(response: T) -> T:
    """Every response carries the hardening headers, refusals included."""
    response.headers.update(SECURITY_HEADERS)
    return response


def refuse(detail: str, status: int) -> Response:
    return hardened(JSONResponse({"detail": detail}, status_code=status))


# Paths served behind the audit credential. One list, because three places in
# the middleware have to agree about it -- the token gate, the generic
# read/rejection logging, and the Cache-Control header. Extending one and not
# the others is how a page ends up logging its own polls as `api.read`, straight
# into the numbers it is displaying.
AUDIT_PATHS = ("/api/audit", "/api/stats")


def audit_gated(path: str) -> bool:
    """Whether a request path is served behind the audit credential."""
    return path.startswith(AUDIT_PATHS)


def audit_api_mode(args: argparse.Namespace) -> str:
    """How the audit API is reachable: 'open', 'token' or 'off'.

    One rule, four consumers: the middleware gate, the two pages that render
    their own mode, and the startup record. 'off' is only reachable when ARGS
    was built by hand -- main() always leaves either a token or an explicit
    opt-out.
    """
    if args.audit_open:
        return "open"
    return "token" if args.audit_token else "off"


@app.middleware("http")
async def guard(request: Request, call_next):
    if not READY:
        return refuse("Server not configured", 503)

    started = time.perf_counter()

    raw_host = request.headers.get("host")
    if not host_allowed(raw_host):
        seen = (raw_host or "").strip()[:100]
        print(f"!  Rejected Host header {seen!r} (add it with --allow-host)")
        audit_rejection(
            "security.host_rejected", request, reason=seen or "missing", status=421
        )
        return refuse(
            f"Unrecognised Host header {seen!r}. "
            "Restart the server with --allow-host for this name.",
            421,
        )

    path = request.url.path
    if path.startswith("/api/"):
        # A custom header forces a CORS preflight, so a hostile page cannot
        # reach these endpoints even with a CORS-safelisted body type.
        site = request.headers.get("sec-fetch-site")
        if site and site not in ("same-origin", "none"):
            audit_rejection("security.cross_site", request, reason=site, status=403)
            return refuse("Cross-site request refused", 403)

        if audit_gated(path):
            # The audit trail has its own credential, deliberately independent
            # of the app token: one can be handed out without the other.
            # --audit-open is the only way past that gate, and it stays
            # explicit: this prefix is where prompt text becomes readable.
            mode = audit_api_mode(ARGS)
            if mode == "off":
                return refuse("Audit API disabled: set --audit-token", 404)
            supplied = request.headers.get("x-audit-token") or ""
            if mode == "token" and not hmac.compare_digest(
                supplied.encode(), ARGS.audit_token.encode()
            ):
                audit_rejection(
                    "security.auth_failed",
                    request,
                    scope="audit",
                    reason="bad-token" if supplied else "missing-token",
                    status=401,
                )
                return refuse("Bad or missing audit token", 401)
        elif ARGS.token:
            supplied = request.headers.get("x-token") or ""
            if not hmac.compare_digest(supplied.encode(), ARGS.token.encode()):
                audit_rejection(
                    "security.auth_failed",
                    request,
                    scope="app",
                    reason="bad-token" if supplied else "missing-token",
                    status=401,
                )
                return refuse("Bad or missing token", 401)

        # Refuse an over-sized body from Content-Length, before Starlette spools
        # the whole multipart payload to a temp file. The endpoint still
        # enforces the exact per-file limit while streaming.
        if request.method == "POST" and path == "/api/jobs":
            declared = request.headers.get("content-length") or ""
            if declared.isdigit():
                # 1 MB of slack for multipart framing and the other form fields.
                cap = ARGS.max_upload_mb * 1024 * 1024 + 1024 * 1024
                declared_bytes = as_int(declared, 0)
                if declared_bytes > cap:
                    audit_rejection(
                        "job.rejected",
                        request,
                        bytes=declared_bytes,
                        reason="body-too-large",
                        status=413,
                    )
                    return refuse(
                        f"Upload exceeds the {ARGS.max_upload_mb} MB limit", 413
                    )

    try:
        response = await call_next(request)
    except Exception:
        # An unhandled 500 used to leave no trace at all.
        audit(
            "request.failed",
            request,
            status=500,
            ms=round((time.perf_counter() - started) * 1000, 1),
        )
        raise

    elapsed = round((time.perf_counter() - started) * 1000, 1)
    hardened(response)

    # The page and the assets it loads must be revalidated, never reused blind.
    # StaticFiles sends ETag and Last-Modified but no Cache-Control, which lets
    # a browser keep its own copy without asking -- and an HTML page cannot be
    # cached that way, because it carries no validators at all. So a reload
    # after an upgrade used to hand the operator the *new* page next to the
    # *old* script: a Transcribe button that did nothing, and a bug fixed in
    # the script still happening, with nothing on screen to say which version
    # was running. no-cache (not no-store) keeps the 304 revalidation cheap.
    if path in ("/", "/audit", "/stats") or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"

    # Polling endpoints are chatty, so reads are opt-in. Everything that
    # changes state or moves a transcript out logs itself at the source.
    if path.startswith("/api/") and not audit_gated(path):
        if response.status_code >= 400:
            if not getattr(request.state, "audit_handled", False):
                audit(
                    "request.rejected",
                    request,
                    status=response.status_code,
                    ms=elapsed,
                )
        elif (
            ARGS.audit_reads
            and request.method == "GET"
            # Endpoints that already logged something specific (an export, say)
            # set audit_handled; don't repeat them as a generic read.
            and not getattr(request.state, "audit_handled", False)
        ):
            audit("api.read", request, status=response.status_code, ms=elapsed)
    return response


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return static_asset("index.html")


def page_with_mode(name: str) -> str:
    """Render a page whose data is behind the audit credential.

    One place for the mode substitution, because both pages must agree: the gate
    ships visible in token mode so it works with no script at all, and every
    other mode starts hidden -- the alternative is a visible prompt for a token
    that does not exist, which is the bug /audit used to have. relock() takes
    over from there, for a server restarted mid-session.
    """
    mode = audit_api_mode(ARGS)
    return (
        static_asset(name)
        .replace("__MODE__", mode)
        .replace("__GATE_CLASS__", "gate" if mode == "token" else "gate locked")
        .replace("__OFF_CLASS__", "gate" if mode == "off" else "gate locked")
    )


@app.get("/audit", response_class=HTMLResponse)
def audit_ui() -> str:
    """The page is public like `/`; the data behind it needs the audit token."""
    return page_with_mode("audit.html")


@app.get("/stats", response_class=HTMLResponse)
def stats_ui() -> str:
    """Public shell, like /audit; the totals behind it need the audit token."""
    return page_with_mode("stats.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    # Served from the cached probe rather than re-queried per poll. This used to
    # call CTranslate2 and attempt `import torch` on every request -- and a
    # *failed* import is not cached in sys.modules, so that was a fresh sys.path
    # scan every 1.2 s, forever, on a machine that will never have torch.
    info = device_details()

    with JOBS_LOCK:
        active = sum(
            1 for j in JOBS.values() if j["state"] in ("queued", "loading", "running")
        )
    with _MODEL_LOCK:
        loaded = [f"{k[0]} / {k[2]}" for k in _MODEL_CACHE]

    return {
        "device": ARGS.device,
        "gpu": gpu_cell(info),
        "cuda": CUDA,
        "device_info": dict(info),
        "compute_type": ARGS.compute_type,
        "default_model": ARGS.model,
        "default_quality": ARGS.quality,
        "models": MODELS,
        "compute_types": PRECISIONS,
        "qualities": list(QUALITIES),
        "allow_model_choice": ARGS.allow_model_choice,
        "allow_precision_choice": ARGS.allow_precision_choice,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "active_jobs": active,
        "loaded_models": loaded,
        "model_cache": ARGS.model_cache,
        "max_upload_mb": ARGS.max_upload_mb,
        "retry_available": ARGS.source_retention != "run",
        "allow_diarize": ARGS.allow_diarize,
        "diarize_max_speakers": DIARIZE_MAX_SPEAKERS,
        "speaker_models": list(DIARIZE_EMBEDDINGS),
        "default_speaker_model": ARGS.diarization_embedding,
        "allow_speaker_model_choice": ARGS.diarization_embedding_choice,
        "prompt_limit": PROMPT_LIMIT,
        "hotwords_limit": HOTWORDS_LIMIT,
        # Only a boolean here: the audit dir and the failure text stay behind the
        # audit token. An operator watching transcription still needs to know
        # the trail is dropping events.
        "audit_degraded": AUDIT is not None and AUDIT.degraded(),
    }


class UploadTooLarge(Exception):
    """Raised by save_upload when a stream exceeds the configured ceiling."""

    def __init__(self, written: int) -> None:
        super().__init__(f"upload exceeded the limit at {written} bytes")
        self.written = written


def save_upload(source: Any, dest: Path, limit: int) -> int:
    """Stream an upload to disk, enforcing the ceiling as it goes.

    Deliberately synchronous: callers run it in a threadpool, because a
    multi-gigabyte copy on the event loop would stall every other request.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(dest.parent, 0o700)  # source audio is not for other local users
    written = 0
    with dest.open("wb") as fh:
        while chunk := source.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                raise UploadTooLarge(written)
            fh.write(chunk)
    with contextlib.suppress(OSError):
        os.chmod(dest, 0o600)
    return written


def discard_file(path: Path) -> None:
    """Remove a partial upload. Safe to call when it is already gone."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


@app.post("/api/jobs")
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(None),
    compute_type: str = Form(None),
    language: str = Form(""),
    vad: str = Form("true"),
    quality: str = Form("balanced"),
    prompt: str = Form(""),
    hotwords: str = Form(""),
    translate: str = Form("false"),
    condition: str = Form("false"),
    word_timestamps: str = Form("false"),
    min_silence_ms: int = Form(2000),
    speech_pad_ms: int = Form(400),
    diarize: str = Form("false"),
    speakers: int = Form(0),
    speaker_model: str = Form(""),
) -> dict[str, Any]:
    opts = build_opts(
        model,
        compute_type,
        language,
        vad,
        quality,
        prompt,
        hotwords,
        translate,
        condition,
        word_timestamps,
        min_silence_ms,
        speech_pad_ms,
        diarize,
        speakers,
        speaker_model,
    )

    with JOBS_LOCK:
        pending = sum(
            1 for j in JOBS.values() if j["state"] in ("queued", "loading", "running")
        )
    if pending >= ARGS.max_queue:
        audit_rejection("job.rejected", request, reason="queue-full", status=429)
        raise HTTPException(
            status_code=429, detail=f"Queue is full ({ARGS.max_queue} jobs)"
        )

    raw_name = Path(file.filename or "audio").name
    safe = "".join(c for c in raw_name if c.isalnum() or c in " ._-").strip()
    # Capped well under NAME_MAX: a long or non-ASCII name would otherwise
    # raise OSError and surface as an opaque 500.
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe[:100] or 'audio'}"

    limit = ARGS.max_upload_mb * 1024 * 1024
    try:
        written = await run_in_threadpool(save_upload, file.file, dest, limit)
    except UploadTooLarge as exc:
        await run_in_threadpool(discard_file, dest)
        audit_rejection(
            "job.rejected",
            request,
            file=raw_name[:200],
            bytes=exc.written,
            reason="upload-too-large",
            status=413,
        )
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {ARGS.max_upload_mb} MB limit",
        ) from None
    except BaseException:
        await run_in_threadpool(discard_file, dest)
        raise

    job_id = new_job(filename=raw_name[:200], path=dest, opts=opts)
    JOB_QUEUE.put(job_id)
    audit(
        "job.created",
        request,
        job=job_id,
        file=raw_name[:200],
        bytes=written,
        opts=audit_opts(opts),
    )
    store_prompt_sidecar(job_id, raw_name[:200], opts, source="upload")
    return {"id": job_id}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(
    job_id: str,
    request: Request,
    model: str = Form(None),
    compute_type: str = Form(None),
    language: str = Form(""),
    vad: str = Form("true"),
    quality: str = Form("balanced"),
    prompt: str = Form(""),
    hotwords: str = Form(""),
    translate: str = Form("false"),
    condition: str = Form("false"),
    word_timestamps: str = Form("false"),
    min_silence_ms: int = Form(2000),
    speech_pad_ms: int = Form(400),
    diarize: str = Form("false"),
    speakers: int = Form(0),
    speaker_model: str = Form(""),
) -> dict[str, Any]:
    """Re-run the same source audio with different settings, no re-upload."""
    old = get_job(job_id)
    source = Path(old["path"])
    if not source.exists():
        audit_rejection(
            "job.retry_rejected", request, job=job_id, reason="source-gone", status=409
        )
        raise HTTPException(
            status_code=409,
            detail="The source audio is no longer on disk; re-upload it",
        )

    opts = build_opts(
        model,
        compute_type,
        language,
        vad,
        quality,
        prompt,
        hotwords,
        translate,
        condition,
        word_timestamps,
        min_silence_ms,
        speech_pad_ms,
        diarize,
        speakers,
        speaker_model,
    )

    with JOBS_LOCK:
        pending = sum(
            1 for j in JOBS.values() if j["state"] in ("queued", "loading", "running")
        )
    if pending >= ARGS.max_queue:
        audit_rejection(
            "job.retry_rejected", request, job=job_id, reason="queue-full", status=429
        )
        raise HTTPException(
            status_code=429, detail=f"Queue is full ({ARGS.max_queue} jobs)"
        )

    new_id = new_job(filename=old["filename"], path=source, opts=opts)
    # Provenance on the job, so its job.done can say it re-transcribed this
    # source rather than being the first pass at it.
    patch_job(new_id, retry_of=job_id)
    JOB_QUEUE.put(new_id)
    audit(
        "job.retried",
        request,
        job=new_id,
        from_job=job_id,
        file=old["filename"],
        opts=audit_opts(opts),
    )
    store_prompt_sidecar(new_id, old["filename"], opts, source="retry", from_job=job_id)
    return {"id": new_id}


@app.post("/api/jobs/{job_id}/speakers")
def relabel_speakers_endpoint(
    job_id: str,
    request: Request,
    speakers: int = Form(0),
    speaker_model: str = Form(""),
) -> dict[str, Any]:
    """Identify the speakers again with another count, without transcribing.

    Auto speaker counting is the unreliable part of diarization -- call audio
    can turn two people into five -- and Retry fixes a wrong count only by
    transcribing the whole file again. This queues a new job over the same
    audio and the transcript the old one already has, so the fix costs the
    speaker pass alone. The old job is left as it was, as Retry leaves it.
    """
    old = get_job(job_id)
    refusal = relabel_refusal(old)
    if refusal is not None:
        audit_rejection(
            "job.relabel_rejected", request, job=job_id, reason=refusal, status=409
        )
        raise HTTPException(status_code=409, detail=refusal)

    with JOBS_LOCK:
        pending = sum(
            1 for j in JOBS.values() if j["state"] in ("queued", "loading", "running")
        )
    if pending >= ARGS.max_queue:
        audit_rejection(
            "job.relabel_rejected", request, job=job_id, reason="queue-full", status=429
        )
        raise HTTPException(
            status_code=429, detail=f"Queue is full ({ARGS.max_queue} jobs)"
        )

    opts = dict(old["opts"])
    opts["diarize"] = True
    opts["speakers"] = clamp(as_int(speakers, 0), 0, DIARIZE_MAX_SPEAKERS)
    # Relabelling with the other model is the cheapest way to compare them.
    opts["speaker_model"] = pick_speaker_model(speaker_model)
    new_id = new_job(filename=old["filename"], path=Path(old["path"]), opts=opts)
    # Whisper's lines, not the last pass's split of them: a new count has to be
    # free to draw the speaker changes somewhere else.
    patch_job(
        new_id,
        relabel_of=job_id,
        transcribed=[dict(seg) for seg in old.get("transcribed") or old["segments"]],
        language=old.get("language"),
        duration=old.get("duration"),
    )
    JOB_QUEUE.put(new_id)
    audit(
        "job.relabelled",
        request,
        job=new_id,
        from_job=job_id,
        file=old["filename"],
        speakers=opts["speakers"],
        speaker_model=opts["speaker_model"],
    )
    store_prompt_sidecar(
        new_id, old["filename"], opts, source="relabel", from_job=job_id
    )
    return {"id": new_id}


@app.post("/api/jobs/{job_id}/speakers/merge")
def merge_speakers(
    job_id: str, request: Request, speaker: int = Form(...), into: int = Form(...)
) -> dict[str, Any]:
    """Give every line of one speaker to another, then renumber 1..N.

    Edits the finished job in place -- a merge moves numbers, nothing else --
    so the segments, the per-speaker totals and every export follow it.
    labels_rev goes up so an open card notices: nothing a poll signs for (state,
    progress, segment count) changes otherwise.
    """
    # Read, change and write under one hold of the lock: two quick merges from
    # two tabs must not both start from the same segments. Nothing in here
    # calls get_job or patch_job, which take the lock themselves.
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        if job["state"] != "done":
            raise HTTPException(
                status_code=409, detail="Only a finished job's speakers can be merged"
            )
        present = {segment_speaker(x) for x in job["segments"]} - {None}
        if speaker == into or speaker not in present or into not in present:
            raise HTTPException(
                status_code=400, detail="Pick two different speakers on this job"
            )
        moved = [
            {**x, "speaker": into} if segment_speaker(x) == speaker else x
            for x in job["segments"]
        ]
        job["segments"], renumbered = renumber_segments(moved)
        job["labels_rev"] = job.get("labels_rev", 0) + 1
        filename = job["filename"]
        # The target keeps its own name, or takes the merged speaker's when it
        # had none; then every name follows its speaker's new number.
        names = dict(job.get("speaker_names") or {})
        if str(speaker) in names:
            names.setdefault(str(into), names[str(speaker)])
            del names[str(speaker)]
        names = {
            str(renumbered[int(k)]): v for k, v in names.items() if int(k) in renumbered
        }
        named = bool(names) or bool(job.get("speaker_names"))
        job["speaker_names"] = names
    audit(
        "job.speakers_merged",
        request,
        job=job_id,
        file=filename,
        speaker=speaker,
        into=into,
        speakers=len(renumbered),
    )
    if named:
        store_speaker_names(job_id, filename, names)
    return {"renumbered": {str(old): new for old, new in renumbered.items()}}


# A name, not a paragraph; and nothing that would break an export -- `<` and `>`
# would end WebVTT's <v Name> voice span, control characters a line.
SPEAKER_NAME_LIMIT = 60
SPEAKER_NAME_BAD = re.compile(r"[<>\x00-\x1f\x7f]")


@app.post("/api/jobs/{job_id}/speakers/name")
def name_speaker(
    job_id: str, request: Request, speaker: int = Form(...), name: str = Form("")
) -> dict[str, Any]:
    """Give a speaker a name, or take it away with an empty one.

    A list of names is a list of who was in the room -- the category of data
    prompts and hotwords are. The app sees it (whoever named the speaker reads
    the transcript anyway), but the main audit log records only its length and
    hash; the text is kept in the job's sidecar, readable with the audit token.
    """
    value = " ".join((name or "").split())
    if len(value) > SPEAKER_NAME_LIMIT or SPEAKER_NAME_BAD.search(name or ""):
        raise HTTPException(
            status_code=400,
            detail=f"A name is up to {SPEAKER_NAME_LIMIT} characters, "
            "without < > or line breaks",
        )
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        if job["state"] != "done":
            raise HTTPException(
                status_code=409, detail="Only a finished job's speakers can be named"
            )
        present = {segment_speaker(x) for x in job["segments"]} - {None}
        if speaker not in present:
            raise HTTPException(status_code=400, detail="No such speaker on this job")
        names = dict(job.get("speaker_names") or {})
        if value:
            names[str(speaker)] = value
        else:
            names.pop(str(speaker), None)
        job["speaker_names"] = names
        job["labels_rev"] = job.get("labels_rev", 0) + 1
        filename = job["filename"]
    audit(
        "job.speaker_named",
        request,
        job=job_id,
        file=filename,
        speaker=speaker,
        cleared=not value,
        name_len=len(value),
        name_sha256=digest(value),
    )
    store_speaker_names(job_id, filename, names)
    return {"speaker_names": names}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)
        return {"jobs": [job_public(j, include_segments=False) for j in jobs]}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str, since: int = 0) -> dict[str, Any]:
    """One job, optionally only the segments after `since`.

    The page polls this while a job runs. Re-sending the whole transcript on
    every tick is quadratic in the length of the recording — an hour-long
    meeting is hundreds of ticks over a transcript that keeps growing — so the
    client says how many segments it already has and gets only the tail.

    `since=0` (the default) still returns everything, so this stays a plain
    detail endpoint for anything that is not the polling loop.
    """
    job = get_job(job_id)
    # One read of the list, and the count, the tail and the label shape all come
    # from it. The worker swaps in a longer list after every segment instead of
    # growing this one, so a second read can see a different list -- and a
    # count from one next to a tail from the other is what the page takes for a
    # transcript that went backwards.
    segments = job["segments"]
    out = job_public(job, include_segments=False)
    total = len(segments)
    start = clamp(as_int(since, 0), 0, total)
    out["segment_count"] = total
    out["segments"] = segments[start:]
    out["segment_start"] = start
    # Speaker labels land in one batch when the diarization pass finishes, so a
    # tail carrying no new segments would never reveal them. Reporting the shape
    # lets the client notice and refetch, which is the only way it can rebuild
    # rows that were already drawn without labels.
    out["speaker_labels"] = any(s.get("speaker") is not None for s in segments)
    # Per-speaker totals for the card's speaker chips. Computed from the same
    # snapshot as the tail, and only read when the list says something changed.
    out["speakers"] = speaker_summary(segments, job.get("speaker_names"))
    return out


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, request: Request) -> dict[str, Any]:
    job = get_job(job_id)
    if job["state"] in ("queued", "loading", "running"):
        patch_job(job_id, state="cancelled", message="Cancelled")
        audit(
            "job.cancelled",
            request,
            job=job_id,
            file=job["filename"],
            state=job["state"],
        )
        # The record stays, and 'job' retention keeps the audio as long as the
        # record does: that is what lets a job cancelled over a wrong model or
        # language be retried. Remove, or eviction, is what lets it go.
        if ARGS.source_retention == "run":
            drop_source(job)
    else:
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        audit(
            "job.deleted",
            request,
            job=job_id,
            file=job["filename"],
            state=job["state"],
            segments=len(job["segments"]),
        )
        drop_source(job)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/text")
def job_text(
    job_id: str, request: Request, format: str = "txt", download: int = 0
) -> Response:
    job = get_job(job_id)
    body, mime = render(job, format)
    # The moment a transcript leaves the machine is the interesting one. It is
    # also the event an app-token holder can repeat in a loop, so a burst from
    # one source is collapsed rather than written out in full.
    audit_repeated(
        "transcript.exported",
        request,
        job=job_id,
        file=job["filename"],
        format=format,
        download=bool(download),
        segments=len(job["segments"]),
        bytes=len(body),
    )
    headers = {}
    if download:
        ext = {"timestamped": "txt"}.get(format, format)
        headers["Content-Disposition"] = content_disposition(
            Path(job["filename"]).stem, ext
        )
    return Response(content=body, media_type=mime, headers=headers)


# --------------------------------------------------------------------------- #
# Audit API
# --------------------------------------------------------------------------- #


@app.get("/api/audit")
def audit_query(
    request: Request,
    date: str = "",
    limit: int = 200,
    offset: int = 0,
    job: str = "",
    q: str = "",
    include_prompts: int = 0,
    before_line: int = 0,
) -> dict[str, Any]:
    if AUDIT is None:
        raise HTTPException(status_code=503, detail="Audit trail unavailable")

    limit = clamp(as_int(limit, 200), 1, 1000)
    offset = max(0, as_int(offset, 0))
    cursor = max(0, as_int(before_line, 0)) or None
    prompt_jobs: list[str] = []
    days = AUDIT.dates()
    day = (date or "").strip() or (
        days[0] if days else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    job = job.strip()
    if job and not SAFE_JOB_ID.fullmatch(job):
        raise HTTPException(status_code=400, detail="Malformed job id")

    needle = q.strip() or None
    page, scan_mode = AUDIT.read_page_mode(
        day,
        limit=limit,
        offset=offset,
        job=job or None,
        needle=needle,
        before_line=cursor,
    )
    raw, total = page.lines, page.total

    events: list[dict[str, Any]] = []
    for line in raw:
        try:
            events.append(json.loads(line))
        except ValueError:
            events.append({"event": "unparsable", "raw": line[:500]})

    if include_prompts:
        # Which jobs' text this read actually disclosed. The aggregate flag is
        # not enough: prompt text is the one thing the audit token gates, so a
        # bulk read has to name what it handed over, exactly as the single-job
        # route does. Never the text itself.
        seen_jobs: set[str] = set()
        for rec in events:
            jid = str(rec.get("job") or "")
            side = AUDIT.read_prompt(jid)
            if side:
                rec["prompt_text"] = side.get("prompt") or ""
                rec["hotwords_text"] = side.get("hotwords") or ""
                if jid and jid not in seen_jobs:
                    seen_jobs.add(jid)
                    prompt_jobs.append(jid)

    audit(
        "audit.accessed",
        request,
        date=day,
        returned=len(events),
        matched=total,
        # Never the search string itself: an audit reader could otherwise copy
        # prompt text (or anything else) into the authoritative record.
        search_len=len(needle) if needle else 0,
        search_sha256=digest(needle or ""),
        job=job or None,
        include_prompts=bool(include_prompts),
        before_line=cursor,
        prompts_returned=len(prompt_jobs) or None,
        prompt_jobs=prompt_jobs[:50] or None,
    )
    return {
        "date": day,
        "dates": days,
        "events": events,
        "total": total,
        "offset": offset,
        "limit": limit,
        # Paging state: hand next_before_line back as before_line to get the next
        # page without the drift an offset has as records land.
        "next_before_line": page.cursor,
        "has_more": page.has_more,
        # Which reader answered: the tail (indexed) or a whole-file scan. A
        # filter forces the scan, and so can the file moving under the tail
        # reader, so this comes from the reader rather than from the filters.
        "scan_mode": scan_mode,
        "retain_days": AUDIT.retain_days,
        "prompts_available": AUDIT.store_prompts,
        "last_error": AUDIT.last_error,
        "lost_total": AUDIT.lost_total,
        "sidecar_lost_total": AUDIT.sidecar_lost_total,
        "degraded": AUDIT.degraded(),
        "full": AUDIT.full,
        "suppressed": AUDIT.suppressed,
        "max_mb": AUDIT.max_mb,
    }


def audit_day_or_400(date: str) -> str:
    """The day to act on: the requested one, else the newest that exists."""
    days = AUDIT.dates() if AUDIT is not None else []
    day = (date or "").strip() or (
        days[0] if days else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    return day


@app.get("/api/audit/verify")
def audit_verify(request: Request, date: str = "") -> dict[str, Any]:
    """Recompute one day's hash chain and report the first break.

    Read-only over the day file; the audit credential gates it like the rest of
    /api/audit, because naming what was edited is itself trail material.
    """
    if AUDIT is None:
        raise HTTPException(status_code=503, detail="Audit trail unavailable")
    day = audit_day_or_400(date)
    if day not in AUDIT.dates():
        raise HTTPException(status_code=404, detail="No audit file for that date")
    result = AUDIT.verify(day)
    # The verdict describes the content that was verified, so it is recorded
    # after the walk. Its `head` is therefore the verified head, not the head
    # including this record; /api/audit/head is the endpoint for anchoring.
    audit(
        "audit.verified",
        request,
        date=day,
        ok=result["ok"],
        checked=result["checked"],
        legacy=result["legacy"],
        reason=result["reason"],
    )
    return result


@app.get("/api/audit/head")
def audit_head(request: Request, date: str = "") -> dict[str, Any]:
    """One day's final link, for keeping an anchor off this machine.

    The chain alone cannot stop someone with write access from recomputing it;
    a head value copied somewhere else is what turns an edit into a detectable
    one.
    """
    if AUDIT is None:
        raise HTTPException(status_code=503, detail="Audit trail unavailable")
    day = audit_day_or_400(date)
    if day not in AUDIT.dates():
        raise HTTPException(status_code=404, detail="No audit file for that date")
    # Log the read before answering. The access record lands in today's file, so
    # reading today's head would otherwise return a value the very next write
    # invalidates -- and the head is meant to be copied somewhere else as an
    # anchor, so it has to be the state a caller would see on disk.
    audit("audit.head_read", request, date=day)
    head = AUDIT.head(day)
    return {"date": day, **head}


@app.get("/api/audit/prompts/{job_id}")
def audit_prompt(job_id: str, request: Request) -> dict[str, Any]:
    if AUDIT is None:
        raise HTTPException(status_code=503, detail="Audit trail unavailable")
    side = AUDIT.read_prompt(job_id)
    if side is None:
        raise HTTPException(status_code=404, detail="No stored prompt for that job")
    audit("audit.prompt_read", request, job=job_id)
    return side


def stats_public(agg: dict[str, Any]) -> dict[str, Any]:
    """An aggregate as the page consumes it: floats rounded, nothing added.

    Deep enough for the nested maps: the caller may be serialising this while
    another request folds the same cached aggregate, and a shared inner dict
    would let the response change under the encoder.
    """
    out: dict[str, Any] = {}
    for key, value in agg.items():
        if isinstance(value, float):
            out[key] = round(value, 2)
        elif isinstance(value, dict):
            out[key] = dict(value)
        else:
            out[key] = value
    return out


@app.get("/api/stats")
def stats(request: Request) -> dict[str, Any]:
    """Totals over this run, and over every retained day.

    Behind the audit credential, like the trail it is derived from. Counts,
    durations and byte totals only -- no filenames, no prompt text, no speaker
    names -- so it can never become a way around the two-token split.

    Sync on purpose: folding a day is file work, and an `async def` here would
    run it in the event loop and stall every poll, exactly like the diarization
    deadlock this project already guards against.
    """
    if AUDIT is None:
        raise HTTPException(status_code=503, detail="Audit trail unavailable")
    days = AUDIT.dates()  # newest first
    # Logged before answering: reading the trail leaves a trace, and that trace
    # lands in today's numbers -- so today's row moves when this is called, and
    # the page must not present it as settled.
    audit("audit.stats_read", request, days=len(days), recording=AUDIT.enabled)
    rows: list[dict[str, Any]] = []
    retained = blank_stats()
    for day in days:
        agg = AUDIT.day_summary(day)
        merge_stats(retained, agg)
        rows.append({"date": day, **stats_public(agg)})
    rows.reverse()  # oldest first: a table or a chart reads left to right
    return {
        "session": stats_public(AUDIT.session.snapshot()),
        "session_since": AUDIT.session.since,
        "days": rows,
        "retained": stats_public(retained),
        # False under --no-audit: the session above still counts what this
        # process did, but there are no files, so there is no history at all.
        "recording": AUDIT.enabled,
        "retain_days": AUDIT.retain_days,
        "max_mb": AUDIT.max_mb,
        "max_sidecars": AUDIT.max_sidecars,
        "lost_total": AUDIT.lost_total,
        "sidecar_lost_total": AUDIT.sidecar_lost_total,
        "degraded": AUDIT.degraded(),
        "full": AUDIT.full,
    }


# --------------------------------------------------------------------------- #
# Front end
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def local_names(bind_host: str, extra: list[str]) -> tuple[set[str], set[str]]:
    """Hostnames this server will answer to. Anything else is a rebinding attempt.

    Returns (exact_names, suffixes). A --allow-host entry starting with
    "*." or "." becomes a suffix match, so --allow-host .trycloudflare.com
    covers the random hostnames `cloudflared tunnel --url` hands out.
    """
    names = {"localhost", "127.0.0.1", "::1", ANY_INTERFACE}
    suffixes: set[str] = set()
    with contextlib.suppress(OSError):
        hostname = socket.gethostname()
        names.add(hostname.lower())
        for info in socket.getaddrinfo(hostname, None):
            names.add(str(info[4][0]).lower())
    if bind_host not in (ANY_INTERFACE, "::"):
        names.add(bind_host.lower())
    for entry in extra:
        e = entry.strip().lower()
        if not e:
            continue
        if e.startswith("*."):
            e = e[1:]  # "*.example.com" -> ".example.com"
        if e.startswith("."):
            suffix = normalize_host(e)
            if suffix:
                suffixes.add("." + suffix.lstrip("."))
        else:
            norm = normalize_host(e)
            if norm:
                names.add(norm)
    return names, suffixes


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Every option the server understands. Config file keys, flag names and env
# vars all resolve into this shape; anything not listed here is rejected rather
# than silently ignored.
DEFAULTS: dict[str, Any] = {
    "config": None,
    "starter_config": True,
    "work_dir": None,
    "host": ANY_INTERFACE,
    "port": 8765,
    "token": "",
    "no_auth": False,
    "allow_host": [],
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "quality": "balanced",
    "model_cache": 1,
    "allow_model_choice": True,
    "allow_precision_choice": False,
    "allow_diarize": True,
    "diarization_embedding": "titanet-small",
    "diarization_embedding_choice": True,
    # None: the embedding model's own calibrated threshold.
    "diarization_threshold": None,
    "diarization_fold_share": DIARIZE_FOLD_SHARE,
    "preload": False,
    "max_upload_mb": 2048,
    "max_queue": 20,
    "max_jobs": 60,
    "source_retention": "job",
    "audit": True,
    "audit_dir": None,
    "audit_reads": False,
    "audit_prompts": True,
    "audit_retain_days": 30,
    # Per-day ceiling on a day file, in MB; 0 keeps everything. A bounded trail
    # that says where it stopped beats a full disk.
    "audit_max_mb": 1024,
    # Ceiling on the prompts/ directory, in files; oldest text first. The day
    # file keeps the length and hash of every prompt, so dropping the oldest
    # text loses the wording, not the evidence a prompt was used. 0 keeps all.
    "audit_max_sidecars": 5000,
    "audit_open": False,
    "audit_token": "",
}

CHOICES: dict[str, list[str]] = {
    "model": MODELS,
    "device": ["cuda", "cpu", "auto"],
    "compute_type": COMPUTE_TYPES,
    "quality": list(QUALITIES),
    "source_retention": RETENTION,
    "diarization_embedding": list(DIARIZE_EMBEDDINGS),
}

# Environment wins over both the config file and the flags, so a service
# wrapper can override whatever is on disk without rewriting it.
ENV_OPTIONS: dict[str, tuple[str, str]] = {
    "TRANSCRIBE_CONFIG": ("config", "str"),
    "TRANSCRIBE_STARTER_CONFIG": ("starter_config", "bool"),
    "TRANSCRIBE_WORK_DIR": ("work_dir", "str"),
    "TRANSCRIBE_HOST": ("host", "str"),
    "TRANSCRIBE_PORT": ("port", "int"),
    "TRANSCRIBE_TOKEN": ("token", "str"),
    "TRANSCRIBE_NO_AUTH": ("no_auth", "bool"),
    "TRANSCRIBE_ALLOW_HOST": ("allow_host", "list"),
    "TRANSCRIBE_MODEL": ("model", "str"),
    "TRANSCRIBE_DEVICE": ("device", "str"),
    "TRANSCRIBE_COMPUTE_TYPE": ("compute_type", "str"),
    "TRANSCRIBE_QUALITY": ("quality", "str"),
    "TRANSCRIBE_MODEL_CACHE": ("model_cache", "int"),
    "TRANSCRIBE_ALLOW_DIARIZE": ("allow_diarize", "bool"),
    "TRANSCRIBE_DIARIZATION_EMBEDDING": ("diarization_embedding", "str"),
    "TRANSCRIBE_DIARIZATION_EMBEDDING_CHOICE": ("diarization_embedding_choice", "bool"),
    "TRANSCRIBE_DIARIZATION_THRESHOLD": ("diarization_threshold", "float"),
    "TRANSCRIBE_DIARIZATION_FOLD_SHARE": ("diarization_fold_share", "float"),
    "TRANSCRIBE_PRELOAD": ("preload", "bool"),
    "TRANSCRIBE_MAX_UPLOAD_MB": ("max_upload_mb", "int"),
    "TRANSCRIBE_MAX_QUEUE": ("max_queue", "int"),
    "TRANSCRIBE_MAX_JOBS": ("max_jobs", "int"),
    "TRANSCRIBE_SOURCE_RETENTION": ("source_retention", "str"),
    "TRANSCRIBE_AUDIT": ("audit", "bool"),
    "TRANSCRIBE_AUDIT_DIR": ("audit_dir", "str"),
    "TRANSCRIBE_AUDIT_READS": ("audit_reads", "bool"),
    "TRANSCRIBE_AUDIT_PROMPTS": ("audit_prompts", "bool"),
    "TRANSCRIBE_AUDIT_RETAIN_DAYS": ("audit_retain_days", "int"),
    "TRANSCRIBE_AUDIT_MAX_MB": ("audit_max_mb", "int"),
    "TRANSCRIBE_AUDIT_MAX_SIDECARS": ("audit_max_sidecars", "int"),
    "TRANSCRIBE_AUDIT_OPEN": ("audit_open", "bool"),
    "TRANSCRIBE_AUDIT_TOKEN": ("audit_token", "str"),
}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [part.strip() for part in str(value).split(",") if part.strip()]


CONVERTERS = {"str": str, "int": int, "float": float, "bool": as_bool, "list": as_list}


def config_path(explicit: str | None, work_dir: str | None = None) -> tuple[Path, bool]:
    """Return (path, required). Only an explicit path must exist.

    Resolution order for the *default* location: an explicit --config, then
    TRANSCRIBE_CONFIG, then the resolved work dir (--work-dir, then
    TRANSCRIBE_WORK_DIR), then ~/.transcribe-server.
    """
    if explicit:
        return Path(explicit).expanduser(), True
    env = os.environ.get("TRANSCRIBE_CONFIG")
    if env:
        return Path(env).expanduser(), True
    base = work_dir or os.environ.get("TRANSCRIBE_WORK_DIR")
    root = Path(base).expanduser() if base else Path.home() / ".transcribe-server"
    return root / "config.toml", False


# Natural spellings for options whose flat name carries its section, so
# [audit] reads = true means audit_reads. The bare name still works too.
CONFIG_ALIASES: dict[str, str] = {
    "audit.enabled": "audit",
    "audit.reads": "audit_reads",
    "audit.dir": "audit_dir",
    "audit.prompts": "audit_prompts",
    "audit.retain_days": "audit_retain_days",
    "audit.max_mb": "audit_max_mb",
    "audit.max_sidecars": "audit_max_sidecars",
    "audit.open": "audit_open",
    "audit.token": "audit_token",
}


def load_config_file(path: Path, required: bool) -> dict[str, Any]:
    """Flatten [section] tables into one dict of option names."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if required:
            raise SystemExit(f"!  No config file at {path}") from None
        return {}
    except OSError as exc:
        raise SystemExit(f"!  Could not read {path}: {exc}") from exc

    try:
        raw = load_toml(text)
    except Exception as exc:  # noqa: BLE001 - tomllib raises several types
        raise SystemExit(f"!  {path} is not valid TOML: {exc}") from exc

    flat: dict[str, Any] = {}
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise SystemExit(f"!  {path}: [{section}] must be a table of options")
        prefix = section.strip().replace("-", "_").lower()
        for key, value in values.items():
            name = key.strip().replace("-", "_").lower()
            name = CONFIG_ALIASES.get(f"{prefix}.{name}", name)
            if name not in DEFAULTS or name == "config":
                name = f"{prefix}_{name}"
            if name not in DEFAULTS or name == "config":
                raise SystemExit(f"!  {path}: unknown option {section}.{key}")
            flat[name] = value
    return flat


# Handed to the operator on first run. Deliberately ships with every key
# commented out: a starter file that wrote the defaults down would pin them,
# so a later release could never move a default for anyone who had started the
# server once. A commented key keeps tracking the default instead.
CONFIG_TEMPLATE = """\
# Configuration for transcribe_server. The server writes this file once, at the
# default location, and never rewrites it -- edit it freely. It is skipped for
# an explicit --config, and --no-starter-config turns it off entirely.
#
# Every key below is commented out and shows its default. Uncomment only what
# you want to change.
#
# Precedence, lowest to highest: defaults < this file < flags < TRANSCRIBE_*
# environment variables. An unknown or misspelled key stops startup instead of
# being ignored, so a typo cannot quietly leave auth off.
#
# The file is created readable only by you. Prefer TRANSCRIBE_TOKEN and
# TRANSCRIBE_AUDIT_TOKEN over the token keys below: a secret in a config file
# survives in backups and is easy to commit by accident.

[server]
# Bind address. 0.0.0.0 is every interface, which is the point on a LAN.
# host = "0.0.0.0"
# port = 8765
# Extra Host header values to accept, for tunnels and reverse proxies. A
# leading "." is a suffix match.
# allow_host = []
# Where uploads and the audit trail live is deliberately not settable here:
# this file was found in that directory, so the key belongs to --work-dir and
# TRANSCRIBE_WORK_DIR, which move the state dir and this file together. Setting
# it here is a startup error, not a silent no-op.
# Pin the access token. Omit it and a fresh one is generated and printed at
# every startup, which logs out every device on restart.
# token = ""
# Serve without a token at all. Only on a network you control: anyone who can
# reach the port can then read every transcript and upload files.
# no_auth = false

[model]
# One of: large-v3, large-v3-turbo, medium, small, base
# model = "large-v3"
# cuda stops startup if the GPU cannot be used; auto falls back to the CPU.
# device = "cuda"
# float16 needs compute capability 7.0+; Pascal and older want int8 or float32.
# compute_type = "float16"
# Default beam size: fast=1, balanced=5, thorough=8
# quality = "balanced"
# Load the model at startup instead of on the first job.
# preload = false
# Models held in VRAM at once. Raising this keeps several model and precision
# combinations resident instead of reloading them.
# model_cache = 1
# Let clients pick the model per job (--pin-model turns this back off).
# allow_model_choice = true
# Expose the precision selector in the UI. Precision is a property of the
# machine, not the recording, so this is off by default.
# allow_precision_choice = false

[diarization]
# Offer speaker identification ("Identify speakers") per job. Labels are
# anonymous and per-recording: Speaker 1 here is not Speaker 1 in the next
# file. Off means the control is hidden and its ~42 MB of models are never
# fetched. A diarized job is slower and needs word-level timings, which the
# server switches on for it.
# allow_diarize = true
# The model that tells voices apart: "titanet-small" (the default) or
# "eres2net-en", smaller, Apache-2.0, and a little better when you pin the
# speaker count. Measured against each other in docs/speaker-diarization.md.
# embedding = "titanet-small"
# Let each job pick the voice model from the page ("Voice model"). Off pins
# every job to the model above, as pin_model does for Whisper.
# embedding_choice = true
# How alike two stretches of voice must be to count as one speaker, when the
# count is left on Auto. Higher merges more, lower splits more; a pinned count
# ignores it. Unset, it follows the model: 0.8 for titanet-small, 0.9 for
# eres2net-en.
# threshold = 0.8
# On Auto, a speaker holding less than this share of the talk time is folded
# into the voice nearest it in time -- usually a laugh or a raised voice heard
# as someone new. 0 turns folding off.
# fold_share = 0.03

[limits]
# max_upload_mb = 2048
# max_queue = 20
# Finished job records kept before the oldest are evicted.
# max_jobs = 60
# run deletes the audio as soon as the job finishes (no retry), job keeps it
# while the record exists, forever never deletes it.
# source_retention = "job"

[audit]
# enabled = true
# Where the daily files and the prompt sidecars live. Default: <work dir>/audit
# dir = "/mnt/big/transcribe-state/audit"
#
# A Windows path needs quoting that TOML accepts. In a double-quoted string a
# backslash starts an escape, so "C:\\Users\\me\\audit" is a parse error (\\U is a
# unicode escape). Single quotes are literal; a double-quoted string needs every
# backslash doubled:
#   'C:\\Users\\me\\audit'
#   "C:\\\\Users\\\\me\\\\audit"
#
# Also log status/list/detail polls. Chatty, so off by default.
# reads = false
# Store prompt and hotword text in the sidecar files. Turning this off keeps
# the hashes and lengths in the main log and records no text anywhere.
# prompts = true
# Delete audit files older than this many days at startup; 0 keeps everything.
# retain_days = 30
# Stop recording once a day file reaches this many MB, after writing one
# audit.full marker; the cap resets at the next day. 0 keeps everything.
# max_mb = 1024
# Keep at most this many prompt sidecar files, oldest text first. The trail
# still records every prompt's length and hash. 0 keeps every sidecar.
# max_sidecars = 5000
# Serve the trail with no credential at all, prompt and hotword text included.
# Only on a network you control. Refused together with the token below.
# open = false
# Separate token for GET /audit and /api/audit. Generated for this run when
# left empty and printed at startup, so the endpoint works unconfigured; pin it
# to keep bookmarks and scripts working across restarts.
# token = ""
"""


def init_starter_config(path: Path, required: bool, enabled: bool) -> bool:
    """Create the commented starter config. True only if this call created it.

    Only the *default* location gets one. An explicit --config names a file the
    operator expects to exist, and creating it would turn a typo into a server
    quietly running on defaults.
    """
    if required or not enabled:
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL makes "only if absent" atomic instead of a stat-then-write
        # race, and sets the mode before anything is in the file.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    except OSError as exc:
        # A read-only or container home is not a reason to refuse to serve.
        print(f"!  Could not create a starter config at {path}: {exc}")
        print("   Continuing with defaults. Create the file yourself, or pass")
        print("   --no-starter-config to stop seeing this.\n")
        return False

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(CONFIG_TEMPLATE)
    except OSError as exc:
        # Half a template is worse than none: the next start would reject it as
        # invalid TOML. The file is ours by construction, so drop it.
        with contextlib.suppress(OSError):
            path.unlink()
        print(f"!  Could not write the starter config at {path}: {exc}\n")
        return False
    return True


def check_default_state_dir(cfg_path: Path, value: Any, target: Path) -> None:
    """Refuse a default-location config that moves the state dir out from under itself.

    The default config path is <work dir>/config.toml, so `work_dir` inside that
    file cannot move the file it was found in: the state would move, this file
    would not, and a config written next to the new state would be ignored with
    nothing on screen to say so. --work-dir and TRANSCRIBE_WORK_DIR move both,
    which is what someone reaching for this key actually wants. An explicit
    --config is fixed independently of the work dir, so the key is honest there
    and stays allowed.
    """
    if target.resolve() == cfg_path.parent.resolve():
        return  # a no-op: the file already sits in the dir it names
    raise SystemExit(
        f"!  {cfg_path} sets work_dir = {value!r}\n"
        f"   This file lives in the state dir, so it cannot move the state dir: the\n"
        f"   state would move, this file would not, and a config written in\n"
        f"   {target} would be silently ignored. Use --work-dir {target} (or\n"
        f"   TRANSCRIBE_WORK_DIR) to move both, or delete the key."
    )


def build_parser() -> argparse.ArgumentParser:
    """Flags use SUPPRESS so an unset flag is absent, letting the config file
    and environment fill it in rather than a hard-coded argparse default."""
    p = argparse.ArgumentParser(
        description="LAN transcription server (faster-whisper)",
        epilog="Options may also come from a TOML config file "
        "(default: <work dir>/config.toml) and from TRANSCRIBE_* "
        "environment variables, which win over flags.",
    )

    net = p.add_argument_group("network and access")
    net.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="TOML config file (default: <work dir>/config.toml)",
    )
    net.add_argument(
        "--no-starter-config",
        dest="starter_config",
        action="store_false",
        default=argparse.SUPPRESS,
        help="never create a starter config at the default location",
    )
    net.add_argument(
        "--work-dir",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="state directory for uploads and the audit trail "
        "(default: ~/.transcribe-server)",
    )
    net.add_argument(
        "--host",
        default=argparse.SUPPRESS,
        help="bind address (default: all interfaces)",
    )
    net.add_argument("--port", type=int, default=argparse.SUPPRESS)
    net.add_argument(
        "--token",
        default=argparse.SUPPRESS,
        help="access token; one is generated if omitted (prefer TRANSCRIBE_TOKEN "
        "over the flag, which is visible in the process list)",
    )
    net.add_argument(
        "--no-auth",
        action="store_true",
        default=argparse.SUPPRESS,
        help="serve without a token (only on a network you control)",
    )
    net.add_argument(
        "--allow-host",
        action="append",
        default=argparse.SUPPRESS,
        metavar="NAME",
        help="extra Host header value to accept; repeatable. "
        'Prefix with "." for a suffix match, e.g. --allow-host '
        ".trycloudflare.com for Cloudflare tunnels.",
    )

    gpu = p.add_argument_group("model and hardware")
    gpu.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        choices=MODELS,
        help="default model",
    )
    gpu.add_argument(
        "--device",
        default=argparse.SUPPRESS,
        choices=["cuda", "cpu", "auto"],
        help="cuda (default) stops startup if the GPU cannot be used; "
        "auto falls back to CPU instead; cpu skips the GPU entirely",
    )
    gpu.add_argument(
        "--compute-type",
        default=argparse.SUPPRESS,
        choices=COMPUTE_TYPES,
        help="float16 needs compute capability 7.0+; use int8 or float32 "
        "on Pascal and older",
    )
    gpu.add_argument(
        "--quality",
        default=argparse.SUPPRESS,
        choices=list(QUALITIES),
        help="default beam size: fast=1, balanced=5, thorough=8",
    )
    gpu.add_argument(
        "--model-cache",
        type=int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="models held in VRAM at once (default 1; raising this lets "
        "several model/precision combinations stay resident)",
    )
    gpu.add_argument(
        "--allow-model-choice",
        dest="allow_model_choice",
        action="store_true",
        default=argparse.SUPPRESS,
        help="let clients pick the model (default)",
    )
    gpu.add_argument(
        "--pin-model",
        dest="allow_model_choice",
        action="store_false",
        default=argparse.SUPPRESS,
        help="force every job to use --model",
    )
    gpu.add_argument(
        "--allow-precision-choice",
        dest="allow_precision_choice",
        action="store_true",
        default=argparse.SUPPRESS,
        help="expose the precision selector in the UI (off by default: "
        "precision is a property of the machine, not the recording)",
    )
    gpu.add_argument(
        "--no-diarize",
        dest="allow_diarize",
        action="store_false",
        default=argparse.SUPPRESS,
        help="remove the speaker-identification control entirely, so the "
        "diarization models are never fetched or loaded",
    )
    gpu.add_argument(
        "--diarization-embedding",
        default=argparse.SUPPRESS,
        choices=list(DIARIZE_EMBEDDINGS),
        help="the speaker-embedding model (default titanet-small)",
    )
    gpu.add_argument(
        "--pin-diarization-embedding",
        dest="diarization_embedding_choice",
        action="store_false",
        default=argparse.SUPPRESS,
        help="use --diarization-embedding for every job; hide the page's "
        "Voice model selector",
    )
    gpu.add_argument(
        "--diarization-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="clustering distance for Auto speaker counts: higher merges more "
        "(default: the embedding model's own, 0.8 for titanet-small)",
    )
    gpu.add_argument(
        "--diarization-fold-share",
        type=float,
        default=argparse.SUPPRESS,
        help="on Auto, fold speakers under this share of the talk time into "
        f"their neighbours; 0 turns it off (default {DIARIZE_FOLD_SHARE:g})",
    )
    gpu.add_argument(
        "--preload",
        action="store_true",
        default=argparse.SUPPRESS,
        help="load the default model at startup instead of on first job",
    )

    lim = p.add_argument_group("limits and storage")
    lim.add_argument(
        "--max-upload-mb",
        type=int,
        default=argparse.SUPPRESS,
        help="per-file upload ceiling",
    )
    lim.add_argument(
        "--max-queue",
        type=int,
        default=argparse.SUPPRESS,
        help="max jobs pending at once",
    )
    lim.add_argument(
        "--max-jobs",
        type=int,
        default=argparse.SUPPRESS,
        help="finished job records retained before the oldest are evicted",
    )
    lim.add_argument(
        "--source-retention",
        default=argparse.SUPPRESS,
        choices=RETENTION,
        help="'run' deletes the upload right after transcription (no retry), "
        "'job' keeps it while the job record exists (default), "
        "'forever' never deletes it",
    )

    aud = p.add_argument_group("audit trail")
    aud.add_argument(
        "--audit-dir",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="where daily audit-YYYY-MM-DD.jsonl files live "
        "(default: <work dir>/audit)",
    )
    aud.add_argument(
        "--no-audit",
        dest="audit",
        action="store_false",
        default=argparse.SUPPRESS,
        help="stop recording the audit trail (existing files stay readable)",
    )
    aud.add_argument(
        "--audit-reads",
        action="store_true",
        default=argparse.SUPPRESS,
        help="also log status/list/detail polls (chatty; off by default)",
    )
    aud.add_argument(
        "--no-audit-prompts",
        dest="audit_prompts",
        action="store_false",
        default=argparse.SUPPRESS,
        help="never write prompt/hotword text to the sidecar files",
    )
    aud.add_argument(
        "--audit-retain-days",
        type=int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="delete audit files older than N days at startup (default 30; "
        "0 keeps everything)",
    )
    aud.add_argument(
        "--audit-max-mb",
        type=int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="stop recording once a day file reaches N MB, after one "
        "audit.full marker (default 1024; 0 keeps everything)",
    )
    aud.add_argument(
        "--audit-max-sidecars",
        type=int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="keep at most N prompt sidecar files, oldest text first "
        "(default 5000; 0 keeps everything)",
    )
    aud.add_argument(
        "--audit-open",
        action="store_true",
        default=argparse.SUPPRESS,
        help="drop the audit credential: /audit and /api/audit become public, "
        "prompt and hotword text included (only on a network you control)",
    )
    aud.add_argument(
        "--audit-token",
        default=argparse.SUPPRESS,
        help="separate token for GET /audit and /api/audit; generated for this "
        "run when unset, like the app token (prefer TRANSCRIBE_AUDIT_TOKEN)",
    )
    return p


def resolve_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, Path, bool]:
    """Layer defaults, config file, flags and environment, in that order."""
    cli = vars(build_parser().parse_args(argv))

    # --work-dir moves the default config location too, not just the state dir.
    path, required = config_path(cli.get("config"), cli.get("work_dir"))
    file_values = load_config_file(path, required)

    merged: dict[str, Any] = dict(DEFAULTS)
    merged.update(file_values)
    merged.update(cli)

    for env_name, (name, kind) in ENV_OPTIONS.items():
        if name == "allow_host":
            continue  # additive, handled below
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            merged[name] = CONVERTERS[kind](raw)
        except (TypeError, ValueError):
            raise SystemExit(f"!  {env_name}={raw!r} is not a valid {kind}") from None

    # --allow-host is repeatable, so it accumulates across every layer instead
    # of a flag silently replacing the configured tunnel hostname.
    hosts: list[str] = []
    hosts += as_list(file_values.get("allow_host") or [])
    hosts += as_list(cli.get("allow_host") or [])
    hosts += as_list(os.environ.get("TRANSCRIBE_ALLOW_HOST") or [])
    merged["allow_host"] = hosts

    # TOML is typed already, but a quoted "8765" should still work.
    for name, default in DEFAULTS.items():
        value = merged.get(name)
        if value is None:
            continue
        if isinstance(default, bool):
            merged[name] = as_bool(value)
        elif isinstance(default, int):
            try:
                merged[name] = int(value)
            except (TypeError, ValueError):
                raise SystemExit(
                    f"!  {name} must be an integer, got {value!r}"
                ) from None
        elif isinstance(default, float):
            try:
                merged[name] = float(value)
            except (TypeError, ValueError):
                raise SystemExit(f"!  {name} must be a number, got {value!r}") from None
        elif isinstance(default, list):
            merged[name] = as_list(value)

    for name, allowed in CHOICES.items():
        if merged[name] not in allowed:
            raise SystemExit(
                f"!  {name}={merged[name]!r} is not one of: {', '.join(allowed)}"
            )

    if not 0 < merged["port"] < 65536:
        raise SystemExit(f"!  port must be 1-65535, got {merged['port']}")
    for name in ("max_upload_mb", "max_queue"):
        if merged[name] < 1:
            raise SystemExit(f"!  {name} must be at least 1, got {merged[name]}")
    for name in (
        "model_cache",
        "max_jobs",
        "audit_retain_days",
        "audit_max_mb",
        "audit_max_sidecars",
    ):
        if merged[name] < 0:
            raise SystemExit(f"!  {name} cannot be negative, got {merged[name]}")
    # Unset follows the model: each embedding has its own distance scale.
    if merged["diarization_threshold"] is None:
        merged["diarization_threshold"] = DIARIZE_EMBEDDINGS[
            merged["diarization_embedding"]
        ]["threshold"]
    try:
        merged["diarization_threshold"] = float(merged["diarization_threshold"])
    except (TypeError, ValueError):
        raise SystemExit(
            f"!  diarization_threshold must be a number, got "
            f"{merged['diarization_threshold']!r}"
        ) from None
    # A cosine-style distance: past 2 nothing ever merges, at 0 nothing does.
    if not 0 < merged["diarization_threshold"] <= 2:
        raise SystemExit(
            "!  diarization_threshold must be above 0 and at most 2, got "
            f"{merged['diarization_threshold']}"
        )
    if not 0 <= merged["diarization_fold_share"] < 0.5:
        raise SystemExit(
            "!  diarization_fold_share must be from 0 (off) to below 0.5, got "
            f"{merged['diarization_fold_share']}"
        )

    # Only the default location is self-referential, and the comparison is on the
    # effective value: --work-dir and TRANSCRIBE_WORK_DIR already moved the config
    # path along with the state, so they can only ever agree here.
    if not required and merged["work_dir"]:
        check_default_state_dir(
            path,
            merged["work_dir"],
            Path(str(merged["work_dir"])).expanduser(),
        )

    # Refused rather than resolved by precedence: the environment beats flags
    # here, so a service wrapper's TRANSCRIBE_AUDIT_TOKEN would otherwise be
    # dropped without a word by [audit] open = true in a config file.
    if merged["audit_open"] and merged["audit_token"]:
        raise SystemExit(
            "!  --audit-open and an audit token are mutually exclusive: "
            "--audit-open would ignore the token. Drop one, or unset "
            "TRANSCRIBE_AUDIT_TOKEN / [audit] token."
        )

    return argparse.Namespace(**merged), path, required


def startup_snapshot(
    args: argparse.Namespace, cfg: Path | None, audit_generated: bool = False
) -> dict[str, Any]:
    """How the server was running, for later forensics. Tokens never appear."""
    return {
        "config": redact(str(cfg), 500) if cfg else None,
        "host": args.host,
        "port": args.port,
        "work_dir": redact(str(WORK_DIR), 500),
        "device": args.device,
        "cuda": CUDA,
        "model": args.model,
        "compute_type": args.compute_type,
        "quality": args.quality,
        "model_cache": args.model_cache,
        "preload": args.preload,
        "allow_model_choice": args.allow_model_choice,
        "allow_precision_choice": args.allow_precision_choice,
        "allow_diarize": args.allow_diarize,
        "diarization_embedding": args.diarization_embedding,
        "diarization_embedding_choice": args.diarization_embedding_choice,
        "diarization_threshold": args.diarization_threshold,
        "diarization_fold_share": args.diarization_fold_share,
        "auth": bool(args.token),
        "audit": args.audit,
        # Where the trail actually wrote itself, and whether the credential was
        # ephemeral. Without these, a run with a custom --audit-dir or a
        # generated audit token cannot be told apart after the fact.
        "audit_dir": redact(str(AUDIT.dir), 500) if AUDIT is not None else None,
        "audit_generated": audit_generated,
        "audit_reads": args.audit_reads,
        "audit_prompts": args.audit_prompts,
        "audit_retain_days": args.audit_retain_days,
        "audit_max_mb": args.audit_max_mb,
        "audit_max_sidecars": args.audit_max_sidecars,
        "audit_api": audit_api_mode(args),
        "max_upload_mb": args.max_upload_mb,
        "max_queue": args.max_queue,
        "max_jobs": args.max_jobs,
        "source_retention": args.source_retention,
        "allowed_hosts": sorted(ALLOWED_HOSTS | ALLOWED_SUFFIXES),
    }


def sweep_uploads() -> tuple[int, int]:
    """Delete uploads left behind by a previous run. Returns (count, bytes)."""
    count = 0
    total = 0
    try:
        entries = list(UPLOAD_DIR.iterdir())
    except OSError:
        return 0, 0
    with JOBS_LOCK:
        referenced = {j["path"] for j in JOBS.values()}
    for path in entries:
        if not path.is_file() or str(path) in referenced:
            continue
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        count += 1
        total += size
    return count, total


def probe_diarization() -> None:
    """Prove the diarizer runs end to end, not merely that it imports.

    Same reasoning as verify_device: these weights are a separate download from
    the Whisper ones, and a truncated file fails deep inside ONNX Runtime rather
    than at load. One second of silence exercises the fetch, the checksum, the
    child process and a real inference — which is all the first job would do.
    """
    audio = WORK_DIR / "diarize-probe.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    try:
        run_diarizer(str(audio), 0, fetch_diarize_models_or_explain())
    finally:
        with contextlib.suppress(OSError):
            audio.unlink()


def preload_model(args: argparse.Namespace) -> None:
    """Load the default model and prove it can encode, or exit with advice.

    Loading alone is not enough: a missing CUDA runtime only surfaces on the
    first encode, so a plain "Model ready." can promise a GPU that cannot
    transcribe anything. Refusing to start beats serving jobs that all fail.
    """
    print(f"Loading {args.model} on {args.device} ({args.compute_type}) ...")
    try:
        model = load_model(args.model, args.device, args.compute_type)
        verify_device(model, args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"\n!  {args.model} on {args.device} cannot run: {exc}")
        if args.device in ("cuda", "auto"):
            print("   The model loaded but failed to encode, which usually means the")
            print("   CUDA runtime is missing. In order of preference:")
            print("     - run through uv so the declared runtime is installed:")
            print("         uv run transcribe_server.py --preload")
            print("     - fall back to the CPU:  --device cpu --compute-type int8")
        print("   Refusing to serve jobs that cannot succeed.\n")
        raise SystemExit(1) from None
    print("Model ready.")

    if not args.allow_diarize:
        return
    print("Checking speaker identification ...")
    try:
        probe_diarization()
    except Exception as exc:  # noqa: BLE001
        # A warning, not a refusal. Transcription is unaffected, and the failure
        # this is most likely to hit in practice -- a machine that cannot reach
        # github.com for the 42 MB of weights -- would otherwise take a working
        # server down with it. run_job treats a failed diarization as a degraded
        # job, so exiting 1 here would contradict that, and a service wrapper
        # would restart-loop without ever showing anyone why.
        print(f"\n!  speaker identification cannot run: {friendly_error(exc)}")
        print("   Transcription is unaffected; jobs will finish without labels.")
        print("   Fix the problem above, or start with --no-diarize to remove the")
        print("   control and stop it being attempted.\n")
    else:
        print("Speaker identification ready.")


def main() -> None:
    global ARGS, ALLOWED_HOSTS, ALLOWED_SUFFIXES, AUDIT, WORK_DIR, UPLOAD_DIR, READY
    global CUDA
    args, cfg_path, cfg_required = resolve_args()

    if args.no_auth:
        args.token = ""
        generated = False
    else:
        generated = not args.token
        if generated:
            args.token = secrets.token_urlsafe(24)

    # The audit API is on by default for the same reason the app API is: an
    # unset credential that silently disables the endpoint is a trap for
    # whoever later goes looking for /audit. --audit-open is the way out.
    audit_generated = False
    if not args.audit_open and not args.audit_token:
        args.audit_token = secrets.token_urlsafe(24)
        audit_generated = True

    WORK_DIR = (
        Path(args.work_dir).expanduser()
        if args.work_dir
        else Path.home() / ".transcribe-server"
    )
    UPLOAD_DIR = WORK_DIR / "uploads"
    audit_dir = (
        Path(args.audit_dir).expanduser() if args.audit_dir else WORK_DIR / "audit"
    )
    AUDIT = AuditLog(
        audit_dir,
        enabled=args.audit,
        retain_days=args.audit_retain_days,
        prompts=args.audit_prompts,
        max_mb=args.audit_max_mb,
        max_sidecars=args.audit_max_sidecars,
    )

    args.model_cache = max(1, args.model_cache)

    # Before anything loads a model: put the CUDA libraries where the loader can
    # see them, then find out whether the GPU is really usable. A broken GPU
    # must not look healthy here -- it would fail minutes into the first job.
    cuda = None
    if args.device != "cpu":
        decision = cuda_startup(args.device)
        args.device = decision["device"]
        if decision["lines"]:
            print()
            print("\n".join(decision["lines"]), flush=True)
            print()
        if decision["fatal"]:
            raise SystemExit(1)
        cuda = decision["report"]

    # A CPU fallback inherits a GPU precision unless we look, and CTranslate2
    # will not run float16 on a CPU at all.
    args.compute_type, precision_notice, precisions = resolve_precision(
        args.device, args.compute_type
    )
    PRECISIONS[:] = precisions  # in place: request handlers hold no reference
    if precision_notice:
        print(precision_notice, flush=True)
        print()

    ARGS = args
    DIARIZE_MODELS["embedding"] = DIARIZE_EMBEDDINGS[args.diarization_embedding]
    READY = True
    CUDA = cuda
    ALLOWED_HOSTS, ALLOWED_SUFFIXES = local_names(args.host, args.allow_host)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(UPLOAD_DIR, 0o700)

    # The state dir exists from here on, which makes this the first safe moment
    # for a starter config -- and it is deliberately after the CUDA check,
    # which must not leave files behind on its way out. The file is inert by
    # construction (every key is commented out), so writing it after the
    # config was read cannot change how this run is configured.
    created = init_starter_config(cfg_path, cfg_required, args.starter_config)
    if created:
        audit("config.created", path=redact(str(cfg_path), 500))

    # Job state is in memory only, so after a restart every file in uploads/ is
    # unreferenced by definition. Without this sweep they accumulate forever,
    # including under --source-retention run.
    swept, swept_bytes = sweep_uploads()

    # prune() is a no-op while auditing is off: --no-audit must not delete a
    # trail it is not managing.
    pruned = AUDIT.prune()

    if not shutil.which("ffmpeg"):
        print("!  ffmpeg not found on PATH. Most formats will fail to decode.")
        print("   winget install Gyan.FFmpeg   (then reopen the terminal)\n")

    threading.Thread(target=worker_loop, daemon=True, name="transcriber").start()

    if args.preload:
        preload_model(args)

    print()
    if cfg_path.exists():
        note = "  (created; every key is commented out)" if created else ""
        print(f"Config     {cfg_path}{note}")
    elif cfg_required:
        print(f"!  Config     {cfg_path} (missing)")

    if args.token:
        print(f"Access token: {args.token}")
        if generated:
            print(
                "(generated for this run; pass --token or TRANSCRIBE_TOKEN to pin it)"
            )
        print(f"\nOpen:  http://<this-machine-ip>:{args.port}/?token={args.token}")
        print("       the token moves out of the URL as soon as the page loads")
    else:
        print("!  Running with --no-auth. Anyone who can reach this port can read")
        print("   every transcript and upload files to this machine.")
        print(f"\nOpen:  http://<this-machine-ip>:{args.port}/")

    print(f"\nModel      {args.model}{'' if args.allow_model_choice else '  (pinned)'}")
    print(
        f"Precision  {args.compute_type}"
        f"{'  (selectable)' if args.allow_precision_choice else '  (pinned)'}"
    )
    detail = ""
    if CUDA:
        detail = (
            f"  {CUDA['device_count']} CUDA device(s)"
            if CUDA["usable"]
            else "  CPU fallback"
        )
    print(f"Device     {args.device}{detail}")
    # The one line a bug report is going to want. device_details() probes here so
    # the banner and the first status poll share a single answer.
    hardware = device_banner(device_details())
    if hardware:
        print(f"GPU        {hardware}")
    print(f"VRAM cache {args.model_cache} model(s)")
    print(
        "Speakers   "
        + (
            f"available ({args.diarization_embedding}, threshold "
            f"{args.diarization_threshold:g}, fold "
            f"{args.diarization_fold_share:g})"
            if args.allow_diarize
            else "disabled"
        )
    )
    print(
        f"Sources    retention={args.source_retention}"
        f"{'  retry enabled' if args.source_retention != 'run' else '  retry disabled'}"
    )

    if args.audit:
        print(f"\nAudit      {audit_dir}")
        print(
            f"           reads={'logged' if args.audit_reads else 'skipped'}"
            f"  prompts={'stored' if args.audit_prompts else 'omitted'}"
            f"  retention={args.audit_retain_days}d"
            f"  cap={str(args.audit_max_mb) + 'MB/day' if args.audit_max_mb else 'none'}"
            f"  sidecars={args.audit_max_sidecars or 'unlimited'}"
        )
        if pruned:
            print(f"           pruned {len(pruned)} expired file(s)")
    else:
        print("\nAudit      off (--no-audit); existing files stay readable")

    # The audit API carries its own credential, so it announces itself on its
    # own lines -- and always, because it stays usable for existing files even
    # when recording is off.
    if args.audit_open:
        print()
        print("!  Audit API open. Anyone who can reach this port can read the")
        print("   prompt and hotword text recorded for every job.")
        print(f"\nAudit UI:  http://<this-machine-ip>:{args.port}/audit")
    else:
        print(f"\nAudit token: {args.audit_token}")
        if audit_generated:
            print(
                "             (generated for this run; pass --audit-token or "
                "TRANSCRIBE_AUDIT_TOKEN to pin it)"
            )
        print(
            f"Audit UI:  http://<this-machine-ip>:{args.port}/audit?token=<audit-token>"
        )

    if swept:
        print(
            f"Uploads    swept {swept} orphaned file(s) from a previous run "
            f"({swept_bytes / 1024 / 1024:.1f} MB)"
        )

    print(f"\nAccepting Host: {', '.join(sorted(ALLOWED_HOSTS | ALLOWED_SUFFIXES))}")
    print("(add more with --allow-host)\n")

    snapshot = startup_snapshot(
        args, cfg_path if cfg_path.exists() else None, audit_generated
    )
    if args.audit:
        audit("server.started", **snapshot)
    else:
        # One unconditional line even with auditing off: a trail that simply
        # stops is otherwise indistinguishable from a server that was down.
        AUDIT.emit("server.started", force=True, **snapshot)
    if pruned:
        audit("audit.pruned", removed=len(pruned), files=pruned[:50])
    if swept:
        audit("uploads.swept", files=swept, bytes=swept_bytes)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
