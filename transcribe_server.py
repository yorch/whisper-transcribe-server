#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi>=0.110",
#     "uvicorn[standard]>=0.27",
#     "python-multipart>=0.0.9",
#     "faster-whisper>=1.0.3",
#     "numpy>=1.24",
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
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from collections.abc import MutableMapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar
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
    """Trailing path components, for records that must not carry machine paths."""
    chunks = Path(path).parts
    return "/".join(chunks[-parts:])


def safe_lib_name(entry: str) -> str:
    """A soname as it is; a directory as its last few components."""
    return short_path(entry) if Path(entry).is_absolute() else entry


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
    ) -> None:
        self.dir = Path(directory)
        self.prompts_dir = self.dir / "prompts"
        self.enabled = enabled
        self.retain_days = max(0, retain_days)
        self.store_prompts = prompts
        self._lock = threading.Lock()
        self._fh: Any | None = None
        self._day: str | None = None
        self._warned: set[tuple[str, str]] = set()
        self.last_error: str | None = None

    # -- writing ---------------------------------------------------------- #

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
        # Retention also runs on rotation: a process that stays up for weeks
        # would otherwise never prune again after startup.
        self.prune()
        return self._fh

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
        if not self.enabled and not force:
            return
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "event": event,
        }
        record.update({k: v for k, v in fields.items() if v is not None})
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if self._day is not None and day < self._day:
                    day = self._day  # never rotate backwards on a boundary race
                if self._day != day or self._fh is None:
                    fh = self._rotate(day)
                    self._day = day
                else:
                    fh = self._fh
                fh.write(line + "\n")
                fh.flush()
        except Exception as exc:  # noqa: BLE001
            self._warn_once("write", exc)

    def write_prompt(self, job_id: str, payload: dict[str, Any]) -> None:
        if not (self.enabled and self.store_prompts):
            return
        if not SAFE_JOB_ID.fullmatch(job_id):
            return
        try:
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(self.prompts_dir, 0o700)
            path = self.prompts_dir / f"{job_id}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            with contextlib.suppress(OSError):
                os.chmod(path, 0o600)
        except Exception as exc:  # noqa: BLE001
            self._warn_once("sidecar write", exc)

    def read_prompt(self, job_id: str) -> dict[str, Any] | None:
        if not SAFE_JOB_ID.fullmatch(job_id):
            return None
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

    def read(
        self,
        day: str,
        limit: int = 200,
        offset: int = 0,
        job: str | None = None,
        needle: str | None = None,
    ) -> tuple[list[str], int]:
        """Return (raw lines newest-first, total matching) for one day.

        Only the requested page is held in memory: a flooded day file can be
        enormous, and the audit UI re-reads every few seconds.
        """
        try:
            fh = (self.dir / f"audit-{day}.jsonl").open(
                "r", encoding="utf-8", errors="replace"
            )
        except OSError:
            return [], 0
        window: deque[str] = deque(maxlen=max(1, limit) + max(0, offset))
        total = 0
        low = needle.lower() if needle else None
        with fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                if low and low not in raw.lower():
                    continue
                if job:
                    try:
                        rec = json.loads(raw)
                    except ValueError:
                        continue
                    if rec.get("job") != job and rec.get("from_job") != job:
                        continue
                total += 1
                window.append(raw)
        lines = list(window)[::-1]
        return lines[offset : offset + limit], total

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
REJECT_LOG_LIMIT = 5
REJECT_WINDOW = 60.0
REJECT_MAX_SOURCES = 1000
_REJECT_STATE: dict[tuple[str, str], tuple[float, int, int]] = {}
_REJECT_LOCK = threading.Lock()


def audit_rejection(event: str, request: Request, **fields: Any) -> None:
    """Record a refused request, collapsing a burst from one source."""
    client = (request.client.host if request.client else "-")[:64]
    key = (event, client)
    now = time.monotonic()
    log_now = False
    suppressed = 0
    with _REJECT_LOCK:
        if len(_REJECT_STATE) > REJECT_MAX_SOURCES:
            _REJECT_STATE.clear()  # bounded memory under a spoofed-source flood
        window_start, count, logged = _REJECT_STATE.get(key, (now, 0, 0))
        if now - window_start > REJECT_WINDOW:
            suppressed = count - logged
            window_start, count, logged = now, 0, 0
        count += 1
        if logged < REJECT_LOG_LIMIT:
            logged += 1
            log_now = True
        _REJECT_STATE[key] = (window_start, count, logged)

    if log_now:
        audit(event, request, **fields)
    if suppressed:
        audit(
            "security.rejected_summary",
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


def job_public(job: dict[str, Any], include_segments: bool = True) -> dict[str, Any]:
    out = {k: v for k, v in job.items() if k not in ("path", "segments", "opts")}
    out["opts"] = public_opts(job["opts"])
    now = time.time()
    started = job.get("started")
    finished = job.get("finished")
    out["elapsed"] = round((finished or now) - started, 1) if started else 0.0
    out["segment_count"] = len(job["segments"])
    out["can_retry"] = (
        job["state"] in ("done", "error", "cancelled") and Path(job["path"]).exists()
    )
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
        "word_timestamps": truthy(word_timestamps),
        "min_silence_ms": clamp(as_int(min_silence_ms, 2000), 100, 10000),
        "speech_pad_ms": clamp(as_int(speech_pad_ms, 400), 0, 2000),
    }


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


def run_job(job_id: str) -> None:
    job = get_job(job_id)
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

    patch_job(job_id, state="running", message="Transcribing")
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

            progress = min(seg.end / duration, 1.0) if duration else 0.0
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

        if not patch_job(
            job_id,
            state="done",
            progress=1.0,
            finished=time.time(),
            message=f"{len(collected)} segments",
        ):
            return  # cancelled while finishing; don't claim success
        audit(
            "job.done",
            job=job_id,
            file=job["filename"],
            model=opts["model"],
            compute_type=opts["compute_type"],
            language=getattr(info, "language", None),
            duration=round(duration, 2),
            segments=len(collected),
            elapsed=round(time.time() - (job.get("started") or time.time()), 1),
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


def render(job: dict[str, Any], fmt: str) -> tuple[str, str]:
    """Return (body, mime) for the requested format."""
    segs = job["segments"]

    if fmt == "txt":
        return "\n".join(s["text"] for s in segs) + "\n", "text/plain; charset=utf-8"

    if fmt == "timestamped":
        lines = [f"[{_stamp(s['start'])}] {s['text']}" for s in segs]
        return "\n".join(lines) + "\n", "text/plain; charset=utf-8"

    if fmt == "srt":
        blocks = []
        for i, s in enumerate(segs, 1):
            blocks.append(
                f"{i}\n{_stamp(s['start'], comma=True)} --> "
                f"{_stamp(s['end'], comma=True)}\n{s['text']}\n"
            )
        return "\n".join(blocks), "application/x-subrip; charset=utf-8"

    if fmt == "vtt":
        blocks = ["WEBVTT\n"]
        for s in segs:
            blocks.append(f"{_stamp(s['start'])} --> {_stamp(s['end'])}\n{s['text']}\n")
        return "\n".join(blocks), "text/vtt; charset=utf-8"

    if fmt == "json":
        payload = {
            "filename": job["filename"],
            "language": job["language"],
            "duration": job["duration"],
            # public_opts, not job["opts"]: the export must not carry prompt
            # or hotword text out to an app-token holder.
            "options": public_opts(job["opts"]),
            "segments": segs,
        }
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
        "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
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


def audit_api_mode(args: argparse.Namespace) -> str:
    """How the audit API is reachable: 'open', 'token' or 'off'.

    One rule, three consumers: the middleware gate, the mode the page renders
    server-side, and the startup record. 'off' is only reachable when ARGS was
    built by hand -- main() always leaves either a token or an explicit opt-out.
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

        if path.startswith("/api/audit"):
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

    # Polling endpoints are chatty, so reads are opt-in. Everything that
    # changes state or moves a transcript out logs itself at the source.
    if path.startswith("/api/") and not path.startswith("/api/audit"):
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
    return PAGE


@app.get("/audit", response_class=HTMLResponse)
def audit_ui() -> str:
    """The page is public like `/`; the data behind it needs the audit token,
    unless the operator opened it with --audit-open. The mode is rendered
    server-side so a switched-off API cannot look like a rejected token."""
    mode = audit_api_mode(ARGS)
    # The gate ships visible, so token mode works with no script at all. Every
    # other mode starts with it hidden: the alternative is a visible prompt for
    # a token that does not exist, which is the bug this page used to have.
    # relock() takes over from there, for a server that restarts mid-session.
    return (
        AUDIT_PAGE.replace("__MODE__", mode)
        .replace("__GATE_CLASS__", "gate" if mode == "token" else "gate locked")
        .replace("__OFF_CLASS__", "gate" if mode == "off" else "gate locked")
    )


@app.get("/api/status")
def status() -> dict[str, Any]:
    gpu = None
    with contextlib.suppress(Exception):
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        gpu = f"{count} CUDA device(s)" if count else None

    with contextlib.suppress(Exception):
        # Optional: only for the pretty device name in the status strip.
        import torch  # pyright: ignore[reportMissingImports]

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)

    with JOBS_LOCK:
        active = sum(
            1 for j in JOBS.values() if j["state"] in ("queued", "loading", "running")
        )
    with _MODEL_LOCK:
        loaded = [f"{k[0]} / {k[2]}" for k in _MODEL_CACHE]

    return {
        "device": ARGS.device,
        "gpu": gpu,
        "cuda": CUDA,
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
        "prompt_limit": PROMPT_LIMIT,
        "hotwords_limit": HOTWORDS_LIMIT,
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


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)
        return {"jobs": [job_public(j, include_segments=False) for j in jobs]}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict[str, Any]:
    return job_public(get_job(job_id))


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
    # The moment a transcript leaves the machine is the interesting one.
    audit(
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
) -> dict[str, Any]:
    if AUDIT is None:
        raise HTTPException(status_code=503, detail="Audit trail unavailable")

    limit = clamp(as_int(limit, 200), 1, 1000)
    offset = max(0, as_int(offset, 0))
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
    raw, total = AUDIT.read(
        day, limit=limit, offset=offset, job=job or None, needle=needle
    )

    events: list[dict[str, Any]] = []
    for line in raw:
        try:
            events.append(json.loads(line))
        except ValueError:
            events.append({"event": "unparsable", "raw": line[:500]})

    if include_prompts:
        for rec in events:
            side = AUDIT.read_prompt(str(rec.get("job") or ""))
            if side:
                rec["prompt_text"] = side.get("prompt") or ""
                rec["hotwords_text"] = side.get("hotwords") or ""

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
    )
    return {
        "date": day,
        "dates": days,
        "events": events,
        "total": total,
        "offset": offset,
        "limit": limit,
        "retain_days": AUDIT.retain_days,
        "prompts_available": AUDIT.store_prompts,
    }


@app.get("/api/audit/prompts/{job_id}")
def audit_prompt(job_id: str, request: Request) -> dict[str, Any]:
    if AUDIT is None:
        raise HTTPException(status_code=503, detail="Audit trail unavailable")
    side = AUDIT.read_prompt(job_id)
    if side is None:
        raise HTTPException(status_code=404, detail="No stored prompt for that job")
    audit("audit.prompt_read", request, job=job_id)
    return side


# --------------------------------------------------------------------------- #
# Front end
# --------------------------------------------------------------------------- #

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Transcription</title>
<style>
  :root{
    --panel:#dcdfe3;
    --panel-hi:#eef0f2;
    --panel-lo:#c3c8cd;
    --ink:#1b2027;
    --muted:#5c646e;
    --amber:#b9610f;
    --amber-dim:#d9b48a;
    --green:#36704a;
    --red:#9d3427;
    --display:"Bahnschrift","DIN Alternate","Roboto Condensed",system-ui,sans-serif;
    --body:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{
    background:var(--panel);
    background-image:linear-gradient(180deg,#e3e6e9 0%,#d3d7dc 100%);
    color:var(--ink);
    font-family:var(--body);
    font-size:15px;
    line-height:1.55;
    min-height:100vh;
    padding:28px 20px 72px;
  }
  .wrap{max-width:660px;margin:0 auto}

  .readout{
    border:1px solid var(--panel-lo);
    border-top-color:var(--panel-hi);
    background:linear-gradient(180deg,#e9ebee,#d6dade);
    border-radius:3px;
    padding:12px 16px;
    display:flex;
    flex-wrap:wrap;
    gap:6px 26px;
    align-items:baseline;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.7);
  }
  .readout dl{margin:0;display:flex;gap:8px;align-items:baseline}
  .readout dt{
    font-family:var(--display);
    font-size:10.5px;
    letter-spacing:.14em;
    text-transform:uppercase;
    color:var(--muted);
  }
  .readout dd{margin:0;font-family:var(--display);font-size:14px;letter-spacing:.01em}
  .lamp{
    width:8px;height:8px;border-radius:50%;
    display:inline-block;margin-right:7px;vertical-align:1px;
    background:var(--amber-dim);
    box-shadow:0 0 0 1px rgba(0,0,0,.18) inset;
  }
  .lamp.on{background:var(--amber);box-shadow:0 0 6px rgba(185,97,15,.55)}
  .lamp.bad{background:var(--red)}

  h1{
    font-family:var(--display);
    font-weight:600;
    font-size:30px;
    letter-spacing:-.01em;
    margin:30px 0 4px;
  }
  .sub{color:var(--muted);margin:0 0 22px;max-width:52ch}

  .gate{
    border:1px solid var(--panel-lo);
    border-left:3px solid var(--amber);
    background:rgba(255,255,255,.5);
    border-radius:3px;
    padding:16px 18px;
    margin-bottom:22px;
  }
  .gate p{margin:0 0 12px}
  .gate .row{display:flex;gap:8px;flex-wrap:wrap}
  input[type=password]{
    flex:1;min-width:200px;
    font-family:var(--body);font-size:15px;
    padding:6px 9px;
    border:1px solid #a9b0b8;border-radius:3px;background:#fff;color:var(--ink);
  }
  input[type=password]:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  .gate-err{color:var(--red);font-size:13.5px;margin:10px 0 0}

  .intake{
    border:2px dashed #a9b0b8;
    border-radius:4px;
    background:rgba(255,255,255,.42);
    padding:40px 24px;
    text-align:center;
    transition:border-color .14s,background .14s;
    cursor:pointer;
  }
  .intake:hover{border-color:#8d959e}
  .intake.hot{border-color:var(--amber);background:rgba(185,97,15,.07)}
  .intake:focus-visible{outline:2px solid var(--amber);outline-offset:3px}
  .intake p{margin:0;font-family:var(--display);font-size:19px}
  .intake small{display:block;margin-top:8px;color:var(--muted);font-size:13px}
  input[type=file]{display:none}

  .controls{
    margin-top:14px;
    display:flex;
    flex-wrap:wrap;
    gap:14px 22px;
    align-items:center;
    padding:12px 2px;
    border-top:1px solid var(--panel-lo);
  }
  label.field{display:flex;align-items:center;gap:8px;font-size:13.5px;color:var(--muted)}
  select{
    font-family:var(--display);
    font-size:14px;
    color:var(--ink);
    background:#fff;
    border:1px solid #a9b0b8;
    border-radius:3px;
    padding:5px 8px;
  }
  select:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  input[type=checkbox]{accent-color:var(--amber);width:15px;height:15px}

  .job{
    margin-top:18px;
    border:1px solid var(--panel-lo);
    border-top-color:var(--panel-hi);
    border-radius:3px;
    background:linear-gradient(180deg,#eceef1,#e1e4e8);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.7);
    overflow:hidden;
  }
  .job-head{display:flex;gap:12px;align-items:baseline;padding:13px 16px 11px}
  .job-name{font-family:var(--display);font-size:17px;flex:1;min-width:0;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .job-meta{font-family:var(--display);font-size:12.5px;color:var(--muted);letter-spacing:.02em;
    white-space:nowrap}

  .meter{display:flex;gap:2px;padding:0 16px 13px}
  .meter i{
    flex:1;height:11px;
    background:#c9ced3;
    box-shadow:inset 0 0 0 1px rgba(0,0,0,.07);
    transition:background .18s;
  }
  .meter i.lit{background:var(--amber)}
  .meter i.done{background:var(--green)}
  .meter i.fail{background:var(--red)}

  .job-body{padding:0 16px 14px}
  .status{font-size:13.5px;color:var(--muted);margin:0 0 10px}
  .status.err{color:var(--red)}
  .transcript{
    background:#fbfbfc;
    border:1px solid #c9ced3;
    border-radius:3px;
    padding:12px 14px;
    max-height:300px;
    overflow:auto;
    font-size:14.5px;
    word-break:break-word;
  }
  .transcript:empty{display:none}
  /* One row per segment: a quiet time gutter, then the text. The gutter keeps
     its width so the words line up down the page instead of stepping in and
     out with the clock. */
  .seg{display:flex;gap:10px;padding:1px 0}
  .seg .ts{
    flex:0 0 auto;width:9ch;color:var(--muted);
    font-family:var(--display);font-size:13px;line-height:1.6;
    font-variant-numeric:tabular-nums;user-select:none;
  }
  .seg .tx{flex:1 1 auto;min-width:0;white-space:pre-wrap}
  /* Following is on but the view is parked somewhere else: saying so beats
     letting it look broken. */
  .transcript.paused{border-color:var(--amber-dim)}
  .actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
  button{
    font-family:var(--display);
    font-size:13px;
    letter-spacing:.03em;
    color:var(--ink);
    background:linear-gradient(180deg,#f4f5f6,#dfe3e7);
    border:1px solid #a9b0b8;
    border-radius:3px;
    padding:6px 12px;
    cursor:pointer;
  }
  button:hover{background:linear-gradient(180deg,#fff,#e7ebef)}
  button:active{background:#d8dce0}
  button:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  button.ghost{background:none;border-color:transparent;color:var(--muted)}
  button.ghost:hover{background:rgba(0,0,0,.05);color:var(--ink)}

  details.advanced{margin-top:2px;border-top:1px solid var(--panel-lo);padding-top:10px}
  details.advanced summary{
    font-family:var(--display);font-size:13px;letter-spacing:.03em;
    color:var(--muted);cursor:pointer;list-style:none;
  }
  details.advanced summary::-webkit-details-marker{display:none}
  details.advanced summary::before{content:"+ ";font-weight:600}
  details.advanced[open] summary::before{content:"\2212 "}
  details.advanced summary:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
  .adv-grid{display:flex;flex-direction:column;gap:12px;padding:14px 0 4px}
  .adv-grid textarea{
    font-family:var(--body);font-size:14px;line-height:1.5;
    padding:8px 10px;border:1px solid #a9b0b8;border-radius:3px;background:#fff;color:var(--ink);
    resize:vertical;min-height:62px;width:100%;
  }
  .adv-grid textarea:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  .adv-row{display:flex;flex-wrap:wrap;gap:12px 22px;align-items:center}
  .hint{display:block;color:var(--muted);font-size:12.5px;margin-top:4px;max-width:56ch}
  .field-block{display:block;font-size:13.5px;color:var(--muted)}
  .field-block > span{display:block;margin-bottom:5px}
  input[type=number]{
    font-family:var(--display);font-size:14px;width:88px;
    padding:5px 7px;border:1px solid #a9b0b8;border-radius:3px;background:#fff;color:var(--ink);
  }
  input[type=number]:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  .empty{margin-top:26px;color:var(--muted);font-size:14px}
  .locked{display:none}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">

  <div class="readout">
    <dl><dt>Device</dt><dd><span class="lamp" id="lamp"></span><span id="r-device">checking</span></dd></dl>
    <dl><dt>Precision</dt><dd id="r-precision">&mdash;</dd></dl>
    <dl><dt>ffmpeg</dt><dd id="r-ffmpeg">&mdash;</dd></dl>
    <dl><dt>Queue</dt><dd id="r-queue">0</dd></dl>
    <dl><dt>Loaded</dt><dd id="r-loaded">none</dd></dl>
  </div>

  <h1>Transcribe audio on the workstation</h1>
  <p class="sub">Drop a recording here and it runs through Whisper on the GPU in this
  machine. Nothing leaves the network.</p>

  <div class="gate locked" id="gate">
    <p>This server needs an access token. It was printed in the terminal when the
    server started.</p>
    <div class="row">
      <input type="password" id="gate-token" autocomplete="off" spellcheck="false"
             placeholder="Access token" aria-label="Access token">
      <button id="gate-go">Unlock</button>
    </div>
    <p class="gate-err locked" id="gate-err">That token was rejected. Check the terminal output.</p>
  </div>

  <div id="main" class="locked">
    <div class="intake" id="intake" tabindex="0" role="button"
         aria-label="Choose files to transcribe">
      <p>Drop files to transcribe</p>
      <small>or click to browse &middot; audio and video both work</small>
      <input type="file" id="picker" multiple
             accept="audio/*,video/*,.m4a,.mp3,.wav,.mp4,.mov,.mkv,.flac,.ogg,.opus,.webm">
    </div>

    <div class="controls">
      <label class="field">Model
        <select id="model"></select>
      </label>
      <label class="field">Language
        <select id="language">
          <option value="">Detect</option>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="pt">Portuguese</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="he">Hebrew</option>
        </select>
      </label>
      <label class="field">Quality
        <select id="quality"></select>
      </label>
      <label class="field locked" id="compute-field">Precision
        <select id="compute"></select>
      </label>
      <label class="field"><input type="checkbox" id="vad" checked> Skip silence</label>
      <label class="field"><input type="checkbox" id="follow" checked> Follow</label>
    </div>

    <details class="advanced">
      <summary>Advanced</summary>
      <div class="adv-grid">
        <label class="field-block">
          <span>Names and terms to expect</span>
          <textarea id="hotwords" rows="2" spellcheck="false"
            placeholder="Kubernetes, CODEOWNERS, MCP gateway, PostgreSQL"></textarea>
          <small class="hint">Comma-separated. Biases the decoder toward words it would
          otherwise mangle &mdash; product names, acronyms, people.</small>
        </label>

        <label class="field-block">
          <span>Context prompt</span>
          <textarea id="prompt" rows="2" spellcheck="false"
            placeholder="A recorded engineering planning call about developer tooling."></textarea>
          <small class="hint">A sentence describing the recording. Nudges style and
          vocabulary more broadly than the term list above.</small>
        </label>

        <div class="adv-row">
          <label class="field"><input type="checkbox" id="translate"> Translate to English</label>
          <label class="field"><input type="checkbox" id="words"> Word-level timings</label>
          <label class="field"><input type="checkbox" id="condition"> Carry context forward</label>
        </div>
        <small class="hint">Carrying context forward is off by default: on long
        recordings it's the usual cause of Whisper repeating a phrase for minutes.
        Word timings cost time, and help if you later run diarization.</small>

        <div class="adv-row" id="vad-tuning">
          <label class="field-block">
            <span>Min silence (ms)</span>
            <input type="number" id="min-silence" value="2000" min="100" max="10000" step="100">
          </label>
          <label class="field-block">
            <span>Speech padding (ms)</span>
            <input type="number" id="speech-pad" value="400" min="0" max="2000" step="50">
          </label>
        </div>
      </div>
    </details>

    <div id="jobs"></div>
    <p class="empty" id="empty">No jobs yet.</p>
  </div>
</div>

<script>
"use strict";
const el = (id) => document.getElementById(id);

/* Every value that reaches innerHTML goes through this. Filenames and error
   strings are attacker-controlled: anyone who can upload can plant markup. */
const ESCAPES = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;","`":"&#96;"};
const esc = (v) => String(v == null ? "" : v).replace(/[&<>"'`]/g, (c) => ESCAPES[c]);

/* ---------- token: accepted from the URL once, then moved out of it ---------- */
let TOKEN = sessionStorage.getItem("tk") || "";
{
  const fromUrl = new URLSearchParams(location.search).get("token");
  if(fromUrl){
    TOKEN = fromUrl;
    sessionStorage.setItem("tk", TOKEN);
    // Keep it out of history, bookmarks and any Referer we might emit.
    history.replaceState(null, "", location.pathname);
  }
}

let unlocked = false;

async function api(path, opts={}){
  const headers = {...(opts.headers||{})};
  if(TOKEN) headers["x-token"] = TOKEN;   // header only: forces a CORS preflight
  const r = await fetch(path, {...opts, headers, credentials:"omit"});
  if(r.status === 401){
    unlocked = false;
    el("gate").classList.remove("locked");
    el("main").classList.add("locked");
    throw new Error("unauthorised");
  }
  return r;
}

el("gate-go").addEventListener("click", submitToken);
el("gate-token").addEventListener("keydown", (e) => { if(e.key === "Enter") submitToken(); });

async function submitToken(){
  const value = el("gate-token").value.trim();
  if(!value) return;
  TOKEN = value;
  try{
    const r = await fetch("/api/status", {headers:{"x-token":TOKEN}, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    sessionStorage.setItem("tk", TOKEN);
    el("gate-err").classList.add("locked");
    el("gate").classList.add("locked");
    el("gate-token").value = "";
    await boot();
  }catch(e){
    el("gate-err").classList.remove("locked");
  }
}

/* ---------- status strip ---------- */
async function refreshStatus(){
  try{
    const s = await (await api("/api/status")).json();
    unlocked = true;
    el("main").classList.remove("locked");
    el("gate").classList.add("locked");

    const cuda = s.cuda || null;
    el("lamp").className = "lamp " + (s.device === "cuda" ? "on" : "bad");
    el("r-device").textContent = s.gpu || s.device;
    el("r-device").title = cuda && !cuda.usable ? (cuda.reason || "CUDA unavailable") : "";
    el("r-precision").textContent = s.compute_type;
    el("r-ffmpeg").textContent = s.ffmpeg ? "found" : "missing";
    el("r-ffmpeg").style.color = s.ffmpeg ? "" : "var(--red)";
    el("r-queue").textContent = s.active_jobs;
    el("r-loaded").textContent = (s.loaded_models && s.loaded_models.length)
      ? s.loaded_models.join(", ") : "none";
    MAX_MB = s.max_upload_mb;
    RETRY_OK = s.retry_available;

    if(!POPULATED){
      POPULATED = true;
      fill("model", s.models, s.default_model);
      fill("quality", s.qualities, s.default_quality);
      fill("compute", s.compute_types, s.compute_type);
      // Precision is a property of this machine, not of a recording. It only
      // appears if the operator explicitly opened it up.
      if(s.allow_precision_choice) el("compute-field").classList.remove("locked");
      if(!s.allow_model_choice){
        el("model").disabled = true;
        el("model").title = "Pinned by the server";
      }
    }
  }catch(e){
    if(e.message !== "unauthorised"){
      el("lamp").className = "lamp bad";
      el("r-device").textContent = "server unreachable";
      el("r-device").title = "";
    }
  }
}

/* ---------- upload ---------- */
let MAX_MB = 0, RETRY_OK = true, POPULATED = false;

function fill(id, values, selected){
  const node = el(id);
  node.textContent = "";
  for(const v of values){
    const o = document.createElement("option");
    o.value = o.textContent = v;
    if(v === selected) o.selected = true;
    node.append(o);
  }
}

/* Whatever is in the controls right now, as a form body. Used for both a fresh
   upload and a retry, so a retry re-runs the same audio under new settings. */
function currentSettings(){
  const fd = new FormData();
  fd.append("model", el("model").value);
  fd.append("compute_type", el("compute").value);
  fd.append("language", el("language").value);
  fd.append("quality", el("quality").value);
  fd.append("vad", el("vad").checked ? "true" : "false");
  fd.append("prompt", el("prompt").value);
  fd.append("hotwords", el("hotwords").value);
  fd.append("translate", el("translate").checked ? "true" : "false");
  fd.append("condition", el("condition").checked ? "true" : "false");
  fd.append("word_timestamps", el("words").checked ? "true" : "false");
  fd.append("min_silence_ms", el("min-silence").value || "2000");
  fd.append("speech_pad_ms", el("speech-pad").value || "400");
  return fd;
}

const intake = el("intake"), picker = el("picker");
intake.addEventListener("click", () => picker.click());
intake.addEventListener("keydown", (e) => {
  if(e.key === "Enter" || e.key === " "){ e.preventDefault(); picker.click(); }
});
picker.addEventListener("change", () => { send([...picker.files]); picker.value = ""; });

for(const ev of ["dragenter","dragover"])
  intake.addEventListener(ev, (e) => { e.preventDefault(); intake.classList.add("hot"); });
for(const ev of ["dragleave","drop"])
  intake.addEventListener(ev, (e) => { e.preventDefault(); intake.classList.remove("hot"); });
intake.addEventListener("drop", (e) => send([...e.dataTransfer.files]));
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => e.preventDefault());

async function send(files){
  for(const f of files){
    if(MAX_MB && f.size > MAX_MB * 1024 * 1024){
      alert(f.name + " is larger than the " + MAX_MB + " MB limit.");
      continue;
    }
    const fd = currentSettings();
    fd.append("file", f);
    try{
      const r = await api("/api/jobs", {method:"POST", body:fd});
      if(!r.ok){
        let detail = r.statusText;
        try{ detail = (await r.json()).detail || detail; }catch(_){}
        throw new Error(detail);
      }
      await r.json();
      await tick();
    }catch(err){
      if(err.message !== "unauthorised") alert("Upload failed for " + f.name + ": " + err.message);
    }
  }
}

el("vad").addEventListener("change", () => {
  el("vad-tuning").style.display = el("vad").checked ? "" : "none";
});

/* ---------- following the transcript ---------- */
/* Auto-scroll for the job previews, on by default and remembered for the
   session: a switch that quietly comes back on after every reload is worse
   than not having one. */
const FOLLOW_KEY = "follow";
try{ el("follow").checked = sessionStorage.getItem(FOLLOW_KEY) !== "0"; }catch(_){}
el("follow").addEventListener("change", () => setFollow(el("follow").checked));

/* ---------- rendering ---------- */
const TICKS = 40;
const known = new Map();
/* One entry per card. The transcript keeps its DOM between polls so a running
   job can be read while it grows: replacing the whole card every 1.2s (what
   this used to do) threw away the scroll position and any selection. */
const views = new Map();

function meter(job){
  const lit = Math.round((job.progress || 0) * TICKS);
  const cls = job.state === "done" ? "done" : job.state === "error" ? "fail" : "lit";
  let html = '<div class="meter">';
  for(let i=0;i<TICKS;i++) html += '<i class="' + (i<lit?cls:"") + '"></i>';
  return html + "</div>";
}

function fmtTime(s){
  s = Math.round(s);
  const m = Math.floor(s/60);
  return m ? m + "m " + String(s%60).padStart(2,"0") + "s" : s + "s";
}

function statusLine(job){
  if(job.state === "queued")  return "Waiting for the GPU";
  if(job.state === "loading") return "Loading " + esc(job.opts.model);
  if(job.state === "running"){
    const pct = Math.round((job.progress||0)*100);
    const eta = job.progress > 0.02
      ? " \u00b7 about " + fmtTime(job.elapsed/job.progress - job.elapsed) + " left"
      : "";
    return pct + "% \u00b7 " + job.segment_count + " segments" + eta;
  }
  if(job.state === "done"){
    const speed = job.duration && job.elapsed
      ? " \u00b7 " + (job.duration/job.elapsed).toFixed(1) + "\u00d7 realtime" : "";
    return "Finished in " + fmtTime(job.elapsed) + speed
      + " \u00b7 " + esc(job.language || "?") + " \u00b7 " + job.segment_count + " segments";
  }
  if(job.state === "cancelled") return "Cancelled";
  return esc(job.message);
}

/* Only show the knobs that were actually off-default, so the line stays short. */
function jobTags(job){
  const o = job.opts || {};
  const bits = [o.model];
  if(o.quality && o.quality !== "balanced") bits.push(o.quality);
  if(o.translate) bits.push("translated");
  if(o.condition) bits.push("context on");
  if(o.word_timestamps) bits.push("word times");
  if(o.has_hotwords) bits.push("terms");
  if(job.duration) bits.push(fmtTime(job.duration));
  return bits.join(" \u00b7 ");
}

/* Seconds to HH:MM:SS: the clock of the timestamped export without the
   milliseconds, which are noise on screen. */
function fmtStamp(seconds){
  const total = Math.max(0, Math.floor(seconds || 0));
  const pad = (n) => String(n).padStart(2,"0");
  return pad(Math.floor(total/3600)) + ":" + pad(Math.floor(total/60)%60)
       + ":" + pad(total%60);
}

function followOn(){ return el("follow").checked; }

function setPaused(view, paused){
  view.paused = paused;
  view.transcript.classList.toggle("paused", paused);
  view.transcript.title = paused
    ? "Following paused \u2014 scroll back to the bottom to catch up" : "";
}

/* Keep the newest line in view, unless Following is off or the operator has
   scrolled away from the bottom. */
function stick(view){
  if(!followOn() || view.paused) return;
  view.transcript.scrollTop = view.transcript.scrollHeight;
}

function setFollow(on){
  try{ sessionStorage.setItem(FOLLOW_KEY, on ? "1" : "0"); }catch(_){}
  for(const view of views.values()){
    setPaused(view, false);
    if(on && view.live) stick(view);   // catch up now, not at the next segment
  }
}

function createCard(job){
  const node = document.createElement("div");
  node.className = "job";
  node.id = "job-" + job.id;
  node.innerHTML =
    '<div class="job-head"></div>' +
    '<div class="meter-slot"></div>' +
    '<div class="job-body">' +
      '<p class="status"></p>' +
      '<div class="transcript"></div>' +
      '<div class="actions"></div>' +
    "</div>";
  el("jobs").prepend(node);

  const view = {
    node,
    transcript: node.querySelector(".transcript"),
    shown: 0,
    paused: false,
    live: false,
  };
  /* Scrolling away from the newest line pauses; coming back re-arms. The
     Follow switch stays the authoritative off switch. */
  view.transcript.addEventListener("scroll", () => {
    const atBottom = view.transcript.scrollHeight - view.transcript.scrollTop
                     - view.transcript.clientHeight <= 8;
    if(atBottom === view.paused) setPaused(view, !atBottom);
  });
  views.set(job.id, view);
  return view;
}

/* Append only what is new. The server grows job["segments"] and never rewrites
   one, so counting is enough. textContent rather than innerHTML: transcript
   text needs no escaping and cannot become markup. */
function appendSegments(view, segments, live){
  view.live = live;
  if(segments.length < view.shown){
    view.transcript.textContent = "";
    view.shown = 0;
  }
  for(let i = view.shown; i < segments.length; i++){
    const seg = segments[i];
    const row = document.createElement("div");
    row.className = "seg";
    const ts = document.createElement("span");
    ts.className = "ts";
    ts.textContent = fmtStamp(seg.start);
    const tx = document.createElement("span");
    tx.className = "tx";
    tx.textContent = seg.text;
    row.append(ts, tx);
    view.transcript.append(row);
  }
  view.shown = segments.length;
  /* Only chase a job that is still producing text: a finished transcript should
     open at its first line, not at its last. */
  if(live) stick(view);
}

function actionsHtml(job){
  const dl = (fmt, label) =>
    '<button data-dl="' + fmt + '" data-id="' + esc(job.id) + '">' + label + "</button>";
  return (job.state === "done"
      ? '<button data-copy="' + esc(job.id) + '">Copy text</button>' +
        dl("txt","Save .txt") + dl("timestamped","Save timestamped") +
        dl("srt","Save .srt") + dl("vtt","Save .vtt") + dl("json","Save .json")
      : "") +
    (RETRY_OK && job.can_retry
      ? '<button data-retry="' + esc(job.id) + '">Retry with these settings</button>'
      : "") +
    '<button class="ghost" data-del="' + esc(job.id) + '">' +
      (["queued","loading","running"].includes(job.state) ? "Cancel" : "Remove") +
    "</button>";
}

function render(job){
  const id = "job-" + job.id;
  let view = views.get(id);
  if(!view || !view.node.isConnected) view = createCard(job);

  const name = esc(job.filename);
  view.node.querySelector(".job-head").innerHTML =
    '<span class="job-name" title="' + name + '">' + name + "</span>" +
    '<span class="job-meta">' + esc(jobTags(job)) + "</span>";
  view.node.querySelector(".meter-slot").innerHTML = meter(job);
  const status = view.node.querySelector(".status");
  status.className = "status" + (job.state === "error" ? " err" : "");
  status.innerHTML = statusLine(job);
  view.node.querySelector(".actions").innerHTML = actionsHtml(job);

  appendSegments(view, job.segments || [],
                 ["queued","loading","running"].includes(job.state));
}

/* Downloads go through fetch so the token stays in a header, never a URL. */
async function download(jobId, fmt){
  const r = await api("/api/jobs/" + encodeURIComponent(jobId)
                      + "/text?format=" + encodeURIComponent(fmt) + "&download=1");
  if(!r.ok) return;
  const blob = await r.blob();
  const cd = r.headers.get("content-disposition") || "";
  let name = "transcript." + (fmt === "timestamped" ? "txt" : fmt);
  const star = /filename\*=UTF-8''([^;]+)/i.exec(cd);
  const plain = /filename="([^"]*)"/i.exec(cd);
  if(star) { try { name = decodeURIComponent(star[1]); } catch(_){} }
  else if(plain) name = plain[1];
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

el("jobs").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if(!b) return;
  try{
    if(b.dataset.dl) await download(b.dataset.id, b.dataset.dl);
    if(b.dataset.copy){
      const r = await api("/api/jobs/" + encodeURIComponent(b.dataset.copy) + "/text?format=txt");
      const body = await r.text();
      if(navigator.clipboard && window.isSecureContext){
        await navigator.clipboard.writeText(body);
      }else{
        // Plain HTTP is not a secure context, so the Clipboard API is unavailable.
        const ta = document.createElement("textarea");
        ta.value = body;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.append(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      b.textContent = "Copied";
      setTimeout(() => (b.textContent = "Copy text"), 1400);
    }
    if(b.dataset.retry){
      b.disabled = true;
      const r = await api("/api/jobs/" + encodeURIComponent(b.dataset.retry) + "/retry",
                          {method:"POST", body:currentSettings()});
      if(!r.ok){
        let detail = r.statusText;
        try{ detail = (await r.json()).detail || detail; }catch(_){}
        alert("Retry failed: " + detail);
        b.disabled = false;
      }
      await tick();
    }
    if(b.dataset.del){
      await api("/api/jobs/" + encodeURIComponent(b.dataset.del), {method:"DELETE"});
      await tick();
    }
  }catch(err){ /* 401 already handled by api() */ }
});

/* ---------- poll ---------- */
async function tick(){
  if(!unlocked) return;
  try{
    const {jobs} = await (await api("/api/jobs")).json();
    const live = jobs.map(j => j.id);
    el("empty").style.display = jobs.length ? "none" : "";

    for(const stale of [...known.keys()].filter(id => !live.includes(id))){
      const n = document.getElementById("job-" + stale);
      if(n) n.remove();
      known.delete(stale);
      views.delete(stale);
    }

    for(const summary of jobs){
      const sig = summary.state + ":" + summary.progress + ":" + summary.segment_count;
      if(known.get(summary.id) === sig) continue;
      known.set(summary.id, sig);
      const full = await (await api("/api/jobs/" + encodeURIComponent(summary.id))).json();
      render(full);
    }
  }catch(e){ /* server blip or 401; next tick retries */ }
}

async function boot(){
  await refreshStatus();
  await tick();
}

boot();
setInterval(tick, 1200);
setInterval(refreshStatus, 5000);
</script>
</body>
</html>
"""


AUDIT_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Audit trail</title>
<style>
  :root{
    --panel:#dcdfe3;--panel-hi:#eef0f2;--panel-lo:#c3c8cd;
    --ink:#1b2027;--muted:#5c646e;--amber:#b9610f;--amber-dim:#d9b48a;
    --green:#36704a;--red:#9d3427;
    --display:"Bahnschrift","DIN Alternate","Roboto Condensed",system-ui,sans-serif;
    --body:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{
    background:var(--panel);
    background-image:linear-gradient(180deg,#e3e6e9 0%,#d3d7dc 100%);
    color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.55;
    min-height:100vh;padding:28px 20px 72px;
  }
  .wrap{max-width:1040px;margin:0 auto}
  .topbar{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
  h1{font-family:var(--display);font-weight:600;font-size:30px;letter-spacing:-.01em;margin:0;flex:1}
  a.back{font-family:var(--display);font-size:13px;color:var(--muted);text-decoration:none}
  a.back:hover{color:var(--amber)}
  .sub{color:var(--muted);margin:6px 0 22px;max-width:70ch}

  .gate{
    border:1px solid var(--panel-lo);border-left:3px solid var(--amber);
    background:rgba(255,255,255,.5);border-radius:3px;padding:16px 18px;margin-bottom:22px;
  }
  .gate p{margin:0 0 12px}
  .gate .row{display:flex;gap:8px;flex-wrap:wrap}
  input[type=password],input[type=text]{
    flex:1;min-width:180px;font-family:var(--body);font-size:15px;padding:6px 9px;
    border:1px solid #a9b0b8;border-radius:3px;background:#fff;color:var(--ink);
  }
  input[type=password]:focus-visible,input[type=text]:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  .gate-err{color:var(--red);font-size:13.5px;margin:10px 0 0}

  .controls{
    display:flex;flex-wrap:wrap;gap:12px 18px;align-items:center;
    padding:12px 14px;border:1px solid var(--panel-lo);border-radius:3px;
    background:linear-gradient(180deg,#e9ebee,#dde1e5);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.7);
  }
  label.field{display:flex;align-items:center;gap:7px;font-size:13.5px;color:var(--muted)}
  select,input[type=number]{
    font-family:var(--display);font-size:14px;color:var(--ink);background:#fff;
    border:1px solid #a9b0b8;border-radius:3px;padding:5px 8px;
  }
  select:focus-visible,input[type=number]:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  input[type=checkbox]{accent-color:var(--amber);width:15px;height:15px}
  button{
    font-family:var(--display);font-size:13px;letter-spacing:.03em;color:var(--ink);
    background:linear-gradient(180deg,#f4f5f6,#dfe3e7);border:1px solid #a9b0b8;
    border-radius:3px;padding:6px 12px;cursor:pointer;
  }
  button:hover{background:linear-gradient(180deg,#fff,#e7ebef)}
  button:active{background:#d8dce0}
  button:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
  button.ghost{background:none;border-color:transparent;color:var(--muted)}
  button.ghost:hover{background:rgba(0,0,0,.05);color:var(--ink)}

  .meta{font-family:var(--display);font-size:12.5px;color:var(--muted);margin:14px 2px 10px;letter-spacing:.02em}
  #events{display:flex;flex-direction:column;gap:8px}
  .ev{
    border:1px solid var(--panel-lo);border-top-color:var(--panel-hi);border-radius:3px;
    background:linear-gradient(180deg,#eceef1,#e2e5e9);padding:10px 14px;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.7);border-left:3px solid #b6bcc3;
  }
  .ev.bad{border-left-color:var(--red)}
  .ev.warn{border-left-color:var(--amber)}
  .ev.good{border-left-color:var(--green)}
  .ev-head{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
  .ev-time{font-family:var(--display);font-size:12.5px;color:var(--muted)}
  .ev-event{font-family:var(--display);font-size:14px;letter-spacing:.01em}
  .ev-src{font-family:var(--display);font-size:12.5px;color:var(--muted);flex:1;text-align:right}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .chip{
    font-size:12px;background:rgba(255,255,255,.7);border:1px solid #c9ced3;
    border-radius:3px;padding:2px 7px;color:var(--muted);
    max-width:100%;overflow-wrap:anywhere;
  }
  .chip b{font-family:var(--display);font-weight:600;color:var(--ink);margin-right:5px}
  .prompt{
    margin-top:8px;background:#fbfbfc;border:1px solid #c9ced3;border-radius:3px;
    padding:9px 11px;font-size:13.5px;white-space:pre-wrap;word-break:break-word;
  }
  .prompt b{font-family:var(--display);font-size:11px;letter-spacing:.12em;
    text-transform:uppercase;color:var(--muted);margin-right:8px}
  .empty{margin-top:22px;color:var(--muted);font-size:14px}
  .locked{display:none}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body data-mode="__MODE__">
<div class="wrap">
  <div class="topbar">
    <h1>Audit trail</h1>
    <a class="back" href="/">&larr; Transcription</a>
  </div>
  <p class="sub">Every action the web app took &mdash; uploads, exports, deletions,
  worker lifecycle and refused requests &mdash; newest first. Access to the trail
  is controlled separately from the app token.</p>

  <div class="__GATE_CLASS__" id="gate">
    <p>Paste the audit token. The server prints it at startup; pass
    <code>--audit-token</code> or <code>TRANSCRIBE_AUDIT_TOKEN</code> to pin it
    across restarts.</p>
    <div class="row">
      <input type="password" id="gate-token" autocomplete="off" spellcheck="false"
             placeholder="Audit token" aria-label="Audit token">
      <button id="gate-go">Unlock</button>
    </div>
    <p class="gate-err locked" id="gate-err">That token was rejected.</p>
  </div>

  <div class="__OFF_CLASS__" id="off">
    <p>This server has the audit API switched off, so there is nothing to show
    here. Restart it with <code>--audit-token</code> (or
    <code>TRANSCRIBE_AUDIT_TOKEN</code>) to turn the endpoint back on.</p>
  </div>

  <div id="main" class="locked">
    <div class="controls">
      <label class="field">Day <select id="date"></select></label>
      <label class="field">Rows
        <select id="limit">
          <option>100</option><option selected>200</option>
          <option>500</option><option>1000</option>
        </select>
      </label>
      <label class="field">Job <input type="text" id="job" size="10"
        placeholder="job id" spellcheck="false"></label>
      <label class="field">Search <input type="text" id="q" size="14"
        placeholder="substring" spellcheck="false"></label>
      <label class="field"><input type="checkbox" id="prompts"> Prompts</label>
      <label class="field"><input type="checkbox" id="auto" checked> Auto</label>
      <button id="reload">Reload</button>
      <button class="ghost" id="more">Load more</button>
    </div>
    <p class="meta" id="meta">&nbsp;</p>
    <div id="events"></div>
    <p class="empty locked" id="empty">No events for this day.</p>
  </div>
</div>

<script>
"use strict";
const el = (id) => document.getElementById(id);
const ESCAPES = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;","`":"&#96;"};
const esc = (v) => String(v == null ? "" : v).replace(/[&<>"'`]/g, (c) => ESCAPES[c]);

/* Same trick as the main page: the token rides in a header, never a URL, and
   a URL copy is stripped from history immediately. */
let TOKEN = sessionStorage.getItem("atk") || "";
{
  const fromUrl = new URLSearchParams(location.search).get("token");
  if(fromUrl){
    TOKEN = fromUrl;
    sessionStorage.setItem("atk", TOKEN);
    history.replaceState(null, "", location.pathname);
  }
}

/* Which of the three modes the server rendered. The page used to infer this
   from a failed request, which made a switched-off API look like a bad token. */
const MODE = document.body.dataset.mode;

let unlocked = false, offset = 0, timer = null;

function relock(){
  unlocked = false;
  clearInterval(timer);
  el("main").classList.add("locked");
  // Only "off" owns the switched-off notice; anything else that lands here is
  // a credential problem, where the gate is the useful thing to show.
  el("gate").classList.toggle("locked", MODE === "off");
  el("off").classList.toggle("locked", MODE !== "off");
}

async function api(path){
  const headers = {};
  if(TOKEN) headers["x-audit-token"] = TOKEN;
  const r = await fetch(path, {headers, credentials:"omit"});
  if(r.status === 401 || r.status === 404){
    relock();
    throw new Error("unauthorised");
  }
  return r;
}

function level(event){
  if(!event) return "";
  if(event.startsWith("security.") || event === "request.rejected" || event.endsWith("error")) return "bad";
  if(event.endsWith("rejected")) return "warn";
  if(event === "job.done" || event === "server.started") return "good";
  return "";
}

const SKIP = new Set(["ts","event","client","client_claimed","host","method","path",
                      "prompt_text","hotwords_text"]);

function fmtVal(v){
  if(v === null || v === undefined) return "";
  if(typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function row(rec){
  const time = String(rec.ts || "").slice(11, 19);
  const where = [];
  if(rec.method) where.push(rec.method + " " + (rec.path || ""));
  // client_claimed is whatever the proxy said; the TCP peer is the fact.
  if(rec.client_claimed) where.push("unverified via " + rec.client_claimed);
  if(rec.client) where.push(rec.client);
  if(rec.host) where.push(rec.host);
  if(rec.status) where.push("HTTP " + rec.status);
  if(rec.ms !== undefined && rec.ms !== null) where.push(rec.ms + "ms");

  const chips = Object.entries(rec)
    .filter(([k, v]) => !SKIP.has(k) && v !== null && v !== undefined && v !== "")
    .map(([k, v]) => '<span class="chip"><b>' + esc(k) + "</b>" + esc(fmtVal(v)) + "</span>")
    .join("");

  let prompt = "";
  if(rec.hotwords_text)
    prompt += '<div><b>hotwords</b>' + esc(rec.hotwords_text) + "</div>";
  if(rec.prompt_text)
    prompt += '<div><b>prompt</b>' + esc(rec.prompt_text) + "</div>";

  return '<div class="ev ' + level(rec.event) + '">' +
    '<div class="ev-head">' +
      '<span class="ev-time">' + esc(time) + "</span>" +
      '<span class="ev-event">' + esc(rec.event) + "</span>" +
      '<span class="ev-src">' + esc(where.join(" \u00b7 ")) + "</span>" +
    "</div>" +
    (chips ? '<div class="chips">' + chips + "</div>" : "") +
    (prompt ? '<div class="prompt">' + prompt + "</div>" : "") +
  "</div>";
}

function fillDates(dates, selected){
  const node = el("date");
  const current = node.value;
  const wanted = selected || current;
  node.textContent = "";
  if(!dates.length){
    const o = document.createElement("option");
    o.value = o.textContent = selected || "";
    node.append(o);
    return;
  }
  for(const d of dates){
    const o = document.createElement("option");
    o.value = o.textContent = d;
    if(d === wanted) o.selected = true;
    node.append(o);
  }
}

async function load(more){
  if(!unlocked) return;
  if(!more) offset = 0;
  const params = new URLSearchParams();
  params.set("date", el("date").value);
  params.set("limit", el("limit").value);
  params.set("offset", String(offset));
  if(el("job").value.trim()) params.set("job", el("job").value.trim());
  if(el("q").value.trim()) params.set("q", el("q").value.trim());
  if(el("prompts").checked) params.set("include_prompts", "1");

  try{
    const r = await api("/api/audit?" + params.toString());
    if(!r.ok){
      let detail = r.statusText;
      try{ detail = (await r.json()).detail || detail; }catch(_){}
      throw new Error(detail);
    }
    const data = await r.json();
    unlocked = true;
    el("gate").classList.add("locked");
    el("main").classList.remove("locked");
    fillDates(data.dates, data.date);

    if(!more) el("events").textContent = "";
    el("events").insertAdjacentHTML("beforeend", data.events.map(row).join(""));
    offset += data.events.length;

    el("empty").classList.toggle("locked", data.total > 0);
    el("more").disabled = offset >= data.total;
    el("meta").textContent = data.total + " event(s) on " + data.date
      + " \u00b7 showing " + Math.min(offset, data.total)
      + " \u00b7 retention " + (data.retain_days ? data.retain_days + " days" : "unlimited")
      + (data.prompts_available ? "" : " \u00b7 prompts not stored");
  }catch(err){
    if(err.message === "unauthorised") return;
    el("meta").textContent = "Could not load the audit trail: " + err.message;
  }
}

function schedule(){
  clearInterval(timer);
  if(el("auto").checked) timer = setInterval(() => {
    if(offset <= Number(el("limit").value)) load(false);
  }, 5000);
}

el("gate-go").addEventListener("click", submitToken);
el("gate-token").addEventListener("keydown", (e) => { if(e.key === "Enter") submitToken(); });

async function submitToken(){
  const value = el("gate-token").value.trim();
  if(!value) return;
  TOKEN = value;
  try{
    const r = await fetch("/api/audit?limit=1", {headers:{"x-audit-token":TOKEN}, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    sessionStorage.setItem("atk", TOKEN);
    el("gate-err").classList.add("locked");
    el("gate-token").value = "";
    unlocked = true;
    await load(false);
    schedule();
  }catch(e){
    el("gate-err").classList.remove("locked");
  }
}

el("reload").addEventListener("click", () => load(false));
// Changing the day starts a fresh page; "Load more" only ever appends.
el("more").addEventListener("click", () => load(true));
el("date").addEventListener("change", () => load(false));
el("job").addEventListener("change", () => load(false));
el("prompts").addEventListener("change", () => load(false));
el("q").addEventListener("keydown", (e) => { if(e.key === "Enter") load(false); });
el("auto").addEventListener("change", schedule);

(async function boot(){
  // No early return on a missing token: --audit-open means there is none to
  // have, and this one probe is what tells the three modes apart.
  try{
    const headers = TOKEN ? {"x-audit-token":TOKEN} : {};
    const r = await fetch("/api/audit?limit=1", {headers, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    unlocked = true;
    await load(false);
    schedule();
  }catch(e){
    relock();
  }
})();
</script>
</body>
</html>
"""


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
    "audit_open": False,
    "audit_token": "",
}

CHOICES: dict[str, list[str]] = {
    "model": MODELS,
    "device": ["cuda", "cpu", "auto"],
    "compute_type": COMPUTE_TYPES,
    "quality": list(QUALITIES),
    "source_retention": RETENTION,
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


CONVERTERS = {"str": str, "int": int, "bool": as_bool, "list": as_list}


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
# Also log status/list/detail polls. Chatty, so off by default.
# reads = false
# Store prompt and hotword text in the sidecar files. Turning this off keeps
# the hashes and lengths in the main log and records no text anywhere.
# prompts = true
# Delete audit files older than this many days at startup; 0 keeps everything.
# retain_days = 30
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
    for name in ("model_cache", "max_jobs", "audit_retain_days"):
        if merged[name] < 0:
            raise SystemExit(f"!  {name} cannot be negative, got {merged[name]}")

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


def startup_snapshot(args: argparse.Namespace, cfg: Path | None) -> dict[str, Any]:
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
        "auth": bool(args.token),
        "audit": args.audit,
        "audit_reads": args.audit_reads,
        "audit_prompts": args.audit_prompts,
        "audit_retain_days": args.audit_retain_days,
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
    print(f"VRAM cache {args.model_cache} model(s)")
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

    snapshot = startup_snapshot(args, cfg_path if cfg_path.exists() else None)
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
