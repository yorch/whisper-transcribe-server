# PyInstaller spec for the Windows tray launcher.
#
# Bundles the launcher, the server script it supervises, and the vendor/ extras
# (uv.exe, ffmpeg.exe). Deliberately does NOT bundle the CUDA runtime: that is
# 2.2 GB and uv resolves it on first run from the script's PEP 723 metadata.
#
#   pyinstaller --clean --noconfirm packaging/transcribe-launcher.spec
#
# Note this builds a *launcher*, not a frozen server. transcribe_server.py is
# shipped as a plain .py and executed by uv, which keeps it a PEP 723 script
# and avoids PyInstaller's CUDA/ctranslate2 discovery problems entirely.

from pathlib import Path

ROOT = Path(SPECPATH).parent  # noqa: F821 - provided by PyInstaller
VENDOR = ROOT / "packaging" / "vendor"

datas = [(str(ROOT / "transcribe_server.py"), ".")]
for extra in ("ffmpeg.exe", "uv.exe"):
    if (VENDOR / extra).is_file():
        datas.append((str(VENDOR / extra), "vendor"))

a = Analysis(  # noqa: F821
    [str(ROOT / "launcher" / "transcribe_tray.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # pystray picks its backend by platform at import time; PyInstaller
        # cannot see through that.
        "pystray._win32",
        "PIL.Image",
        "PIL.ImageDraw",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # The launcher never transcribes; keep the heavy stack out of the exe.
        "ctranslate2",
        "faster_whisper",
        "torch",
        "numpy",
        "onnxruntime",
        "av",
        "nvidia",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Transcription Server",
    debug=False,
    strip=False,
    upx=False,
    # A tray app: no console window. Everything the server prints goes to
    # %LOCALAPPDATA%\TranscriptionServer\server.log instead.
    console=False,
    disable_windowed_traceback=False,
    icon=str(ROOT / "packaging" / "icon.ico")
    if (ROOT / "packaging" / "icon.ico").is_file()
    else None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="TranscriptionServer",
)
