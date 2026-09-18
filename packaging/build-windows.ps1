# Build the Windows tray launcher.
#
#   pwsh -File packaging/build-windows.ps1              # build
#   pwsh -File packaging/build-windows.ps1 -Installer   # build + Inno Setup
#
# Produces dist\TranscriptionServer\ (~50 MB) plus, with -Installer, a single
# setup .exe. Run on Windows: PyInstaller cannot cross-compile.

[CmdletBinding()]
param(
    [switch]$Installer,
    [switch]$SkipVendor,   # reuse an existing packaging\vendor
    [string]$Python = "3.12"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Vendor = Join-Path $PSScriptRoot "vendor"

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "!   $msg" -ForegroundColor Yellow }

# --- prerequisites ---------------------------------------------------------- #

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required to build: winget install astral-sh.uv"
}

# --- vendor: uv.exe and a static ffmpeg ------------------------------------- #
# Both are bundled so the first run needs no winget step and no uv download.

if (-not $SkipVendor) {
    New-Item -ItemType Directory -Force -Path $Vendor | Out-Null

    if (-not (Test-Path (Join-Path $Vendor "uv.exe"))) {
        Step "Fetching uv.exe"
        $zip = Join-Path $env:TEMP "uv-windows.zip"
        Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath (Join-Path $env:TEMP "uv-extract") -Force
        $found = Get-ChildItem -Path (Join-Path $env:TEMP "uv-extract") -Filter "uv.exe" -Recurse | Select-Object -First 1
        if (-not $found) { throw "uv.exe not found in the release archive" }
        Copy-Item $found.FullName (Join-Path $Vendor "uv.exe")
        Remove-Item $zip -Force
    } else {
        Step "uv.exe already vendored"
    }

    if (-not (Test-Path (Join-Path $Vendor "ffmpeg.exe"))) {
        Step "Fetching a static ffmpeg (~80 MB)"
        # gyan.dev "essentials" is the usual static Windows build.
        $zip = Join-Path $env:TEMP "ffmpeg-release-essentials.zip"
        Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $zip
        $extract = Join-Path $env:TEMP "ffmpeg-extract"
        Expand-Archive -Path $zip -DestinationPath $extract -Force
        $found = Get-ChildItem -Path $extract -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
        if (-not $found) { throw "ffmpeg.exe not found in the archive" }
        Copy-Item $found.FullName (Join-Path $Vendor "ffmpeg.exe")
        Remove-Item $zip -Force
        Warn "ffmpeg is vendored, not auto-updated: libav is attack surface, so rebuild periodically."
    } else {
        Step "ffmpeg.exe already vendored"
    }
}

# --- build ------------------------------------------------------------------ #

Step "Building with PyInstaller"
Push-Location $Root
try {
    uv run --no-project --python $Python --with pyinstaller --with pystray --with pillow `
        pyinstaller --clean --noconfirm packaging/transcribe-launcher.spec
} finally {
    Pop-Location
}

$dist = Join-Path $Root "dist\TranscriptionServer"
if (-not (Test-Path $dist)) { throw "Build did not produce $dist" }

$exe = Join-Path $dist "Transcription Server.exe"
if (-not (Test-Path $exe)) { throw "Launcher exe missing from $dist" }

$size = (Get-ChildItem $dist -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Step ("Built {0:N0} MB -> {1}" -f $size, $dist)

# --- smoke test ------------------------------------------------------------- #
# The launcher's self-test does not need a tray or a display, so it is a cheap
# way to catch a broken bundle before it reaches a user.

Step "Running the launcher self-test from the bundle"
& $exe --self-test
if ($LASTEXITCODE -ne 0) { throw "Bundled launcher failed its self-test" }

# --- installer -------------------------------------------------------------- #

if ($Installer) {
    $iscc = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1

    if (-not $iscc) {
        Warn "Inno Setup 6 not found; skipping the installer."
        Warn "Install it with: winget install JRSoftware.InnoSetup"
    } else {
        Step "Building the installer"
        & $iscc (Join-Path $PSScriptRoot "transcribe-server.iss")
        if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
        Step "Installer written to packaging\output"
    }
}

Step "Done."
