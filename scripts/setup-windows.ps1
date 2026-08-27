#Requires -Version 5.1
<#
.SYNOPSIS
  Bootstraps build prerequisites for italy-rs on Windows and runs the first
  build. Mirrors humans.md's "Prerequisites"/"Build" sections for Linux —
  keep this in sync if either changes. Not yet verified on a real Windows
  machine (this project has only ever been built on Linux so far) — if a
  step below turns out wrong on real hardware, fix the step, not the intent.

.DESCRIPTION
  What this does NOT do: download the OptiX SDK. NVIDIA gates that installer
  behind a free Developer Program login, so it can't be scripted — this
  script checks for it and prints the manual steps if it's missing.

  Installs via winget: NVIDIA CUDA Toolkit, CMake, Ninja, and VS Build Tools'
  C++ workload (needed as nvcc's host compiler on Windows). Requires winget
  (ships with modern Windows 10/11; if missing, get "App Installer" from the
  Microsoft Store).
#>
$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Error $msg; exit 1 }

function Test-Cmd($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

if (-not (Test-Cmd winget)) {
  Fail "winget not found — install 'App Installer' from the Microsoft Store, then re-run."
}

# --- NVIDIA driver -------------------------------------------------------
if (-not (Test-Cmd nvidia-smi)) {
  Fail "nvidia-smi not found — install the NVIDIA driver first (this project is NVIDIA-only: CUDA + OptiX, no AMD/Vulkan)."
}
Write-Step "NVIDIA driver OK: $(& nvidia-smi -L | Select-Object -First 1)"

# --- CMake + Ninja ---------------------------------------------------------
if (-not (Test-Cmd cmake)) {
  Write-Step "installing CMake"
  winget install --id Kitware.CMake -e --silent --accept-package-agreements --accept-source-agreements
  if (-not (Test-Cmd cmake)) {
    Fail "cmake still not on PATH after install — open a new PowerShell (winget PATH updates need a fresh shell) and re-run."
  }
}
if (-not (Test-Cmd ninja)) {
  Write-Step "installing Ninja"
  winget install --id Ninja-build.Ninja -e --silent --accept-package-agreements --accept-source-agreements
  if (-not (Test-Cmd ninja)) {
    Fail "ninja still not on PATH after install — open a new PowerShell (winget PATH updates need a fresh shell) and re-run."
  }
}

# --- CUDA Toolkit ----------------------------------------------------------
if (-not (Test-Cmd nvcc)) {
  Write-Step "installing NVIDIA CUDA Toolkit (winget id: Nvidia.CUDA)"
  winget install --id Nvidia.CUDA -e --silent --accept-package-agreements --accept-source-agreements
  if (-not (Test-Cmd nvcc)) {
    Fail "nvcc still not on PATH — winget usually needs a fresh shell (or a reboot) to pick up the new PATH. Close this window, open a new PowerShell, and re-run this script."
  }
}
Write-Step "CUDA Toolkit OK: $((& nvcc --version | Select-Object -Last 1))"

# --- MSVC (nvcc's Windows host compiler) -----------------------------------
# monkey-boy: picks whatever `vswhere -latest` reports; doesn't handle
# multiple side-by-side VS installs or ARM64 Windows.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vcToolsId = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
$vsInstallPath = $null
if (Test-Path $vswhere) {
  $vsInstallPath = & $vswhere -latest -products * -requires $vcToolsId -property installationPath
}
if (-not $vsInstallPath) {
  Write-Step "installing Visual Studio Build Tools (C++ workload) — this can take a while"
  winget install --id Microsoft.VisualStudio.2022.BuildTools -e --silent --wait `
    --accept-package-agreements --accept-source-agreements `
    --override "--quiet --wait --add $vcToolsId"
  $vsInstallPath = & $vswhere -latest -products * -requires $vcToolsId -property installationPath
  if (-not $vsInstallPath) {
    Fail "VS Build Tools install didn't produce a detectable VC++ toolset — install manually from https://visualstudio.microsoft.com/downloads/ (Build Tools, 'Desktop development with C++' workload) and re-run."
  }
}
Write-Step "MSVC toolset OK: $vsInstallPath"

# Import the VS dev environment into *this* process so cl.exe/INCLUDE/LIB
# are on PATH for nvcc and for CMake's own compiler checks.
Import-Module (Join-Path $vsInstallPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vsInstallPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null

# --- OptiX SDK: cannot be scripted, only checked ---------------------------
function Test-OptixRoot($optixPath) {
  if (-not $optixPath) { return $false }
  return (Test-Path (Join-Path $optixPath "include\optix.h")) -and
         (Test-Path (Join-Path $optixPath "SDK\sutil\vec_math.h"))
}

if (-not (Test-OptixRoot $env:OPTIX_ROOT)) {
  # Default installer location; only auto-adopt it if exactly one is found.
  $candidates = @(Get-ChildItem "$env:ProgramData\NVIDIA Corporation" -Directory -Filter "OptiX SDK*" -ErrorAction SilentlyContinue)
  if ($candidates.Count -eq 1) {
    $env:OPTIX_ROOT = $candidates[0].FullName
  }
}
if (-not (Test-OptixRoot $env:OPTIX_ROOT)) {
  Write-Host @"
error: OPTIX_ROOT is not set (or doesn't point at a full SDK install).

NVIDIA gates the OptiX installer behind a free Developer Program login, so
this step is manual:
  1. Log in at https://developer.nvidia.com/designworks/optix/download
  2. Download and run the Windows installer (defaults to
     C:\ProgramData\NVIDIA Corporation\OptiX SDK <version>\).
  3. setx OPTIX_ROOT "C:\ProgramData\NVIDIA Corporation\OptiX SDK <version>"
     (setx only persists for *new* shells -- this one still needs OPTIX_ROOT
     set directly before you re-run this script.)
"@ -ForegroundColor Red
  exit 1
}
Write-Step "OptiX SDK OK: $env:OPTIX_ROOT"

# --- Build -------------------------------------------------------------
Write-Step "configuring"
cmake -B build -G Ninja
Write-Step "building"
cmake --build build

Write-Step "done — run .\build\italy-rs.exe (or .\build\italy-rs.exe assets\test.glb)"
