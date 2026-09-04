#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Error $msg; exit 1 }

function Test-Cmd($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

if (-not (Test-Cmd winget)) {
  Fail "winget not found — install 'App Installer' from the Microsoft Store, then re-run."
}

if (-not (Test-Cmd nvidia-smi)) {
  Fail "nvidia-smi not found — install the NVIDIA driver first (this project is NVIDIA-only: CUDA + OptiX, no AMD/Vulkan)."
}
Write-Step "NVIDIA driver OK: $(& nvidia-smi -L | Select-Object -First 1)"

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

if (-not (Test-Cmd nvcc)) {
  Write-Step "installing NVIDIA CUDA Toolkit (winget id: Nvidia.CUDA)"
  winget install --id Nvidia.CUDA -e --silent --accept-package-agreements --accept-source-agreements
  if (-not (Test-Cmd nvcc)) {
    Fail "nvcc still not on PATH — winget usually needs a fresh shell (or a reboot) to pick up the new PATH. Close this window, open a new PowerShell, and re-run this script."
  }
}
Write-Step "CUDA Toolkit OK: $((& nvcc --version | Select-Object -Last 1))"

# claudia: picks whatever `vswhere -latest` reports; doesn't handle multiple side-by-side VS installs or ARM64 Windows.
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

Import-Module (Join-Path $vsInstallPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vsInstallPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null

function Test-OptixRoot($optixPath) {
  if (-not $optixPath) { return $false }
  return (Test-Path (Join-Path $optixPath "include\optix.h")) -and
         (Test-Path (Join-Path $optixPath "SDK\sutil\vec_math.h"))
}

if (-not (Test-OptixRoot $env:OPTIX_ROOT)) {
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

Write-Step "configuring"
cmake -B build -G Ninja
Write-Step "building"
cmake --build build

Write-Step "done — run .\build\italy-rs.exe (or .\build\italy-rs.exe assets\test.glb)"
