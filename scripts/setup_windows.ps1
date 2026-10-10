<#
.SYNOPSIS
  Prepare puyotrain for training on Windows.

.DESCRIPTION
  Run from anywhere inside the cloned repository:

      powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

  Steps:
    1. Check prerequisites (Git, Rust, MSVC Build Tools, Python).
    2. Create .venv with the requested Python version.
    3. Install PyTorch. With an NVIDIA GPU it tries the CUDA builds in -CudaIndexes
       order and keeps the first one where torch.cuda.is_available() is True.
    4. Build the Rust extension and install the RL dependencies (maturin develop).
    5. Run the Rust and Python test suites.
    6. Measure training speed on CPU and GPU (bench_train.py).

  Safe to run again: existing .venv is reused, finished steps are quick.

.PARAMETER Python
  Python version to use, e.g. 3.12. Default: auto (3.12 preferred, then 3.13 / 3.11 / 3.10,
  through the py launcher or "python" on PATH).
.PARAMETER Cpu
  Install CPU-only PyTorch even if an NVIDIA GPU is present.
.PARAMETER CudaIndexes
  PyTorch CUDA wheel indexes to try, newest first.
.PARAMETER SkipTests
  Skip cargo test / pytest.
.PARAMETER SkipBench
  Skip the training speed measurement.
#>
param(
    [string]$Python = "",
    [switch]$Cpu,
    [string[]]$CudaIndexes = @("cu130", "cu128", "cu126"),
    [switch]$SkipTests,
    [switch]$SkipBench
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Fail([string]$Message) { Write-Host "`nERROR: $Message" -ForegroundColor Red; exit 1 }
function Has([string]$Name) { [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# Run a native command and stop on a non-zero exit code.
# (A simple function on purpose: arguments such as "-m" are passed through as-is.)
function Invoke-Native {
    $exe = $args[0]
    $rest = @($args | Select-Object -Skip 1)
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { Fail "'$exe $($rest -join ' ')' failed (exit $LASTEXITCODE)" }
}

# Run a native command, hide its output, and return the exit code. Messages on stderr
# must not stop the script (Windows PowerShell 5.1 turns redirected stderr into errors).
function Invoke-Quiet {
    $exe = $args[0]
    $rest = @($args | Select-Object -Skip 1)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $exe @rest 2>&1 | Out-Null } finally { $ErrorActionPreference = $old }
    return $LASTEXITCODE
}

# Run a native command and return its combined output as text (never stops the script).
function Get-QuietOutput {
    $exe = $args[0]
    $rest = @($args | Select-Object -Skip 1)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { return ((& $exe @rest 2>&1 | ForEach-Object { "$_" }) -join "`n").Trim() }
    catch { return "" }
    finally { $ErrorActionPreference = $old }
}

# ------------------------------------------------------------------ 1. prerequisites
Step "Checking prerequisites"
$missing = @()
if (-not (Has git)) {
    $missing += "Git:        winget install --id Git.Git -e"
}
if (-not (Has cargo)) {
    $missing += "Rust:       winget install --id Rustlang.Rustup -e   (then open a new terminal)"
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vcPath = $null
if (Test-Path $vswhere) {
    $vcPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $vcPath) {
    $missing += ("MSVC:       winget install --id Microsoft.VisualStudio.2022.BuildTools -e " +
                 "--override ""--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended""")
}
# Find a Python that PyTorch supports. $pyCmd is the command (with launcher flag) to run it.
$pyCmd = $null
$wanted = @("3.12", "3.13", "3.11", "3.10")
if ($Python) { $wanted = @($Python) }
if (Has py) {
    foreach ($v in $wanted) {
        if ((Invoke-Quiet py "-$v" -c "import sys") -eq 0) { $pyCmd = @("py", "-$v"); break }
    }
}
if (-not $pyCmd -and (Has python)) {
    # "python" on PATH (may be the Microsoft Store placeholder, which fails this check)
    $range = "(3,10) <= sys.version_info[:2] <= (3,13)"
    if ($Python) { $range = "'%d.%d' % sys.version_info[:2] == '$Python'" }
    if ((Invoke-Quiet python -c "import sys; sys.exit(0 if $range else 1)") -eq 0) { $pyCmd = @("python") }
}
if (-not $pyCmd) {
    $found = @()
    if (Has py) { $found += ("py --list:`n" + (Get-QuietOutput py --list)) }
    if (Has python) { $found += ("python on PATH: " + (Get-QuietOutput python --version)) }
    if ($found.Count -eq 0) { $found = @("(no python / py found)") }
    Write-Host "Python found on this PC:" -ForegroundColor Yellow
    $found | ForEach-Object { Write-Host "  $_" }
    $missing += "Python 3.12: winget install --id Python.Python.3.12 -e   (then open a new terminal)"
}
function Invoke-BasePython {
    $exe = $pyCmd[0]
    $rest = @($pyCmd | Select-Object -Skip 1) + @($args)
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { Fail "'$($pyCmd -join ' ') $($args -join ' ')' failed (exit $LASTEXITCODE)" }
}
if ($missing.Count -gt 0) {
    Write-Host "Missing prerequisites. Install them, open a NEW terminal, and run this script again:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host ("git: {0}" -f (git --version))
Write-Host ("rust: {0}" -f (rustc --version))
Write-Host ("MSVC: {0}" -f $vcPath)
Write-Host ("python: {0} ({1})" -f (Invoke-BasePython --version | Out-String).Trim(), ($pyCmd -join ' '))

# ------------------------------------------------------------------ 2. venv
Step "Creating virtual environment (.venv)"
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Invoke-BasePython -m venv .venv
} else {
    Write-Host "reusing existing .venv"
}
$env:VIRTUAL_ENV = Join-Path $Root ".venv"
$env:PATH = (Join-Path $Root ".venv\Scripts") + ";" + $env:PATH
Invoke-Native $py -m pip install --upgrade pip maturin

# ------------------------------------------------------------------ 3. PyTorch
Step "Installing PyTorch"
$hasGpu = (-not $Cpu) -and (Has nvidia-smi)
$cudaOk = $false
if ($hasGpu) {
    Write-Host "NVIDIA GPU:"
    & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    foreach ($idx in $CudaIndexes) {
        Write-Host "trying PyTorch CUDA build: $idx"
        Invoke-Quiet $py -m pip uninstall -y torch | Out-Null
        & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$idx"
        if ($LASTEXITCODE -ne 0) { Write-Warning "no torch wheel from $idx; trying the next one"; continue }
        $probe = Invoke-Quiet $py -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
        if ($probe -eq 0) { $cudaOk = $true; break }
        Write-Warning "torch from $idx installed but CUDA is not usable (GPU driver too old?); trying the next one"
    }
    if (-not $cudaOk) {
        Write-Warning "No CUDA build worked. Falling back to CPU PyTorch. Updating the NVIDIA driver usually fixes this."
    }
}
if (-not $cudaOk) {
    Invoke-Quiet $py -m pip uninstall -y torch | Out-Null
    Invoke-Native $py -m pip install torch
}
Invoke-Native $py -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# ------------------------------------------------------------------ 4. build
Step "Building the Rust extension and installing RL dependencies"
Invoke-Native $py -m maturin develop --release -E "rl,test"

# ------------------------------------------------------------------ 5. tests
if (-not $SkipTests) {
    Step "Running Rust tests"
    Invoke-Native cargo test --workspace --release
    Step "Running Python tests"
    Invoke-Native $py -m pytest -q
}

# ------------------------------------------------------------------ 6. speed check
$cores = [Environment]::ProcessorCount
if (-not $SkipBench) {
    Step "Measuring training speed (about 1-3 minutes)"
    $devices = @("cpu")
    if ($cudaOk) { $devices += "cuda" }
    foreach ($n in @(8, 16)) {
        Invoke-Native $py scripts\bench_train.py --devices @devices --steps 20480 --set "env.n_envs=$n"
    }
    Write-Host "Pick the fastest device / n_envs above for start_training.ps1 (-Device, -NEnvs)."
}

Step "Setup finished"
Write-Host "Start (or resume) training:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\start_training.ps1"
Write-Host "Watch progress in another terminal:"
Write-Host "  .venv\Scripts\tensorboard.exe --logdir runs"
