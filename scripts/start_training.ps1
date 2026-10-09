<#
.SYNOPSIS
  Start a long training run, or resume it if it already has checkpoints.

.DESCRIPTION
      powershell -ExecutionPolicy Bypass -File scripts\start_training.ps1
      powershell -ExecutionPolicy Bypass -File scripts\start_training.ps1 -Device cuda -NEnvs 16

  The run lives in runs\<RunName>. Running the same command again after stopping
  (Ctrl+C, reboot) resumes from the latest checkpoint. Raise -TotalSteps to train longer.
  Extra settings: -Set "ppo.learning_rate=0.0001","eval.max_steps=1000"

.PARAMETER RunName          Directory name under runs\ (default solo_main).
.PARAMETER TotalSteps       Target total environment steps (default 50,000,000).
.PARAMETER NEnvs            Parallel environments (default 16).
.PARAMETER VecEnv           dummy (single process, default) or subproc.
.PARAMETER Device           auto / cpu / cuda (default auto).
.PARAMETER CheckpointEvery  Steps between checkpoints (default 500,000).
.PARAMETER VideoEvery       Make videos every N checkpoints (default 4).
.PARAMETER Set              Additional config overrides (key=value).
#>
param(
    [string]$RunName = "solo_main",
    [long]$TotalSteps = 50000000,
    [int]$NEnvs = 16,
    [ValidateSet("dummy", "subproc")][string]$VecEnv = "dummy",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [long]$CheckpointEvery = 500000,
    [int]$VideoEvery = 4,
    [string[]]$Set = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: .venv not found. Run scripts\setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

$run = Join-Path "runs" $RunName
$overrides = @(
    "ppo.total_timesteps=$TotalSteps",
    "env.n_envs=$NEnvs",
    "env.vec_env=$VecEnv",
    "device=$Device",
    "checkpoint.every_steps=$CheckpointEvery",
    "video.every_checkpoints=$VideoEvery"
) + $Set

$cmd = @("scripts\train.py")
$hasCheckpoint = Test-Path (Join-Path $run "checkpoints\step_*.zip")
if ($hasCheckpoint) {
    Write-Host "Resuming $run (target $TotalSteps steps)" -ForegroundColor Cyan
    $cmd += @("--resume", $run)
} else {
    Write-Host "Starting new run $run (target $TotalSteps steps)" -ForegroundColor Cyan
    $cmd += @("--run-dir", $run)
}
foreach ($o in $overrides) { $cmd += @("--set", $o) }

Write-Host "TensorBoard (another terminal): .venv\Scripts\tensorboard.exe --logdir runs"
Write-Host "Stop with Ctrl+C; the current weights are saved. Run this script again to resume.`n"
& $py @cmd
exit $LASTEXITCODE
