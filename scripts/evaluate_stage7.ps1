param(
    [string]$ExperimentRoot = 'runs\stage7_cuda_v1',
    [string]$OutputDir = 'reports\stage7_cuda_v1'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$Host.UI.RawUI.WindowTitle = 'UJIIndoorLoc - Stage 7 - Evaluation and benchmarks'
& '.\.venv-cuda\Scripts\python.exe' -u scripts/evaluate_stage7.py --experiment-root $ExperimentRoot --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Stage 7 evaluation failed. Read evaluation_terminal.log.' }
