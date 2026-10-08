param([string]$OutputRoot = 'runs\stage7_cuda_v1')
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$Host.UI.RawUI.WindowTitle = 'UJIIndoorLoc - Stage 7 - GPU epoch logs'
& '.\.venv-cuda\Scripts\python.exe' -u scripts/train_stage7.py --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw 'Stage 7 training stopped. Read terminal.log.' }
