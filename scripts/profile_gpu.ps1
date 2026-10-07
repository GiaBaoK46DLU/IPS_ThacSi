param(
    [string]$Python = '.\.venv-cuda\Scripts\python.exe',
    [string]$OutputRoot = 'runs\gpu_calibration_v1'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
# Full train/validation, 3 epochs each; no test and no change to base settings.
# Separate from final 100-epoch comparison; seed 42 is a timing calibration only.
foreach ($modelName in @('mlp','mlp_mixer','kan_mixer')) {
    $runDirectory = Join-Path $OutputRoot ($modelName + '_seed42')
    & $Python -m ips train --model $modelName --seed 42 --device cuda --epochs 3 --run-dir $runDirectory
    if ($LASTEXITCODE -ne 0) { throw ('GPU calibration failed: ' + $runDirectory) }
}
& $Python scripts/summarize_gpu_profile.py --root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw 'GPU calibration report failed.' }
