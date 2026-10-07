param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$OutputRoot = 'runs\experiment_v1',
    [string]$Device = 'cpu',
    [int]$Epochs = 100,
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
if ($Smoke) { $modelSeeds = @(42) } else { $modelSeeds = @(42,43,44) }
foreach ($modelName in @('mlp','mlp_mixer','kan_mixer')) {
    foreach ($modelSeed in $modelSeeds) {
        $runDirectory = Join-Path $OutputRoot ($modelName + '_seed' + $modelSeed)
        $arguments = @('-m','ips','train','--model',$modelName,'--seed',$modelSeed,
                       '--device',$Device,'--epochs',$Epochs,'--run-dir',$runDirectory)
        if ($Smoke) { $arguments += '--smoke' }
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) { throw ('Training failed: ' + $runDirectory) }
    }
}
