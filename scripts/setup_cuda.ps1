param([string]$BasePython = '.\.venv\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
if (-not (Test-Path '.\.venv-cuda\Scripts\python.exe')) {
    & $BasePython -m venv .venv-cuda
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create CUDA environment.' }
}
$cudaPython = '.\.venv-cuda\Scripts\python.exe'
& $cudaPython -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Cannot install CUDA PyTorch.' }
& $cudaPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Cannot install project dependencies.' }
& $cudaPython -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print('VRAM bytes:', torch.cuda.get_device_properties(0).total_memory)"
if ($LASTEXITCODE -ne 0) { throw 'CUDA device check failed.' }
& $cudaPython -m ips check --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Model forward check failed.' }
