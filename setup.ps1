param([Parameter(Mandatory=$true)][ValidateRange(1,7)][int]$Lab, [string]$Python, [switch]$Cuda, [switch]$SkipModels, [switch]$SkipServices)
$ErrorActionPreference = 'Stop'
$names = @('lab1', 'lab2_vllm', 'lab3_chroma_rag', 'lab4_img2img', 'lab5_multimodal_captioning', 'lab6_llamaindex_agents', 'lab7_multimodal_studio')
$labRoot = Join-Path $PSScriptRoot $names[$Lab - 1]
if (-not $Python) {
    $Python = (& py -3.13 -c 'import sys; print(sys.executable)')
    if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.13 or pass -Python <executable>.' }
}
& $Python -c 'import sys; assert sys.version_info[:2] == (3,13), "Python 3.13 required"'
if ($LASTEXITCODE -ne 0) { throw 'Use Python 3.13.' }
$venvPython = Join-Path $labRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv (Join-Path $labRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
$requirements = Join-Path $labRoot 'requirements.lock.txt'
if (-not (Test-Path -LiteralPath $requirements)) { $requirements = Join-Path $labRoot 'requirements.txt' }
& $venvPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed' }
& $venvPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if ($Cuda) {
    & $venvPython -m pip install --no-deps torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel installation failed' }
    & $venvPython -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA validation failed' }
}
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts' }

if (-not $SkipModels) {
    $models = switch ($Lab) { 3 { @('minilm') }; 4 { @('diffusion') }; 5 { @('llava') }; 7 { @('llava','diffusion') }; default { @() } }
    foreach ($model in $models) {
        $modelArgs = @((Join-Path $PSScriptRoot 'scripts/fetch_models.py'), $model)
        if ($model -eq 'diffusion' -and $Cuda) { $modelArgs += '--fp16' }
        & $venvPython @modelArgs
        if ($LASTEXITCODE -ne 0) { throw 'Model download/checksum verification failed' }
    }
}
if ($Lab -in @(2,3,6,7) -and -not $SkipServices) {
    & ollama pull qwen3:4b
    if ($LASTEXITCODE -ne 0) { throw 'Start Ollama and rerun setup' }
    $expected = (Get-Content -Raw (Join-Path $PSScriptRoot 'services.lock.json') | ConvertFrom-Json).ollama.source_digest
    $installed = (Invoke-RestMethod http://127.0.0.1:11434/api/tags).models | Where-Object { $_.name -eq 'qwen3:4b' }
    if ($installed.digest -ne $expected) { throw 'Ollama source digest differs from services.lock.json; do not silently regenerate historical results with different weights.' }
    & ollama create qwen3:4b-studio -f (Join-Path $PSScriptRoot 'lab7_multimodal_studio/Modelfile')
    if ($LASTEXITCODE -ne 0) { throw 'Ollama CPU model creation failed' }
}
Write-Output "Interpreter: $venvPython"
