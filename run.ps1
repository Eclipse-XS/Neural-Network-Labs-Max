param([Parameter(Mandatory=$true)][ValidateRange(1,7)][int]$Lab, [switch]$Smoke)
$ErrorActionPreference = 'Stop'
$names = @('lab1', 'lab2_vllm', 'lab3_chroma_rag', 'lab4_img2img', 'lab5_multimodal_captioning', 'lab6_llamaindex_agents', 'lab7_multimodal_studio')
$labRoot = Join-Path $PSScriptRoot $names[$Lab - 1]
$python = Join-Path $labRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Run ./setup.ps1 -Lab $Lab first" }
if ($Smoke) {
    & $python (Join-Path $PSScriptRoot 'scripts/smoke.py') --lab $Lab
} elseif ($Lab -eq 7) {
    Push-Location $labRoot
    try { & $python -m streamlit run (Join-Path $labRoot 'app.py') --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false } finally { Pop-Location }
} elseif ($Lab -eq 2) {
    & $python (Join-Path $labRoot 'cli.py')
} else {
    & $python (Join-Path $PSScriptRoot 'scripts/run_notebooks.py') $labRoot
}
if ($LASTEXITCODE -ne 0) { throw "Lab $Lab failed with exit code $LASTEXITCODE" }
