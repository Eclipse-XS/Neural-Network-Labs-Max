param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../run.ps1') -Lab 7 -Smoke:$Smoke
