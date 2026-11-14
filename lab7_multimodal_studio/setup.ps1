param([string]$Python, [switch]$Cpu, [switch]$SkipModels, [switch]$SkipServices)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../setup.ps1') -Lab 7 -Python $Python -Cuda:(-not $Cpu) -SkipModels:$SkipModels -SkipServices:$SkipServices
