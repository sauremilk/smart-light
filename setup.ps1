#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Einrichtungsskript für das Emotion-Light-Projekt (Windows PowerShell).
    Erstellt das venv, installiert Abhängigkeiten und führt einen Syntax-Check durch.
.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Force       # Löscht und erstellt das venv neu
#>

param(
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$VenvPath    = Join-Path $ProjectRoot "venv"
$PythonExe   = Join-Path $VenvPath "Scripts\python.exe"
$PipExe      = Join-Path $VenvPath "Scripts\pip.exe"
$ReqFile     = Join-Path $ProjectRoot "requirements.txt"

Write-Host "=== Emotion-Light Setup ===" -ForegroundColor Cyan
Write-Host "Projektordner: $ProjectRoot"

# --- Venv löschen (wenn -Force) ---
if ($Force -and (Test-Path $VenvPath)) {
    Write-Host "[1/4] Altes venv wird entfernt..." -ForegroundColor Yellow
    Remove-Item $VenvPath -Recurse -Force
}

# --- Venv erstellen ---
if (-not (Test-Path $VenvPath)) {
    Write-Host "[1/4] Erstelle virtuelles Environment..." -ForegroundColor Green
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "venv-Erstellung fehlgeschlagen." }
} else {
    Write-Host "[1/4] venv bereits vorhanden, überspringe Erstellung." -ForegroundColor DarkGray
}

# --- Pip upgraden ---
Write-Host "[2/4] Pip upgraden..." -ForegroundColor Green
& $PythonExe -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { throw "Pip-Upgrade fehlgeschlagen." }

# --- Abhängigkeiten installieren ---
Write-Host "[3/4] Installiere Abhängigkeiten aus requirements.txt..." -ForegroundColor Green
& $PipExe install -r $ReqFile
if ($LASTEXITCODE -ne 0) { throw "Paket-Installation fehlgeschlagen." }

# --- Syntax-Check ---
Write-Host "[4/4] Syntax-Check (main.py, config.py)..." -ForegroundColor Green
& $PythonExe -m py_compile (Join-Path $ProjectRoot "main.py")
if ($LASTEXITCODE -ne 0) { throw "Syntaxfehler in main.py!" }
& $PythonExe -m py_compile (Join-Path $ProjectRoot "config.py")
if ($LASTEXITCODE -ne 0) { throw "Syntaxfehler in config.py!" }

Write-Host ""
Write-Host "=== Setup abgeschlossen! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Naechste Schritte:" -ForegroundColor Cyan
Write-Host "  1. Bridge-IP in config.py anpassen:  HUE_BRIDGE_IP = '<deine-bridge-ip>'"
Write-Host "  2. Lampen-ID pruefen:                HUE_LIGHT_ID = <id>"
Write-Host ""
Write-Host "Starten (mit echter Hardware):" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\python.exe main.py"
Write-Host ""
Write-Host "Starten (Mock-Modus, kein Hardware noetig):" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\python.exe main.py --mock"
Write-Host ""
Write-Host "Weitere Optionen:" -ForegroundColor DarkGray
Write-Host "  .\venv\Scripts\python.exe main.py --help"
