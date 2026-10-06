@echo off
rem CrispFrame launcher: run GUI with the venv pythonw (no console)
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo [CrispFrame] .venv not found. Create the venv first.
  pause
  exit /b 1
)
start "CrispFrame" ".venv\Scripts\pythonw.exe" gui.py
