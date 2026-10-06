@echo off
rem CrispFrame 双击启动器：用 venv 里的 pythonw 无控制台运行 GUI
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo [CrispFrame] 找不到 .venv，请先在项目目录创建虚拟环境。
  pause
  exit /b 1
)
start "CrispFrame" ".venv\Scripts\pythonw.exe" gui.py
