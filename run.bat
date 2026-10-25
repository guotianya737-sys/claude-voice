@echo off
chcp 65001 >nul
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [pi-voice] .venv not found, creating and installing deps...
  "C:\Program Files\Python312\python.exe" -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" main.py %*
endlocal
