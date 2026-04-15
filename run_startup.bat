@echo off
REM Starts dictation in the background (no console). Use this target for Windows Startup.
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0main.py"
  exit /b 0
)
if exist "%~dp0venv\Scripts\pythonw.exe" (
  start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0main.py"
  exit /b 0
)
echo No venv pythonw found. Create .venv or venv in this folder, or run run.bat instead.
pause
