@echo off
cd /d "%~dp0"
if exist "%~dp0venv\Scripts\activate.bat" (
  call "%~dp0venv\Scripts\activate.bat"
)
python main.py
