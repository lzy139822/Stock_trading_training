@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" main.py
if errorlevel 1 (
  echo.
  echo [ERROR] Exit code %errorlevel%
  pause
)
