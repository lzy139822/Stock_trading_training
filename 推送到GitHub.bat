@echo off
cd /d "%~dp0"

rem ============================================================
rem  Push to GitHub: lzy139822/Stock_trading_training
rem
rem  Why this script:
rem  The bundled Git credential helpers (helper-selector / Git
rem  Credential Manager) can hang forever waiting for a login
rem  dialog, which makes "git push" look frozen with no output.
rem  Pointing GIT_CONFIG_GLOBAL / GIT_CONFIG_SYSTEM at a clean
rem  config that only uses the gh CLI credential helper avoids
rem  the interactive helpers entirely.
rem
rem  This file is intentionally pure ASCII so cmd.exe reads it
rem  correctly under any code page.
rem ============================================================

set "GIT_CONFIG_GLOBAL=%~dp0.workbuddy\tmp\gitconfig-push"
set "GIT_CONFIG_SYSTEM=%~dp0.workbuddy\tmp\gitconfig-system-empty"
set GIT_TERMINAL_PROMPT=0

echo [1/2] Pushing branch master ...
git push origin master --progress
if errorlevel 1 goto :err

echo.
echo [2/2] Pushing tags ...
git push --tags origin
if errorlevel 1 goto :err

echo.
echo Done: https://github.com/lzy139822/Stock_trading_training
pause
exit /b 0

:err
echo.
echo [ERROR] Push failed, exit code %errorlevel%
pause
exit /b 1
