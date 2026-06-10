@echo off
setlocal EnableExtensions EnableDelayedExpansion

if "%PORT%"=="" set PORT=8010
if "%HOST%"=="" set HOST=127.0.0.1

set "SCRIPT_DIR=%~dp0"
echo %SCRIPT_DIR% | findstr /B /I "\\\\wsl.localhost\\Ubuntu\\" >nul
if not errorlevel 1 (
  where wsl.exe >nul 2>&1
  if errorlevel 1 (
    echo This project is under WSL, but wsl.exe was not found.
    echo Please run ./start.sh inside WSL, or copy the project to a normal Windows folder.
    pause
    exit /b 1
  )

  for /f "delims=" %%I in ('powershell -NoProfile -Command "$p=$env:SCRIPT_DIR.TrimEnd([char]92); $p=$p -replace '^\\\\wsl\.localhost\\Ubuntu',''; $p=$p -replace '\\','/'; if ($p -eq '') { $p='/' }; Write-Output $p"') do set "WSL_PROJECT_DIR=%%I"

  echo Detected WSL project path.
  echo Starting via WSL at !WSL_PROJECT_DIR!
  echo.
  wsl bash -lc "cd '!WSL_PROJECT_DIR!' && chmod +x start.sh && PORT='%PORT%' HOST='%HOST%' ARK_API_KEY='%ARK_API_KEY%' ARK_TEXT_ENDPOINT_ID='%ARK_TEXT_ENDPOINT_ID%' ARK_VIDEO_ENDPOINT_ID='%ARK_VIDEO_ENDPOINT_ID%' AIGC_DISABLE_LLM='%AIGC_DISABLE_LLM%' AIGC_DISABLE_VIDEO_MODEL='%AIGC_DISABLE_VIDEO_MODEL%' AIGC_DISABLE_BACKGROUND_REMOVAL='%AIGC_DISABLE_BACKGROUND_REMOVAL%' AIGC_SKIP_INSTALL='%AIGC_SKIP_INSTALL%' ./start.sh"
  set "EXIT_CODE=!ERRORLEVEL!"
  if not "!EXIT_CODE!"=="0" (
    echo.
    echo WSL startup failed with error code !EXIT_CODE!.
    pause
  )
  exit /b !EXIT_CODE!
)

pushd "%SCRIPT_DIR%" >nul 2>&1
if errorlevel 1 (
  echo Failed to enter project directory:
  echo %SCRIPT_DIR%
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  py -3.12 -m venv .venv
  if errorlevel 1 py -3 -m venv .venv
  if errorlevel 1 python -m venv .venv
  if errorlevel 1 goto fail
)

if "%AIGC_SKIP_INSTALL%"=="1" (
  echo Skipping dependency installation because AIGC_SKIP_INSTALL=1.
) else if not exist ".venv\.aigc_deps_installed" (
  echo Installing dependencies. This may take a few minutes on the first run...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto fail

  ".venv\Scripts\python.exe" -m pip install -e ".[dev]"
  if errorlevel 1 goto fail

  echo ok> ".venv\.aigc_deps_installed"
) else (
  echo Dependencies already installed. Set AIGC_SKIP_INSTALL=0 and delete .venv\.aigc_deps_installed to reinstall.
)

if "%ARK_API_KEY%"=="" (
  if "%AIGC_DISABLE_LLM%"=="" set AIGC_DISABLE_LLM=1
  if "%AIGC_DISABLE_VIDEO_MODEL%"=="" set AIGC_DISABLE_VIDEO_MODEL=1
)

echo.
echo Starting AIGC Video System at http://%HOST%:%PORT%
echo Keep this window open while using the demo.
echo.
".venv\Scripts\python.exe" task_creation_demo_app.py
set EXIT_CODE=%ERRORLEVEL%
popd >nul 2>&1
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Server exited with error code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%

:fail
echo.
echo Startup failed. Check the error above.
echo Common fixes:
echo - Install Python 3.12 and make sure python or py is in PATH. Python 3.14 may not have all dependency wheels yet.
echo - Run this file from a normal Windows folder, or run ./start.sh inside WSL.
echo - If dependency installation fails, check your network or pip mirror.
popd >nul 2>&1
pause
exit /b 1
