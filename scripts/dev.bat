@echo off
REM ============================================================
REM  Local development verification (double-click to run)
REM
REM  Syncs dependencies, runs unit tests, then runs the full
REM  pipeline on a STEP file. The STEP file can be passed as an
REM  argument, or entered interactively when prompted.
REM
REM  Usage:
REM     dev.bat                          prompt for the STEP file
REM     dev.bat data\input\Part.STEP     run with the given file
REM ============================================================
setlocal
set "EXIT_CODE=0"
set "STEP_FILE=%~1"
cd /d "%~dp0\.."

if "%STEP_FILE%"=="" goto prompt_step
goto have_step

:prompt_step
echo.
echo ============================================================
echo   Path Planner - Local Development
echo ============================================================
echo   This script will:
echo     1) sync dependencies
echo     2) run unit tests
echo     3) run the full pipeline on your STEP file
echo ============================================================
echo.
set /p "STEP_FILE=Enter the STEP file path: "
if "%STEP_FILE%"=="" goto no_step

:have_step
echo.
echo [1/3] Syncing dependencies...
uv sync
if errorlevel 1 goto failed

echo [2/3] Running unit tests...
uv run python -m unittest discover -s tests
if errorlevel 1 goto failed

echo [3/3] Running full pipeline on: %STEP_FILE%
uv run python main.py --step "%STEP_FILE%"
if errorlevel 1 goto failed

echo.
echo ============================================
echo   DEV VERIFICATION PASSED
echo ============================================
goto finish

:no_step
echo.
echo Error: no STEP file provided.
set "EXIT_CODE=2"
goto finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo ============================================
echo   DEV VERIFICATION FAILED (exit code %EXIT_CODE%)
echo ============================================

:finish
echo.
pause
exit /b %EXIT_CODE%
