@echo off
REM ============================================================
REM  Online / production build (double-click to run)
REM
REM  Builds the exe with PyInstaller and copies config.yaml.
REM  Previous results (dist\PathPlanner\out) and inputs
REM  (dist\PathPlanner\data) are preserved across builds.
REM
REM  After building, run the exe manually:
REM     dist\PathPlanner\PathPlanner.exe
REM     dist\PathPlanner\PathPlanner.exe --step D:\parts\PartA.step
REM ============================================================
setlocal
set "EXIT_CODE=0"
cd /d "%~dp0\.."

echo.
echo ============================================================
echo   Path Planner - Production Build
echo ============================================================
echo   This script will:
echo     1) clean previous build
echo     2) build the exe with PyInstaller
echo     3) copy config.yaml
echo ============================================================
echo.

REM Kill any running instance first
taskkill /f /im PathPlanner.exe >nul 2>nul

REM Back up previous results and inputs so they survive the rebuild
set "BACKUP_DIR=%CD%\.build_backup"
if exist "%BACKUP_DIR%" rmdir /s /q "%BACKUP_DIR%"
if exist "dist\PathPlanner\out" (
    mkdir "%BACKUP_DIR%" 2>nul
    move "dist\PathPlanner\out" "%BACKUP_DIR%\out" >nul
)
if exist "dist\PathPlanner\data" (
    if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%" 2>nul
    move "dist\PathPlanner\data" "%BACKUP_DIR%\data" >nul
)

echo [1/3] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist dist goto clean_failed
if exist main.spec del /q main.spec
if exist PathPlanner.spec del /q PathPlanner.spec

echo [2/3] Building exe...
uv run pyinstaller --noconfirm --console --collect-all OCP --collect-all shapely --name PathPlanner main.py
if errorlevel 1 goto build_failed

echo [3/3] Copying config.yaml...
copy /y config.yaml dist\PathPlanner\config.yaml >nul
if errorlevel 1 goto build_failed

REM Restore previous results and inputs
call :restore_backup

echo.
echo ============================================
echo   BUILD PASSED
echo ============================================
echo   Output: dist\PathPlanner\PathPlanner.exe
echo   Run it manually, e.g.:
echo     dist\PathPlanner\PathPlanner.exe
echo     dist\PathPlanner\PathPlanner.exe --step D:\parts\PartA.step
echo ============================================
goto finish

:clean_failed
echo.
echo Error: cannot remove dist\ folder. Another program is using it.
echo Close any File Explorer or console window opened in dist\, then retry.
set "EXIT_CODE=1"
call :restore_backup
goto finish

:build_failed
set "EXIT_CODE=%ERRORLEVEL%"
call :restore_backup
echo.
echo ============================================
echo   BUILD FAILED (exit code %EXIT_CODE%)
echo ============================================
goto finish

:restore_backup
mkdir "dist\PathPlanner" 2>nul
if exist "%BACKUP_DIR%\out" move "%BACKUP_DIR%\out" "dist\PathPlanner\out" >nul
if exist "%BACKUP_DIR%\data" move "%BACKUP_DIR%\data" "dist\PathPlanner\data" >nul
if exist "%BACKUP_DIR%" rmdir /s /q "%BACKUP_DIR%" 2>nul
exit /b

:finish
echo.
pause
exit /b %EXIT_CODE%
