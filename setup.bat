@echo off
setlocal EnableDelayedExpansion

REM --- Check Python version ---
echo Checking Python version...

for /f "tokens=2 delims== " %%I in ('"python --version 2>&1"') do set PYVERSION=%%I

for /f "tokens=1,2 delims=." %%A in ("%PYVERSION%") do (
    set MAJOR=%%A
    set MINOR=%%B
)

if "%MAJOR%.%MINOR%" NEQ "3.10" (
    echo.
    echo [ERROR] Python 3.10.11 is required. Current version is %PYVERSION%.
    echo Please install Python 3.10.11 and try again.
    exit /b 1
)

REM --- Confirm exact version ---
for /f "tokens=2 delims= " %%V in ('python --version') do set PYFULL=%%V
if "%PYFULL%" NEQ "3.10.11" (
    echo.
    echo [ERROR] Python 3.10.11 is required. Detected version: %PYFULL%
    echo Please install the exact version 3.10.11 and try again.
    exit /b 1
)

echo Python version verified: %PYFULL%
echo.

REM --- Create virtual environment ---
echo Creating virtual environment in .venv...
python -m venv .venv

if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)

REM --- Activate virtual environment ---
echo Activating virtual environment...
call .venv\Scripts\activate

if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

REM --- Install requirements ---
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt

    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        exit /b 1
    )

    echo.
    echo [SUCCESS] Setup completed successfully!
) else (
    echo [ERROR] requirements.txt not found.
    exit /b 1
)

endlocal
