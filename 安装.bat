@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ==================================================
echo AI Video Summary Assistant - Python Environment Setup
echo ==================================================
echo [1] Standard - Whisper GPU transcription and speaker diarization
echo [2] Optional - Standard with vocal separation - larger environment
echo.
choice /C 12 /N /M "Select an installation profile [1/2]: "
if errorlevel 2 (
    set "SETUP_PROFILE=vocal"
) else (
    set "SETUP_PROFILE=core"
)

echo.
echo This creates a project-local environment and installs Python packages online.
echo It does not download Python, Whisper models, CUDA files, FFmpeg, or voice models.
echo.

set "PYTHON_COMMAND="
py -3.10 -c "import struct; raise SystemExit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_COMMAND=py -3.10"

if not defined PYTHON_COMMAND (
    python -c "import struct, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_COMMAND=python"
)

if not defined PYTHON_COMMAND (
    echo Python 3.10 64-bit was not found.
    echo Install Python 3.10 64-bit first. Python 3.10.11 is recommended.
    pause
    exit /b 1
)

%PYTHON_COMMAND% -B "%~dp0scripts\setup_env.py" --profile "%SETUP_PROFILE%"
set "SETUP_EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%SETUP_EXIT_CODE%"=="0" (
    echo Python environment setup did not finish. Keep the error messages above.
) else (
    echo Python dependencies installed successfully.
    echo Next, follow docs\RESOURCE_DOWNLOADS.md to add models, CUDA files, and FFmpeg.
)
pause
exit /b %SETUP_EXIT_CODE%
