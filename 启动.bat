@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PROJECT_ROOT=%~dp0"
set "ACTIVE_PROFILE_FILE=%PROJECT_ROOT%runtime\active-profile.txt"
set "ACTIVE_PROFILE="

if exist "%ACTIVE_PROFILE_FILE%" set /p ACTIVE_PROFILE=<"%ACTIVE_PROFILE_FILE%"
if /I not "%ACTIVE_PROFILE%"=="core" if /I not "%ACTIVE_PROFILE%"=="vocal" set "ACTIVE_PROFILE=core"

set "PRIVATE_PYTHON=%PROJECT_ROOT%runtime\envs\%ACTIVE_PROFILE%\Scripts\python.exe"
if not exist "%PRIVATE_PYTHON%" (
    set "ACTIVE_PROFILE=core"
    set "PRIVATE_PYTHON=%PROJECT_ROOT%runtime\envs\core\Scripts\python.exe"
)
if not exist "%PRIVATE_PYTHON%" (
    echo No usable project-local runtime environment was found.
    echo Run the installation batch file first.
    pause
    exit /b 1
)

set "TEMP=%PROJECT_ROOT%runtime\tmp"
set "TMP=%PROJECT_ROOT%runtime\tmp"
set "PIP_CACHE_DIR=%PROJECT_ROOT%runtime\pip-cache"
set "HF_HOME=%PROJECT_ROOT%runtime\huggingface-cache"
set "HUGGINGFACE_HUB_CACHE=%PROJECT_ROOT%runtime\huggingface-cache\hub"
set "TORCH_HOME=%PROJECT_ROOT%runtime\torch-cache"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "AI_VIDEO_PYTHON=%PRIVATE_PYTHON%"
set "AI_VIDEO_CUDA_BIN_DIR=%PROJECT_ROOT%tools\cuda\bin"
set "PATH=%PROJECT_ROOT%tools\cuda\bin;%PROJECT_ROOT%tools\ffmpeg\bin;%PATH%"

echo Starting with the %ACTIVE_PROFILE% environment...
"%PRIVATE_PYTHON%" -B "%PROJECT_ROOT%launcher.py"
set "LAUNCH_EXIT_CODE=%ERRORLEVEL%"
if not "%LAUNCH_EXIT_CODE%"=="0" (
    echo.
    echo Startup failed. Follow the messages above to check the installation.
    pause
)
exit /b %LAUNCH_EXIT_CODE%
