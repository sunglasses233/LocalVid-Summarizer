@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PROJECT_ROOT=%~dp0"
set "ACTIVE_PROFILE=core"

if exist "%PROJECT_ROOT%runtime\active-profile.txt" set /p ACTIVE_PROFILE=<"%PROJECT_ROOT%runtime\active-profile.txt"
if /I not "%ACTIVE_PROFILE%"=="core" if /I not "%ACTIVE_PROFILE%"=="vocal" set "ACTIVE_PROFILE=core"

set "PRIVATE_PYTHON=%PROJECT_ROOT%runtime\envs\%ACTIVE_PROFILE%\Scripts\python.exe"
if not exist "%PRIVATE_PYTHON%" (
    echo No project-local Python environment was found for %ACTIVE_PROFILE%.
    echo Run the installation batch file first.
    pause
    exit /b 1
)

set "TEMP=%PROJECT_ROOT%runtime\tmp"
set "TMP=%PROJECT_ROOT%runtime\tmp"
set "HF_HOME=%PROJECT_ROOT%runtime\huggingface-cache"
set "HUGGINGFACE_HUB_CACHE=%PROJECT_ROOT%runtime\huggingface-cache\hub"
set "TORCH_HOME=%PROJECT_ROOT%runtime\torch-cache"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "AI_VIDEO_PROFILE=%ACTIVE_PROFILE%"
set "AI_VIDEO_CUDA_BIN_DIR=%PROJECT_ROOT%tools\cuda\bin"
set "PATH=%PROJECT_ROOT%tools\cuda\bin;%PROJECT_ROOT%tools\ffmpeg\bin;%PATH%"

echo Checking the %ACTIVE_PROFILE% environment...
"%PRIVATE_PYTHON%" -B "%PROJECT_ROOT%runtime_check.py" --profile "%ACTIVE_PROFILE%"
set "CHECK_EXIT_CODE=%ERRORLEVEL%"
echo.
if "%CHECK_EXIT_CODE%"=="0" (
    echo Environment check passed. You can run the startup batch file.
) else (
    echo Environment check failed. Follow the messages above to fix missing resources.
)
pause
exit /b %CHECK_EXIT_CODE%
