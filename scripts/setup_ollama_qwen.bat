@echo off
REM Install Ollama and pull qwen2.5:7b (works in cmd, no PowerShell policy needed)

set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not exist "%OLLAMA%" set "OLLAMA=C:\Program Files\Ollama\ollama.exe"

if not exist "%OLLAMA%" (
    echo Ollama not found. Launching installer...
    if not exist "%TEMP%\OllamaSetup.exe" (
        echo Download from https://ollama.com/download and install, then re-run this file.
        start "" "https://ollama.com/download"
        pause
        exit /b 1
    )
    start /wait "" "%TEMP%\OllamaSetup.exe"
    set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    if not exist "%OLLAMA%" (
        echo Ollama install not detected. Close this window, open a NEW terminal, run again.
        pause
        exit /b 1
    )
)

echo Using Ollama: %OLLAMA%
echo Pulling qwen2.5:7b ^(~4.7 GB, may take a while^)...
"%OLLAMA%" pull qwen2.5:7b

echo.
echo Installed models:
"%OLLAMA%" list

echo.
echo Done! Set in .env:
echo   LLM_PROVIDER=ollama
echo   LLM_MODEL=qwen2.5:7b
pause
