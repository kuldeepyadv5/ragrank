# Install Ollama (if needed) and download qwen2.5:7b
# Run in PowerShell: .\scripts\setup_ollama_qwen.ps1

$ErrorActionPreference = "Stop"

function Get-OllamaExe {
    $paths = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "C:\Program Files\Ollama\ollama.exe"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$ollama = Get-OllamaExe

if (-not $ollama) {
    Write-Host "Ollama not found. Downloading installer..."
    $installer = "$env:TEMP\OllamaSetup.exe"
    if (-not (Test-Path $installer)) {
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer
    }
    Write-Host "Launching Ollama installer — complete the setup, then re-run this script."
    Start-Process $installer -Wait
    $ollama = Get-OllamaExe
    if (-not $ollama) {
        Write-Error "Ollama still not found. Restart your terminal after install and run this script again."
    }
}

Write-Host "Using Ollama at: $ollama"
Write-Host "Pulling qwen2.5:7b (about 4.7 GB, may take several minutes)..."
& $ollama pull qwen2.5:7b

Write-Host "`nInstalled models:"
& $ollama list

Write-Host "`nDone! Update your .env:"
Write-Host "  LLM_PROVIDER=ollama"
Write-Host "  LLM_MODEL=qwen2.5:7b"
