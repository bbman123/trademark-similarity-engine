# PowerShell script to start the Trademark Similarity API

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Trademark Similarity Engine API" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path ".\.venv")) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create it first: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Check if models exist
if (-Not (Test-Path ".\models\cnn_encoder.keras")) {
    Write-Host ""
    Write-Host "Warning: Trained models not found!" -ForegroundColor Red
    Write-Host "Please train the models first by running the notebook:" -ForegroundColor Yellow
    Write-Host "  Non_hybrid.ipynb (Cell 4)" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# Start API server
Write-Host ""
Write-Host "Starting API server..." -ForegroundColor Green
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python api_server.py
