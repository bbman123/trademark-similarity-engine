# PowerShell script to run the API server

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Trademark Similarity Engine - API Server" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment active: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠ Virtual environment not active. Activating..." -ForegroundColor Yellow
    & ".\.venv\Scripts\Activate.ps1"
}

Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""

# Run uvicorn
python -m uvicorn src.api_service:app --reload --host 0.0.0.0 --port 8000
