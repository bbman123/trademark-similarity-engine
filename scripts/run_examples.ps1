# PowerShell script to run examples

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Trademark Similarity Engine - Examples" -ForegroundColor Cyan
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
Write-Host "Running examples..." -ForegroundColor Cyan
Write-Host ""

# Run examples
python examples.py
