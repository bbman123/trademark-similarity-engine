#!/usr/bin/env pwsh
# Trademark Similarity Engine - Setup Verification Script
# This script verifies that all required files are present and the environment is ready

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    Trademark Similarity Engine - Setup Verification         ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$errors = 0
$warnings = 0

# Function to check file existence
function Test-FileExists {
    param($Path, $Description)
    if (Test-Path $Path) {
        Write-Host "✓" -ForegroundColor Green -NoNewline
        Write-Host " $Description" -ForegroundColor White
        return $true
    } else {
        Write-Host "✗" -ForegroundColor Red -NoNewline
        Write-Host " $Description - MISSING!" -ForegroundColor Red
        $script:errors++
        return $false
    }
}

# Function to check Python version
function Test-PythonVersion {
    try {
        $version = python --version 2>&1
        if ($version -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 11) {
                Write-Host "✓" -ForegroundColor Green -NoNewline
                Write-Host " Python $major.$minor (>= 3.11)" -ForegroundColor White
                return $true
            }
        }
        Write-Host "✗" -ForegroundColor Red -NoNewline
        Write-Host " Python version too old. Need 3.11+" -ForegroundColor Red
        $script:errors++
        return $false
    } catch {
        Write-Host "✗" -ForegroundColor Red -NoNewline
        Write-Host " Python not found!" -ForegroundColor Red
        $script:errors++
        return $false
    }
}

# Function to check Node.js version
function Test-NodeVersion {
    try {
        $version = node --version 2>&1
        if ($version -match "v(\d+)") {
            $major = [int]$Matches[1]
            if ($major -ge 18) {
                Write-Host "✓" -ForegroundColor Green -NoNewline
                Write-Host " Node.js v$major (>= 18)" -ForegroundColor White
                return $true
            }
        }
        Write-Host "⚠" -ForegroundColor Yellow -NoNewline
        Write-Host " Node.js version recommended: 18+" -ForegroundColor Yellow
        $script:warnings++
        return $false
    } catch {
        Write-Host "✗" -ForegroundColor Red -NoNewline
        Write-Host " Node.js not found!" -ForegroundColor Red
        $script:errors++
        return $false
    }
}

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "1. Checking Prerequisites" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

Test-PythonVersion
Test-NodeVersion

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "2. Checking Backend Files" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

Test-FileExists "backend\api.py" "API server"
Test-FileExists "backend\requirements.txt" "Python dependencies"
Test-FileExists "backend\start.ps1" "PowerShell startup script"
Test-FileExists "backend\start.bat" "CMD startup script"
Test-FileExists "backend\start.sh" "Linux/Mac startup script"

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "3. Checking Model Files" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

Test-FileExists "backend\models\cnn_encoder.keras" "CNN model (~3MB)"
Test-FileExists "backend\models\cnn_encoder_tokenizer.pkl" "Tokenizer"
Test-FileExists "backend\models\hybrid_svm.pkl" "SVM model (~7MB)"

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "4. Checking Frontend Files" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

Test-FileExists "frontend\package.json" "Node.js configuration"
Test-FileExists "frontend\vite.config.js" "Vite configuration"
Test-FileExists "frontend\tailwind.config.js" "Tailwind configuration"
Test-FileExists "frontend\index.html" "HTML entry point"
Test-FileExists "frontend\src\main.jsx" "React entry point"
Test-FileExists "frontend\src\App.jsx" "Main application"
Test-FileExists "frontend\src\index.css" "Global styles"

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "5. Checking React Components" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

Test-FileExists "frontend\src\components\Header.jsx" "Header component"
Test-FileExists "frontend\src\components\ApiStatus.jsx" "API status component"
Test-FileExists "frontend\src\components\SimilarityChecker.jsx" "Similarity checker"
Test-FileExists "frontend\src\components\BatchChecker.jsx" "Batch checker"
Test-FileExists "frontend\src\components\ResultCard.jsx" "Result card"
Test-FileExists "frontend\src\components\AnalysisDetails.jsx" "Analysis details"

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "6. Checking Documentation" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

Test-FileExists "README.md" "Main documentation"
Test-FileExists "QUICKSTART.md" "Quick start guide"
Test-FileExists "DEPLOYMENT_CHECKLIST.md" "Deployment checklist"
Test-FileExists "PACKAGE_SUMMARY.md" "Package summary"

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "Summary" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "✓ All checks passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host " Ready to Start!" -ForegroundColor Green -NoNewline
    Write-Host " Follow these steps:" -ForegroundColor White
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Terminal 1 (Backend):" -ForegroundColor Cyan
    Write-Host "  cd backend" -ForegroundColor White
    Write-Host "  python -m venv venv" -ForegroundColor White
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "  pip install -r requirements.txt" -ForegroundColor White
    Write-Host "  python api.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Terminal 2 (Frontend):" -ForegroundColor Cyan
    Write-Host "  cd frontend" -ForegroundColor White
    Write-Host "  npm install" -ForegroundColor White
    Write-Host "  npm run dev" -ForegroundColor White
    Write-Host ""
    Write-Host "Then open: " -ForegroundColor White -NoNewline
    Write-Host "http://localhost:3000" -ForegroundColor Cyan
    Write-Host ""
} elseif ($errors -eq 0) {
    Write-Host "⚠ Setup complete with $warnings warning(s)" -ForegroundColor Yellow
    Write-Host "  You can proceed, but some features may not work optimally." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "✗ Setup incomplete - $errors error(s), $warnings warning(s)" -ForegroundColor Red
    Write-Host "  Please resolve the errors above before proceeding." -ForegroundColor Red
    Write-Host ""
}

Write-Host "For detailed instructions, see: README.md" -ForegroundColor Gray
Write-Host ""
