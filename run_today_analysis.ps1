param(
    [Parameter(Mandatory=$true)]
    [string]$Ticker
)

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "   Today's Analysis Runner" -ForegroundColor Green  
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

# Get today's date in YYYY-MM-DD format
$TodayDate = Get-Date -Format "yyyy-MM-dd"

Write-Host "📊 Running TODAY'S analysis for: $Ticker" -ForegroundColor Cyan
Write-Host "📅 Date: $TodayDate" -ForegroundColor Yellow
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\activate.bat")) {
    Write-Host "❌ Error: Virtual environment not found at 'venv\Scripts\activate.bat'" -ForegroundColor Red
    Write-Host "Please ensure you are running this from the TradingAgents directory" -ForegroundColor Red
    Write-Host "and that the virtual environment is set up." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if main.py exists  
if (-not (Test-Path "main.py")) {
    Write-Host "❌ Error: main.py not found in current directory" -ForegroundColor Red
    Write-Host "Please ensure you are running this from the TradingAgents directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🔄 Activating virtual environment..." -ForegroundColor Blue

# Activate virtual environment and run analysis
& "venv\Scripts\activate.bat"
& python main.py $Ticker $TodayDate

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Analysis completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Analysis failed!" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to close" 