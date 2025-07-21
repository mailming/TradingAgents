param(
    [Parameter(Mandatory=$true)]
    [string]$Ticker
)

Write-Host ""
Write-Host "===================================="
Write-Host "   Today's Analysis Runner"
Write-Host "===================================="
Write-Host ""

# Get today's date in YYYY-MM-DD format
$TodayDate = Get-Date -Format "yyyy-MM-dd"

Write-Host "Running TODAY analysis for: $Ticker"
Write-Host "Date: $TodayDate"
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\activate.bat")) {
    Write-Host "Error: Virtual environment not found"
    Write-Host "Please ensure you are in the TradingAgents directory"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if main.py exists  
if (-not (Test-Path "main.py")) {
    Write-Host "Error: main.py not found"
    Write-Host "Please ensure you are in the TradingAgents directory"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Activating virtual environment..."

# Run the analysis directly with python
& python main.py $Ticker $TodayDate

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Analysis completed successfully!"
} else {
    Write-Host ""
    Write-Host "Analysis failed!"
}

Write-Host ""
Read-Host "Press Enter to close" 