param(
    [string[]]$Tickers = @("MSFT", "TSLA", "AAPL", "NVDA"),
    [string]$LogPath = ".\logs"
)

# Create log directory if it doesn't exist
if (-not (Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
}

# Set up logging
$LogFile = Join-Path $LogPath "daily_analysis_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').log"
$TodayDate = Get-Date -Format "yyyy-MM-dd"
$StartTime = Get-Date

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    Write-Host $LogEntry
    Add-Content -Path $LogFile -Value $LogEntry
}

function Test-Prerequisites {
    Write-Log "Checking prerequisites..." "INFO"
    
    # Check if in correct directory
    if (-not (Test-Path "main.py")) {
        Write-Log "main.py not found. Please run from TradingAgents directory" "ERROR"
        return $false
    }
    
    # Check virtual environment
    if (-not (Test-Path "venv\Scripts\activate.bat")) {
        Write-Log "Virtual environment not found at 'venv\Scripts\activate.bat'" "ERROR"
        return $false
    }
    
    # Check .env file
    if (-not (Test-Path ".env")) {
        Write-Log ".env file not found. API keys may not be available" "WARN"
    }
    
    Write-Log "Prerequisites check completed" "INFO"
    return $true
}

function Run-TickerAnalysis {
    param([string]$Ticker)
    
    Write-Log "Starting analysis for $Ticker..." "INFO"
    $TickerStartTime = Get-Date
    
    try {
        # Run the analysis
        $Process = Start-Process -FilePath "python" -ArgumentList "main.py", $Ticker, $TodayDate -Wait -PassThru -NoNewWindow
        $TickerEndTime = Get-Date
        $Duration = ($TickerEndTime - $TickerStartTime).TotalSeconds
        
        if ($Process.ExitCode -eq 0) {
            Write-Log "✅ $Ticker analysis completed successfully in $([math]::Round($Duration, 1))s" "SUCCESS"
            return $true
        } else {
            Write-Log "❌ $Ticker analysis failed with exit code $($Process.ExitCode)" "ERROR"
            return $false
        }
    }
    catch {
        Write-Log "❌ $Ticker analysis failed with exception: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Main execution
Write-Log "========================================" "INFO"
Write-Log "Daily Trading Analysis Started" "INFO"
Write-Log "Date: $TodayDate" "INFO"
Write-Log "Tickers: $($Tickers -join ', ')" "INFO"
Write-Log "========================================" "INFO"

# Check prerequisites
if (-not (Test-Prerequisites)) {
    Write-Log "Prerequisites check failed. Exiting." "ERROR"
    exit 1
}

# Track results
$Results = @{}
$SuccessCount = 0
$FailCount = 0

# Process each ticker
for ($i = 0; $i -lt $Tickers.Count; $i++) {
    $Ticker = $Tickers[$i]
    $Success = Run-TickerAnalysis -Ticker $Ticker
    $Results[$Ticker] = $Success
    
    if ($Success) {
        $SuccessCount++
    } else {
        $FailCount++
    }
    
    # Add delay between analyses to avoid rate limiting (except for last ticker)
    if ($i -lt ($Tickers.Count - 1)) {
        Write-Log "Waiting 10 seconds before next analysis..." "INFO"
        Start-Sleep -Seconds 10
    }
}

# Summary
$EndTime = Get-Date
$TotalDuration = ($EndTime - $StartTime).TotalMinutes

Write-Log "========================================" "INFO"
Write-Log "Daily Analysis Summary" "INFO"
Write-Log "Total Duration: $([math]::Round($TotalDuration, 1)) minutes" "INFO"
Write-Log "Successful: $SuccessCount" "INFO"
Write-Log "Failed: $FailCount" "INFO"
Write-Log "Results:" "INFO"

foreach ($Ticker in $Tickers) {
    $Status = if ($Results[$Ticker]) { "✅ SUCCESS" } else { "❌ FAILED" }
    Write-Log "  $Ticker: $Status" "INFO"
}

Write-Log "Log file: $LogFile" "INFO"
Write-Log "========================================" "INFO"

# Exit with appropriate code
if ($FailCount -eq 0) {
    Write-Log "All analyses completed successfully!" "SUCCESS"
    exit 0
} else {
    Write-Log "$FailCount analyses failed. Check logs for details." "ERROR"
    exit 1
} 