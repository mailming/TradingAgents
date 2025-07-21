param(
    [string]$Ticker = "NVDA",
    [int]$Year = 2025,
    [string[]]$Months = @("05", "06"),  # May and June
    [string]$LogPath = ".\logs"
)

# Create log directory if it doesn't exist
if (-not (Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
}

# Set up logging
$LogFile = Join-Path $LogPath "nvda_historical_analysis_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').log"
$StartTime = Get-Date

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    Write-Host $LogEntry -ForegroundColor $(switch($Level) { "ERROR" {"Red"} "WARN" {"Yellow"} "SUCCESS" {"Green"} default {"White"} })
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

function Get-TradingDaysInMonth {
    param([int]$Year, [string]$Month)
    
    $DaysInMonth = [DateTime]::DaysInMonth($Year, [int]$Month)
    $TradingDays = @()
    
    for ($Day = 1; $Day -le $DaysInMonth; $Day++) {
        $Date = Get-Date -Year $Year -Month [int]$Month -Day $Day
        $DayOfWeek = $Date.DayOfWeek
        
        # Skip weekends (Saturday = 6, Sunday = 0)
        if ($DayOfWeek -ne "Saturday" -and $DayOfWeek -ne "Sunday") {
            $DateString = $Date.ToString("yyyy-MM-dd")
            $TradingDays += $DateString
        }
    }
    
    return $TradingDays
}

function Clear-NVDACache {
    Write-Log "Clearing NVDA cache to ensure fresh data..." "INFO"
    
    try {
        $ClearResult = python -c "
from tradingagents.dataflows.interface import clear_cache_data
result = clear_cache_data(symbol='NVDA')
print(result)
"
        Write-Log "NVDA cache cleared: $ClearResult" "SUCCESS"
        return $true
    }
    catch {
        Write-Log "Failed to clear NVDA cache: $($_.Exception.Message)" "WARN"
        return $false
    }
}

function Run-HistoricalAnalysis {
    param([string]$AnalysisDate)
    
    Write-Log "Starting analysis for $Ticker on $AnalysisDate..." "INFO"
    $AnalysisStartTime = Get-Date
    
    try {
        # Run the analysis
        $Process = Start-Process -FilePath "python" -ArgumentList "main.py", $Ticker, $AnalysisDate -Wait -PassThru -NoNewWindow
        $AnalysisEndTime = Get-Date
        $Duration = ($AnalysisEndTime - $AnalysisStartTime).TotalSeconds
        
        if ($Process.ExitCode -eq 0) {
            Write-Log "✅ $Ticker analysis completed successfully for $AnalysisDate in $([math]::Round($Duration, 1))s" "SUCCESS"
            return $true
        } else {
            Write-Log "❌ $Ticker analysis failed for $AnalysisDate with exit code $($Process.ExitCode)" "ERROR"
            return $false
        }
    }
    catch {
        Write-Log "❌ $Ticker analysis failed for $AnalysisDate with exception: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Main execution
Write-Log "========================================" "INFO"
Write-Log "NVDA Historical Analysis Started" "INFO"
Write-Log "Ticker: $Ticker" "INFO"
Write-Log "Year: $Year" "INFO"
Write-Log "Months: $($Months -join ', ')" "INFO"
Write-Log "========================================" "INFO"

# Check prerequisites
if (-not (Test-Prerequisites)) {
    Write-Log "Prerequisites check failed. Exiting." "ERROR"
    exit 1
}

# Clear NVDA cache
Clear-NVDACache

# Track results
$AllResults = @{}
$TotalSuccessCount = 0
$TotalFailCount = 0
$TotalSkippedCount = 0

# Process each month
foreach ($Month in $Months) {
    $MonthName = (Get-Date -Month [int]$Month -Day 1).ToString("MMMM")
    Write-Log "========================================" "INFO"
    Write-Log "Processing $MonthName $Year" "INFO"
    Write-Log "========================================" "INFO"
    
    # Get all trading days in this month
    $TradingDays = Get-TradingDaysInMonth -Year $Year -Month $Month
    Write-Log "Found $($TradingDays.Count) trading days in $MonthName $Year" "INFO"
    
    $MonthSuccessCount = 0
    $MonthFailCount = 0
    
    # Process each trading day
    for ($i = 0; $i -lt $TradingDays.Count; $i++) {
        $AnalysisDate = $TradingDays[$i]
        $Success = Run-HistoricalAnalysis -AnalysisDate $AnalysisDate
        $AllResults[$AnalysisDate] = $Success
        
        if ($Success) {
            $MonthSuccessCount++
            $TotalSuccessCount++
        } else {
            $MonthFailCount++
            $TotalFailCount++
        }
        
        # Add delay between analyses to avoid rate limiting (except for last day of month)
        if ($i -lt ($TradingDays.Count - 1)) {
            Write-Log "Waiting 15 seconds before next analysis..." "INFO"
            Start-Sleep -Seconds 15
        }
    }
    
    # Month summary
    Write-Log "Month Summary for $MonthName ${Year}:" "INFO"
    Write-Log "  Successful: $MonthSuccessCount" "INFO"
    Write-Log "  Failed: $MonthFailCount" "INFO"
    $TradingDayCount = $TradingDays.Count
    Write-Log "  Trading Days Processed: $TradingDayCount" "INFO"
}

# Final Summary
$EndTime = Get-Date
$TotalDuration = ($EndTime - $StartTime).TotalMinutes

Write-Log "========================================" "INFO"
Write-Log "HISTORICAL ANALYSIS FINAL SUMMARY" "INFO"
Write-Log "========================================" "INFO"
Write-Log "Ticker: $Ticker" "INFO"
$MonthsList = $Months -join ', '
Write-Log "Period: $MonthsList/${Year}" "INFO"
$RoundedDuration = [math]::Round($TotalDuration, 1)
Write-Log "Total Duration: $RoundedDuration minutes" "INFO"
Write-Log "Total Successful: $TotalSuccessCount" "INFO"
Write-Log "Total Failed: $TotalFailCount" "INFO"
$TotalAnalyses = $TotalSuccessCount + $TotalFailCount
if ($TotalAnalyses -gt 0) {
    $SuccessRate = [math]::Round(($TotalSuccessCount / $TotalAnalyses * 100), 1)
    Write-Log "Success Rate: $SuccessRate%" "INFO"
} else {
    Write-Log "Success Rate: N/A" "INFO"
}

# Detailed results by month
foreach ($Month in $Months) {
    $MonthName = (Get-Date -Month [int]$Month -Day 1).ToString("MMMM")
    Write-Log "Results for $MonthName ${Year}:" "INFO"
    
    $MonthDays = Get-TradingDaysInMonth -Year $Year -Month $Month
    foreach ($Day in $MonthDays) {
        if ($AllResults[$Day]) {
            $Status = "✅ SUCCESS"
        } else {
            $Status = "❌ FAILED"
        }
        Write-Log "  ${Day}: $Status" "INFO"
    }
}

Write-Log "Log file: $LogFile" "INFO"
Write-Log "Analysis results saved to: C:\Users\USER\Workspace\zzsheepTrader\analysis_results\json\" "INFO"
Write-Log "========================================" "INFO"

# Exit with appropriate code
if ($TotalFailCount -eq 0) {
    Write-Log "All historical analyses completed successfully! 🎉" "SUCCESS"
    exit 0
} else {
    Write-Log "$TotalFailCount analyses failed. Check logs for details." "ERROR"
    exit 1
} 