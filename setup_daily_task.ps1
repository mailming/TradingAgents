# Setup Daily Trading Analysis Task
# This script creates a Windows Scheduled Task to run daily analysis at 3 PM

param(
    [string]$TaskName = "Daily Trading Analysis",
    [string]$TradingAgentsPath = "C:\Users\USER\Workspace\TradingAgents",
    [string]$RunTime = "15:00",  # 3:00 PM in 24-hour format
    [string]$UserName = $env:USERNAME
)

Write-Host "Setting up Daily Trading Analysis Scheduled Task..." -ForegroundColor Green
Write-Host "Task Name: $TaskName" -ForegroundColor Cyan
Write-Host "Path: $TradingAgentsPath" -ForegroundColor Cyan
Write-Host "Run Time: $RunTime daily" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "⚠️  WARNING: Not running as administrator. Some features may not work properly." -ForegroundColor Yellow
    Write-Host "   For best results, run this script as administrator." -ForegroundColor Yellow
    Write-Host ""
}

# Check if TradingAgents directory exists
if (-not (Test-Path $TradingAgentsPath)) {
    Write-Host "❌ Error: TradingAgents directory not found at: $TradingAgentsPath" -ForegroundColor Red
    Write-Host "   Please update the TradingAgentsPath parameter or create the directory." -ForegroundColor Red
    exit 1
}

# Check if required files exist
$RequiredFiles = @(
    "$TradingAgentsPath\run_daily_analysis.bat",
    "$TradingAgentsPath\run_daily_analysis_simple.ps1",
    "$TradingAgentsPath\main.py"
)

foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File)) {
        Write-Host "❌ Error: Required file not found: $File" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ All required files found" -ForegroundColor Green

try {
    # Remove existing task if it exists
    $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($ExistingTask) {
        Write-Host "🔄 Removing existing task: $TaskName" -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    # Create the action (what to run)
    $Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"cd /d `"$TradingAgentsPath`" && run_daily_analysis.bat`""

    # Create the trigger (when to run - daily at 3 PM)
    $Trigger = New-ScheduledTaskTrigger -Daily -At $RunTime

    # Create the principal (run as current user)
    $Principal = New-ScheduledTaskPrincipal -UserId $UserName -LogonType Interactive

    # Create task settings
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable

    # Register the scheduled task
    Write-Host "📅 Creating scheduled task..." -ForegroundColor Blue
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "Daily trading analysis for MSFT, TSLA, AAPL, and NVDA at 3 PM"

    Write-Host ""
    Write-Host "✅ SUCCESS: Daily Trading Analysis task created!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Task Details:" -ForegroundColor Cyan
    Write-Host "   Name: $TaskName"
    Write-Host "   Schedule: Daily at $RunTime"
    Write-Host "   Command: $TradingAgentsPath\run_daily_analysis.bat"
    Write-Host "   User: $UserName"
    Write-Host ""
    Write-Host "🎯 The task will analyze: MSFT, TSLA, AAPL, NVDA" -ForegroundColor Yellow
    Write-Host "📁 Logs will be saved to: $TradingAgentsPath\logs\" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔧 Management Commands:" -ForegroundColor Cyan
    Write-Host "   View task:    Get-ScheduledTask -TaskName '$TaskName'"
    Write-Host "   Run now:      Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "   Disable:      Disable-ScheduledTask -TaskName '$TaskName'"
    Write-Host "   Remove:       Unregister-ScheduledTask -TaskName '$TaskName'"
    Write-Host ""
    
    # Ask if user wants to test the task
    $TestNow = Read-Host "Would you like to test the task now? (y/n)"
    if ($TestNow -eq 'y' -or $TestNow -eq 'Y') {
        Write-Host "🚀 Starting test run..." -ForegroundColor Blue
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "✅ Task started. Check the logs folder for results." -ForegroundColor Green
    }

} catch {
    Write-Host "❌ Error creating scheduled task: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   1. Run PowerShell as Administrator"
    Write-Host "   2. Check if Task Scheduler service is running"
    Write-Host "   3. Verify the path: $TradingAgentsPath"
    Write-Host "   4. Ensure all required files exist"
    exit 1
}

Write-Host "🎉 Setup complete! Your daily trading analysis will run automatically at 3 PM." -ForegroundColor Green 