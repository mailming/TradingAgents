@echo off
setlocal

REM =====================================================
REM Daily Trading Analysis Runner
REM Runs analysis for MSFT, TSLA, AAPL, and NVDA
REM =====================================================

echo [%date% %time%] Starting daily trading analysis...

REM Change to the TradingAgents directory (adjust path as needed)
cd /d "C:\Users\USER\Workspace\TradingAgents"

REM Run the PowerShell script
powershell.exe -ExecutionPolicy Bypass -File ".\run_daily_analysis_simple.ps1"

set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==0 (
    echo [%date% %time%] Daily analysis completed successfully
) else (
    echo [%date% %time%] Daily analysis failed with exit code %EXIT_CODE%
)

REM Log the completion
echo [%date% %time%] Exit Code: %EXIT_CODE% >> logs\daily_analysis_batch.log

exit /b %EXIT_CODE% 