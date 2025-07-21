@echo off
setlocal

echo ========================================
echo Daily Trading Analysis - All Tickers
echo ========================================
echo Starting analysis for MSFT, TSLA, AAPL, NVDA...
echo Time: %date% %time%
echo.

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Log start time
echo [%date% %time%] Starting daily analysis for all tickers >> logs\daily_analysis_batch.log

REM Counter for success/failure tracking
set SUCCESS_COUNT=0
set FAIL_COUNT=0

REM Run MSFT Analysis
echo [1/4] Analyzing MSFT...
powershell -ExecutionPolicy Bypass -File ".\run_today_simple.ps1" MSFT
if %ERRORLEVEL%==0 (
    echo ✅ MSFT analysis completed successfully
    set /a SUCCESS_COUNT+=1
) else (
    echo ❌ MSFT analysis failed
    set /a FAIL_COUNT+=1
)
echo [%date% %time%] MSFT analysis - Exit Code: %ERRORLEVEL% >> logs\daily_analysis_batch.log

echo Waiting 10 seconds before next analysis...
timeout /t 10 /nobreak > nul

REM Run TSLA Analysis  
echo [2/4] Analyzing TSLA...
powershell -ExecutionPolicy Bypass -File ".\run_today_simple.ps1" TSLA
if %ERRORLEVEL%==0 (
    echo ✅ TSLA analysis completed successfully
    set /a SUCCESS_COUNT+=1
) else (
    echo ❌ TSLA analysis failed
    set /a FAIL_COUNT+=1
)
echo [%date% %time%] TSLA analysis - Exit Code: %ERRORLEVEL% >> logs\daily_analysis_batch.log

echo Waiting 10 seconds before next analysis...
timeout /t 10 /nobreak > nul

REM Run AAPL Analysis
echo [3/4] Analyzing AAPL...
powershell -ExecutionPolicy Bypass -File ".\run_today_simple.ps1" AAPL
if %ERRORLEVEL%==0 (
    echo ✅ AAPL analysis completed successfully
    set /a SUCCESS_COUNT+=1
) else (
    echo ❌ AAPL analysis failed
    set /a FAIL_COUNT+=1
)
echo [%date% %time%] AAPL analysis - Exit Code: %ERRORLEVEL% >> logs\daily_analysis_batch.log

echo Waiting 10 seconds before next analysis...
timeout /t 10 /nobreak > nul

REM Run NVDA Analysis
echo [4/4] Analyzing NVDA...
powershell -ExecutionPolicy Bypass -File ".\run_today_simple.ps1" NVDA
if %ERRORLEVEL%==0 (
    echo ✅ NVDA analysis completed successfully
    set /a SUCCESS_COUNT+=1
) else (
    echo ❌ NVDA analysis failed
    set /a FAIL_COUNT+=1
)
echo [%date% %time%] NVDA analysis - Exit Code: %ERRORLEVEL% >> logs\daily_analysis_batch.log

REM Summary
echo.
echo ========================================
echo Daily Analysis Summary
echo ========================================
echo Successful: %SUCCESS_COUNT%
echo Failed: %FAIL_COUNT%
echo Total: 4 tickers (MSFT, TSLA, AAPL, NVDA)
echo Completed: %date% %time%
echo.

REM Log summary
echo [%date% %time%] Daily analysis completed - Success: %SUCCESS_COUNT%, Failed: %FAIL_COUNT% >> logs\daily_analysis_batch.log

REM Exit with appropriate code
if %FAIL_COUNT%==0 (
    echo 🎉 All analyses completed successfully!
    echo Results exported to zzsheepTrader frontend
    exit /b 0
) else (
    echo ⚠️ %FAIL_COUNT% analyses failed. Check logs for details.
    exit /b 1
)

pause 