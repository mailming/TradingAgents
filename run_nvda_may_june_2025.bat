@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo NVDA Historical Analysis - May & June 2025
echo ========================================
echo.

REM Clear NVDA cache first
echo Clearing NVDA cache...
python -c "from tradingagents.dataflows.interface import clear_cache_data; result = clear_cache_data(symbol='NVDA'); print(f'Cache cleared: {result}')"
echo.

REM Create logs directory
if not exist "logs" mkdir logs

REM Set log file
set LOGFILE=logs\nvda_historical_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%.log
echo Log file: %LOGFILE%
echo.

REM May 2025 trading days (excluding weekends)
echo ========================================
echo Processing May 2025
echo ========================================

set MAY_DAYS=2025-05-01 2025-05-02 2025-05-05 2025-05-06 2025-05-07 2025-05-08 2025-05-09 2025-05-12 2025-05-13 2025-05-14 2025-05-15 2025-05-16 2025-05-19 2025-05-20 2025-05-21 2025-05-22 2025-05-23 2025-05-27 2025-05-28 2025-05-29 2025-05-30

for %%d in (%MAY_DAYS%) do (
    echo.
    echo [%TIME%] Starting NVDA analysis for %%d...
    echo [%TIME%] Starting NVDA analysis for %%d... >> "%LOGFILE%"
    
    python main.py NVDA %%d
    if !ERRORLEVEL! EQU 0 (
        echo [%TIME%] SUCCESS: NVDA analysis completed for %%d
        echo [%TIME%] SUCCESS: NVDA analysis completed for %%d >> "%LOGFILE%"
    ) else (
        echo [%TIME%] FAILED: NVDA analysis failed for %%d
        echo [%TIME%] FAILED: NVDA analysis failed for %%d >> "%LOGFILE%"
    )
    
    echo Waiting 15 seconds before next analysis...
    timeout /t 15 /nobreak >nul
)

echo.
echo ========================================
echo Processing June 2025  
echo ========================================

set JUNE_DAYS=2025-06-02 2025-06-03 2025-06-04 2025-06-05 2025-06-06 2025-06-09 2025-06-10 2025-06-11 2025-06-12 2025-06-13 2025-06-16 2025-06-17 2025-06-18 2025-06-19 2025-06-20 2025-06-23 2025-06-24 2025-06-25 2025-06-26 2025-06-27 2025-06-30

for %%d in (%JUNE_DAYS%) do (
    echo.
    echo [%TIME%] Starting NVDA analysis for %%d...
    echo [%TIME%] Starting NVDA analysis for %%d... >> "%LOGFILE%"
    
    python main.py NVDA %%d
    if !ERRORLEVEL! EQU 0 (
        echo [%TIME%] SUCCESS: NVDA analysis completed for %%d
        echo [%TIME%] SUCCESS: NVDA analysis completed for %%d >> "%LOGFILE%"
    ) else (
        echo [%TIME%] FAILED: NVDA analysis failed for %%d  
        echo [%TIME%] FAILED: NVDA analysis failed for %%d >> "%LOGFILE%"
    )
    
    echo Waiting 15 seconds before next analysis...
    timeout /t 15 /nobreak >nul
)

echo.
echo ========================================
echo NVDA Historical Analysis Complete
echo ========================================
echo Results saved to: C:\Users\USER\Workspace\zzsheepTrader\analysis_results\json\
echo Log file: %LOGFILE%
echo.
pause 