# Daily Trading Analysis Automation Setup

This guide shows how to set up automated daily analysis for **MSFT, TSLA, AAPL, and NVDA** at **3:00 PM** every day using Windows Task Scheduler.

## 📁 Files Created

### Core Scripts
- ✅ `run_today_simple.ps1` - Single ticker analysis (working)
- ✅ `run_daily_analysis.bat` - Batch wrapper for automation
- ✅ `main.py` - Updated with dotenv support
- ✅ `.env` - API keys configuration

### Setup Scripts
- `setup_daily_task.ps1` - Windows Task Scheduler setup
- `DAILY_ANALYSIS_SETUP.md` - This documentation

## 🚀 Quick Setup (Manual Method)

### Step 1: Verify Working Scripts
Test the single ticker analysis first:
```powershell
.\run_today_simple.ps1 TSLA
```

### Step 2: Create Windows Scheduled Task Manually

1. **Open Task Scheduler**
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create Basic Task**
   - Click "Create Basic Task" in the right panel
   - Name: `Daily Trading Analysis`
   - Description: `Daily analysis for MSFT, TSLA, AAPL, NVDA at 3 PM`

3. **Set Trigger**
   - When: `Daily`
   - Start date: Today
   - Time: `3:00:00 PM`
   - Recur every: `1 days`

4. **Set Action**
   - Action: `Start a program`
   - Program/script: `cmd.exe`
   - Arguments: `/c "cd /d "C:\Users\USER\Workspace\TradingAgents" && run_daily_analysis.bat"`
   - Start in: `C:\Users\USER\Workspace\TradingAgents`

5. **Finish Setup**
   - Check "Open the Properties dialog" before clicking Finish
   - In Properties:
     - Security options: "Run whether user is logged on or not"
     - Configure for: "Windows 10"
     - Check "Run with highest privileges"

## 📊 Manual Multi-Ticker Analysis

### Option 1: Run All Tickers in Sequence
```powershell
# Run each ticker with 10-second delays
.\run_today_simple.ps1 MSFT
Start-Sleep 10
.\run_today_simple.ps1 TSLA  
Start-Sleep 10
.\run_today_simple.ps1 AAPL
Start-Sleep 10
.\run_today_simple.ps1 NVDA
```

### Option 2: Create Simple Batch Script
Create `run_all_tickers.bat`:
```batch
@echo off
echo Starting daily analysis for all tickers...
powershell -ExecutionPolicy Bypass -File ".\run_today_simple.ps1" MSFT
timeout /t 10 /nobreak
powershell -ExecutionPolicy Bypass -File ".\run_today_simple.ps1" TSLA
timeout /t 10 /nobreak  
powershell -ExecutionPolicy Bypass -File ".\run_today_simple.ps1" AAPL
timeout /t 10 /nobreak
powershell -ExecutionPolicy Bypass -File ".\run_today_simple.ps1" NVDA
echo All analyses complete!
pause
```

## 🗂️ Directory Structure

```
TradingAgents/
├── main.py                           # ✅ Updated with dotenv
├── .env                              # ✅ API keys
├── run_today_simple.ps1              # ✅ Single ticker analysis
├── run_daily_analysis.bat            # ✅ Batch wrapper
├── run_all_tickers.bat               # 📝 Create manually
├── setup_daily_task.ps1              # 🔧 Task scheduler setup
├── logs/                             # 📁 Analysis logs
│   ├── daily_analysis_*.log          # PowerShell logs
│   └── daily_analysis_batch.log      # Batch logs
└── venv/                             # Python environment
```

## 📋 Task Management Commands

### PowerShell Commands
```powershell
# View the scheduled task
Get-ScheduledTask -TaskName "Daily Trading Analysis"

# Run task immediately (for testing)
Start-ScheduledTask -TaskName "Daily Trading Analysis"

# Check task history
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-TaskScheduler/Operational'; ID=201}

# Disable task
Disable-ScheduledTask -TaskName "Daily Trading Analysis"

# Remove task
Unregister-ScheduledTask -TaskName "Daily Trading Analysis" -Confirm:$false
```

## 📊 Expected Results

### Analysis Output for Each Ticker
- **🎯 Decision**: BUY/SELL/HOLD
- **💰 Current Price**: Real-time market price
- **📊 Volume**: Trading volume
- **📉 Volatility**: Calculated volatility percentage  
- **🎯 Trend**: Bullish/Bearish/Neutral
- **📰 Sentiment**: Positive/Negative/Neutral
- **⏱️ Duration**: Analysis completion time

### Export Locations
- **JSON Files**: `C:\Users\USER\Workspace\zzsheepTrader\analysis_results\json\`
- **Log Files**: `C:\Users\USER\Workspace\TradingAgents\logs\`

## 🔧 Troubleshooting

### Common Issues

1. **"ANTHROPIC_API_KEY not found"**
   - Verify `.env` file exists and contains valid API keys
   - Check file encoding (should be ASCII/UTF-8)

2. **"Virtual environment not found"**
   - Ensure `venv\Scripts\activate.bat` exists
   - Recreate virtual environment if needed

3. **Task doesn't run automatically**
   - Check Task Scheduler service is running
   - Verify user permissions
   - Check "Run whether user is logged on or not" option

4. **PowerShell execution policy errors**
   - Run as Administrator: `Set-ExecutionPolicy RemoteSigned`
   - Or use bypass flag: `-ExecutionPolicy Bypass`

### Log Locations
- **PowerShell Logs**: `.\logs\daily_analysis_*.log`
- **Batch Logs**: `.\logs\daily_analysis_batch.log`
- **Windows Task Logs**: Task Scheduler > Task History

## 🎯 Testing the Setup

### Test Individual Ticker
```powershell
.\run_today_simple.ps1 TSLA
```

### Test Batch Wrapper
```batch
run_daily_analysis.bat
```

### Test Scheduled Task
```powershell
Start-ScheduledTask -TaskName "Daily Trading Analysis"
```

## 📈 Expected Timeline

For all 4 tickers (MSFT, TSLA, AAPL, NVDA):
- **Analysis Time**: ~2-3 minutes per ticker
- **Total Duration**: ~10-15 minutes (including delays)
- **Rate Limiting**: 10-second delays between tickers
- **Completion**: Analysis + export + logging

## 🎉 Success Indicators

✅ **Task Created**: Shows in Task Scheduler  
✅ **Logs Generated**: Files appear in `logs/` directory  
✅ **Analysis Complete**: JSON files exported to zzsheepTrader  
✅ **No Errors**: Exit code 0 in logs  
✅ **All Tickers**: MSFT, TSLA, AAPL, NVDA processed  

Your automated daily trading analysis system is now ready! 🚀 