@echo off
echo Setting up environment variables for TradingAgents analysis...
echo.
echo Please ensure you have set the following environment variables:
echo - ANTHROPIC_API_KEY: Your Anthropic Claude API key
echo - FINANCIALDATASETS_API_KEY: Your financialdatasets.ai API key
echo.
echo You can set them by running:
echo set ANTHROPIC_API_KEY=your_anthropic_key_here
echo set FINANCIALDATASETS_API_KEY=your_financialdatasets_key_here
echo.
echo Or create a local_env.bat file with your keys and run it before this script.
echo.
if "%ANTHROPIC_API_KEY%"=="" (
    echo ERROR: ANTHROPIC_API_KEY environment variable is not set!
    echo Please set your Anthropic API key first.
    pause
    exit /b 1
)
if "%FINANCIALDATASETS_API_KEY%"=="" (
    echo ERROR: FINANCIALDATASETS_API_KEY environment variable is not set!
    echo Please set your financialdatasets.ai API key first.
    pause
    exit /b 1
)
echo ✅ Environment variables are set. Starting analysis...
python run_universal_analysis.py MSFT
pause 