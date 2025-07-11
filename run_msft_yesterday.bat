@echo off
echo Setting environment variables...
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
set FINANCIALDATASETS_API_KEY=your_financialdatasets_api_key_here
set OPENAI_API_KEY=your_openai_api_key_here
set PYTHONIOENCODING=utf-8

echo Running MSFT analysis for 2025-07-10...
python run_universal_analysis.py MSFT 2025-07-10

echo Analysis complete!
pause 