#!/bin/zsh

# Quick script to run analysis for TODAY
# Usage: ./run_today_analysis.sh TICKER

if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide a ticker symbol"
    echo "Usage: ./run_today_analysis.sh NVDA"
    exit 1
fi

TICKER=$1
TODAY=$(date +%Y-%m-%d)

echo "🚀 Running analysis for $TICKER on $TODAY..."

# Activate virtual environment and run analysis
source venv/bin/activate && python main.py $TICKER $TODAY

echo "✅ Analysis complete for $TICKER on $TODAY" 