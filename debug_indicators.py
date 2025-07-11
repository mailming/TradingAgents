#!/usr/bin/env python3
"""
Debug technical indicators calculation for AAPL
"""

from tradingagents.agents.utils.market_data_utils import (
    _calculate_technical_indicators_from_data,
    extract_market_data_from_reports
)

print('🔍 Testing AAPL technical indicators - Step by step debugging...')

# Step 1: Test direct calculation
print('\n=== STEP 1: Direct calculation ===')
direct_result = _calculate_technical_indicators_from_data('AAPL', '2025-07-11')
print('📊 Direct calculation result:')
for key, value in direct_result.items():
    print(f'  {key}: {value}')

print(f'\n🎯 SMA-20 from direct calculation: {direct_result.get("sma_20", "N/A")}')

# Step 2: Test through extract_market_data_from_reports
print('\n=== STEP 2: Full extract_market_data_from_reports ===')

# Create a more realistic mock final_state
test_final_state = {
    "market_report": "Apple stock trading at $211.16 with strong momentum indicators",
    "fundamentals_report": "Apple shows strong financial health with growing revenue",
    "company_of_interest": "AAPL",
    "news_report": "Positive sentiment around Apple's AI initiatives",
    "sentiment_report": "Overall bullish sentiment with investor confidence"
}

market_data = extract_market_data_from_reports(test_final_state, "AAPL", "2025-07-11")

print('📊 Market data full result:')
print(f'  current_price: {market_data.get("current_price", "N/A")}')
print(f'  volume: {market_data.get("volume", "N/A")}')
print(f'  volatility: {market_data.get("volatility", "N/A")}')

print('\n📊 Market data technical indicators:')
for key, value in market_data.get('technical_indicators', {}).items():
    print(f'  {key}: {value}')

print(f'\n🎯 SMA-20 from extract_market_data: {market_data.get("technical_indicators", {}).get("sma_20", "N/A")}')
print(f'🎯 SMA-50 from extract_market_data: {market_data.get("technical_indicators", {}).get("sma_50", "N/A")}')

# Step 3: Test specific scenario from the JSON
print('\n=== STEP 3: Testing JSON scenario ===')

# Check if there's any difference between the values
direct_sma_20 = direct_result.get("sma_20", "N/A")
extract_sma_20 = market_data.get("technical_indicators", {}).get("sma_20", "N/A")

print(f'Direct SMA-20: {direct_sma_20}')
print(f'Extract SMA-20: {extract_sma_20}')
print(f'Are they equal? {direct_sma_20 == extract_sma_20}')

if direct_sma_20 != extract_sma_20:
    print('❌ BUG FOUND: Values are different!')
    print(f'   Direct: {direct_sma_20}')
    print(f'   Extract: {extract_sma_20}')
else:
    print('✅ Values match - bug must be elsewhere') 