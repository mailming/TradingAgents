#!/usr/bin/env python3
import json
import os

json_file = "C:/Users/USER/Workspace/zzsheepTrader/analysis_results/json/MSFT-2025-07-10_EOD.json"

try:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("📊 Full Report Length Analysis:")
    print("=" * 50)
    
    # Check market analysis report
    market_report_len = len(data['analysis_components']['market_analysis']['full_report'])
    print(f"📈 Market Analysis Report Length: {market_report_len:,} characters")
    
    # Check news analysis report  
    news_report_len = len(data['analysis_components']['news_analysis']['full_report'])
    print(f"📰 News Analysis Report Length: {news_report_len:,} characters")
    
    # Check fundamental analysis report
    fundamental_report_len = len(data['analysis_components']['fundamental_analysis']['full_report'])
    print(f"💼 Fundamental Analysis Report Length: {fundamental_report_len:,} characters")
    
    print()
    print("📋 Verification:")
    print(f"   ✅ Market Report > 500 chars: {market_report_len > 500}")
    print(f"   ✅ News Report > 500 chars: {news_report_len > 500}")
    print(f"   ✅ Fundamental Report > 500 chars: {fundamental_report_len > 500}")
    
    # Show first 200 characters of each report
    print()
    print("📄 Sample Content (first 200 characters):")
    print("=" * 50)
    print("📈 Market Analysis:")
    print(data['analysis_components']['market_analysis']['full_report'][:200] + "...")
    print()
    print("📰 News Analysis:")
    print(data['analysis_components']['news_analysis']['full_report'][:200] + "...")
    print()
    print("💼 Fundamental Analysis:")
    print(data['analysis_components']['fundamental_analysis']['full_report'][:200] + "...")
    
except Exception as e:
    print(f"❌ Error reading JSON file: {e}") 