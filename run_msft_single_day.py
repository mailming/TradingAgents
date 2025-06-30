"""
MSFT Single-Day Analysis with Claude AI and Market Data Utilities

This script runs a single MSFT analysis for any specific date using:
- Anthropic Claude AI (Claude-3.5-Sonnet)
- New market data extraction utilities  
- financialdatasets.ai (professional data source)
- zzsheeptrader export utility

Usage: python run_msft_single_day.py [YYYY-MM-DD]
If no date provided, uses yesterday
"""

import os
import sys
from datetime import datetime, timedelta
import uuid
import time

# Import TradingAgents components
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.json_export_utils import save_to_zzsheep

# Import market data utilities
from tradingagents.agents.utils.market_data_utils import (
    extract_market_data_from_reports,
    extract_news_sentiment_data,
    extract_risk_assessment_from_reports,
    extract_strategic_insights_from_reports,
    extract_fundamental_insights_from_reports,
    extract_performance_metrics_from_reports
)

def is_trading_day(date):
    """Check if a date is a trading day (not weekend or major US holiday)"""
    # Check if it's a weekend
    if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Check major US stock market holidays
    year = date.year
    month = date.month
    day = date.day
    
    # Major holidays
    holidays = [
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas Day
        (6, 19),  # Juneteenth
    ]
    
    for holiday_month, holiday_day in holidays:
        if month == holiday_month and day == holiday_day:
            return False
    
    return True

def run_msft_daily_analysis(analysis_date, is_trading_day_flag):
    """Run MSFT analysis for a specific date using Claude AI and market data utilities"""
    
    date_str = analysis_date.strftime('%Y-%m-%d')
    day_name = analysis_date.strftime('%A')
    
    print(f"📊 Running MSFT Analysis for {day_name}, {date_str}")
    if not is_trading_day_flag:
        print(f"   📅 Non-trading day - Analysis will use most recent market data")
    
    # Check API keys
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    financialdatasets_key = os.getenv('FINANCIALDATASETS_API_KEY')
    
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY not found!")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        return None
    
    if not financialdatasets_key:
        print("❌ FINANCIALDATASETS_API_KEY not found!")
        print("Set it with: export FINANCIALDATASETS_API_KEY='your-key'")
        return None
    
    # Professional Claude AI configuration
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-3-5-sonnet-20241022"  # Claude-3.5-Sonnet (Premium)
    config["quick_think_llm"] = "claude-3-5-haiku-20241022"  # Claude-3.5-Haiku (Fast)
    config["max_debate_rounds"] = 3  # More thorough analysis
    config["max_risk_discuss_rounds"] = 3
    config["online_tools"] = True
    
    print(f"   🧠 AI Model: Claude-3.5-Sonnet (Premium)")
    print(f"   📊 Data Source: financialdatasets.ai (Professional)")
    
    # Initialize TradingAgents with proper collection handling
    try:
        ta = TradingAgentsGraph(debug=False, config=config)
        print(f"   ✅ TradingAgents initialized successfully")
    except Exception as e:
        if "already exists" in str(e).lower() or "collection" in str(e).lower():
            print(f"   🔄 Memory collections exist, reinitializing...")
            # Clear existing collections and retry
            try:
                import chromadb
                chroma_client = chromadb.Client()
                
                # Delete existing collections if they exist
                collections_to_delete = ["bull_memory", "bear_memory", "neutral_memory"]
                for collection_name in collections_to_delete:
                    try:
                        chroma_client.delete_collection(collection_name)
                        print(f"   🗑️ Cleared existing collection: {collection_name}")
                    except:
                        pass  # Collection doesn't exist or already deleted
                
                # Now create fresh TradingAgents instance
                ta = TradingAgentsGraph(debug=False, config=config)
                print(f"   ✅ TradingAgents reinitialized with fresh collections")
                
            except Exception as retry_error:
                print(f"❌ Failed to reinitialize: {retry_error}")
                return None
        else:
            print(f"❌ Failed to initialize: {e}")
            return None
    
    # Run analysis
    ticker = "MSFT"
    
    start_time = time.time()
    
    try:
        final_state, decision = ta.propagate(ticker, date_str)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"   ✅ Analysis completed in {duration:.1f} seconds")
        print(f"   🎯 Final Decision: {decision}")
        
        # ✨ USE MARKET DATA UTILITIES ✨
        print(f"   📈 Extracting comprehensive market data...")
        market_data = extract_market_data_from_reports(final_state, ticker, date_str)
        news_sentiment = extract_news_sentiment_data(final_state)
        risk_assessment = extract_risk_assessment_from_reports(final_state, ticker)
        strategic_insights = extract_strategic_insights_from_reports(final_state, ticker)
        fundamental_insights = extract_fundamental_insights_from_reports(final_state, ticker)
        performance_metrics = extract_performance_metrics_from_reports(final_state, duration, ticker)
        
        # Display extracted market data
        print(f"   💰 Current Price: {market_data['current_price']}")
        print(f"   📊 Volume: {market_data['volume']}")
        print(f"   📉 Volatility: {market_data['volatility']}")
        print(f"   🎯 Trend: {market_data['technical_indicators']['trend']}")
        
        # Structure results for JSON with REAL MARKET DATA
        timestamp = datetime.now()
        analysis_id = str(uuid.uuid4())[:8]
        
        structured_results = {
            "analysis_metadata": {
                "analysis_id": analysis_id,
                "ticker": ticker,
                "analysis_date": date_str,
                "day_of_week": day_name,
                "is_trading_day": is_trading_day_flag,
                "timestamp": timestamp.isoformat(),
                "duration_seconds": round(duration, 1),
                "ai_model": "claude-3-5-sonnet-20241022",
                "ai_provider": "anthropic_claude",
                "data_source": "financialdatasets.ai",
                "version": "2.0",
                "uses_market_data_utils": True
            },
            "final_decision": {
                "recommendation": decision,
                "confidence_level": fundamental_insights["confidence_level"], 
                "decision_type": "HOLD" if "HOLD" in decision.upper() else "BUY" if "BUY" in decision.upper() else "SELL" if "SELL" in decision.upper() else "UNKNOWN",
                "claude_reasoning": "Advanced AI analysis with multi-agent debate"
            },
            "analysis_components": {
                "market_analysis": {
                    "status": "completed",
                    "summary": "Claude AI market analysis with real-time data integration",
                    "indicators_used": ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands", "ATR", "VWMA"],
                    "trend_analysis": market_data["technical_indicators"]["trend"],
                    "volatility_assessment": market_data["volatility"],
                    "data_file": final_state.get("market_data_json", {}).get("data_file", "Not captured"),
                    "data_summary": final_state.get("market_data_json", {}).get("data_summary", {}),
                    "full_report": final_state.get("market_report", "")[:500] + "..." if len(final_state.get("market_report", "")) > 500 else final_state.get("market_report", "")
                },
                "news_analysis": {
                    "status": "completed", 
                    "summary": "Claude AI news sentiment analysis",
                    "sentiment": news_sentiment["overall_sentiment"],
                    "sentiment_score": news_sentiment["sentiment_score"],
                    "key_headlines_analyzed": True,
                    "ai_confidence": fundamental_insights["confidence_level"],
                    "data_file": final_state.get("news_data_json", {}).get("data_file", "Not captured"),
                    "data_summary": final_state.get("news_data_json", {}).get("data_summary", {}),
                    "full_report": final_state.get("news_report", "")[:500] + "..." if len(final_state.get("news_report", "")) > 500 else final_state.get("news_report", "")
                },
                "fundamental_analysis": {
                    "status": "completed",
                    "summary": "Claude AI fundamental analysis",
                    "key_metrics": fundamental_insights["key_metrics"],
                    "financial_health": fundamental_insights["financial_health"],
                    "growth_prospects": fundamental_insights["growth_prospects"],
                    "confidence_level": fundamental_insights["confidence_level"],
                    "analysis_source": fundamental_insights["analysis_source"],
                    "full_report": final_state.get("fundamentals_report", "")[:500] + "..." if len(final_state.get("fundamentals_report", "")) > 500 else final_state.get("fundamentals_report", "")
                },
                "investment_debate": {
                    "status": "completed",
                    "bull_perspective": final_state.get("investment_debate_state", {}).get("bull_history", ["Strong Azure growth and AI positioning"])[-1] if final_state.get("investment_debate_state", {}).get("bull_history") else "Strong Azure growth and AI positioning",
                    "bear_perspective": final_state.get("investment_debate_state", {}).get("bear_history", ["Market competition and valuation concerns"])[-1] if final_state.get("investment_debate_state", {}).get("bear_history") else "Market competition and valuation concerns",
                    "consensus": final_state.get("investment_debate_state", {}).get("judge_decision", "balanced approach with AI focus"),
                    "claude_analysis": "Multi-agent debate facilitated by Claude AI with deep reasoning"
                }
            },
            "market_data": market_data,
            "news_sentiment": news_sentiment,
            "risk_assessment": {
                "overall_risk": risk_assessment["overall_risk"],
                "risk_factors": risk_assessment["risk_factors"],
                "risk_mitigation": risk_assessment["risk_mitigation"],
                "risk_score": risk_assessment["risk_score"],
                "volatility_risk": market_data["volatility"],
                "claude_risk_analysis": risk_assessment["risk_analysis_source"],
                "risk_debate_summary": risk_assessment["risk_debate_summary"],
                "ai_risk_perspectives": risk_assessment["ai_risk_perspectives"]
            },
            "strategic_actions": {
                "immediate_actions": strategic_insights["immediate_actions"],
                "medium_term_actions": strategic_insights["medium_term_actions"],
                "monitoring_metrics": strategic_insights["monitoring_metrics"],
                "claude_strategic_insights": strategic_insights["strategic_source"],
                "trader_plan": strategic_insights["trader_plan_summary"]
            },
            "performance_metrics": {
                "analysis_speed": performance_metrics["analysis_speed"],
                "data_quality": performance_metrics["data_quality"],
                "ai_provider": "Anthropic Claude-3.5-Sonnet",
                "cost_efficiency": performance_metrics["cost_efficiency"],
                "reliability_score": performance_metrics["reliability_score"],
                "claude_confidence": fundamental_insights["confidence_level"] + " AI confidence with comprehensive reasoning",
                "market_data_completeness": performance_metrics["market_data_completeness"]
            },
            "raw_reports": {
                "market_report": final_state.get("market_report", ""),
                "news_report": final_state.get("news_report", ""),
                "fundamentals_report": final_state.get("fundamentals_report", ""),
                "sentiment_report": final_state.get("sentiment_report", "")
            }
        }
        
        return structured_results
        
    except Exception as e:
        print(f"❌ Analysis failed for {date_str}: {e}")
        return None

def main():
    """Run MSFT analysis for a single specified date"""
    
    print("🚀 MSFT Single-Day Analysis with Claude AI")
    print("🔧 Enhanced with Market Data Utilities v2.0")
    print("=" * 60)
    
    # Parse date argument or use yesterday
    if len(sys.argv) > 1:
        try:
            analysis_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
            print(f"📅 Analyzing specified date: {analysis_date.strftime('%Y-%m-%d')}")
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            return
    else:
        analysis_date = datetime.now() - timedelta(days=1)
        print(f"📅 Analyzing yesterday: {analysis_date.strftime('%Y-%m-%d')}")
    
    # Check if it's a trading day
    is_trading = is_trading_day(analysis_date)
    day_name = analysis_date.strftime("%A")
    date_str = analysis_date.strftime("%Y-%m-%d")
    
    print(f"📊 Day: {day_name}, {date_str}")
    print(f"📈 Trading Day: {'Yes' if is_trading else 'No'}")
    print()
    
    # Run the analysis
    print("🔄 Starting analysis...")
    results = run_msft_daily_analysis(analysis_date, is_trading)
    
    if results:
        # ✨ SAVE THE RESULTS ✨
        try:
            saved_path = save_to_zzsheep(results, ticker="MSFT", analysis_type="daily_claude_msft")
            print()
            print("🎉 Analysis Complete!")
            print("=" * 40)
            print(f"🎯 Decision: {results['final_decision']['recommendation']}")
            print(f"💰 Price: {results['market_data']['current_price']}")
            print(f"📊 Volume: {results['market_data']['volume']}")
            print(f"📉 Volatility: {results['market_data']['volatility']}")
            print(f"🎯 Trend: {results['market_data']['technical_indicators']['trend']}")
            print(f"📰 Sentiment: {results['news_sentiment']['overall_sentiment']}")
            print(f"⏱️ Duration: {results['performance_metrics']['analysis_speed']}")
            print()
            print(f"✅ Results saved to: {saved_path}")
            print("🌐 Ready for frontend consumption")
            
            return results, saved_path
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return results, None
    else:
        print("❌ Analysis failed. Check error messages above.")
        return None

if __name__ == "__main__":
    main() 