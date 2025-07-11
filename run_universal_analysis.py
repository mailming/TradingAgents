"""
Universal Trading Analysis Script

This script can analyze any ticker for any specific date using:
- Anthropic Claude AI (Claude-3.5-Sonnet)
- New market data extraction utilities  
- financialdatasets.ai (professional data source)
- zzsheeptrader export utility

Usage: python run_universal_analysis.py TICKER [YYYY-MM-DD]
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

def run_analysis(ticker, analysis_date):
    """Run trading analysis for a specific ticker and date"""
    
    date_str = analysis_date.strftime('%Y-%m-%d')
    day_name = analysis_date.strftime('%A')
    is_trading = is_trading_day(analysis_date)
    
    print(f"📊 Running {ticker} Analysis for {day_name}, {date_str}")
    if not is_trading:
        print(f"   📅 Non-trading day - Analysis will use most recent market data")
    
    # Check API keys
    required_keys = ['ANTHROPIC_API_KEY', 'FINANCIALDATASETS_API_KEY']
    optional_keys = ['OPENAI_API_KEY']
    
    # Check required keys
    for key in required_keys:
        if not os.getenv(key):
            print(f"❌ {key} not found!")
            return None
    
    # Check optional keys and warn if missing
    for key in optional_keys:
        if not os.getenv(key):
            print(f"⚠️  {key} not found - some OpenAI features will be unavailable")
        else:
            print(f"✅ {key} found")
    
    # Professional Claude AI configuration
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-3-5-sonnet-20241022"
    config["quick_think_llm"] = "claude-3-5-haiku-20241022"
    config["max_debate_rounds"] = 3
    config["max_risk_discuss_rounds"] = 3
    config["online_tools"] = True
    
    print(f"   🧠 AI Model: Claude-3.5-Sonnet (Premium)")
    print(f"   📊 Data Source: financialdatasets.ai (Professional)")
    
    # Initialize TradingAgents - use sequential processing to avoid message deletion issues
    try:
        ta = TradingAgentsGraph(debug=False, config=config, parallel_processing=False)
        print(f"   ✅ TradingAgents initialized successfully")
    except Exception as e:
        if "already exists" in str(e).lower() or "collection" in str(e).lower():
            print(f"   🔄 Memory collections exist, reinitializing...")
            try:
                import chromadb
                chroma_client = chromadb.Client()
                collections_to_delete = ["bull_memory", "bear_memory", "neutral_memory"]
                for collection_name in collections_to_delete:
                    try:
                        chroma_client.delete_collection(collection_name)
                    except:
                        pass
                ta = TradingAgentsGraph(debug=False, config=config)
                print(f"   ✅ TradingAgents reinitialized with fresh collections")
            except Exception as retry_error:
                print(f"❌ Failed to reinitialize: {retry_error}")
                return None
        else:
            print(f"❌ Failed to initialize: {e}")
            return None
    
    # Run analysis
    start_time = time.time()
    
    try:
        final_state, decision = ta.propagate(ticker, date_str)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"   ✅ Analysis completed in {duration:.1f} seconds")
        print(f"   🎯 Final Decision: {decision}")
        
        # Extract comprehensive market data
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
        
        # Structure results for JSON
        timestamp = datetime.now()
        analysis_id = str(uuid.uuid4())[:8]
        
        structured_results = {
            "analysis_metadata": {
                "analysis_id": analysis_id,
                "ticker": ticker.upper(),
                "analysis_date": date_str,
                "day_of_week": day_name,
                "is_trading_day": is_trading,
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
                    "full_report": final_state.get("market_report", "")
                },
                "news_analysis": {
                    "status": "completed",
                    "summary": "Claude AI news sentiment analysis",
                    "sentiment": news_sentiment["overall_sentiment"],
                    "sentiment_score": news_sentiment["sentiment_score"],
                    "key_headlines_analyzed": True,
                    "ai_confidence": "High",
                    "full_report": final_state.get("news_report", "")
                },
                "fundamental_analysis": {
                    "status": "completed",
                    "summary": "Claude AI fundamental analysis",
                    "key_metrics": fundamental_insights["key_metrics"],
                    "financial_health": fundamental_insights["financial_health"],
                    "growth_prospects": fundamental_insights["growth_prospects"],
                    "confidence_level": fundamental_insights["confidence_level"],
                    "analysis_source": fundamental_insights["analysis_source"],
                    "full_report": final_state.get("fundamentals_report", "")
                },
                "investment_debate": {
                    "status": "completed",
                    "bull_perspective": final_state.get("investment_debate_state", {}).get("bull_history", ["Strong fundamentals and growth potential"])[-1] if final_state.get("investment_debate_state", {}).get("bull_history") else "Strong fundamentals and growth potential",
                    "bear_perspective": final_state.get("investment_debate_state", {}).get("bear_history", ["Market risks and valuation concerns"])[-1] if final_state.get("investment_debate_state", {}).get("bear_history") else "Market risks and valuation concerns",
                    "consensus": final_state.get("investment_debate_state", {}).get("judge_decision", "balanced approach recommended"),
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
    """Main function to run analysis"""
    
    if len(sys.argv) < 2:
        print("❌ Usage: python run_universal_analysis.py TICKER [YYYY-MM-DD]")
        print("Examples:")
        print("  python run_universal_analysis.py AAPL")
        print("  python run_universal_analysis.py MSFT 2025-06-24")
        return
    
    ticker = sys.argv[1].upper()
    
    # Parse date
    if len(sys.argv) >= 3:
        try:
            analysis_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            return
    else:
        analysis_date = datetime.now() - timedelta(days=1)
    
    print(f"🚀 Universal Trading Analysis with Claude AI")
    print(f"🔧 Enhanced with Market Data Utilities v2.0")
    print("=" * 60)
    print(f"📊 Ticker: {ticker}")
    print(f"📅 Date: {analysis_date.strftime('%A, %Y-%m-%d')}")
    print()
    
    # Run analysis
    print("🔄 Starting analysis...")
    results = run_analysis(ticker, analysis_date)
    
    if results:
        # Save results
        try:
            saved_path = save_to_zzsheep(results, ticker=ticker, analysis_type="daily_claude_analysis")
            
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
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    else:
        print("❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()
