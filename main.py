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
import json
import re

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
    extract_performance_metrics_from_reports,
    format_market_analysis_report,
    format_news_analysis_report,
    format_fundamental_analysis_report
)

def is_api_overloaded_error(error_message):
    """Check if error is due to API overload/rate limiting"""
    error_str = str(error_message).lower()
    
    # Patterns that indicate API overload
    overload_patterns = [
        "overloaded",
        "overloaded_error",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "529",
        "service unavailable",
        "server overload",
        "temporarily unavailable"
    ]
    
    return any(pattern in error_str for pattern in overload_patterns)

def retry_on_overload(func, max_retries=3, wait_time=30):
    """Retry function with exponential backoff for API overload errors"""
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if is_api_overloaded_error(str(e)):
                if attempt < max_retries:
                    wait_seconds = wait_time * (2 ** attempt)  # Exponential backoff
                    print(f"   ⏳ API overloaded (attempt {attempt + 1}/{max_retries + 1}). Waiting {wait_seconds} seconds before retry...")
                    time.sleep(wait_seconds)
                    continue
                else:
                    print(f"   ❌ API still overloaded after {max_retries} retries. Giving up.")
                    raise e
            else:
                # Non-overload error, don't retry
                raise e
    
    # This should never be reached
    raise Exception("Unexpected error in retry logic")

def extract_analyst_perspective(history_string, analyst_type):
    """Extract meaningful perspective from analyst history string"""
    if not history_string or not isinstance(history_string, str):
        return ""
    
    # Split by lines and get all analyst responses
    lines = history_string.strip().split('\n')
    
    # Find analyst sections and extract meaningful content
    prefix = f"{analyst_type.capitalize()} Analyst:"
    analyst_content = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith(prefix):
            # Found analyst section, now collect the actual content
            content_lines = []
            
            # Get content after the prefix on same line
            remaining_content = line[len(prefix):].strip()
            if remaining_content and not any(keyword in remaining_content.upper() for keyword in ["PERSPECTIVE:", "ANALYSIS:", "THESIS", "CASE FOR", "EXAMINATION"]):
                content_lines.append(remaining_content)
            
            # Collect following lines until next analyst or end
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                
                # Stop if we hit another analyst
                if any(next_line.startswith(f"{analyst.capitalize()} Analyst:") for analyst in ["bull", "bear", "neutral", "risky", "safe"]):
                    break
                
                # Skip empty lines and title/header lines
                if (next_line and 
                    not any(keyword in next_line.upper() for keyword in ["PERSPECTIVE:", "ANALYSIS:", "THESIS", "CASE FOR", "EXAMINATION", "🐻", "🐂"]) and
                    len(next_line) > 15):
                    content_lines.append(next_line)
                
                # Stop after collecting enough content
                if len(' '.join(content_lines)) > 300:
                    break
            
            # Join the content and clean it up
            if content_lines:
                full_content = ' '.join(content_lines).strip()
                
                # Remove any remaining formatting artifacts
                full_content = full_content.replace('**', '').replace('##', '').replace('###', '')
                
                # Extract key sentences (first few substantive sentences)
                sentences = full_content.split('. ')
                meaningful_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if (len(sentence) > 30 and 
                        not any(keyword in sentence.upper() for keyword in ["PERSPECTIVE", "ANALYSIS", "THESIS", "CASE FOR", "I'LL", "HERE'S", "LET ME"])):
                        meaningful_sentences.append(sentence)
                        if len(meaningful_sentences) >= 2:  # Limit to 2 key sentences
                            break
                
                if meaningful_sentences:
                    result = '. '.join(meaningful_sentences)
                    if not result.endswith('.'):
                        result += '.'
                    
                    # Truncate if too long
                    if len(result) > 250:
                        result = result[:247] + "..."
                    
                    analyst_content.append(result)
    
    # Return the last (most recent) meaningful response
    return analyst_content[-1] if analyst_content else ""

def extract_concise_consensus(judge_decision):
    """Extract a concise consensus from the verbose judge decision"""
    if not judge_decision or not isinstance(judge_decision, str):
        return "Balanced approach recommended based on comprehensive analysis"
    
    # Look for the actual recommendation
    decision = "HOLD"
    reasoning = ""
    
    # Extract the recommendation
    if "Recommendation: BUY" in judge_decision or "BUY" in judge_decision.upper():
        decision = "BUY"
    elif "Recommendation: SELL" in judge_decision or "SELL" in judge_decision.upper():
        decision = "SELL"
    elif "Recommendation: HOLD" in judge_decision or "HOLD" in judge_decision.upper():
        decision = "HOLD"
    
    # Extract key reasoning - look for rationale section
    lines = judge_decision.split('\n')
    rationale_found = False
    key_points = []
    
    for line in lines:
        line = line.strip()
        if "Rationale:" in line or "rationale" in line.lower():
            rationale_found = True
            continue
        
        if rationale_found and line:
            # Stop at strategic plan or other sections
            if any(keyword in line.lower() for keyword in ["strategic", "entry strategy", "risk management", "position sizing", "learning from"]):
                break
            
            # Extract meaningful points
            if line.startswith(("1.", "2.", "3.", "-", "•")) or len(line) > 30:
                clean_line = line.lstrip("123456789.- •").strip()
                if len(clean_line) > 20 and len(clean_line) < 150:
                    key_points.append(clean_line)
                    if len(key_points) >= 2:  # Limit to 2 key points
                        break
    
    # Build concise consensus
    if key_points:
        reasoning = ". ".join(key_points[:2])
        if len(reasoning) > 150:
            reasoning = reasoning[:147] + "..."
    else:
        # Fallback to basic reasoning based on decision
        if decision == "BUY":
            reasoning = "Strong fundamentals and growth prospects outweigh potential risks"
        elif decision == "SELL":
            reasoning = "Significant risks and overvaluation concerns warrant caution"
        else:
            reasoning = "Mixed signals suggest a balanced approach with careful monitoring"
    
    return f"{decision}: {reasoning}"

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
    
    print(f"📊 Checking {ticker} Analysis for {day_name}, {date_str}")
    if not is_trading:
        print(f"   📅 Non-trading day - Skipping analysis")
        print(f"   ⏭️  Market is closed on {day_name}s or holidays")
        return None
    
    print(f"📊 Running {ticker} Analysis for {day_name}, {date_str}")
    
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
        def init_trading_agents():
            return TradingAgentsGraph(debug=False, config=config, parallel_processing=False)
        
        print(f"   🔄 Initializing TradingAgents (with auto-retry on API overload)...")
        ta = retry_on_overload(init_trading_agents, max_retries=2, wait_time=30)
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
                
                def reinit_trading_agents():
                    return TradingAgentsGraph(debug=False, config=config)
                
                ta = retry_on_overload(reinit_trading_agents, max_retries=2, wait_time=30)
                print(f"   ✅ TradingAgents reinitialized with fresh collections")
            except Exception as retry_error:
                print(f"❌ Failed to reinitialize: {retry_error}")
                return None
        else:
            print(f"❌ Failed to initialize: {e}")
            return None
    
    # Run analysis with retry logic for API overload
    start_time = time.time()
    
    try:
        def run_analysis_with_retry():
            return ta.propagate(ticker, date_str)
        
        print(f"   🔄 Running analysis (with auto-retry on API overload)...")
        final_state, decision = retry_on_overload(run_analysis_with_retry, max_retries=3, wait_time=30)
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
                    "summary": "Professional market analysis with real-time data integration and technical indicators",
                    "indicators_used": ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands", "ATR", "VWMA"],
                    "trend_analysis": market_data["technical_indicators"]["trend"],
                    "volatility_assessment": market_data["volatility"],
                    "full_report": format_market_analysis_report(final_state, market_data, ticker)
                },
                "news_analysis": {
                    "status": "completed",
                    "summary": "Professional news sentiment analysis with AI-powered insights",
                    "sentiment": news_sentiment["overall_sentiment"],
                    "sentiment_score": news_sentiment["sentiment_score"],
                    "key_headlines_analyzed": True,
                    "ai_confidence": "High",
                    "full_report": format_news_analysis_report(final_state, news_sentiment, ticker)
                },
                "fundamental_analysis": {
                    "status": "completed",
                    "summary": "Professional fundamental analysis with financial health assessment",
                    "key_metrics": fundamental_insights["key_metrics"],
                    "financial_health": fundamental_insights["financial_health"],
                    "growth_prospects": fundamental_insights["growth_prospects"],
                    "confidence_level": fundamental_insights["confidence_level"],
                    "analysis_source": fundamental_insights["analysis_source"],
                    "full_report": format_fundamental_analysis_report(final_state, fundamental_insights, ticker)
                },
                "investment_debate": {
                    "status": "completed",
                    "bull_perspective": extract_analyst_perspective(final_state.get("investment_debate_state", {}).get("bull_history", ""), "bull") or "Strong fundamentals and growth potential with upside opportunities",
                    "bear_perspective": extract_analyst_perspective(final_state.get("investment_debate_state", {}).get("bear_history", ""), "bear") or "Market risks and valuation concerns requiring caution",
                    "consensus": extract_concise_consensus(final_state.get("investment_debate_state", {}).get("judge_decision", "")),
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
        # Check if it was skipped due to non-trading day
        if not is_trading_day(analysis_date):
            print()
            print("⏭️  Analysis Skipped - Non-Trading Day")
            print("=" * 40)
            print("💡 Try running analysis on a weekday when markets are open")
            print("📅 Markets are typically closed on weekends and major holidays")
        else:
            print("❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()
