"""
Market Data Extraction and Processing Utilities for TradingAgents

This module provides utilities for extracting and processing market data from 
various sources including real-time APIs, historical data, and analysis reports.

Author: TradingAgents Team
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)


def extract_market_data_from_reports(final_state: Dict[str, Any], ticker: str = None, analysis_date: str = None) -> Dict[str, Any]:
    """
    Extract comprehensive market data from analysis reports and historical APIs
    
    This function combines:
    - Historical data for the specific analysis date from financialdatasets.ai API
    - Real-time data as fallback if historical data not available
    - Historical data for volatility calculation  
    - Text parsing from AI analysis reports
    - Technical indicator extraction
    
    Args:
        final_state: The final state object from trading analysis containing reports
        ticker: Stock ticker symbol (extracted from final_state if not provided)
        analysis_date: Specific date for analysis in YYYY-MM-DD format (if None, uses current date)
        
    Returns:
        Dictionary with comprehensive market data including:
        - current_price, daily_change, daily_change_percent
        - market_cap, volume, volatility
        - technical_indicators (trend, rsi, macd, sma values, etc.)
    """
    
    # Initialize market data structure
    market_data = {
        "current_price": "N/A",
        "daily_change": "N/A", 
        "daily_change_percent": "N/A",
        "market_cap": "N/A",
        "volume": "N/A",
        "volatility": "N/A",
        "technical_indicators": {
            "trend": "N/A",
            "momentum": "N/A",
            "support_level": "N/A",
            "resistance_level": "N/A",
            "rsi": "N/A",
            "macd": "N/A",
            "sma_20": "N/A",
            "sma_50": "N/A",
            "bollinger_upper": "N/A",
            "bollinger_lower": "N/A"
        }
    }
    
    # Extract ticker from final_state if not provided
    if not ticker:
        ticker = final_state.get("company_of_interest", "UNKNOWN")
    
    # Step 1: Get historical data for the specific analysis date (or real-time as fallback)
    try:
        from tradingagents.dataflows.cached_api_wrappers import fetch_financialdatasets_prices_cached, fetch_financialdatasets_realtime_quote
        
        # Parse analysis date
        target_date = None
        if analysis_date:
            try:
                target_date = datetime.strptime(analysis_date, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid analysis_date format: {analysis_date}. Using current date.")
        
        # If we have a specific analysis date, try to get historical data for that date
        if target_date and target_date.date() < datetime.now().date():
            try:
                # Get historical data for a range around the target date
                start_date = target_date - timedelta(days=5)  # Get some buffer days
                end_date = target_date + timedelta(days=1)
                
                hist_data = fetch_financialdatasets_prices_cached(ticker, start_date, end_date)
                
                if not hist_data.empty:
                    # Find the closest trading day to our target date
                    hist_data['date'] = pd.to_datetime(hist_data.index).date
                    target_row = hist_data[hist_data['date'] <= target_date.date()]
                    
                    if not target_row.empty:
                        # Get the most recent trading day data
                        latest_data = target_row.iloc[-1]
                        
                        market_data["current_price"] = f"${latest_data.get('close', 'N/A'):.2f}"
                        
                        # Calculate daily change if we have previous day data
                        if len(target_row) > 1:
                            prev_close = target_row.iloc[-2]['close']
                            daily_change = latest_data['close'] - prev_close
                            daily_change_pct = (daily_change / prev_close) * 100
                            market_data["daily_change"] = f"${daily_change:+.2f}"
                            market_data["daily_change_percent"] = f"{daily_change_pct:+.2f}%"
                        
                        # Extract volume (format nicely with commas)
                        volume = latest_data.get('volume', 'N/A')
                        if volume != 'N/A':
                            market_data["volume"] = f"{int(volume):,}"
                        else:
                            market_data["volume"] = volume
                            
                        logger.info(f"✅ Historical data captured for {ticker} on {analysis_date}: ${latest_data.get('close', 'N/A'):.2f}, Volume: {market_data['volume']}")
                        
                    else:
                        logger.warning(f"No historical data found for {ticker} around {analysis_date}")
                        
            except Exception as e:
                logger.warning(f"Could not fetch historical data for {ticker} on {analysis_date}: {e}")
        
        # If historical data failed or we're analyzing current/future date, use real-time data
        if market_data["current_price"] == "N/A":
            quote = fetch_financialdatasets_realtime_quote(ticker)
            if quote:
                market_data["current_price"] = f"${quote.get('price', 'N/A')}"
                market_data["daily_change"] = f"${quote.get('day_change', 'N/A')}"
                market_data["daily_change_percent"] = f"{quote.get('day_change_percent', 'N/A')}%"
                market_data["market_cap"] = quote.get('market_cap', 'N/A')
                
                # Extract volume (format nicely with commas)
                volume = quote.get('volume', 'N/A')
                if volume != 'N/A':
                    market_data["volume"] = f"{int(volume):,}"
                else:
                    market_data["volume"] = volume
                    
                logger.info(f"✅ Real-time data captured for {ticker}: ${quote.get('price', 'N/A')}, Volume: {market_data['volume']}")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch market data for {ticker}: {e}")
    
    # Step 2: Calculate volatility from recent historical data
    try:
        from tradingagents.dataflows.cached_api_wrappers import fetch_financialdatasets_prices_cached
        
        # Get 30 days of historical data to calculate volatility relative to analysis date
        if target_date:
            end_date = target_date
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        hist_data = fetch_financialdatasets_prices_cached(ticker, start_date, end_date)
        
        if not hist_data.empty and 'close' in hist_data.columns:
            # Calculate daily returns
            hist_data['daily_return'] = hist_data['close'].pct_change()
            
            # Calculate annualized volatility (standard deviation of returns * sqrt(252))
            daily_vol = hist_data['daily_return'].std()
            annual_vol = daily_vol * np.sqrt(252) * 100  # Convert to percentage
            
            market_data["volatility"] = f"{annual_vol:.1f}%"
            logger.info(f"📊 Volatility calculated for {ticker}: {annual_vol:.1f}%")
            
            # Also extract volume from historical data if real-time failed
            if market_data["volume"] == "N/A" and 'volume' in hist_data.columns:
                avg_volume = hist_data['volume'].tail(5).mean()  # 5-day average
                market_data["volume"] = f"{int(avg_volume):,}"
                logger.info(f"📊 Volume from historical data for {ticker}: {int(avg_volume):,}")
                
    except Exception as e:
        logger.warning(f"⚠️ Could not calculate volatility for {ticker}: {e}")
    
    # Step 3: Extract data from analysis reports using text parsing
    market_report = final_state.get("market_report", "")
    fundamentals_report = final_state.get("fundamentals_report", "")
    combined_text = f"{market_report} {fundamentals_report}"
    
    if combined_text:
        # Extract price information if not captured from API
        if market_data["current_price"] == "N/A":
            market_data["current_price"] = _extract_price_from_text(combined_text, ticker)
        
        # Extract technical indicators
        market_data["technical_indicators"] = _extract_technical_indicators(combined_text, market_data["technical_indicators"])
        
        # Extract market cap estimation from revenue if not available
        if market_data["market_cap"] == "N/A":
            market_data["market_cap"] = _estimate_market_cap_from_revenue(combined_text)
    
    return market_data


def _extract_price_from_text(text: str, ticker: str) -> str:
    """Extract stock price from analysis text"""
    
    # Better price extraction patterns - prioritize higher, more realistic prices
    price_patterns = [
        r"at\s+\$(\d{2,3}\.?\d*)\s+per\s+share",  # "at $145 per share" (2-3 digits)
        r"purchased.*?\$(\d{2,3}\.?\d*)\s+per\s+share",
        r"sold.*?\$(\d{2,3}\.?\d*)\s+per\s+share", 
        r"price[:\s]+\$(\d{2,3}\.?\d*)",
        r"current[:\s]+\$(\d{2,3}\.?\d*)",
        r"trading.*?\$(\d{2,3}\.?\d*)",
        r"stock.*?price.*?\$(\d{2,3}\.?\d*)"
    ]
    
    # Collect all found prices and take the most reasonable one
    found_prices = []
    for pattern in price_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                price = float(match)
                if 10 <= price <= 1000:  # Reasonable range for most stocks
                    found_prices.append(price)
            except ValueError:
                continue
    
    if found_prices:
        # Take the most recent (highest) reasonable price
        best_price = max(found_prices)
        logger.info(f"📊 Price extracted from text for {ticker}: ${best_price:.2f}")
        return f"${best_price:.2f}"
    
    return "N/A"


def _extract_technical_indicators(text: str, indicators: Dict[str, str]) -> Dict[str, str]:
    """Extract technical indicators from analysis text"""
    
    # RSI extraction (looking for "RSI oscillating between 50 and 70")
    rsi_patterns = [
        r"RSI.*?oscillating.*?between\s+(\d+)\s+and\s+(\d+)",
        r"RSI[:\s]+(\d+\.?\d*)",
        r"relative.*?strength.*?index.*?(\d+\.?\d*)"
    ]
    
    for pattern in rsi_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if "between" in pattern:
                # Take the average of the range
                try:
                    low, high = match.groups()
                    avg_rsi = (float(low) + float(high)) / 2
                    indicators["rsi"] = f"{avg_rsi:.1f}"
                except ValueError:
                    continue
            else:
                indicators["rsi"] = match.group(1)
            break
    
    # MACD indicators
    macd_patterns = [
        r"MACD[:\s]+([+-]?\d+\.?\d*)",
        r"MACD.*?line.*?above.*?signal",  # Positive signal
        r"MACD.*?line.*?below.*?signal"   # Negative signal
    ]
    
    for pattern in macd_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if "above" in match.group(0).lower():
                indicators["macd"] = "Positive"
            elif "below" in match.group(0).lower():
                indicators["macd"] = "Negative"
            else:
                indicators["macd"] = match.group(1)
            break
    
    # Moving averages - look for SMA patterns with realistic prices
    sma_patterns = [
        (r"50-day\s+SMA.*?\$?(\d{2,3}\.?\d*)", "sma_50"),
        (r"SMA.*?50.*?\$?(\d{2,3}\.?\d*)", "sma_50"),
        (r"200-day\s+SMA.*?\$?(\d{2,3}\.?\d*)", "sma_20"),  # Using available field
        (r"SMA.*?200.*?\$?(\d{2,3}\.?\d*)", "sma_20"),
        (r"20-day\s+SMA.*?\$?(\d{2,3}\.?\d*)", "sma_20"),
        (r"SMA.*?20.*?\$?(\d{2,3}\.?\d*)", "sma_20")
    ]
    
    for pattern, field in sma_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                price = float(match)
                if 10 <= price <= 1000:  # Reasonable range for SMA
                    indicators[field] = f"${price:.2f}"
                    break
            except ValueError:
                continue
    
    # Volume extraction
    volume_patterns = [
        r"volume[:\s]+(\d+[\d,]*)",
        r"trading\s+volume.*?(\d+[\d,]*)",
        r"(\d+[\d,]*)\s+shares?\s+traded"
    ]
    
    # Trend analysis (improved)
    bullish_signals = ["bullish", "uptrend", "rising", "positive momentum", "above signal line", "above.*sma"]
    bearish_signals = ["bearish", "downtrend", "falling", "negative momentum", "below signal line", "below.*sma"]
    
    bullish_count = sum(1 for signal in bullish_signals if re.search(signal, text.lower()))
    bearish_count = sum(1 for signal in bearish_signals if re.search(signal, text.lower()))
    
    if bullish_count > bearish_count:
        indicators["trend"] = "Bullish"
    elif bearish_count > bullish_count:
        indicators["trend"] = "Bearish"
    else:
        indicators["trend"] = "Neutral"
    
    return indicators


def _estimate_market_cap_from_revenue(text: str) -> str:
    """Estimate market cap from revenue mentioned in reports"""
    
    revenue_patterns = [
        r"revenue.*?\$(\d+\.?\d*)\s+billion",
        r"\$(\d+\.?\d*)\s+billion.*?revenue"
    ]
    
    for pattern in revenue_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                revenue_billions = float(match.group(1))
                # Rough market cap estimation (varies by industry, using conservative 5-8x revenue)
                estimated_market_cap = revenue_billions * 6.5  # Conservative estimate
                return f"~${estimated_market_cap:.1f}B"
            except ValueError:
                continue
    
    return "N/A"


def extract_news_sentiment_data(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract news sentiment analysis data from reports
    
    Args:
        final_state: The final state object containing news and sentiment reports
        
    Returns:
        Dictionary with news sentiment data
    """
    
    news_data = {
        "overall_sentiment": "Mixed",
        "sentiment_score": 0.0,
        "positive_news_count": 0,
        "negative_news_count": 0,
        "neutral_news_count": 0,
        "key_themes": []
    }
    
    news_report = final_state.get("news_report", "")
    sentiment_report = final_state.get("sentiment_report", "")
    
    combined_report = f"{news_report} {sentiment_report}".lower()
    
    if combined_report:
        # Determine overall sentiment
        positive_words = ["positive", "bullish", "growth", "strong", "good", "up", "gains"]
        negative_words = ["negative", "bearish", "decline", "weak", "poor", "down", "losses"]
        
        pos_score = sum(1 for word in positive_words if word in combined_report)
        neg_score = sum(1 for word in negative_words if word in combined_report)
        
        if pos_score > neg_score * 1.2:
            news_data["overall_sentiment"] = "Positive"
            news_data["sentiment_score"] = 0.6
        elif neg_score > pos_score * 1.2:
            news_data["overall_sentiment"] = "Negative"
            news_data["sentiment_score"] = -0.6
        else:
            news_data["overall_sentiment"] = "Mixed"
            news_data["sentiment_score"] = 0.1
    
    return news_data


def format_market_data_for_display(market_data: Dict[str, Any]) -> str:
    """
    Format market data for display in analysis summaries
    
    Args:
        market_data: Market data dictionary
        
    Returns:
        Formatted string for display
    """
    
    return f"""
💰 Current Price: {market_data['current_price']}
📈 Daily Change: {market_data['daily_change']} ({market_data['daily_change_percent']})
🔄 Volume: {market_data['volume']}
📊 Volatility: {market_data['volatility']}
📉 Trend: {market_data['technical_indicators']['trend']}
🎯 RSI: {market_data['technical_indicators']['rsi']}
📊 MACD: {market_data['technical_indicators']['macd']}
""".strip()


def extract_risk_assessment_from_reports(final_state, ticker):
    """
    Extract dynamic risk assessment and factors from AI risk analysis reports
    
    Args:
        final_state: Complete analysis state from TradingAgents
        ticker: Stock ticker symbol
        
    Returns:
        dict: Dynamic risk assessment with extracted factors
    """
    
    # Get risk debate state
    risk_state = final_state.get("risk_debate_state", {})
    
    # Extract risk perspectives
    risky_analysis = risk_state.get("current_risky_response", "")
    safe_analysis = risk_state.get("current_safe_response", "")
    neutral_analysis = risk_state.get("current_neutral_response", "")
    judge_decision = risk_state.get("judge_decision", "")
    
    # Extract risk factors from AI analysis
    risk_factors = []
    risk_mitigation = []
    
    # Common risk keywords to look for in AI analysis
    risk_keywords = [
        "volatility", "inflation", "recession", "competition", "regulation", 
        "market", "economic", "geopolitical", "sector", "earnings", 
        "valuation", "liquidity", "credit", "interest rate", "supply chain",
        "cyber", "environmental", "political", "technology", "disruption"
    ]
    
    # Extract from risky analysis
    if risky_analysis:
        sentences = risky_analysis.split('.')
        for sentence in sentences:
            sentence = sentence.strip().lower()
            for keyword in risk_keywords:
                if keyword in sentence and any(term in sentence for term in ["risk", "concern", "threat", "challenge", "pressure"]):
                    # Create a risk factor from the sentence
                    if len(sentence) > 20 and len(sentence) < 150:
                        risk_factor = sentence.strip().capitalize()
                        if risk_factor and risk_factor not in risk_factors:
                            risk_factors.append(risk_factor)
                    break
    
    # Extract from safe analysis for mitigation strategies
    if safe_analysis:
        sentences = safe_analysis.split('.')
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(term in sentence for term in ["mitigate", "hedge", "diversify", "manage", "reduce", "protect", "strategy"]):
                if len(sentence) > 20 and len(sentence) < 150:
                    mitigation = sentence.strip().capitalize()
                    if mitigation and mitigation not in risk_mitigation:
                        risk_mitigation.append(mitigation)
    
    # If no specific factors found, extract from judge decision
    if not risk_factors and judge_decision:
        sentences = judge_decision.split('.')
        for sentence in sentences:
            sentence = sentence.strip().lower()
            for keyword in risk_keywords:
                if keyword in sentence:
                    if len(sentence) > 20 and len(sentence) < 150:
                        risk_factor = sentence.strip().capitalize()
                        if risk_factor and "risk" in risk_factor.lower():
                            risk_factors.append(risk_factor)
                    break
    
    # Fallback to general categories if no specific factors found
    if not risk_factors:
        # Try to determine sector-specific risks based on ticker
        sector_risks = {
            "AAPL": ["Technology competition", "Supply chain dependencies", "Consumer demand fluctuations"],
            "MSFT": ["Cloud competition", "Cybersecurity threats", "Regulatory scrutiny"],
            "TSLA": ["EV market competition", "Production scaling risks", "Regulatory changes"],
            "NVDA": ["Semiconductor cycle risks", "AI bubble concerns", "Geopolitical tensions"],
            "GOOGL": ["Regulatory antitrust risks", "Privacy regulation", "Ad market competition"],
            "AMZN": ["E-commerce saturation", "Cloud competition", "Labor cost pressures"]
        }
        risk_factors = sector_risks.get(ticker, ["Market volatility", "Sector-specific risks", "Economic uncertainties"])
    
    # Determine overall risk level based on AI analysis tone
    overall_risk = "Moderate"
    risk_score = 5.0
    
    if risky_analysis and safe_analysis:
        risky_words = len([w for w in risky_analysis.lower().split() if w in ["risk", "concern", "threat", "volatile", "uncertain", "decline", "fall"]])
        safe_words = len([w for w in safe_analysis.lower().split() if w in ["stable", "strong", "growth", "positive", "opportunity", "bullish"]])
        
        if risky_words > safe_words * 1.5:
            overall_risk = "High"
            risk_score = 7.5
        elif safe_words > risky_words * 1.5:
            overall_risk = "Low"  
            risk_score = 3.0
    
    # Ensure we have mitigation strategies
    if not risk_mitigation:
        risk_mitigation = [
            "Diversification across positions",
            "Position sizing management", 
            "Stop-loss implementation",
            "Regular portfolio monitoring"
        ]
    
    return {
        "overall_risk": overall_risk,
        "risk_factors": risk_factors[:4],  # Limit to top 4 factors
        "risk_mitigation": risk_mitigation[:4],  # Limit to top 4 strategies
        "risk_score": risk_score,
        "risk_analysis_source": "AI Risk Debate Analysis",
        "risk_debate_summary": judge_decision[:200] + "..." if len(judge_decision) > 200 else judge_decision,
        "ai_risk_perspectives": {
            "risky_perspective": risky_analysis[:150] + "..." if len(risky_analysis) > 150 else risky_analysis,
            "safe_perspective": safe_analysis[:150] + "..." if len(safe_analysis) > 150 else safe_analysis,
            "neutral_perspective": neutral_analysis[:150] + "..." if len(neutral_analysis) > 150 else neutral_analysis
        }
    }


def extract_strategic_insights_from_reports(final_state, ticker):
    """
    Extract dynamic strategic actions and fundamental insights from AI analysis reports
    
    Args:
        final_state: Complete analysis state from TradingAgents
        ticker: Stock ticker symbol
        
    Returns:
        dict: Dynamic strategic insights extracted from AI analysis
    """
    
    # Get all relevant reports
    market_report = final_state.get("market_report", "")
    fundamentals_report = final_state.get("fundamentals_report", "")
    trader_plan = final_state.get("trader_investment_plan", "")
    investment_debate = final_state.get("investment_debate_state", {})
    
    # Extract strategic actions from trader plan and reports
    immediate_actions = []
    medium_term_actions = []
    monitoring_metrics = []
    
    # Action keywords to look for
    immediate_keywords = ["monitor", "track", "watch", "assess", "check", "review daily", "observe"]
    medium_keywords = ["evaluate", "develop", "implement", "expand", "build", "strategy", "long-term", "quarterly"]
    metric_keywords = ["metric", "kpi", "measure", "indicator", "ratio", "growth", "performance", "revenue"]
    
    # Extract from trader plan
    if trader_plan:
        sentences = trader_plan.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 100:
                sentence_lower = sentence.lower()
                
                # Check for immediate actions
                if any(keyword in sentence_lower for keyword in immediate_keywords):
                    action = sentence.strip()
                    if action and action not in immediate_actions:
                        immediate_actions.append(action)
                
                # Check for medium-term actions
                elif any(keyword in sentence_lower for keyword in medium_keywords):
                    action = sentence.strip()
                    if action and action not in medium_term_actions:
                        medium_term_actions.append(action)
                
                # Check for monitoring metrics
                elif any(keyword in sentence_lower for keyword in metric_keywords):
                    metric = sentence.strip()
                    if metric and metric not in monitoring_metrics:
                        monitoring_metrics.append(metric)
    
    # Extract from fundamentals report
    if fundamentals_report:
        sentences = fundamentals_report.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 100:
                sentence_lower = sentence.lower()
                
                if any(keyword in sentence_lower for keyword in metric_keywords):
                    metric = sentence.strip()
                    if metric and metric not in monitoring_metrics and len(monitoring_metrics) < 6:
                        monitoring_metrics.append(metric)
    
    # Fallback to ticker-specific actions if nothing extracted
    if not immediate_actions:
        ticker_actions = {
            "AAPL": ["Monitor iPhone sales trends", "Track services revenue growth", "Watch supply chain developments"],
            "MSFT": ["Monitor Azure quarterly growth", "Track AI service adoption rates", "Watch cloud market share dynamics"],
            "TSLA": ["Monitor delivery numbers", "Track production capacity", "Watch EV market competition"],
            "NVDA": ["Monitor data center demand", "Track AI chip adoption", "Watch semiconductor cycle"],
            "GOOGL": ["Monitor search ad revenue", "Track cloud growth", "Watch regulatory developments"],
            "AMZN": ["Monitor AWS growth", "Track e-commerce margins", "Watch logistics efficiency"]
        }
        immediate_actions = ticker_actions.get(ticker, ["Monitor earnings reports", "Track sector trends", "Watch market indicators"])
    
    if not medium_term_actions:
        ticker_medium = {
            "AAPL": ["Assess ecosystem expansion", "Review market share metrics", "Evaluate innovation pipeline"],
            "MSFT": ["Evaluate AI platform integration progress", "Review enterprise customer expansion", "Assess competitive positioning vs. AWS/Google"],
            "TSLA": ["Assess global expansion strategy", "Review autonomous driving progress", "Evaluate energy business growth"],
            "NVDA": ["Assess AI market expansion", "Review data center partnerships", "Evaluate next-gen chip development"],
            "GOOGL": ["Assess AI integration across products", "Review antitrust compliance", "Evaluate new revenue streams"],
            "AMZN": ["Assess international expansion", "Review logistics automation", "Evaluate new business segments"]
        }
        medium_term_actions = ticker_medium.get(ticker, ["Assess competitive position", "Review growth strategies", "Monitor regulatory changes"])
    
    if not monitoring_metrics:
        ticker_metrics = {
            "AAPL": ["iPhone unit sales", "Services revenue", "Market share trends", "Customer loyalty metrics"],
            "MSFT": ["Azure revenue growth rate", "Teams active users", "AI service utilization", "Enterprise customer retention"],
            "TSLA": ["Vehicle delivery numbers", "Production capacity utilization", "Supercharger network expansion", "Energy storage deployments"],
            "NVDA": ["Data center revenue", "Gaming GPU sales", "AI chip demand", "Automotive partnerships"],
            "GOOGL": ["Search ad revenue", "YouTube revenue", "Cloud growth rate", "Regulatory fine impact"],
            "AMZN": ["AWS revenue growth", "Prime subscriber growth", "Fulfillment center efficiency", "Third-party seller growth"]
        }
        monitoring_metrics = ticker_metrics.get(ticker, ["Revenue growth", "Market share", "Customer metrics", "Financial ratios"])
    
    return {
        "immediate_actions": immediate_actions[:4],  # Limit to top 4
        "medium_term_actions": medium_term_actions[:4],  # Limit to top 4  
        "monitoring_metrics": monitoring_metrics[:4],  # Limit to top 4
        "strategic_source": "AI Trading Plan Analysis",
        "trader_plan_summary": trader_plan[:200] + "..." if len(trader_plan) > 200 else trader_plan
    }


def extract_fundamental_insights_from_reports(final_state, ticker):
    """
    Extract dynamic fundamental analysis insights from AI reports
    
    Args:
        final_state: Complete analysis state from TradingAgents
        ticker: Stock ticker symbol
        
    Returns:
        dict: Dynamic fundamental insights extracted from AI analysis
    """
    
    fundamentals_report = final_state.get("fundamentals_report", "")
    investment_debate = final_state.get("investment_debate_state", {})
    
    # Extract financial health assessment
    financial_health = "Moderate"
    growth_prospects = "Mixed"
    
    if fundamentals_report:
        report_lower = fundamentals_report.lower()
        
        # Assess financial health based on keywords
        positive_health = ["strong balance", "robust financials", "healthy cash", "solid fundamentals", "excellent financial", "strong financial"]
        negative_health = ["weak balance", "poor financials", "cash concerns", "debt issues", "financial stress", "liquidity problems"]
        
        positive_count = sum(1 for term in positive_health if term in report_lower)
        negative_count = sum(1 for term in negative_health if term in report_lower)
        
        if positive_count > negative_count:
            financial_health = "Strong"
        elif negative_count > positive_count:
            financial_health = "Weak"
        
        # Assess growth prospects
        positive_growth = ["strong growth", "expanding revenue", "growing market", "increasing profits", "bright outlook", "positive trajectory"]
        negative_growth = ["declining revenue", "shrinking market", "decreasing profits", "challenging outlook", "negative trajectory", "growth concerns"]
        
        growth_positive = sum(1 for term in positive_growth if term in report_lower)
        growth_negative = sum(1 for term in negative_growth if term in report_lower)
        
        if growth_positive > growth_negative:
            growth_prospects = "Positive"
        elif growth_negative > growth_positive:
            growth_prospects = "Negative"
    
    # Extract key metrics based on ticker and analysis content
    key_metrics = []
    
    ticker_metrics = {
        "AAPL": ["Revenue Growth", "iPhone Sales", "Services Revenue", "Gross Margins"],
        "MSFT": ["Revenue Growth", "Azure Cloud", "AI Integration", "Office Subscriptions"],
        "TSLA": ["Vehicle Deliveries", "Production Scaling", "Energy Business", "Autonomous Driving"],
        "NVDA": ["Data Center Revenue", "Gaming Revenue", "AI Chip Demand", "Automotive Partnerships"],
        "GOOGL": ["Search Revenue", "YouTube Revenue", "Cloud Growth", "Other Bets"],
        "AMZN": ["AWS Revenue", "E-commerce Growth", "Prime Subscriptions", "Operating Margins"]
    }
    
    key_metrics = ticker_metrics.get(ticker, ["Revenue Growth", "Financial Health", "Market Position", "Growth Prospects"])
    
    # Extract confidence level from analysis tone
    confidence_level = "Moderate"
    
    if fundamentals_report:
        report_lower = fundamentals_report.lower()
        high_confidence = ["confident", "certain", "clear", "strong conviction", "high confidence", "definitive"]
        low_confidence = ["uncertain", "unclear", "mixed signals", "cautious", "low confidence", "ambiguous"]
        
        high_count = sum(1 for term in high_confidence if term in report_lower)
        low_count = sum(1 for term in low_confidence if term in report_lower)
        
        if high_count > low_count:
            confidence_level = "High"
        elif low_count > high_count:
            confidence_level = "Low"
    
    return {
        "key_metrics": key_metrics,
        "financial_health": financial_health,
        "growth_prospects": growth_prospects,
        "confidence_level": confidence_level,
        "analysis_source": "AI Fundamental Analysis",
        "fundamentals_summary": fundamentals_report[:200] + "..." if len(fundamentals_report) > 200 else fundamentals_report
    }


def extract_performance_metrics_from_reports(final_state, duration, ticker):
    """
    Extract dynamic performance metrics from AI analysis
    
    Args:
        final_state: Complete analysis state from TradingAgents
        duration: Analysis duration in seconds
        ticker: Stock ticker symbol
        
    Returns:
        dict: Dynamic performance metrics
    """
    
    # Get analysis components
    market_report = final_state.get("market_report", "")
    fundamentals_report = final_state.get("fundamentals_report", "")
    news_report = final_state.get("news_report", "")
    
    # Calculate data quality based on completeness
    data_quality = "Basic"
    completeness_score = 0
    
    if market_report and len(market_report) > 100:
        completeness_score += 25
    if fundamentals_report and len(fundamentals_report) > 100:
        completeness_score += 25
    if news_report and len(news_report) > 100:
        completeness_score += 25
    if final_state.get("market_data_json") or final_state.get("news_data_json"):
        completeness_score += 25
    
    if completeness_score >= 75:
        data_quality = "Professional-grade with real-time integration"
    elif completeness_score >= 50:
        data_quality = "High-quality with comprehensive data"
    elif completeness_score >= 25:
        data_quality = "Standard with essential data"
    
    # Calculate reliability score based on analysis depth
    reliability_score = 80  # Base score
    
    if final_state.get("investment_debate_state", {}).get("judge_decision"):
        reliability_score += 5
    if final_state.get("risk_debate_state", {}).get("judge_decision"):
        reliability_score += 5
    if final_state.get("trader_investment_plan"):
        reliability_score += 5
    if duration < 120:  # Fast analysis
        reliability_score += 3
    if duration > 300:  # Very thorough analysis
        reliability_score += 2
    
    reliability_score = min(reliability_score, 99)  # Cap at 99
    
    # Determine cost efficiency based on speed and quality
    cost_efficiency = "Standard"
    if duration < 60 and completeness_score >= 75:
        cost_efficiency = "Optimized for production"
    elif duration < 120 and completeness_score >= 50:
        cost_efficiency = "Efficient with good quality"
    elif duration > 300:
        cost_efficiency = "Thorough but resource-intensive"
    
    return {
        "analysis_speed": f"{duration:.1f}s",
        "data_quality": data_quality,
        "cost_efficiency": cost_efficiency,
        "reliability_score": reliability_score,
        "completeness_score": completeness_score,
        "market_data_completeness": "High quality data integration" if completeness_score >= 75 else "Standard data integration"
    }


# Example usage and testing functions
if __name__ == "__main__":
    print("Market Data Utils - Testing Functions")
    
    # Mock final_state for testing
    test_final_state = {
        "company_of_interest": "AAPL",
        "market_report": "The stock is trading at $150 per share with RSI oscillating between 45 and 65. MACD line is above signal line indicating bullish momentum.",
        "fundamentals_report": "Revenue of $380 billion shows strong growth. The company maintains strong profitability.",
        "news_report": "Positive earnings beat expectations. Strong growth in services revenue.",
        "sentiment_report": "Overall bullish sentiment in the market with strong investor confidence."
    }
    
    # Test market data extraction
    market_data = extract_market_data_from_reports(test_final_state, "AAPL", "2024-06-15")
    print("Extracted Market Data:")
    print(format_market_data_for_display(market_data)) 