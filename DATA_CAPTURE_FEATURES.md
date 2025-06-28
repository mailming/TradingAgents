# TradingAgents Data Capture Features

## 🔍 Overview

The TradingAgents system now automatically captures and saves the underlying raw data used by analysts, in addition to generating analysis reports. This provides complete transparency and data traceability for all trading decisions.

## 📊 Market Data Capture

### **Market Analyst Enhanced Features**
The `market_analyst.py` now captures:

#### **Data Types Captured:**
- **Price Data**: Historical stock prices from Yahoo Finance
  - Open, High, Low, Close, Volume
  - Date range and parameters used
- **Technical Indicators**: StockStats calculations
  - SMA, EMA, MACD, RSI, Bollinger Bands, ATR, VWMA
  - All calculated indicator values and parameters

#### **File Structure:**
```json
{
  "price_data": {
    "data": {/* DataFrame converted to dict */},
    "tool_used": "get_YFin_data",
    "parameters": {"symbol": "AAPL", "start_date": "2024-12-01", "end_date": "2024-12-28"}
  },
  "technical_indicators": {
    "data": {/* Technical indicators data */},
    "tool_used": "get_stockstats_indicators_report",
    "parameters": {"indicators": ["sma", "ema", "macd", "rsi"]}
  },
  "analysis_metadata": {
    "ticker": "AAPL",
    "analysis_date": "2024-12-28",
    "timestamp": "2024-12-28T15:30:45.123456",
    "analyst_type": "market_analyst",
    "data_source": "yahoo_finance_stockstats",
    "tools_used": ["get_YFin_data", "get_stockstats_indicators_report"]
  }
}
```

#### **Saved Files:**
- **Location**: `zzsheepTrader/analysis_results/json/`
- **Naming**: `{TICKER}_{DATE}_{ID}_market_data.json`
- **Example**: `AAPL_2024-12-28_abc123_market_data.json`

---

## 📰 News Feed Capture

### **News Analyst Enhanced Features**
The `news_analyst.py` now captures:

#### **Data Types Captured:**
- **Finnhub News**: Financial news from Finnhub API
- **Google News**: General news from Google News
- **Reddit News**: Social sentiment from Reddit
- **Global News**: International news sources

#### **File Structure:**
```json
{
  "finnhub_news": [
    {
      "tool_used": "get_finnhub_news",
      "parameters": {"ticker": "AAPL", "lookback_days": 7},
      "timestamp": "2024-12-28T15:30:45.123456",
      "data": [/* Array of news articles */]
    }
  ],
  "google_news": [/* Google news entries */],
  "reddit_news": [/* Reddit posts */],
  "global_news": [/* Global news articles */],
  "analysis_metadata": {
    "ticker": "AAPL",
    "analysis_date": "2024-12-28",
    "timestamp": "2024-12-28T15:30:45.123456",
    "analyst_type": "news_analyst",
    "data_source": "multiple_news_sources",
    "tools_used": ["get_finnhub_news", "get_google_news"]
  }
}
```

#### **Saved Files:**
- **Location**: `zzsheepTrader/analysis_results/json/`
- **Naming**: `{TICKER}_{DATE}_{ID}_news_feed.json`
- **Example**: `AAPL_2024-12-28_def456_news_feed.json`

---

## 🔗 Integration with Analysis Reports

### **Enhanced JSON Structure**
The main analysis reports now include references to the captured data:

```json
{
  "analysis_components": {
    "market_analysis": {
      "status": "completed",
      "summary": "AI-generated market analysis",
      "indicators_used": ["SMA", "EMA", "MACD", "RSI"],
      "data_file": "/path/to/AAPL_2024-12-28_abc123_market_data.json",
      "data_summary": {
        "has_price_data": true,
        "has_indicators": true,
        "tools_executed": 2
      }
    },
    "news_analysis": {
      "status": "completed",
      "summary": "AI-generated news analysis",
      "sentiment": "Mixed",
      "data_file": "/path/to/AAPL_2024-12-28_def456_news_feed.json",
      "data_summary": {
        "total_news_items": 25,
        "finnhub_articles": 10,
        "google_articles": 8,
        "reddit_posts": 5,
        "global_articles": 2,
        "tools_executed": 3,
        "sources_used": ["get_finnhub_news", "get_google_news", "get_reddit_news"]
      }
    }
  }
}
```

---

## 🎯 Benefits

### **Complete Data Traceability**
- **Source Attribution**: Every piece of data is linked to its source
- **Parameter Tracking**: All tool parameters and settings recorded
- **Timestamp Precision**: Exact capture times for data freshness verification

### **Enhanced Analysis Quality**
- **Data Verification**: Raw data can be inspected and verified
- **Reproducibility**: Analysis can be reproduced with same data
- **Debugging**: Issues can be traced back to data sources

### **Frontend Integration**
- **Rich Dashboards**: Frontend can display both analysis and raw data
- **Data Visualization**: Charts can be built from captured market data
- **News Feeds**: Real news articles can be displayed alongside analysis

---

## 🔧 Technical Implementation

### **Agent State Updates**
New fields added to `AgentState`:
```python
market_data_json: Annotated[dict, "Market data and technical indicators captured as JSON"]
news_data_json: Annotated[dict, "News feed data captured as JSON"]
```

### **Error Handling**
- **Graceful Degradation**: If data capture fails, analysis continues
- **Warning Messages**: Issues logged but don't stop analysis
- **Fallback Behavior**: System works with or without data capture

### **Performance Considerations**
- **Efficient Execution**: Data capture integrated into existing tool calls
- **Minimal Overhead**: No duplicate API calls or processing
- **Asynchronous Saving**: JSON files saved without blocking analysis

---

## 📁 File Organization

### **Directory Structure**
```
zzsheepTrader/analysis_results/json/
├── AAPL_2024-12-28_14-30-15_abc123.json          # Main analysis
├── AAPL_2024-12-28_abc123_market_data.json       # Market data
├── AAPL_2024-12-28_def456_news_feed.json         # News feed
├── MSFT_2024-12-27_ghi789_claude.json            # Main analysis
├── MSFT_2024-12-27_ghi789_market_data.json       # Market data
└── MSFT_2024-12-27_jkl012_news_feed.json         # News feed
```

### **Naming Conventions**
- **Main Analysis**: `{TICKER}_{DATE}_{TIME}_{ID}.json`
- **Market Data**: `{TICKER}_{DATE}_{ID}_market_data.json`
- **News Feed**: `{TICKER}_{DATE}_{ID}_news_feed.json`
- **Claude Analysis**: `{TICKER}_{DATE}_{ID}_claude.json`

---

## 🚀 Usage Examples

### **Accessing Market Data**
```python
import json

# Load main analysis
with open('AAPL_2024-12-28_14-30-15_abc123.json') as f:
    analysis = json.load(f)

# Get market data file path
market_data_file = analysis['analysis_components']['market_analysis']['data_file']

# Load raw market data
with open(market_data_file) as f:
    market_data = json.load(f)

# Access price data
price_data = market_data['price_data']['data']
indicators = market_data['technical_indicators']['data']
```

### **Accessing News Data**
```python
# Get news data file path
news_data_file = analysis['analysis_components']['news_analysis']['data_file']

# Load raw news data
with open(news_data_file) as f:
    news_data = json.load(f)

# Access different news sources
finnhub_articles = news_data['finnhub_news']
google_articles = news_data['google_news']
reddit_posts = news_data['reddit_news']
```

---

## 🔄 Backward Compatibility

- **Existing Code**: All existing analysis code continues to work
- **Optional Features**: Data capture is additional, not required
- **API Stability**: No breaking changes to existing APIs
- **Migration**: No migration needed for existing analyses

---

## 📈 Future Enhancements

### **Planned Features**
- **Fundamental Data Capture**: Add similar capture for fundamental analyst
- **Social Media Data**: Enhanced social sentiment data capture
- **Data Aggregation**: Tools to combine and analyze captured data
- **Historical Comparison**: Compare data across different analysis dates

### **Integration Opportunities**
- **Real-time Updates**: Live data feeds for dashboard updates
- **Data Warehouse**: Centralized storage for historical analysis data
- **Machine Learning**: Use captured data for model training
- **Compliance**: Audit trails for regulatory requirements

---

## 🎯 Key Features Summary

✅ **Automatic Data Capture**: Market data and news feeds saved automatically  
✅ **Complete Traceability**: Full audit trail from data to decisions  
✅ **Frontend Ready**: JSON files optimized for dashboard consumption  
✅ **Error Resilient**: Graceful handling of capture failures  
✅ **Performance Optimized**: Minimal overhead on analysis speed  
✅ **Backward Compatible**: Existing code continues to work  
✅ **ZZSheep Integration**: Automatic saving to zzsheepTrader project  

The enhanced data capture features provide complete transparency in the TradingAgents analysis process, enabling better decision-making, debugging, and frontend integration while maintaining the system's speed and reliability. 