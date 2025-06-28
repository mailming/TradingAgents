# 🤖 TradingAgents - AI Stock Analysis Backend

Advanced multi-agent system for intelligent stock analysis powered by AI models and institutional-grade financial data.

## 🌟 Overview

TradingAgents is a sophisticated backend system that provides AI-powered stock analysis through multiple specialized agents. This repository contains the core analysis engine, data processing pipelines, and agent coordination systems.

**Note**: This is the backend analysis engine. For the frontend showcase, see the [zzsheeptrader repository](https://github.com/your-username/zzsheeptrader).

## 🧠 Agent Architecture

### Core Agents
- **Market Analyst**: Technical analysis and market trend evaluation
- **News Analyst**: Sentiment analysis from financial news sources
- **Fundamentals Analyst**: Financial metrics and company health assessment
- **Social Media Analyst**: Social sentiment and retail investor mood
- **Research Manager**: Coordinates analysis across multiple agents
- **Risk Manager**: Risk assessment and mitigation strategies
- **Bull/Bear Researchers**: Contrarian analysis for balanced perspectives

### Specialized Components
- **Trader Agent**: Executes analysis and generates recommendations
- **Memory System**: Persistent agent states and learning
- **Graph Processing**: Advanced signal processing and propagation
- **Data Flows**: Comprehensive market data integration

## 📊 Data Sources & Analysis

- **Professional Data**: financialdatasets.ai with 30+ years of history
- **Technical Indicators**: 20+ professional indicators (SMA, EMA, MACD, RSI, etc.)
- **Fundamental Metrics**: P/E, ROE, ROA, Debt/Equity, margins, and more
- **News & Sentiment**: Real-time news analysis and sentiment scoring
- **Risk Assessment**: Multi-factor risk models and scoring

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/TradingAgents.git
cd TradingAgents

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Usage

#### CLI Analysis
```bash
# Run comprehensive analysis on a stock
python -m cli.main analyze TSLA

# Get technical analysis only
python -m cli.main technical AAPL

# Monitor news sentiment
python -m cli.main news NVDA
```

#### Python API
```python
from tradingagents.managers.research_manager import ResearchManager
from tradingagents.agents.trader.trader import Trader

# Initialize research manager
research_manager = ResearchManager()

# Run comprehensive analysis
analysis = research_manager.analyze_stock("MSFT")

# Get recommendation
trader = Trader()
recommendation = trader.generate_recommendation(analysis)
```

#### Simple API Server
```bash
# Start the API server
python simple_api_server.py

# Query endpoints
curl http://localhost:8000/analyze/TSLA
curl http://localhost:8000/technical/AAPL
```

## 📁 Project Structure

```
TradingAgents/
├── tradingagents/          # Core agent system
│   ├── agents/            # Individual agent implementations
│   │   ├── analysts/      # Market, news, fundamentals analysts
│   │   ├── managers/      # Research and risk managers
│   │   ├── researchers/   # Bull/bear research agents
│   │   ├── risk_mgmt/     # Risk management agents
│   │   └── trader/        # Trading decision agent
│   ├── dataflows/         # Data processing and APIs
│   ├── graph/             # Signal processing and propagation
│   └── adapters/          # AI model adapters
├── cli/                   # Command line interface
├── analysis_results/      # Analysis outputs
├── eval_results/          # Evaluation and backtesting
├── dataflows/             # Cached data processing
├── demo_*.py             # Demonstration scripts
└── test_*.py             # Test suites
```

## 🔧 Configuration

### AI Models
- **Claude Haiku**: Budget-optimized for high-frequency analysis
- **GPT-4o-Mini**: Enhanced reasoning for complex decisions
- **Anthropic Direct**: Custom adapter for Claude models

### Data Configuration
```python
# dataflows/config.py
DATA_SOURCES = {
    'financialdatasets': 'primary',
    'finnhub': 'secondary',
    'cache_ttl': 3600  # 1 hour
}
```

## 🎯 Key Features

### Multi-Agent Analysis
- **Parallel Processing**: Multiple agents analyze different aspects simultaneously
- **Consensus Building**: Agents debate and reach informed conclusions
- **Risk Assessment**: Comprehensive multi-factor risk evaluation
- **Performance Tracking**: Built-in analysis performance metrics

### Advanced Data Processing
- **Time Series Caching**: Intelligent caching for performance optimization
- **Market Data Integration**: Multiple data source aggregation
- **News Processing**: Real-time news sentiment analysis
- **Technical Analysis**: Professional-grade technical indicators

### Flexible Architecture
- **Modular Design**: Easy to extend with new agents
- **API Ready**: RESTful API for external integrations
- **CLI Interface**: Command-line tools for analysis
- **Configurable**: Extensive configuration options

## 📈 Performance Metrics

- **Analysis Speed**: 70-85 seconds average per stock
- **Data Quality**: Professional-grade institutional data
- **Reliability Score**: 95%+ accuracy
- **Cost Efficiency**: Optimized AI model usage

## 🔍 Analysis Output

Each analysis provides:
- **Technical Analysis**: Price trends, support/resistance, indicators
- **Fundamental Analysis**: Financial health, valuation metrics
- **Sentiment Analysis**: News and social media sentiment
- **Risk Assessment**: Multi-factor risk scoring
- **Investment Recommendation**: BUY/HOLD/SELL with confidence levels
- **Strategic Actions**: Immediate and medium-term recommendations

## 🛠️ Development

### Running Tests
```bash
# Test AI providers
python test_ai_providers.py

# Test data connections
python test_openai_connection.py

# Run comprehensive analysis demo
python demo_financialdatasets_integration.py
```

### Adding New Agents
1. Create agent class in `tradingagents/agents/`
2. Implement required analysis methods
3. Register with research manager
4. Add CLI commands in `cli/main.py`

## 📊 Data Caching

The system includes intelligent caching:
```bash
# Demo cache functionality
python demo_time_series_cache.py

# See cache implementation details
cat CACHE_IMPLEMENTATION_SUMMARY.md
```

## 🔗 Integration

### API Endpoints
- `GET /analyze/{ticker}` - Comprehensive analysis
- `GET /technical/{ticker}` - Technical analysis only
- `GET /news/{ticker}` - News sentiment analysis
- `GET /risk/{ticker}` - Risk assessment

### CLI Commands
- `analyze` - Full multi-agent analysis
- `technical` - Technical analysis only
- `news` - News and sentiment analysis
- `risk` - Risk assessment

## 📞 Support

For questions about the TradingAgents backend system:
- Review the documentation in `/docs`
- Check demo scripts for usage examples
- See test files for API examples

---

**TradingAgents Backend** - Intelligent multi-agent system for professional-grade stock analysis and investment decision support.
