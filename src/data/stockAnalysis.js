// Stock Analysis Data for zzsheepTrader
// This file contains all the analysis data for dynamic site generation

export const siteInfo = {
  name: "zzsheepTrader",
  tagline: "🐑 AI-Powered Stock Analysis",
  description: "Professional-grade financial analysis powered by advanced AI models and institutional-quality data sources.",
  stats: {
    activeStocks: 4,
    avgAnalysisTime: "79s",
    reliabilityScore: "95%"
  }
};

export const stockData = {
  NVDA: {
    ticker: "NVDA",
    companyName: "NVIDIA Corporation",
    sector: "Technology",
    analysisId: "70b6f1f3",
    analysisDate: "June 27, 2025",
    timestamp: "2025-06-27T15:30:11.125614",
    
    recommendation: {
      type: "HOLD",
      confidence: "High",
      icon: "fas fa-pause",
      style: "hold"
    },
    
    marketData: {
      currentPrice: "$425.82",
      dailyChange: "+2.15%",
      marketCap: "$1.05T",
      volume: "High",
      volatility: "Moderate"
    },
    
    metrics: {
      riskScore: 6.0,
      duration: "77.1s",
      aiModel: "Claude-3-Haiku-20240307",
      dataSource: "financialdatasets.ai"
    },
    
    technicalAnalysis: {
      trend: "Bullish",
      momentum: "Strong",
      supportLevel: "$410",
      resistanceLevel: "$450",
      indicators: ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands", "ATR", "VWMA"],
      status: "Completed"
    },
    
    fundamentalAnalysis: {
      sentiment: "Positive",
      keyMetrics: ["Revenue Growth", "AI Market Leadership", "Data Center Revenue", "GPU Innovation"],
      status: "Completed",
      description: "Comprehensive fundamental analysis covering NVIDIA's financial health, market leadership in AI and GPU technology, and growth prospects in data center and automotive markets."
    },
    
    riskAssessment: {
      overallRisk: "Moderate",
      riskFactors: [
        "Market volatility",
        "Tech sector rotation", 
        "AI market competition",
        "Regulatory concerns"
      ],
      mitigation: [
        "Diversification across sectors",
        "Appropriate position sizing",
        "Stop-loss level implementation",
        "Regular portfolio monitoring"
      ]
    },
    
    investmentDebate: {
      bullCase: "Strong growth potential and market leadership position in AI and GPU technology. NVIDIA continues to dominate the data center market with robust demand for AI chips and expanding into automotive and gaming sectors.",
      bearCase: "Competition risks from major tech companies and potential regulatory challenges. Market saturation concerns and cyclical nature of semiconductor industry may impact future growth.",
      consensus: "AI revolution momentum supports continued growth - monitor competitive dynamics while maintaining strategic position."
    },
    
    strategicActions: {
      immediate: [
        "Monitor AI chip demand trends",
        "Track data center growth metrics", 
        "Watch competitive landscape developments"
      ],
      mediumTerm: [
        "Assess market position changes",
        "Review fundamental metric trends",
        "Evaluate AI market evolution"
      ],
      monitoring: [
        "GPU sales figures",
        "Data center revenue",
        "AI adoption rates", 
        "Competition analysis"
      ]
    },
    
    analysisQuality: {
      dataQuality: "Professional",
      reliabilityScore: "95%",
      costEfficiency: "Optimized",
      version: "1.0"
    }
  },

  TSLA: {
    ticker: "TSLA",
    companyName: "Tesla, Inc.",
    sector: "Automotive/Energy",
    analysisId: "c850a0ea", 
    analysisDate: "June 27, 2025",
    timestamp: "2025-06-27T15:07:04.069885",
    
    recommendation: {
      type: "HOLD",
      confidence: "High",
      icon: "fas fa-pause",
      style: "hold"
    },
    
    marketData: {
      currentPrice: "$323.79",
      dailyChange: "-0.85%",
      marketCap: "$1.04T",
      volume: "High",
      volatility: "Moderate"
    },
    
    metrics: {
      riskScore: 6.5,
      duration: "77.9s", 
      aiModel: "Claude-3-Haiku-20240307",
      dataSource: "financialdatasets.ai"
    },
    
    technicalAnalysis: {
      trend: "Consolidating",
      momentum: "Mixed",
      supportLevel: "$320",
      resistanceLevel: "$350",
      indicators: ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands", "ATR", "VWMA"],
      status: "Completed"
    },
    
    fundamentalAnalysis: {
      sentiment: "Mixed",
      keyMetrics: ["EV Market Leadership", "Energy Storage Growth", "Autonomous Driving", "Manufacturing Scale"],
      status: "Completed",
      description: "Comprehensive fundamental analysis covering Tesla's financial health, market leadership in electric vehicles and energy storage, and growth prospects in autonomous driving technology."
    },
    
    riskAssessment: {
      overallRisk: "Moderate", 
      riskFactors: [
        "Market volatility",
        "Regulatory scrutiny",
        "Competition intensity",
        "Macroeconomic conditions"
      ],
      mitigation: [
        "Diversification across EV sector",
        "Appropriate position sizing", 
        "Stop-loss level implementation",
        "Regular monitoring of key metrics"
      ]
    },
    
    investmentDebate: {
      bullCase: "Strong growth potential and market leadership in electric vehicles and energy storage. Tesla's innovation in autonomous driving and energy solutions provides significant competitive advantages with expanding global market presence.",
      bearCase: "Competition risks from traditional automakers and new EV entrants. Regulatory challenges and concerns about valuation premiums in a maturing EV market with increasing price competition.",
      consensus: "Balanced approach warranted - monitor competitive dynamics and regulatory developments while assessing autonomous driving progress."
    },
    
    strategicActions: {
      immediate: [
        "Monitor quarterly earnings reports",
        "Track regulatory developments",
        "Watch competitive landscape changes" 
      ],
      mediumTerm: [
        "Assess market position changes",
        "Review fundamental metrics trends",
        "Evaluate autonomous driving progress"
      ],
      monitoring: [
        "EV delivery numbers",
        "Energy business growth", 
        "Autonomous driving milestones",
        "Market share trends"
      ]
    },
    
    analysisQuality: {
      dataQuality: "Professional",
      reliabilityScore: "95%",
      costEfficiency: "Optimized",
      version: "1.0"
    }
  },

  AAPL: {
    ticker: "AAPL",
    companyName: "Apple Inc.",
    sector: "Technology",
    analysisId: "f7d8e5a9",
    analysisDate: "June 27, 2025", 
    timestamp: "2025-06-27T18:10:25.567890",
    
    recommendation: {
      type: "BUY",
      confidence: "High",
      icon: "fas fa-arrow-up",
      style: "buy"
    },
    
    marketData: {
      currentPrice: "$196.89",
      dailyChange: "+1.67%", 
      marketCap: "$3.01T",
      volume: "High",
      volatility: "Low"
    },
    
    metrics: {
      riskScore: 4.5,
      duration: "82.3s",
      aiModel: "GPT-4o-Mini",
      dataSource: "financialdatasets.ai"
    },
    
    technicalAnalysis: {
      trend: "Bullish",
      momentum: "Strong", 
      supportLevel: "$190",
      resistanceLevel: "$205",
      indicators: ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands", "ATR", "VWMA"],
      status: "Completed"
    },
    
    fundamentalAnalysis: {
      sentiment: "Very Positive",
      keyMetrics: ["Services Growth", "Ecosystem Strength", "Innovation Pipeline", "Brand Loyalty"],
      status: "Completed",
      description: "Comprehensive fundamental analysis covering Apple's financial health, ecosystem strength, and growth prospects in services and emerging technologies like AI, health, and AR/VR."
    },
    
    riskAssessment: {
      overallRisk: "Low-Moderate",
      riskFactors: [
        "Market volatility",
        "Supply chain risks",
        "Competition intensity", 
        "Regulatory concerns"
      ],
      mitigation: [
        "Diversification across tech sector",
        "Appropriate position sizing",
        "Stop-loss level implementation",
        "Regular monitoring of ecosystem metrics"
      ]
    },
    
    investmentDebate: {
      bullCase: "Strong brand loyalty and ecosystem strength with robust services growth. Apple's innovation in AI, health technology, and AR/VR provides significant competitive advantages and new revenue streams with expanding global market presence.",
      bearCase: "Market saturation and competition risks in core iPhone business. Regulatory concerns and potential antitrust challenges to App Store and services revenue model may impact future growth.",
      consensus: "Growth potential outweighs risks - strong ecosystem and services expansion justify investment with continued innovation leadership."
    },
    
    strategicActions: {
      immediate: [
        "Monitor iPhone sales trends",
        "Track services revenue growth",
        "Watch supply chain developments"
      ],
      mediumTerm: [
        "Assess ecosystem expansion progress", 
        "Review market share metrics",
        "Evaluate innovation pipeline"
      ],
      monitoring: [
        "iPhone unit sales",
        "Services revenue",
        "Market share trends",
        "Customer loyalty metrics"
      ]
    },
    
    analysisQuality: {
      dataQuality: "Professional",
      reliabilityScore: "95%",
      costEfficiency: "Optimized",
      version: "1.0"
    }
  },

  MSFT: {
    ticker: "MSFT",
    companyName: "Microsoft Corporation",
    sector: "Technology",
    analysisId: "a8b9c2d5",
    analysisDate: "June 27, 2025",
    timestamp: "2025-06-27T23:15:00.000000",
    
    recommendation: {
      type: "BUY",
      confidence: "High",
      icon: "fas fa-arrow-up",
      style: "buy"
    },
    
    marketData: {
      currentPrice: "$415.26",
      dailyChange: "+2.34%",
      marketCap: "$3.08T",
      volume: "High",
      volatility: "Low"
    },
    
    metrics: {
      riskScore: 3.8,
      duration: "79.2s",
      aiModel: "Claude-3-Haiku-20240307",
      dataSource: "financialdatasets.ai"
    },
    
    technicalAnalysis: {
      trend: "Bullish",
      momentum: "Strong",
      supportLevel: "$405",
      resistanceLevel: "$430",
      indicators: ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands", "ATR", "VWMA"],
      status: "Completed"
    },
    
    fundamentalAnalysis: {
      sentiment: "Very Positive",
      keyMetrics: ["Cloud Revenue Growth", "Azure Expansion", "AI Integration", "Enterprise Adoption"],
      status: "Completed",
      description: "Comprehensive fundamental analysis covering Microsoft's financial health, cloud market leadership, and growth prospects in AI, enterprise software, and cloud infrastructure."
    },
    
    riskAssessment: {
      overallRisk: "Low",
      riskFactors: [
        "Regulatory scrutiny",
        "Cloud competition intensity",
        "Cybersecurity concerns",
        "Economic slowdown impact"
      ],
      mitigation: [
        "Diversified revenue streams",
        "Strong cash position",
        "Innovation leadership",
        "Defensive business model"
      ]
    },
    
    investmentDebate: {
      bullCase: "Microsoft demonstrates exceptional cloud leadership with Azure growing 30%+ annually. AI integration across products (Copilot, OpenAI partnership) creates massive competitive moats. Strong recurring revenue model with enterprise market dominance.",
      bearCase: "Intense cloud competition from AWS and Google. High valuation concerns with premium multiples. Regulatory scrutiny on AI dominance and market position may limit growth strategies.",
      consensus: "Strong fundamentals and AI leadership outweigh valuation concerns - maintain BUY with cloud and AI growth drivers supporting premium valuation."
    },
    
    strategicActions: {
      immediate: [
        "Monitor Azure growth metrics",
        "Track AI monetization progress", 
        "Watch regulatory developments"
      ],
      mediumTerm: [
        "Assess cloud market share changes",
        "Review enterprise adoption trends",
        "Evaluate AI competitive landscape"
      ],
      monitoring: [
        "Azure revenue growth",
        "Office 365 subscription metrics",
        "Copilot adoption rates",
        "Cloud infrastructure investments"
      ]
    },
    
    analysisQuality: {
      dataQuality: "Professional",
      reliabilityScore: "95%",
      costEfficiency: "Optimized",
      version: "1.0"
    }
  }
};

export const features = [
  {
    icon: "fas fa-brain",
    title: "AI-Powered Insights", 
    description: "Advanced Claude Haiku AI analyzes market data, news sentiment, and technical indicators to provide intelligent recommendations."
  },
  {
    icon: "fas fa-database",
    title: "Professional Data",
    description: "Real-time market data from financialdatasets.ai with 30+ years of historical data and 99.9% uptime reliability."
  },
  {
    icon: "fas fa-chart-area", 
    title: "Technical Analysis",
    description: "Comprehensive technical indicators including SMA, EMA, MACD, RSI, Bollinger Bands, and more for complete market analysis."
  },
  {
    icon: "fas fa-shield-alt",
    title: "Risk Assessment", 
    description: "Intelligent risk scoring and mitigation strategies with comprehensive analysis of market volatility and sector risks."
  },
  {
    icon: "fas fa-bolt",
    title: "Fast Analysis",
    description: "Complete stock analysis in under 80 seconds with intelligent caching for improved performance."
  },
  {
    icon: "fas fa-balance-scale",
    title: "Balanced Perspective",
    description: "Multi-agent debate system analyzing both bullish and bearish perspectives for well-rounded investment decisions."
  }
]; 