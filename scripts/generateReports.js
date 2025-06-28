#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import the stock data
const stockDataPath = path.join(__dirname, '../src/data/stockAnalysis.js');
const stockDataContent = fs.readFileSync(stockDataPath, 'utf8');

// Extract stockData object from the file
const stockDataMatch = stockDataContent.match(/export const stockData = ({[\s\S]*?});/);
if (!stockDataMatch) {
    console.error('Could not extract stockData from file');
    process.exit(1);
}

const stockData = eval(`(${stockDataMatch[1]})`);

function generateReportHTML(stock) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${stock.companyName} (${stock.ticker}) Comprehensive Analysis | zzsheepTrader</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #667eea;
            --primary-light: #764ba2;
            --bg-primary: #1a1a3a;
            --bg-secondary: #2d2d5a;
            --bg-card: #353575;
            --text-primary: #ffffff;
            --text-secondary: #b8b8d1;
            --border-color: #4a4a7a;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --hold-color: #f59e0b;
            --buy-color: #10b981;
            --sell-color: #ef4444;
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--bg-primary);
            min-height: 100vh;
        }

        .header {
            background: rgba(26, 26, 58, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 0;
            position: sticky;
            top: 0;
            z-index: 1000;
            border-bottom: 1px solid var(--border-color);
        }

        .header-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            text-decoration: none;
        }

        .logo-icon {
            font-size: 2rem;
        }

        .nav-link {
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        .nav-link:hover, .nav-link.active {
            color: var(--text-primary);
            background: rgba(102, 126, 234, 0.2);
        }

        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .report-header {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
        }

        .stock-info {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.5rem;
        }

        .stock-details h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stock-subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }

        .recommendation-badge {
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .recommendation-badge.buy {
            background: rgba(16, 185, 129, 0.1);
            color: var(--buy-color);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .recommendation-badge.hold {
            background: rgba(245, 158, 11, 0.1);
            color: var(--hold-color);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .recommendation-badge.sell {
            background: rgba(239, 68, 68, 0.1);
            color: var(--sell-color);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .key-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .metric-card {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--border-color);
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .metric-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .analysis-section {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
        }

        .section-title {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .section-icon {
            background: var(--primary-gradient);
            padding: 0.5rem;
            border-radius: 8px;
            font-size: 1rem;
        }

        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }

        .analysis-card {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }

        .tag-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .tag {
            background: rgba(102, 126, 234, 0.2);
            color: var(--primary-color);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }

        .risk-tag {
            background: rgba(245, 158, 11, 0.2);
            color: var(--warning-color);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .risk-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .risk-score {
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
        }

        .risk-score.low {
            background: var(--success-color);
        }

        .risk-score.medium {
            background: var(--warning-color);
        }

        .risk-score.high {
            background: var(--danger-color);
        }

        .debate-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 1.5rem;
        }

        .bull-case {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 1.5rem;
            border-radius: 12px;
        }

        .bear-case {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 1.5rem;
            border-radius: 12px;
        }

        .case-title {
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .bull-case .case-title {
            color: var(--buy-color);
        }

        .bear-case .case-title {
            color: var(--sell-color);
        }

        .back-button {
            background: var(--primary-gradient);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 2rem;
        }

        .back-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        .status-completed {
            color: var(--success-color);
            font-weight: 600;
        }

        .buy-summary {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 2rem;
            border-radius: 12px;
            margin-top: 1.5rem;
        }

        .buy-summary h3 {
            color: var(--buy-color);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .consensus-box {
            margin-top: 1.5rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        @media (max-width: 768px) {
            .stock-info {
                flex-direction: column;
                gap: 1rem;
            }

            .debate-section {
                grid-template-columns: 1fr;
            }

            .main-container {
                padding: 1rem;
            }

            .analysis-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-container">
            <a href="../index.html" class="logo">
                <div class="logo-icon">🐑</div>
                zzsheepTrader
            </a>
            <nav>
                <a href="../index.html" class="nav-link">← Back to Home</a>
            </nav>
        </div>
    </header>

    <div class="main-container">
        <a href="../index.html" class="back-button">
            <i class="fas fa-arrow-left"></i>
            Back to Dashboard
        </a>

        <!-- Report Header -->
        <div class="report-header">
            <div class="stock-info">
                <div class="stock-details">
                    <h1>${stock.ticker}</h1>
                    <div class="stock-subtitle">${stock.companyName} - Comprehensive Analysis Report</div>
                    <div class="stock-subtitle">Analysis Date: ${stock.analysisDate} • Analysis ID: ${stock.analysisId}</div>
                </div>
                <div class="recommendation-badge ${stock.recommendation.style}">
                    <i class="${stock.recommendation.icon}"></i>
                    ${stock.recommendation.type}
                </div>
            </div>

            <div class="key-metrics">
                <div class="metric-card">
                    <div class="metric-value">${stock.marketData.currentPrice}</div>
                    <div class="metric-label">Current Price</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${stock.marketData.dailyChange}</div>
                    <div class="metric-label">Daily Change</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${stock.metrics.riskScore}/10</div>
                    <div class="metric-label">Risk Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${stock.marketData.marketCap}</div>
                    <div class="metric-label">Market Cap</div>
                </div>
            </div>
        </div>

        ${stock.recommendation.type === 'BUY' ? `
        <!-- Executive Summary -->
        <div class="analysis-section">
            <h2 class="section-title">
                <div class="section-icon"><i class="fas fa-star"></i></div>
                Executive Summary & ${stock.recommendation.type} Recommendation
            </h2>
            <div class="buy-summary">
                <h3>
                    <i class="fas fa-thumbs-up"></i>
                    Strong ${stock.recommendation.type} Recommendation
                </h3>
                <p>
                    <strong>${stock.companyName} (${stock.ticker})</strong> presents a compelling investment opportunity with strong fundamentals and positive growth outlook.
                </p>
            </div>
        </div>
        ` : ''}

        <!-- Market & Technical Analysis Section -->
        <div class="analysis-section">
            <h2 class="section-title">
                <div class="section-icon"><i class="fas fa-chart-line"></i></div>
                Market & Technical Analysis
            </h2>
            <div class="analysis-grid">
                <div class="analysis-card">
                    <div class="card-title">Technical Indicators</div>
                    <p><strong>Status:</strong> <span class="status-completed">${stock.technicalAnalysis.status}</span></p>
                    <p><strong>Trend:</strong> ${stock.technicalAnalysis.trend}</p>
                    <p><strong>Momentum:</strong> ${stock.technicalAnalysis.momentum}</p>
                    <p><strong>Volume:</strong> ${stock.marketData.volume}</p>
                    <p><strong>Volatility:</strong> ${stock.marketData.volatility}</p>
                    <div class="tag-list">
                        ${stock.technicalAnalysis.indicators.map(indicator => `<span class="tag">${indicator}</span>`).join('')}
                    </div>
                </div>
                <div class="analysis-card">
                    <div class="card-title">Price Levels & Performance</div>
                    <p><strong>Current Price:</strong> ${stock.marketData.currentPrice}</p>
                    <p><strong>Daily Change:</strong> ${stock.marketData.dailyChange}</p>
                    <p><strong>Market Cap:</strong> ${stock.marketData.marketCap}</p>
                    <p><strong>Support Level:</strong> ${stock.technicalAnalysis.supportLevel}</p>
                    <p><strong>Resistance Level:</strong> ${stock.technicalAnalysis.resistanceLevel}</p>
                </div>
            </div>
        </div>

        <!-- Fundamental Analysis Section -->
        <div class="analysis-section">
            <h2 class="section-title">
                <div class="section-icon"><i class="fas fa-building"></i></div>
                Fundamental Analysis
            </h2>
            <div class="analysis-grid">
                <div class="analysis-card">
                    <div class="card-title">Key Metrics</div>
                    <p><strong>Status:</strong> <span class="status-completed">${stock.fundamentalAnalysis.status}</span></p>
                    <p><strong>Sector:</strong> ${stock.sector}</p>
                    <p>${stock.fundamentalAnalysis.description}</p>
                    <div class="tag-list">
                        ${stock.fundamentalAnalysis.keyMetrics.map(metric => `<span class="tag">${metric}</span>`).join('')}
                    </div>
                </div>
                <div class="analysis-card">
                    <div class="card-title">Market Sentiment</div>
                    <p><strong>Status:</strong> <span class="status-completed">${stock.fundamentalAnalysis.status}</span></p>
                    <p><strong>Overall Sentiment:</strong> ${stock.fundamentalAnalysis.sentiment}</p>
                    <p>Market sentiment analysis reflects current investor confidence and outlook for ${stock.companyName}'s strategic initiatives and market position.</p>
                </div>
            </div>
        </div>

        <!-- Risk Assessment Section -->
        <div class="analysis-section">
            <h2 class="section-title">
                <div class="section-icon"><i class="fas fa-shield-alt"></i></div>
                Risk Assessment
            </h2>
            <div class="analysis-grid">
                <div class="analysis-card">
                    <div class="card-title">Risk Profile</div>
                    <p><strong>Overall Risk:</strong> ${stock.riskAssessment.overallRisk}</p>
                    <div class="risk-indicator">
                        <span>Risk Score:</span>
                        <div class="risk-score ${stock.metrics.riskScore <= 4 ? 'low' : stock.metrics.riskScore <= 7 ? 'medium' : 'high'}">${stock.metrics.riskScore}/10</div>
                    </div>
                    <div class="tag-list">
                        ${stock.riskAssessment.riskFactors.map(factor => `<span class="tag risk-tag">${factor}</span>`).join('')}
                    </div>
                </div>
                <div class="analysis-card">
                    <div class="card-title">Risk Mitigation</div>
                    <p>Recommended strategies to manage investment risk:</p>
                    <ul style="margin-top: 1rem; color: var(--text-secondary); line-height: 1.8; padding-left: 1rem;">
                        ${stock.riskAssessment.mitigation.map(strategy => `<li>${strategy}</li>`).join('')}
                    </ul>
                </div>
            </div>
        </div>

        <!-- Investment Debate Section -->
        <div class="analysis-section">
            <h2 class="section-title">
                <div class="section-icon"><i class="fas fa-balance-scale"></i></div>
                Investment Debate Analysis
            </h2>
            <div class="debate-section">
                <div class="bull-case">
                    <div class="case-title">
                        <i class="fas fa-arrow-up"></i>
                        Bull Case
                    </div>
                    <p>${stock.investmentDebate.bullCase}</p>
                </div>
                <div class="bear-case">
                    <div class="case-title">
                        <i class="fas fa-arrow-down"></i>
                        Bear Case
                    </div>
                    <p>${stock.investmentDebate.bearCase}</p>
                </div>
            </div>
            <div class="consensus-box">
                <strong>Consensus:</strong> ${stock.investmentDebate.consensus}
            </div>
        </div>

        <!-- Strategic Actions Section -->
        <div class="analysis-section">
            <h2 class="section-title">
                <div class="section-icon"><i class="fas fa-tasks"></i></div>
                Strategic Actions & Monitoring
            </h2>
            <div class="analysis-grid">
                <div class="analysis-card">
                    <div class="card-title">Immediate Actions</div>
                    <ul style="color: var(--text-secondary); line-height: 1.8; margin-top: 1rem; padding-left: 1rem;">
                        ${stock.strategicActions.immediate.map(action => `<li>${action}</li>`).join('')}
                    </ul>
                </div>
                <div class="analysis-card">
                    <div class="card-title">Medium-term Actions</div>
                    <ul style="color: var(--text-secondary); line-height: 1.8; margin-top: 1rem; padding-left: 1rem;">
                        ${stock.strategicActions.mediumTerm.map(action => `<li>${action}</li>`).join('')}
                    </ul>
                </div>
                <div class="analysis-card">
                    <div class="card-title">Key Monitoring Metrics</div>
                    <div class="tag-list">
                        ${stock.strategicActions.monitoring.map(metric => `<span class="tag">${metric}</span>`).join('')}
                    </div>
                </div>
            </div>
        </div>

        <!-- Performance Metrics Section -->
        <div class="analysis-section">
            <h2 class="section-title">
                <div class="section-icon"><i class="fas fa-tachometer-alt"></i></div>
                Analysis Performance Metrics
            </h2>
            <div class="analysis-grid">
                <div class="analysis-card">
                    <div class="card-title">Analysis Quality</div>
                    <p><strong>Data Source:</strong> ${stock.metrics.dataSource}</p>
                    <p><strong>Data Quality:</strong> ${stock.analysisQuality.dataQuality}</p>
                    <p><strong>Reliability Score:</strong> ${stock.analysisQuality.reliabilityScore}</p>
                    <p><strong>Cost Efficiency:</strong> ${stock.analysisQuality.costEfficiency}</p>
                </div>
                <div class="analysis-card">
                    <div class="card-title">Technical Details</div>
                    <p><strong>Analysis Version:</strong> ${stock.analysisQuality.version}</p>
                    <p><strong>Timestamp:</strong> ${stock.timestamp}</p>
                    <p><strong>Processing Time:</strong> ${stock.metrics.duration}</p>
                    <p><strong>AI Model:</strong> ${stock.metrics.aiModel}</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>`;
}

// Generate reports for all stocks
const reportsDir = path.join(__dirname, '../reports');

// Create reports directory if it doesn't exist
if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
}

// Generate a report for each stock
Object.keys(stockData).forEach(ticker => {
    const stock = stockData[ticker];
    const htmlContent = generateReportHTML(stock);
    const fileName = `${ticker.toLowerCase()}_comprehensive_report.html`;
    const filePath = path.join(reportsDir, fileName);
    
    fs.writeFileSync(filePath, htmlContent);
    console.log(`✅ Generated ${fileName}`);
});

console.log(`\n🎉 Successfully generated ${Object.keys(stockData).length} comprehensive reports!`); 