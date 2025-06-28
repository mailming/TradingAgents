"""
Example: Using ZZSheep JSON Export Utility

This script demonstrates how to use the ZZSheep JSON export utility
to save trading analysis results to the zzsheepTrader project for
frontend consumption.
"""

from datetime import datetime
import uuid
from tradingagents.dataflows.json_export_utils import (
    create_exporter, 
    save_to_zzsheep, 
    save_multi_day_to_zzsheep
)


def create_sample_analysis_result(ticker="AAPL", analysis_date=None):
    """Create a sample analysis result for demonstration"""
    
    if not analysis_date:
        analysis_date = datetime.now().strftime("%Y-%m-%d")
    
    sample_result = {
        "analysis_metadata": {
            "analysis_id": str(uuid.uuid4())[:8],
            "ticker": ticker,
            "analysis_date": analysis_date,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": 45.3,
            "ai_model": "claude-3-5-sonnet-20241022",
            "data_source": "financialdatasets.ai",
            "version": "1.0"
        },
        "final_decision": {
            "recommendation": "BUY - Strong fundamentals and positive momentum",
            "confidence_level": "High",
            "decision_type": "BUY"
        },
        "analysis_components": {
            "market_analysis": {
                "status": "completed",
                "summary": f"Technical analysis completed for {ticker}",
                "indicators_used": ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands"]
            },
            "fundamental_analysis": {
                "status": "completed",
                "summary": "Strong financial metrics detected",
                "key_metrics": ["Revenue Growth", "Profitability", "Cash Position"]
            }
        },
        "market_data": {
            "current_price": "$150.45",
            "daily_change": "+2.3%",
            "volume": "High",
            "volatility": "Moderate"
        },
        "risk_assessment": {
            "overall_risk": "Moderate",
            "risk_score": 6.2,
            "risk_factors": ["Market volatility", "Competition"]
        }
    }
    
    return sample_result


def example_basic_export():
    """Example 1: Basic export using convenience function"""
    
    print("📝 Example 1: Basic Export")
    print("=" * 40)
    
    # Create sample analysis result
    results = create_sample_analysis_result("AAPL")
    
    # Save using convenience function
    saved_path = save_to_zzsheep(results, ticker="AAPL", analysis_type="demo_analysis")
    
    print(f"✅ Saved to: {saved_path}")
    print()


def example_advanced_export():
    """Example 2: Advanced export using exporter class"""
    
    print("📝 Example 2: Advanced Export with Custom Options")
    print("=" * 50)
    
    # Create exporter instance
    exporter = create_exporter()
    
    # Create sample analysis result
    results = create_sample_analysis_result("TSLA")
    
    # Save with custom filename
    saved_path = exporter.save_analysis_results(
        results,
        ticker="TSLA",
        analysis_type="custom_analysis",
        custom_filename="TSLA_custom_demo_analysis.json"
    )
    
    print(f"✅ Saved to: {saved_path}")
    print()


def example_multi_day_export():
    """Example 3: Multi-day analysis export"""
    
    print("📝 Example 3: Multi-Day Analysis Export")
    print("=" * 45)
    
    # Create multiple daily results
    daily_results = []
    for i in range(3):
        day_date = f"2024-12-{25 + i:02d}"  # Dec 25, 26, 27
        daily_result = create_sample_analysis_result("MSFT", day_date)
        daily_result["analysis_metadata"]["analysis_date"] = day_date
        daily_results.append(daily_result)
    
    # Create summary data
    summary_data = {
        "summary_metadata": {
            "ticker": "MSFT",
            "analysis_period": {
                "start_date": "2024-12-25",
                "end_date": "2024-12-27",
                "trading_days_analyzed": 3
            },
            "ai_model": "claude-3-5-sonnet-20241022",
            "timestamp": datetime.now().isoformat()
        },
        "daily_results": [
            {
                "date": "2024-12-25",
                "decision": "BUY",
                "confidence": "High"
            },
            {
                "date": "2024-12-26", 
                "decision": "HOLD",
                "confidence": "Medium"
            },
            {
                "date": "2024-12-27",
                "decision": "BUY", 
                "confidence": "High"
            }
        ],
        "analysis_summary": {
            "total_analyses": 3,
            "decision_distribution": {
                "BUY": 2,
                "HOLD": 1,
                "SELL": 0
            }
        }
    }
    
    # Save multi-day analysis
    daily_paths, summary_path = save_multi_day_to_zzsheep(
        daily_results, 
        summary_data, 
        "MSFT", 
        "3day_demo"
    )
    
    print(f"✅ Daily files saved: {len(daily_paths)}")
    print(f"✅ Summary saved: {summary_path}")
    print()


def show_export_configuration():
    """Show the current export configuration"""
    
    print("🔧 Export Configuration")
    print("=" * 30)
    
    exporter = create_exporter()
    info = exporter.get_export_info()
    
    print(f"📁 zzsheepTrader Project: {info['zzsheep_project_dir']}")
    print(f"📊 Analysis Results Dir: {info['analysis_results_dir']}")  
    print(f"📄 JSON Export Dir: {info['json_export_dir']}")
    print()
    
    print("✅ Directory Status:")
    for dir_name, exists in info['directories_exist'].items():
        status = "✅" if exists else "❌"
        print(f"   {status} {dir_name}: {exists}")
    print()


def main():
    """Run all examples"""
    
    print("🌟 ZZSheep JSON Export Utility - Examples")
    print("=" * 60)
    print("🎯 This script demonstrates saving analysis results")
    print("   to the zzsheepTrader project for frontend use.")
    print()
    
    # Show configuration
    show_export_configuration()
    
    # Run examples
    example_basic_export()
    example_advanced_export()
    example_multi_day_export()
    
    print("🎉 All examples completed!")
    print("📁 Check the zzsheepTrader/analysis_results/json/ directory")
    print("   to see the saved files.")
    print()
    
    # Show usage tips
    print("💡 Usage Tips:")
    print("=" * 15)
    print("1. Import: from tradingagents.dataflows.json_export_utils import save_to_zzsheep")
    print("2. Basic:  save_to_zzsheep(results, ticker='AAPL')")
    print("3. Custom: exporter.save_analysis_results(results, custom_filename='my_analysis.json')")
    print("4. Multi:  save_multi_day_to_zzsheep(daily_results, summary, 'MSFT')")


if __name__ == "__main__":
    main() 