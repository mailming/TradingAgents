"""
JSON Export Utilities for TradingAgents Analysis Results

This module provides utilities for saving analysis results to JSON files
in the zzsheepTrader project for frontend consumption.

Features:
- Automatic directory creation
- Consistent file naming conventions
- Metadata enrichment
- Error handling
- Cross-project file management
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import uuid


class ZZSheepJSONExporter:
    """JSON exporter that saves results to zzsheepTrader project"""
    
    def __init__(self):
        """Initialize the exporter with paths to zzsheepTrader project"""
        # Get the parent directory of TradingAgents (Workspace)
        current_dir = Path(__file__).parent.parent.parent  # Go up from tradingagents/dataflows/
        workspace_dir = current_dir.parent
        
        # Path to zzsheepTrader project
        self.zzsheep_project_dir = workspace_dir / "zzsheepTrader"
        self.analysis_results_dir = self.zzsheep_project_dir / "analysis_results"
        self.json_dir = self.analysis_results_dir / "json"
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        try:
            self.analysis_results_dir.mkdir(parents=True, exist_ok=True)
            self.json_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Export directories ready: {self.json_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create directories: {e}")
            print(f"   Falling back to local analysis_results/")
            # Fallback to local directory
            local_results = Path("analysis_results")
            self.json_dir = local_results / "json"
            self.json_dir.mkdir(parents=True, exist_ok=True)
    
    def save_analysis_results(
        self, 
        results: Dict[str, Any], 
        ticker: Optional[str] = None,
        analysis_type: str = "trading_analysis",
        custom_filename: Optional[str] = None
    ) -> Path:
        """
        Save analysis results to JSON file in zzsheepTrader project
        
        Args:
            results: Dictionary containing analysis results
            ticker: Stock ticker (extracted from results if not provided)
            analysis_type: Type of analysis (for filename)
            custom_filename: Custom filename (overrides auto-generation)
        
        Returns:
            Path to saved JSON file
        """
        
        # Extract ticker from results if not provided
        if not ticker:
            ticker = self._extract_ticker(results)
        
        # Generate filename
        if custom_filename:
            filename = custom_filename if custom_filename.endswith('.json') else f"{custom_filename}.json"
        else:
            filename = self._generate_filename(ticker, analysis_type, results)
        
        # Full path to save file
        json_path = self.json_dir / filename
        
        # Enrich results with export metadata
        enriched_results = self._add_export_metadata(results, ticker, analysis_type)
        
        try:
            # Save JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_results, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Analysis results saved to zzsheepTrader:")
            print(f"   📄 File: {json_path}")
            print(f"   🎯 Ticker: {ticker}")
            print(f"   📊 Type: {analysis_type}")
            print(f"   🌐 Ready for frontend consumption")
            
            return json_path
            
        except Exception as e:
            print(f"❌ Error saving JSON file: {e}")
            raise
    
    def save_multi_day_analysis(
        self, 
        daily_results: list, 
        summary_data: Dict[str, Any],
        ticker: str,
        period_description: str = "multi_day"
    ) -> tuple[list, Path]:
        """
        Save multiple daily analysis results and a summary
        
        Args:
            daily_results: List of daily analysis result dictionaries
            summary_data: Summary analysis data
            ticker: Stock ticker
            period_description: Description of the analysis period
        
        Returns:
            Tuple of (list of daily file paths, summary file path)
        """
        
        daily_paths = []
        
        # Save each daily result
        for i, daily_result in enumerate(daily_results):
            try:
                analysis_date = daily_result.get("analysis_metadata", {}).get("analysis_date", f"day_{i+1}")
                filename = f"{ticker}_{analysis_date}_{str(uuid.uuid4())[:8]}_daily.json"
                
                daily_path = self.save_analysis_results(
                    daily_result, 
                    ticker=ticker,
                    analysis_type="daily_analysis",
                    custom_filename=filename
                )
                daily_paths.append(daily_path)
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to save daily result {i+1}: {e}")
        
        # Save summary report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f"{ticker}_{period_description}_summary_{timestamp}.json"
        
        summary_path = self.save_analysis_results(
            summary_data,
            ticker=ticker,
            analysis_type="summary_report",
            custom_filename=summary_filename
        )
        
        print(f"📋 Multi-day analysis complete:")
        print(f"   📅 Daily files: {len(daily_paths)}")
        print(f"   📊 Summary: {summary_path.name}")
        
        return daily_paths, summary_path
    
    def _extract_ticker(self, results: Dict[str, Any]) -> str:
        """Extract ticker from results metadata"""
        
        # Try different possible locations for ticker
        possible_paths = [
            ["analysis_metadata", "ticker"],
            ["metadata", "ticker"],
            ["ticker"],
            ["symbol"],
            ["company_ticker"]
        ]
        
        for path in possible_paths:
            current = results
            try:
                for key in path:
                    current = current[key]
                if current:
                    return str(current).upper()
            except (KeyError, TypeError):
                continue
        
        # Default if not found
        return "UNKNOWN"
    
    def _generate_filename(self, ticker: str, analysis_type: str, results: Dict[str, Any]) -> str:
        """Generate a consistent filename for the analysis results"""
        
        # Get timestamp
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H-%M-%S")
        
        # Try to get analysis ID from results
        analysis_id = "unknown"
        try:
            analysis_id = results.get("analysis_metadata", {}).get("analysis_id", str(uuid.uuid4())[:8])
        except:
            analysis_id = str(uuid.uuid4())[:8]
        
        # Generate filename
        if analysis_type == "trading_analysis":
            filename = f"{ticker}_{date_str}_{time_str}_{analysis_id}.json"
        else:
            filename = f"{ticker}_{analysis_type}_{date_str}_{time_str}_{analysis_id}.json"
        
        return filename
    
    def _add_export_metadata(self, results: Dict[str, Any], ticker: str, analysis_type: str) -> Dict[str, Any]:
        """Add export metadata to results"""
        
        enriched_results = results.copy()
        
        # Add export metadata
        export_metadata = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "export_destination": "zzsheepTrader",
                "export_path": str(self.json_dir),
                "ticker": ticker,
                "analysis_type": analysis_type,
                "exporter_version": "1.0",
                "ready_for_frontend": True
            }
        }
        
        # Merge with existing results
        enriched_results.update(export_metadata)
        
        return enriched_results
    
    def get_export_info(self) -> Dict[str, Any]:
        """Get information about the export configuration"""
        
        return {
            "zzsheep_project_dir": str(self.zzsheep_project_dir),
            "analysis_results_dir": str(self.analysis_results_dir),
            "json_export_dir": str(self.json_dir),
            "directories_exist": {
                "zzsheep_project": self.zzsheep_project_dir.exists(),
                "analysis_results": self.analysis_results_dir.exists(),
                "json_dir": self.json_dir.exists()
            },
            "example_usage": {
                "basic": "exporter.save_analysis_results(results, ticker='AAPL')",
                "custom": "exporter.save_analysis_results(results, analysis_type='daily', custom_filename='custom.json')",
                "multi_day": "exporter.save_multi_day_analysis(daily_results, summary, 'MSFT', '7day')"
            }
        }


# Convenience functions for easy import and use
def create_exporter() -> ZZSheepJSONExporter:
    """Create and return a new JSON exporter instance"""
    return ZZSheepJSONExporter()


def save_to_zzsheep(results: Dict[str, Any], ticker: str = None, analysis_type: str = "trading_analysis") -> Path:
    """
    Quick function to save results to zzsheepTrader project
    
    Args:
        results: Analysis results dictionary
        ticker: Stock ticker
        analysis_type: Type of analysis
    
    Returns:
        Path to saved file
    """
    exporter = create_exporter()
    return exporter.save_analysis_results(results, ticker, analysis_type)


def save_multi_day_to_zzsheep(daily_results: list, summary: Dict[str, Any], ticker: str, period: str = "multi_day") -> tuple:
    """
    Quick function to save multi-day analysis to zzsheepTrader project
    
    Args:
        daily_results: List of daily analysis results
        summary: Summary analysis data
        ticker: Stock ticker
        period: Period description
    
    Returns:
        Tuple of (daily_paths, summary_path)
    """
    exporter = create_exporter()
    return exporter.save_multi_day_analysis(daily_results, summary, ticker, period)


# Example usage
if __name__ == "__main__":
    # Create exporter and show configuration
    exporter = create_exporter()
    info = exporter.get_export_info()
    
    print("🔧 ZZSheep JSON Exporter Configuration:")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key}: {value}") 