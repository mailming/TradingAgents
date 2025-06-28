# ZZSheep JSON Export Utility

## 🎯 Overview

The ZZSheep JSON Export Utility is a powerful system that automatically saves TradingAgents analysis results to the **zzsheepTrader** project for frontend consumption. This enables seamless integration between the backend AI analysis system and the frontend trading dashboard.

## 📁 File Structure

```
TradingAgents/
├── tradingagents/dataflows/json_export_utils.py  # Main utility module
├── example_zzsheep_export.py                     # Usage examples
├── run_aapl_analysis.py                          # Updated to use ZZSheep exporter
├── run_msft_7day_analysis.py                     # Updated to use ZZSheep exporter
└── ZZSHEEP_JSON_EXPORT_README.md                 # This file

zzsheepTrader/
└── analysis_results/
    └── json/                                      # 🎯 Results saved here!
        ├── AAPL_2024-12-28_analysis.json
        ├── MSFT_2024-12-27_claude.json
        └── summary_reports.json
```

## ✨ Features

- **🚀 Automatic Cross-Project Saving**: Results saved directly to zzsheepTrader project
- **📝 Consistent File Naming**: Standardized filename conventions across all analyses
- **🔧 Flexible Configuration**: Support for custom filenames and analysis types
- **📊 Multi-Day Analysis Support**: Handle complex multi-day trading analysis workflows
- **🛡️ Error Handling**: Graceful fallback to local directories if zzsheepTrader not available
- **📈 Metadata Enrichment**: Automatic addition of export metadata for tracking
- **🎨 Frontend Ready**: JSON structure optimized for frontend consumption

## 🚀 Quick Start

### Basic Usage

```python
from tradingagents.dataflows.json_export_utils import save_to_zzsheep

# Your analysis results
results = {
    "analysis_metadata": {
        "ticker": "AAPL",
        "analysis_date": "2024-12-28",
        # ... other metadata
    },
    "final_decision": {
        "recommendation": "BUY",
        # ... decision details
    }
}

# Save to zzsheepTrader project
saved_path = save_to_zzsheep(results, ticker="AAPL")
print(f"Results saved to: {saved_path}")
```

### Advanced Usage

```python
from tradingagents.dataflows.json_export_utils import create_exporter

# Create exporter instance
exporter = create_exporter()

# Save with custom options
saved_path = exporter.save_analysis_results(
    results,
    ticker="TSLA",
    analysis_type="momentum_analysis",
    custom_filename="TSLA_special_analysis.json"
)
```

### Multi-Day Analysis

```python
from tradingagents.dataflows.json_export_utils import save_multi_day_to_zzsheep

# Daily results list
daily_results = [day1_results, day2_results, day3_results]

# Summary data
summary = {
    "summary_metadata": {
        "ticker": "MSFT",
        "analysis_period": {"start_date": "2024-12-25", "end_date": "2024-12-27"}
    },
    "daily_results": [...],
    "analysis_summary": {...}
}

# Save multi-day analysis
daily_paths, summary_path = save_multi_day_to_zzsheep(
    daily_results, 
    summary, 
    "MSFT", 
    "7day_momentum"
)
```

## 📊 File Naming Conventions

The utility uses consistent naming patterns:

| Analysis Type | Filename Format | Example |
|---------------|----------------|---------|
| **Single Analysis** | `{TICKER}_{DATE}_{TIME}_{ID}.json` | `AAPL_2024-12-28_14-30-15_abc123.json` |
| **Custom Analysis** | `{TICKER}_{TYPE}_{DATE}_{TIME}_{ID}.json` | `TSLA_momentum_2024-12-28_14-30-15_def456.json` |
| **Daily Analysis** | `{TICKER}_{DATE}_{ID}_daily.json` | `MSFT_2024-12-27_ghi789_daily.json` |
| **Summary Report** | `{TICKER}_{PERIOD}_summary_{TIMESTAMP}.json` | `MSFT_7day_summary_20241228_143015.json` |
| **Claude Analysis** | `{TICKER}_{DATE}_{ID}_claude.json` | `NVDA_2024-12-28_jkl012_claude.json` |

## 🔧 Integration Examples

### Update Existing Analysis Scripts

**Before:**
```python
# Old local saving
def save_analysis_results(results):
    output_dir = Path("analysis_results")
    json_dir = output_dir / "json"
    # ... local file operations
```

**After:**
```python
# New ZZSheep exporter
from tradingagents.dataflows.json_export_utils import save_to_zzsheep

# In your analysis function
saved_path = save_to_zzsheep(results, ticker="AAPL", analysis_type="momentum")
```

### Updated Scripts

The following analysis scripts have been updated to use ZZSheep exporter:

1. **`run_aapl_analysis.py`** → Uses `save_to_zzsheep()` for single AAPL analysis
2. **`run_msft_7day_analysis.py`** → Uses `create_exporter()` for multi-day Claude analysis

## 📋 JSON Structure Enhancement

The exporter automatically adds metadata to your results:

```json
{
  "analysis_metadata": {
    "ticker": "AAPL",
    "analysis_date": "2024-12-28",
    // ... your original metadata
  },
  "final_decision": {
    // ... your analysis results
  },
  "export_info": {
    "exported_at": "2024-12-28T14:30:15.123456",
    "export_destination": "zzsheepTrader",
    "export_path": "/Users/user/Workspace/zzsheepTrader/analysis_results/json",
    "ticker": "AAPL",
    "analysis_type": "trading_analysis",
    "exporter_version": "1.0",
    "ready_for_frontend": true
  }
}
```

## 🛠️ Configuration

### Path Detection

The utility automatically detects the zzsheepTrader project:

```python
# Automatic path resolution
workspace_dir = current_dir.parent  # Go up to Workspace
zzsheep_project = workspace_dir / "zzsheepTrader"
export_dir = zzsheep_project / "analysis_results" / "json"
```

### Fallback Behavior

If zzsheepTrader project is not found:
- ⚠️ Warning message displayed
- 🔄 Automatic fallback to local `analysis_results/json/`
- ✅ Analysis continues without interruption

## 🧪 Testing

Run the example script to test the utility:

```bash
python example_zzsheep_export.py
```

This will:
1. Show export configuration
2. Create sample analysis results
3. Test basic, advanced, and multi-day exports
4. Verify files are saved to zzsheepTrader project

## 🔍 Verification

Check that files are saved correctly:

```bash
# List recent files in zzsheepTrader
ls -la ../zzsheepTrader/analysis_results/json/ | tail -10

# View configuration
python -c "from tradingagents.dataflows.json_export_utils import create_exporter; exporter = create_exporter(); print(exporter.get_export_info())"
```

## 🎯 Benefits

### For Backend Development
- **🔗 Seamless Integration**: No manual file copying between projects
- **📝 Consistent Output**: Standardized JSON structure for all analyses
- **🔧 Easy Integration**: Simple imports and function calls
- **🛡️ Error Resilience**: Graceful handling of missing directories

### For Frontend Development
- **📊 Real-time Data**: Fresh analysis results automatically available
- **🎨 Optimized Structure**: JSON format designed for frontend consumption
- **📈 Rich Metadata**: Export information and analysis tracking
- **🔄 Automatic Updates**: Latest results always available in expected location

## 📚 API Reference

### Classes

#### `ZZSheepJSONExporter`
Main exporter class with full functionality.

**Methods:**
- `save_analysis_results(results, ticker, analysis_type, custom_filename)` → Path
- `save_multi_day_analysis(daily_results, summary_data, ticker, period)` → (list[Path], Path)
- `get_export_info()` → Dict[str, Any]

### Functions

#### `create_exporter() → ZZSheepJSONExporter`
Create and return a new exporter instance.

#### `save_to_zzsheep(results, ticker, analysis_type) → Path`
Quick function for single analysis results.

#### `save_multi_day_to_zzsheep(daily_results, summary, ticker, period) → tuple`
Quick function for multi-day analysis results.

## 🚀 Future Enhancements

- **📡 Real-time Synchronization**: Live updates to frontend
- **🔄 Version Control**: Analysis result versioning and history
- **📊 Batch Operations**: Bulk analysis result processing
- **🎨 Custom Templates**: Configurable JSON output formats
- **📈 Analytics**: Export and usage statistics

---

## 💡 Usage Tips

1. **Always specify ticker**: Even if it's in metadata, explicit ticker parameter ensures correct categorization
2. **Use descriptive analysis_type**: Helps organize different types of analyses in the frontend
3. **Custom filenames for special cases**: Use when you need specific naming for particular analyses
4. **Check export_info**: Use `get_export_info()` to verify configuration before running large analyses
5. **Test with examples**: Run `example_zzsheep_export.py` when setting up a new environment

## 🤝 Integration with Analysis Scripts

The utility is designed to be a drop-in replacement for local JSON saving:

```python
# Replace this pattern:
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

# With this:
save_to_zzsheep(results, ticker="AAPL")
```

This ensures all TradingAgents analysis results are automatically available to the zzsheepTrader frontend for visualization and decision making. 