# Batch Trading Analysis Script

This shell script automates running trading analysis for a specific ticker across multiple days.

## Features

- ✅ Creates and manages virtual environment automatically
- ✅ Loads API keys from `.env` file
- ✅ Runs analysis sequentially for multiple dates
- ✅ Skips non-trading days (weekends/holidays)
- ✅ Colored output with progress tracking
- ✅ Comprehensive error handling and reporting
- ✅ Automatic API retry on overload (30s wait with exponential backoff)
- ✅ Automatic results organization

## Usage

```bash
./run_batch_analysis.sh TICKER DAYS
```

### Examples

```bash
# Analyze AAPL for the last 5 days
./run_batch_analysis.sh AAPL 5

# Analyze MSFT for the last 10 days
./run_batch_analysis.sh MSFT 10

# Analyze GOOGL for the last 7 days
./run_batch_analysis.sh GOOGL 7
```

## Setup

### 1. API Keys Setup

Create a `.env` file in the project root with your API keys:

```bash
# Required API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
FINANCIALDATASETS_API_KEY=your_financialdatasets_api_key_here

# Optional API Keys
OPENAI_API_KEY=your_openai_api_key_here
```

**Get API keys from:**
- [Anthropic Console](https://console.anthropic.com/)
- [FinancialDatasets.ai](https://financialdatasets.ai/)
- [OpenAI Platform](https://platform.openai.com/api-keys)

### 2. Make Script Executable

```bash
chmod +x run_batch_analysis.sh
```

### 3. Run the Script

```bash
./run_batch_analysis.sh TICKER DAYS
```

## What the Script Does

1. **Environment Setup**
   - Creates virtual environment if it doesn't exist
   - Activates virtual environment
   - Installs/updates requirements from `requirements.txt`

2. **API Key Validation**
   - Loads API keys from `.env` file
   - Validates required keys are present
   - Exits with error if keys are missing

3. **Date Generation**
   - Generates list of dates going backwards from yesterday
   - Handles both macOS and Linux date commands

4. **Sequential Analysis**
   - Runs `main.py` for each date with the specified ticker
   - Handles successful runs, skipped runs, and failures
   - Adds 2-second delay between runs to be respectful to APIs

5. **Results Summary**
   - Tracks successful, skipped, and failed analyses
   - Provides comprehensive summary at the end
   - Organizes results in `analysis_results/` directory

## Output

The script provides colored, structured output:

- 🔵 **INFO**: General information and progress
- 🟢 **SUCCESS**: Successful operations
- 🟡 **WARNING**: Non-critical issues (like skipped non-trading days)
- 🔴 **ERROR**: Critical errors that require attention

## Error Handling

- **Missing API Keys**: Script exits with clear error message
- **Invalid Arguments**: Validates ticker and days parameters
- **Non-Trading Days**: Automatically skips weekends and holidays
- **API Overload**: Automatic retry with 30s wait + exponential backoff (up to 3 retries)
- **Failed Analyses**: Continues with remaining dates, reports in summary

### API Retry Mechanism

The script includes intelligent retry logic for API overload situations:

- **Detects overload errors**: 529 errors, rate limits, "overloaded" messages
- **Wait times**: 30s, 60s, 120s (exponential backoff)
- **Max retries**: 3 attempts for analysis, 2 for initialization
- **Smart detection**: Only retries on actual overload, not other errors
- **Clear feedback**: Shows retry progress and wait times

## Results

- Results are saved to `analysis_results/` directory
- Each successful analysis generates a JSON file
- Files are organized by ticker and date
- Ready for frontend consumption via zzsheep export format

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x run_batch_analysis.sh
   ```

2. **Missing API Keys**
   - Check your `.env` file exists
   - Verify API keys are correctly formatted
   - Test keys individually

3. **Python Environment Issues**
   - Script creates fresh virtual environment
   - Ensure `python3` is available
   - Check `requirements.txt` exists

4. **Date Format Issues**
   - Script handles both macOS and Linux date commands
   - Dates are generated automatically going backwards from yesterday

### Debug Mode

To see more detailed output, you can modify the script to add debug mode:

```bash
# Add this after the shebang line
set -x  # Enable debug mode
```

## Integration

This script works seamlessly with:
- `main.py` - The core analysis script
- `tradingagents/` - The trading agents framework
- `zzsheep` export format - For frontend consumption
- Market data utilities - For comprehensive analysis

## Performance

- **Virtual Environment**: Isolated dependencies
- **API Rate Limiting**: 2-second delays between calls + automatic retry delays
- **Memory Management**: Each analysis runs independently
- **Error Recovery**: Continues on individual failures
- **Retry Logic**: May extend execution time but ensures completion during API overload

## Next Steps

After running batch analysis:
1. Check `analysis_results/` directory for JSON files
2. Review summary statistics
3. Load results into your frontend application
4. Use zzsheep format for data visualization 