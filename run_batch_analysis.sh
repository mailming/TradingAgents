#!/bin/zsh

# Batch Trading Analysis Script
# Usage: ./run_batch_analysis.sh TICKER DAYS
# Example: ./run_batch_analysis.sh AAPL 5

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check arguments
if [ $# -ne 2 ]; then
    print_error "Usage: $0 TICKER DAYS"
    print_error "Example: $0 AAPL 5"
    exit 1
fi

TICKER=$1
DAYS=$2

# Validate arguments
if ! [[ "$DAYS" =~ ^[0-9]+$ ]] || [ "$DAYS" -lt 1 ]; then
    print_error "DAYS must be a positive integer"
    exit 1
fi

print_status "Starting batch analysis for $TICKER over $DAYS days"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "Working directory: $SCRIPT_DIR"

# Step 1: Create/activate virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip and install requirements
print_status "Installing/updating requirements..."
pip install --upgrade pip > /dev/null 2>&1
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    print_success "Requirements installed"
else
    print_warning "requirements.txt not found, skipping package installation"
fi

# Step 2: Load .env file for API keys
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    print_status "Loading API keys from $ENV_FILE..."
    set -a  # Automatically export all variables
    source "$ENV_FILE"
    set +a  # Stop automatically exporting
    print_success "API keys loaded"
else
    print_warning "$ENV_FILE not found, relying on system environment variables"
fi

# Check for required API keys
REQUIRED_KEYS=("ANTHROPIC_API_KEY" "FINANCIALDATASETS_API_KEY")
MISSING_KEYS=()

for key in "${REQUIRED_KEYS[@]}"; do
    if [ -z "${(P)key}" ]; then
        MISSING_KEYS+=("$key")
    fi
done

if [ ${#MISSING_KEYS[@]} -gt 0 ]; then
    print_error "Missing required API keys: ${MISSING_KEYS[*]}"
    print_error "Please add them to your .env file or set as environment variables"
    exit 1
fi

print_success "All required API keys found"

# Step 3: Generate date list (going backwards from yesterday)
print_status "Generating date list for the last $DAYS days..."
DATES=()
for i in $(seq 1 $DAYS); do
    DATE=$(date -j -v-${i}d +%Y-%m-%d 2>/dev/null || date -d "$i days ago" +%Y-%m-%d)
    DATES+=("$DATE")
done

print_success "Generated ${#DATES[@]} dates: ${DATES[*]}"

# Step 4: Run analysis for each date
print_status "Starting sequential analysis..."
echo ""
echo "=================================================="
echo "BATCH ANALYSIS: $TICKER"
echo "DATES: ${DATES[*]}"
echo "=================================================="
echo ""

SUCCESSFUL_RUNS=0
SKIPPED_RUNS=0
FAILED_RUNS=0
RESULTS_DIR="analysis_results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

for DATE in "${DATES[@]}"; do
    echo ""
    echo "🔄 Processing $TICKER for $DATE..."
    echo "----------------------------------------"
    
    # Run the analysis
    if python main.py "$TICKER" "$DATE"; then
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        print_success "✅ Analysis completed for $DATE"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            # If exit code is 0 but we're here, it might be a skipped run
            SKIPPED_RUNS=$((SKIPPED_RUNS + 1))
            print_warning "⏭️  Analysis skipped for $DATE (non-trading day)"
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
            print_error "❌ Analysis failed for $DATE"
        fi
    fi
    
    # Small delay between runs to be respectful to APIs
    # Note: Individual analyses now include auto-retry with 30s delays for API overload
    sleep 2
done

# Summary
echo ""
echo "=================================================="
echo "BATCH ANALYSIS SUMMARY"
echo "=================================================="
echo "📊 Ticker: $TICKER"
echo "📅 Date Range: $DAYS days"
echo "✅ Successful: $SUCCESSFUL_RUNS"
echo "⏭️  Skipped: $SKIPPED_RUNS"
echo "❌ Failed: $FAILED_RUNS"
echo "📁 Results Directory: $RESULTS_DIR"
echo ""

if [ $SUCCESSFUL_RUNS -gt 0 ]; then
    print_success "Batch analysis completed! Check $RESULTS_DIR for saved results."
else
    print_warning "No successful analyses. Check the dates and ensure markets were open."
fi

# Deactivate virtual environment
deactivate 2>/dev/null || true

print_status "Virtual environment deactivated"
print_success "Batch analysis script completed" 