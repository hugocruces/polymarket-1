#!/bin/bash
#
# run_political_bias_scan.sh
# Runs the Polymarket political bias scanner with Haiku model for cost-effective screening
# Activates venv, executes the scanner, and saves output to output/latest_political_scan.md
#
# Usage: ./run_political_bias_scan.sh
# Schedule with cron: 0 9 * * * /path/to/run_political_bias_scan.sh

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the project directory
cd "$SCRIPT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting political bias scan..."

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found at $SCRIPT_DIR/venv"
    exit 1
fi

source venv/bin/activate

# Verify the scanner can be imported
python -c "import polymarket_agent" || {
    echo "ERROR: polymarket_agent module not found. Check venv activation."
    exit 1
}

# Create output directory if it doesn't exist
mkdir -p output

# Run the scanner with the political keywords config
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running scanner with political keywords filter..."
python -m polymarket_agent.scan \
    --output output/latest_political_scan.md \
    --min-volume 500 \
    --model claude-haiku-4-5

if [ $? -eq 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✓ Political bias scan completed successfully"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Output saved to: output/latest_political_scan.md"
    exit 0
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✗ Political bias scan failed"
    exit 1
fi
