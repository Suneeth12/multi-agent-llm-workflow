#!/usr/bin/env bash
# Generate the sales data and run the multi-agent analysis workflow.
set -e
cd "$(dirname "$0")"

pip install -r requirements.txt
python3 src/generate_data.py
echo ""
python3 src/orchestrator.py
echo ""
echo "Self-correction demo:"
python3 src/orchestrator.py "Give me a statistical summary of the data"
