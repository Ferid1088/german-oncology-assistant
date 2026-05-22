#!/bin/bash
# Launches the evaluation dashboard on port 8502.
# (Port 8502 avoids conflict with the main Streamlit app on 8501.)
#
# Usage:
#   bash evaluations/run_eval_dashboard.sh

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Evaluation Dashboard ==="
echo "Opening at http://localhost:8502"
echo "Press Ctrl+C to stop."
echo ""

.venv/bin/streamlit run evaluations/ui/app.py \
    --server.port 8502 \
    --server.address localhost \
    --browser.gatherUsageStats false
