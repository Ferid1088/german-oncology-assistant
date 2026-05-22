#!/bin/bash
# Runs the full evaluation pipeline:
#   1. Starts the API in the background
#   2. Waits until the API is ready
#   3. Runs the evaluation script
#   4. Stops the API
#
# Usage:
#   bash evaluations/run_eval.sh
#   API_KEY=my-key LIMIT=5 bash evaluations/run_eval.sh

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_KEY="${API_KEY:-dev-secret-key}"
SPLIT="${SPLIT:-eval}"
LIMIT="${LIMIT:-}"
READY_ONLY="${READY_ONLY:-1}"
TIMEOUT="${TIMEOUT:-120}"
RETRIES="${RETRIES:-1}"
RETRY_DELAY="${RETRY_DELAY:-3}"
THROTTLE_SECONDS="${THROTTLE_SECONDS:-2}"
API_URL="http://localhost:8000"

echo "=== Oncology Evaluation Pipeline ==="
echo "Split:   $SPLIT"
echo "API URL: $API_URL"
echo "API Key: $API_KEY"
[ -n "$LIMIT" ] && echo "Limit:   $LIMIT items"
[ "$READY_ONLY" = "1" ] && echo "Ready only: yes"
echo "Timeout: $TIMEOUT s | Retries: $RETRIES | Retry delay: $RETRY_DELAY s"
echo ""

# ── Step 1: Generate / check datasets ────────────────────────────────────────
echo "── Step 1: Generating evaluation datasets ──────────────────────"
.venv/bin/python scripts/generate_eval_dataset.py >/dev/null

echo "── Step 2: Checking dataset ─────────────────────────────────────"
.venv/bin/python evaluations/scripts/check_dataset.py --split "$SPLIT"
echo ""

# ── Step 3: Start API in background ──────────────────────────────────────────
echo "── Step 3: Starting API ─────────────────────────────────────────"

API_PID=""
API_STARTED_BY_US=0

if curl -sf "$API_URL/health" > /dev/null 2>&1; then
    echo "API already running at $API_URL — reusing it."
else
    .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level warning &
    API_PID=$!
    API_STARTED_BY_US=1

    echo "Waiting for API to start..."
    READY=0
    for i in $(seq 1 30); do
        if curl -sf "$API_URL/health" > /dev/null 2>&1; then
            READY=1
            echo "API ready after ${i}s."
            break
        fi
        sleep 1
    done

    if [ "$READY" -eq 0 ]; then
        echo "ERROR: API did not start within 30 seconds." >&2
        kill "$API_PID" 2>/dev/null || true
        exit 1
    fi
fi

cleanup() {
    if [ "$API_STARTED_BY_US" -eq 1 ] && [ -n "$API_PID" ]; then
        echo ""
        echo "Stopping API (PID $API_PID)..."
        kill "$API_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
echo ""

# ── Step 4: Run evaluation ────────────────────────────────────────────────────
echo "── Step 4: Running evaluation ───────────────────────────────────"
ARGS="--api-url $API_URL --api-key $API_KEY --split $SPLIT --timeout $TIMEOUT --retries $RETRIES --retry-delay $RETRY_DELAY --throttle-seconds $THROTTLE_SECONDS"
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"
[ "$READY_ONLY" = "1" ] && ARGS="$ARGS --ready-only"

.venv/bin/python evaluations/scripts/run_eval.py $ARGS
echo ""

echo "=== Evaluation complete. Results saved to evaluations/results/ ==="
echo "Launch dashboard: bash evaluations/run_eval_dashboard.sh"
