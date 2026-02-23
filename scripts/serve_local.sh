#!/usr/bin/env bash
# Start the full inference stack locally (no Docker).
# Usage:  bash scripts/serve_local.sh [--gpus 1,2] [--port 9000]
#
# Starts:  Redis → 2 GPU workers → FastAPI server
# Stops:   all background PIDs on SIGINT/EXIT

set -euo pipefail

REDIS_BIN="${CONDA_PREFIX:-/tmp2/b11902156/miniconda3}/bin/redis-server"
REDIS_PORT=6379
API_PORT=9000
GPUS="${GPUS:-1,2}"           # comma-separated physical GPU indices
LOG_DIR="logs/serve"

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus) GPUS="$2"; shift 2 ;;
    --port) API_PORT="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
  echo ""
  echo "Shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# --- Redis ---
echo "[1/3] Starting Redis on port ${REDIS_PORT} ..."
"$REDIS_BIN" --port "$REDIS_PORT" --daemonize no \
  --loglevel warning \
  > "$LOG_DIR/redis.log" 2>&1 &
PIDS+=($!)
sleep 1

# --- Workers (one per GPU) ---
IFS=',' read -ra GPU_LIST <<< "$GPUS"
echo "[2/3] Starting ${#GPU_LIST[@]} workers (GPUs: ${GPUS}) ..."
for i in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$i]}"
  log="$LOG_DIR/worker_gpu${gpu}.log"
  echo "      worker $i → GPU $gpu  (log: $log)"
  CUDA_VISIBLE_DEVICES="$gpu" \
  WORKER_GPU_ID="$i" \
  REDIS_HOST=localhost \
  REDIS_PORT="$REDIS_PORT" \
    uv run python -m inference.worker.worker \
    > "$log" 2>&1 &
  PIDS+=($!)
done

# Give workers time to load the model (~15s each, but they run in parallel)
echo "      waiting for workers to load model (~15s) ..."
sleep 20

# --- API server ---
echo "[3/3] Starting API server on port ${API_PORT} ..."
REDIS_HOST=localhost \
REDIS_PORT="$REDIS_PORT" \
  uv run uvicorn inference.api.server:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --log-level info \
  > "$LOG_DIR/api.log" 2>&1 &
PIDS+=($!)
sleep 2

echo ""
echo "Stack is up:"
echo "  API    → http://localhost:${API_PORT}"
echo "  Health → http://localhost:${API_PORT}/health"
echo "  Docs   → http://localhost:${API_PORT}/docs"
echo ""
echo "Run load test:  uv run scripts/load_test.py --url http://localhost:${API_PORT} --n 4"
echo ""
echo "Press Ctrl+C to stop."
wait
