#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/nohup"
LOG_FILE="$LOG_DIR/learn.nohup.log"
PID_FILE="$LOG_DIR/learn.pid"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-codes/learn.py}"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "A process is already running with PID $(cat "$PID_FILE")"
  echo "Use nohup/stop.sh first or remove $PID_FILE if that PID is stale."
  exit 1
fi

nohup "$PYTHON_BIN" "$ROOT_DIR/$SCRIPT_PATH" "$@" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Started $SCRIPT_PATH in the background."
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
