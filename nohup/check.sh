#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/nohup"
JOB_NAME="${JOB_NAME:-learn}"
LOG_FILE="$LOG_DIR/${JOB_NAME}.nohup.log"
PID_FILE="$LOG_DIR/${JOB_NAME}.pid"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "learn.py is running with PID $PID"
  else
    echo "PID file exists but the process is not running."
  fi
else
  echo "No PID file found."
fi

if [[ -f "$LOG_FILE" ]]; then
  echo "Last 50 log lines from $LOG_FILE:"
  tail -n 50 "$LOG_FILE"
else
  echo "No log file found at $LOG_FILE"
fi
