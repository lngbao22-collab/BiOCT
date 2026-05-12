#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/nohup"
PID_FILE="$LOG_DIR/learn.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped process $PID"
else
  echo "Process $PID is not running."
fi

rm -f "$PID_FILE"
