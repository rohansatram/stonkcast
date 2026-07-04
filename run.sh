#!/usr/bin/env bash
# Start stonkcast (one server serves both the UI and the API at :8000).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT/.server.pid"

# Clean restart: stop any instance already running.
"$ROOT/stop.sh" >/dev/null 2>&1 || true

cd "$ROOT/backend"
nohup uv run python src/api.py > "$ROOT/server.log" 2>&1 &
echo $! > "$PID_FILE"

echo "stonkcast running at http://127.0.0.1:8000  (UI + API, pid $(cat "$PID_FILE"))"
echo "logs: server.log    stop: ./stop.sh"
