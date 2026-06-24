#!/usr/bin/env bash
# Stop stonkcast.
ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT/.server.pid"
stopped=0

if [ -f "$PID_FILE" ]; then
  kill "$(cat "$PID_FILE")" 2>/dev/null && stopped=1 || true
  rm -f "$PID_FILE"
fi

# uv may spawn a child process, so also free port 8000 directly.
port_pids="$(lsof -ti tcp:8000 2>/dev/null || true)"
if [ -n "$port_pids" ]; then
  kill $port_pids 2>/dev/null || true
  stopped=1
fi

[ "$stopped" = 1 ] && echo "stonkcast stopped." || echo "stonkcast was not running."
