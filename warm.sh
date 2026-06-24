#!/usr/bin/env bash
# Pre-warm the disk cache (run before a demo so nothing is cold live).
# Usage: ./warm.sh [EXTRA TICKERS...]
cd "$(dirname "$0")/backend"
uv run python src/warm_cache.py "$@"
