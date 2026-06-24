"""
Pre-warm the disk cache before a demo: fetch every sector-basket ticker + SPY
(plus any extra tickers passed as args) concurrently, so nothing is cold during a
live showcase run.

    uv run python src/warm_cache.py              # all sector baskets + SPY
    uv run python src/warm_cache.py NFLX DIS     # plus these
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import SECTOR_BASKETS  # noqa: E402
from backtest import warm_cache  # noqa: E402

if __name__ == "__main__":
    extra = [arg.upper() for arg in sys.argv[1:]]
    tickers = sorted({ticker for basket in SECTOR_BASKETS.values() for ticker in basket} | {"SPY", *extra})
    print(f"Warming cache for {len(tickers)} tickers (concurrent)...")
    started = time.time()
    warm_cache(tickers)
    print(f"Done in {time.time() - started:.1f}s")
