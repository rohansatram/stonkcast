"""
Disk cache for fetched ticker data.

We cache FULL history per ticker (not cutoff-filtered), keyed by ticker only.
The same cached blob then serves any cutoff: the live Jan-2026 run, a 2025
backtest, and the future accuracy-verifier (which needs post-cutoff prices to
measure what actually happened). Point-in-time filtering is applied downstream,
never at fetch time. This is what makes the cache reusable across cutoffs and
keeps us off yfinance's rate limit.
"""

import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.pkl"


def _record_age(record: dict) -> timedelta | None:
    fetched_at = record.get("fetched_at")
    if not fetched_at:
        return None
    try:
        return datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)
    except ValueError:
        return None


def fetched_at(ticker: str) -> str | None:
    """When `ticker`'s cache blob was written (ISO UTC), or None if not cached."""
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        with path.open("rb") as cache_file:
            return pickle.load(cache_file).get("fetched_at")
    except Exception:
        return None


def cached_fetch(ticker: str, fetch_fn: Callable[[str], dict], refresh: bool = False,
                 max_age: timedelta | None = None) -> dict:
    """
    Return cached data for `ticker`, or fetch + cache it if absent.

    fetch_fn(ticker) -> dict is called only on a cache miss (or refresh=True).
    A corrupt/unreadable cache file is treated as a miss and re-fetched.
    `max_age`, when set, treats a blob older than it as stale and re-fetches it
    (the live run uses a short TTL so an "as-of-today" score is never built from
    a days-old blob left over from backtest prep).
    """
    path = _cache_path(ticker)

    if not refresh and path.exists():
        try:
            with path.open("rb") as cache_file:
                record = pickle.load(cache_file)
            age = _record_age(record)
            stale = max_age is not None and (age is None or age > max_age)
            if not stale:
                return record["data"]
        except Exception:
            pass  # corrupt cache -> fall through and re-fetch

    data = fetch_fn(ticker)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    record = {"fetched_at": datetime.now(timezone.utc).isoformat(), "ticker": ticker.upper(), "data": data}
    temp_path = path.with_suffix(".pkl.tmp")
    with temp_path.open("wb") as cache_file:
        pickle.dump(record, cache_file)
    temp_path.replace(path)  # atomic write so a crash mid-write can't corrupt the cache

    return data


def is_cached(ticker: str) -> bool:
    return _cache_path(ticker).exists()


def clear_cache(ticker: str | None = None) -> None:
    """Delete one ticker's cache, or the whole cache dir if ticker is None."""
    if ticker is not None:
        _cache_path(ticker).unlink(missing_ok=True)
        return
    if CACHE_DIR.exists():
        for cached_file in CACHE_DIR.glob("*.pkl"):
            cached_file.unlink(missing_ok=True)
