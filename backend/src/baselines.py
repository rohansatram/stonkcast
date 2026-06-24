"""
Point-in-time sector baselines for z-score normalisation.

A raw metric is meaningless alone (a P/E of 40 is cheap for software, dear for a
utility). So we score each metric relative to its sector: compute the same
metrics for a basket of sector peers AS OF the cutoff, then express the target's
value as a z-score against that peer distribution.

Baselines are computed per (sector, cutoff, excluded ticker) and memoised
in-process. The peer data itself is disk-cached by ticker, so building a baseline
costs nothing after the first run for a sector.
"""

import logging
import math
import statistics
from datetime import datetime

import polars as pl

from fetch.fetchStockData import fetch_cached
from metrics import compute_metrics

logger = logging.getLogger(__name__)

MIN_PEERS_FOR_BASELINE = 3  # need a few peers for a meaningful distribution

# yfinance's 11 sector names -> representative large/liquid baskets.
SECTOR_BASKETS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "INTC"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "T", "CMCSA", "CHTR", "EA"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "GM"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "MDLZ", "CL", "MO", "KMB", "GIS"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", "SCHW", "SPGI"],
    "Industrials": ["CAT", "HON", "UPS", "BA", "GE", "RTX", "DE", "LMT", "UNP", "MMM"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB"],
    "Basic Materials": ["LIN", "SHW", "APD", "ECL", "FCX", "NEM", "DOW", "DD", "NUE", "CTVA"],
    "Real Estate": ["PLD", "AMT", "EQIX", "PSA", "O", "WELL", "SPG", "CCI", "DLR", "VICI"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG"],
}

# Whether higher is better (+1) or worse (-1) for each scored metric.
METRIC_DIRECTION: dict[str, int] = {
    "momentum_6m_vs_spy": +1,         # horizon-aligned relative momentum
    "momentum_vs_spy": +1,
    "momentum_30d": +1,
    "fifty_two_week_position": +1,    # relative strength: near 52w high tends bullish
    "revenue_growth": +1,
    "eps_trend": +1,
    "earnings_surprise": +1,
    "ebitda_margin": +1,
    "pe_ratio": -1,          # cheaper (lower) is better (verify sign against backtest: value-trap risk)
    "debt_to_equity": -1,    # less leverage is better
}

_baseline_cache: dict[tuple[str, str, str], dict] = {}


def _center_scale(values: list[float]) -> tuple[float, float, int] | None:
    """Robust (center, scale, n) for a small peer basket. Uses median + scaled MAD
    (1.4826 * median abs deviation, the std-consistent estimator) so one wild peer
    can't shift the distribution the way mean/std would at n of 3-10. Falls back to
    sample std only when the MAD collapses (more than half the peers identical)."""
    valid_values = [value for value in values if value is not None and math.isfinite(value)]
    n = len(valid_values)
    if n < MIN_PEERS_FOR_BASELINE:
        return None
    median = statistics.median(valid_values)
    mad = statistics.median([abs(value - median) for value in valid_values]) * 1.4826
    if mad > 0:
        return median, mad, n
    mean_value = sum(valid_values) / n
    std = (sum((value - mean_value) ** 2 for value in valid_values) / (n - 1)) ** 0.5
    return median, std, n


def sector_baseline(sector: str, cutoff: datetime, spy_prices: pl.DataFrame,
                    exclude: str | None = None, refresh: bool = False) -> dict:
    """
    {metric: {"mean": m, "std": s}} for the sector's basket as of the cutoff.
    Returns {} for unknown sectors. Peers with std == 0 are dropped (no spread,
    nothing to normalise against).
    """
    if sector not in SECTOR_BASKETS:
        return {}

    # Key MUST include `exclude`: each target ticker is removed from its own peer
    # distribution, so two tickers in the same sector get different baselines.
    excluded_ticker = (exclude or "").upper()
    cache_key = (sector, cutoff.date().isoformat(), excluded_ticker)
    if not refresh and cache_key in _baseline_cache:
        return _baseline_cache[cache_key]

    peers = [peer for peer in SECTOR_BASKETS[sector] if peer != excluded_ticker]
    baseline_metrics = list(METRIC_DIRECTION) + ["volatility"]  # volatility feeds confidence
    values_by_metric: dict[str, list[float]] = {metric: [] for metric in baseline_metrics}

    dropped: list[tuple[str, str]] = []
    for peer in peers:
        try:
            peer_data = fetch_cached(peer, refresh=refresh)
            peer_metrics = compute_metrics(peer_data, spy_prices, cutoff)
        except Exception as exc:  # keep one bad peer from sinking the basket, but don't hide it
            dropped.append((peer, type(exc).__name__))
            continue
        if peer_metrics is None:
            dropped.append((peer, "no_metrics"))
            continue
        for metric in baseline_metrics:
            values_by_metric[metric].append(peer_metrics.get(metric))

    # Silently shrinking the peer set narrows the spread and inflates z-scores toward
    # the clip rails, so surface drops rather than swallowing them (the old behaviour).
    if dropped:
        logger.warning("sector_baseline(%s @ %s): dropped %d/%d peers: %s",
                       sector, cutoff.date().isoformat(), len(dropped), len(peers), dropped)

    baseline: dict[str, dict] = {}
    for metric, values in values_by_metric.items():
        stats = _center_scale(values)
        if stats and stats[1] > 0:
            baseline[metric] = {"mean": stats[0], "std": stats[1], "n": stats[2]}

    _baseline_cache[cache_key] = baseline
    return baseline
