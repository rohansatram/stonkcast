"""
Phase 1 orchestrator: ticker + cutoff -> a deterministic 1-5 score from pure
math (no LLM). This is the standalone baseline that Phase 2 (Nova reasoning)
must beat to justify itself.

    from phase1 import score_ticker
    result = score_ticker("NVDA", datetime(2026, 1, 1, tzinfo=timezone.utc))
"""

import time
from datetime import datetime, timezone

from fetch.fetchStockData import fetch_cached
from fetch.fetchCongressTrades import congress_signal, congress_score
from cache import fetched_at
from metrics import compute_metrics
from baselines import sector_baseline
from scoring import score

DEFAULT_CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)

# The 8 metrics the brief requires (analyst consensus is display-only / leaky).
REQUIRED_METRICS = [
    "pe_ratio", "eps_trend", "revenue_growth", "ebitda_margin",
    "debt_to_equity", "fifty_two_week_position", "momentum_30d",
]


def score_ticker(ticker: str, cutoff: datetime = DEFAULT_CUTOFF, refresh: bool = False) -> dict:
    started = time.time()

    data = fetch_cached(ticker, refresh=refresh)
    spy_data = fetch_cached("SPY", refresh=refresh)
    spy_prices = spy_data["prices"]

    metrics = compute_metrics(data, spy_prices, cutoff)
    if metrics is None:
        return {
            "ticker": ticker.upper(),
            "cutoff": cutoff.date().isoformat(),
            "error": f"no price history for {ticker.upper()} before {cutoff.date().isoformat()} "
                     f"(the ticker may be too new, delisted, or the cutoff too early)",
            "latency_sec": round(time.time() - started, 3),
        }

    # Congressional trades: a direct (non-z-scored) math signal, point-in-time by
    # disclosure date. Neutral (None) when no dataset is present in backend/data/.
    congress = congress_signal(ticker, cutoff)
    metrics["congress"] = congress_score(congress)

    sector = data["info"].get("sector")
    baseline = sector_baseline(sector, cutoff, spy_prices, exclude=ticker, refresh=refresh) if sector else {}
    result = score(metrics, baseline)

    last_price_date = metrics.get("_last_price_date")
    return {
        "ticker": ticker.upper(),
        "name": data["info"].get("long_name"),
        "sector": sector,
        "sector_note": "fetch-time classification (not point-in-time); routes peer baseline only",
        "cutoff": cutoff.date().isoformat(),
        "score": result["score"],
        "label": result["label"],
        "raw_score": result["raw_score"],
        "confidence": result["confidence"],
        "coverage": result["coverage"],
        "families": result["families"],
        "median_peer_count": result["median_peer_count"],
        "data_asof": last_price_date.date().isoformat() if last_price_date is not None else None,
        "data_fetched_at": fetched_at(ticker.upper()),
        "metrics": {metric: metrics.get(metric) for metric in REQUIRED_METRICS},
        "extra_signals": {
            "momentum_6m_vs_spy": metrics.get("momentum_6m_vs_spy"),
            "momentum_vs_spy": metrics.get("momentum_vs_spy"),
            "earnings_surprise": metrics.get("earnings_surprise"),
            "volatility": metrics.get("volatility"),
            "congress": metrics.get("congress"),
        },
        "congress": {
            "available": congress.get("available"),
            "signal": congress.get("signal"),
            "purchases": congress.get("purchases"),
            "sales": congress.get("sales"),
            "net_trades": congress.get("net_trades"),
            "n_members": congress.get("n_members"),
            "window_days": congress.get("window_days"),
            "recent": congress.get("recent", []),
        } if congress else None,
        "analyst_consensus_display_only": {
            "recommendation_mean": data["info"].get("recommendation_mean"),
            "recommendation_key": data["info"].get("recommendation_key"),
            "note": "live value, excluded from score (not point-in-time)",
        },
        "breakdown": result["breakdown"],
        "tokens": 0,       # Phase 1 uses no LLM; populated in Phase 2
        "cost_usd": 0.0,
        "latency_sec": round(time.time() - started, 3),
    }


def _fmt(value, pct=False):
    if value is None:
        return "n/a"
    return f"{value*100:+.1f}%" if pct else f"{value:.2f}"


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    cutoff = DEFAULT_CUTOFF
    if len(sys.argv) > 2:
        cutoff = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)

    result = score_ticker(ticker, cutoff)
    if "error" in result:
        print(f"{result['ticker']}: {result['error']}")
        raise SystemExit(1)

    print(f"\n{'='*56}")
    print(f" {result['name']} ({result['ticker']})  |  {result['sector']}")
    print(f" Cutoff: {result['cutoff']}")
    print(f"{'='*56}")
    print(f" SCORE: {result['score']}/5  {result['label']}   (raw {result['raw_score']:+.2f})")
    print(f" Confidence: {result['confidence']}  |  metric coverage: {result['coverage']*100:.0f}%")
    print(f"{'-'*56}")
    print(" Required metrics:")
    pct_metrics = {"eps_trend", "revenue_growth", "ebitda_margin", "momentum_30d", "fifty_two_week_position"}
    for metric in REQUIRED_METRICS:
        print(f"   {metric:26s} {_fmt(result['metrics'][metric], pct=metric in pct_metrics)}")
    print(" Extra signals:")
    for name, value in result["extra_signals"].items():
        print(f"   {name:26s} {_fmt(value, pct=name != 'earnings_surprise')}")
    analyst_consensus = result["analyst_consensus_display_only"]
    print(f" Analyst consensus (display-only): {analyst_consensus['recommendation_key']} ({analyst_consensus['recommendation_mean']})")
    print(f"{'-'*56}")
    print(" Score breakdown (signal x weight):")
    for entry in sorted(result["breakdown"], key=lambda item: -abs(item["contribution"])):
        signal_text = "n/a" if entry["signal"] is None else f"{entry['signal']:+.2f}"
        print(f"   {entry['metric']:26s} signal={signal_text:>6s}  w={entry['weight']:.2f}  contrib={entry['contribution']:+.3f}")
    print(f"{'-'*56}")
    print(f" Latency: {result['latency_sec']}s  |  tokens: {result['tokens']}  |  cost: ${result['cost_usd']}")
    print(f"{'='*56}\n")
