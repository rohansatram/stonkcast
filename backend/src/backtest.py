"""
Backtest harness: grade a scorer against reality.

For each (ticker, cutoff) it runs the scorer AT the cutoff (point-in-time, so no
leakage) and compares the score to the stock's ACTUAL forward alpha over the
horizon (stock return minus SPY return). Outcomes are allowed to use post-cutoff
prices here, because this is the verifier, not the predictor.

The scorer is pluggable: run_backtest(scorer=..., score_key=...) grades either the
Phase 1 math model (score_ticker / "score") or the Phase 2 Nova model
(score_ticker_v2 / "final_score"). compare() runs both on the same grid so you can
see whether Nova actually beats the math baseline, and at what token cost.

    from backtest import run_backtest, compare, fit_and_evaluate
    report = run_backtest()            # JSON-serialisable dict: predictions + summary
"""

import hashlib
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import polars as pl

from fetch.fetchStockData import fetch_cached
from phase1 import score_ticker
import scoring

UTC = timezone.utc
HORIZON_DAYS = 182  # ~6 months, matching the contest's Jan->Jun window
MAX_JOB_WORKERS = 6   # (ticker, cutoff) jobs run concurrently; bounded for yfinance/Bedrock limits
MAX_WARM_WORKERS = 8  # cache pre-warm concurrency
DEAD_BAND = 0.03    # |alpha| <= this is a true "Hold" outcome (shared with outcome_bucket)

# Default universe = the cached tech + financial baskets (fast first run).
# SURVIVORSHIP CAVEAT: these are today's surviving large caps, so the headline
# hit-rate is optimistic vs a universe that included delisted/shrunken names.
# Expand (and add laggards) for a real run; cost is one fetch per new ticker.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "INTC",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", "SCHW", "SPGI",
]
DEFAULT_CUTOFFS = ["2023-07-03", "2024-01-02", "2024-07-01", "2025-01-02", "2025-07-01"]
# Held-out split for calibration: fit thresholds on the earlier cutoffs, evaluate
# on the latest (a genuine forward-in-time test, not in-sample).
DEFAULT_FIT_CUTOFFS = DEFAULT_CUTOFFS[:-1]
DEFAULT_EVAL_CUTOFFS = DEFAULT_CUTOFFS[-1:]

# actual-alpha thresholds -> the bucket the scorer "should" have produced
OUTCOME_THRESHOLDS = [(0.10, 5), (DEAD_BAND, 4), (-DEAD_BAND, 3), (-0.10, 2)]  # else 1

CALIBRATION_FILE = scoring.CALIBRATION_FILE


def _price_on_or_after(prices: pl.DataFrame, target_date: datetime) -> float | None:
    matches = prices.filter(pl.col("Date") >= target_date)
    return matches["Close"][0] if matches.height else None


def forward_alpha(prices: pl.DataFrame, spy_prices: pl.DataFrame, cutoff: datetime, horizon_days: int) -> float | None:
    """Stock return minus SPY return, from the first tradable day at/after the
    cutoff to the first tradable day at/after cutoff+horizon. None if the window
    extends past available data."""
    exit_date = cutoff + timedelta(days=horizon_days)
    stock_entry, stock_exit = _price_on_or_after(prices, cutoff), _price_on_or_after(prices, exit_date)
    market_entry, market_exit = _price_on_or_after(spy_prices, cutoff), _price_on_or_after(spy_prices, exit_date)
    if None in (stock_entry, stock_exit, market_entry, market_exit) or stock_entry == 0 or market_entry == 0:
        return None
    return (stock_exit / stock_entry - 1.0) - (market_exit / market_entry - 1.0)


def outcome_bucket(alpha: float) -> int:
    for threshold, bucket in OUTCOME_THRESHOLDS:
        if alpha >= threshold:
            return bucket
    return 1


def _raw_score_of(result: dict) -> float | None:
    """raw_score lives at the top level for Phase 1 and under 'phase1' for Phase 2."""
    if result.get("raw_score") is not None:
        return result["raw_score"]
    return result.get("phase1", {}).get("raw_score")


def warm_cache(tickers, max_workers: int = MAX_WARM_WORKERS) -> None:
    """Fetch each ticker's full history into the disk cache concurrently. Front-loads
    all network I/O once (bounded), so subsequent scoring is cache-fast. Used by the
    backtest and the pre-demo warm script. Errors per ticker are swallowed."""
    def _safe(ticker):
        try:
            fetch_cached(ticker)
        except Exception:
            pass
    unique = sorted(set(tickers))
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(unique)))) as pool:
        list(pool.map(_safe, unique))


def run_backtest(tickers: list[str] | None = None, cutoffs: list[str] | None = None,
                 horizon_days: int = HORIZON_DAYS, scorer=None, score_key: str = "score",
                 label: str = "phase1", max_workers: int = MAX_JOB_WORKERS) -> dict:
    tickers = tickers or DEFAULT_UNIVERSE
    cutoffs = cutoffs or DEFAULT_CUTOFFS
    scorer = scorer or score_ticker

    # Pre-warm the universe + SPY once (front-loads the cold fetches under one bounded
    # pool), so the parallel scoring below is mostly cache hits, not nested network.
    warm_cache(list(tickers) + ["SPY"])
    spy_prices = fetch_cached("SPY")["prices"]

    def run_one(job: tuple[str, str]) -> dict | None:
        cutoff_str, ticker = job
        cutoff = datetime.fromisoformat(cutoff_str).replace(tzinfo=UTC)
        try:
            result = scorer(ticker, cutoff)
            if "error" in result:
                return None
            score_value = result.get(score_key)
            if score_value is None:
                return None
            prices = fetch_cached(ticker)["prices"]
            alpha = forward_alpha(prices, spy_prices, cutoff, horizon_days)
            if alpha is None:
                return None
        except Exception:
            return None
        return {
            "ticker": ticker,
            "cutoff": cutoff_str,
            "sector": result.get("sector"),
            "score": score_value,
            "raw_score": _raw_score_of(result),
            "confidence": result.get("confidence"),
            "actual_alpha": round(alpha, 4),
            "true_bucket": outcome_bucket(alpha),
            "tokens": result.get("tokens", 0),
            "cost_usd": result.get("cost_usd", 0.0),
            "latency_sec": result.get("latency_sec"),
        }

    jobs = [(cutoff_str, ticker) for cutoff_str in cutoffs for ticker in tickers]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        predictions = [p for p in pool.map(run_one, jobs) if p is not None]

    return {"label": label, "universe": tickers, "cutoffs": cutoffs,
            "summary": _summarise(predictions, horizon_days), "predictions": predictions}


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #

def _summarise(predictions: list[dict], horizon_days: int) -> dict:
    total = len(predictions)
    if total == 0:
        return {"n": 0, "note": "no predictions (cache cold or windows beyond data)"}

    # Directional accuracy, excluding Holds (score == 3) AND pushes (|alpha| within
    # the dead-band, which are true Holds by our own outcome definition - counting a
    # +0.4% move as a correct "Buy" would inflate the headline).
    directional = [p for p in predictions if p["score"] != 3]
    pushes = [p for p in directional if abs(p["actual_alpha"]) <= DEAD_BAND]
    decided = [p for p in directional if abs(p["actual_alpha"]) > DEAD_BAND]
    correct = sum(
        1 for p in decided
        if (p["score"] >= 4 and p["actual_alpha"] > DEAD_BAND)
        or (p["score"] <= 2 and p["actual_alpha"] < -DEAD_BAND)
    )
    directional_accuracy = correct / len(decided) if decided else None

    bucket_mae = sum(abs(p["score"] - p["true_bucket"]) for p in predictions) / total
    correlation = _pearson([p["raw_score"] for p in predictions if p.get("raw_score") is not None],
                           [p["actual_alpha"] for p in predictions if p.get("raw_score") is not None])

    calibration = {}
    for bucket_score in range(1, 6):
        group = [p["actual_alpha"] for p in predictions if p["score"] == bucket_score]
        if group:
            calibration[bucket_score] = {
                "count": len(group),
                "mean_actual_alpha": round(sum(group) / len(group), 4),
                "median_actual_alpha": round(statistics.median(group), 4),
            }

    summary = {
        "n": total,
        "horizon_days": horizon_days,
        "directional_accuracy_excl_holds": round(directional_accuracy, 3) if directional_accuracy is not None else None,
        "directional_accuracy_ci95": _wilson_interval(correct, len(decided)),
        "directional_calls": len(directional),
        "decided_calls": len(decided),
        "pushes": len(pushes),
        "holds": total - len(directional),
        "bucket_mae": round(bucket_mae, 3),
        "bucket_mae_note": "score is z-score-space, true_bucket is alpha-space; lead with rank_ic / calibration, not this",
        "corr_raw_vs_alpha": round(correlation, 3) if correlation is not None else None,
        "rank_ic": _rank_ic(predictions),
        "quantile_spread": _quantile_spread(predictions),
        "calibration_by_score": calibration,
        "baselines": _baseline_strategies(predictions),
        "cost": _cost_summary(predictions),
        "warnings": _warnings(predictions, len(decided), calibration),
    }
    return summary


def _baseline_strategies(predictions: list[dict]) -> dict:
    """What trivial strategies would score on the SAME predictions - so 'beat a coin
    flip' is demonstrated, not asserted."""
    total = len(predictions)
    decided = [p for p in predictions if abs(p["actual_alpha"]) > DEAD_BAND]
    always_buy = (sum(1 for p in decided if p["actual_alpha"] > DEAD_BAND) / len(decided)) if decided else None
    # Deterministic pseudo-random buckets (seeded by ticker+cutoff, no Math.random):
    rng_correct = 0
    for p in decided:
        # Deterministic across runs (Python's str hash is salted), so the baseline is reproducible.
        digest = hashlib.md5(f"{p.get('ticker', '')}|{p.get('cutoff', '')}".encode()).hexdigest()
        pseudo = (int(digest, 16) % 5) + 1  # 1..5
        if (pseudo >= 4 and p["actual_alpha"] > DEAD_BAND) or (pseudo <= 2 and p["actual_alpha"] < -DEAD_BAND):
            rng_correct += 1
    return {
        "always_hold_bucket_mae": round(sum(abs(3 - p["true_bucket"]) for p in predictions) / total, 3) if total else None,
        "always_buy_directional_accuracy": round(always_buy, 3) if always_buy is not None else None,
        "random_directional_accuracy": round(rng_correct / len(decided), 3) if decided else None,
        "note": "directional accuracies are over decided calls (|alpha| > dead-band)",
    }


def _cost_summary(predictions: list[dict]) -> dict:
    """The tokens/latency/cost the judges put on the scoreboard."""
    latencies = [p["latency_sec"] for p in predictions if p.get("latency_sec") is not None]
    return {
        "total_tokens": sum(p.get("tokens", 0) or 0 for p in predictions),
        "total_cost_usd": round(sum(p.get("cost_usd", 0.0) or 0.0 for p in predictions), 6),
        "mean_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "mean_tokens_per_call": round(sum(p.get("tokens", 0) or 0 for p in predictions) / len(predictions), 1) if predictions else None,
    }


def _warnings(predictions: list[dict], n_decided: int, calibration: dict) -> list[str]:
    warnings = []
    if n_decided < 30:
        warnings.append(f"low sample: only {n_decided} decided directional calls; treat accuracy as indicative, not significant")
    thin = [bucket for bucket, entry in calibration.items() if entry["count"] < 5]
    if thin:
        warnings.append(f"thin calibration buckets {thin} (<5 obs); per-bucket alphas are noisy")
    # Repeated tickers across adjacent ~6-month cutoffs make windows regime-correlated,
    # so the effective sample is smaller than n.
    n_tickers = len({p.get("ticker") for p in predictions})
    if predictions and len(predictions) > n_tickers:
        warnings.append("windows overlap/repeat per ticker across adjacent cutoffs; effective n < n (correlated outcomes)")
    return warnings


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def _pearson(x_values: list[float], y_values: list[float]) -> float | None:
    count = len(x_values)
    if count < 3:
        return None
    mean_x, mean_y = sum(x_values) / count, sum(y_values) / count
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    std_x = sum((x - mean_x) ** 2 for x in x_values) ** 0.5
    std_y = sum((y - mean_y) ** 2 for y in y_values) ** 0.5
    return covariance / (std_x * std_y) if std_x > 0 and std_y > 0 else None


def _ranks(values: list[float]) -> list[float]:
    """Average (tie-corrected) ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _rank_ic(predictions: list[dict]) -> dict:
    """Spearman rank correlation of raw_score vs actual_alpha WITHIN each cutoff,
    averaged across cutoffs. This matches the cross-sectional (peer-relative) design
    better than a pooled Pearson that mixes in cross-period regime drift."""
    by_cutoff: dict[str, list[dict]] = {}
    for p in predictions:
        if p.get("raw_score") is not None:
            by_cutoff.setdefault(p.get("cutoff", "_"), []).append(p)
    ics = []
    for group in by_cutoff.values():
        if len(group) < 3:
            continue
        ic = _pearson(_ranks([p["raw_score"] for p in group]), _ranks([p["actual_alpha"] for p in group]))
        if ic is not None:
            ics.append(ic)
    if not ics:
        return {"mean": None, "n_cutoffs": 0}
    mean_ic = sum(ics) / len(ics)
    return {
        "mean": round(mean_ic, 3),
        "n_cutoffs": len(ics),
        "stdev": round(statistics.stdev(ics), 3) if len(ics) > 1 else None,
        "per_cutoff": {cutoff: round(_pearson(_ranks([p["raw_score"] for p in g]),
                                              _ranks([p["actual_alpha"] for p in g])) or 0.0, 3)
                       for cutoff, g in by_cutoff.items() if len(g) >= 3},
    }


def _quantile_spread(predictions: list[dict]) -> dict:
    """Mean alpha of top-bucket (4/5) calls minus bottom-bucket (1/2) calls. A
    calibrated scorer makes this clearly positive."""
    top = [p["actual_alpha"] for p in predictions if p["score"] >= 4]
    bottom = [p["actual_alpha"] for p in predictions if p["score"] <= 2]
    if not top or not bottom:
        return {"top_minus_bottom": None}
    return {
        "top_mean_alpha": round(sum(top) / len(top), 4),
        "bottom_mean_alpha": round(sum(bottom) / len(bottom), 4),
        "top_minus_bottom": round(sum(top) / len(top) - sum(bottom) / len(bottom), 4),
    }


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion (sane at small n, unlike normal)."""
    if n == 0:
        return None
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return [round(center - margin, 3), round(center + margin, 3)]


# --------------------------------------------------------------------------- #
# Per-slice breakdowns
# --------------------------------------------------------------------------- #

def breakdowns(report: dict, horizon_days: int = HORIZON_DAYS) -> dict:
    """Per-sector and per-cutoff sub-summaries, so a single blended number can't
    hide regime- or sector-dependence."""
    predictions = report["predictions"]
    by_sector, by_cutoff = {}, {}
    for p in predictions:
        by_sector.setdefault(p.get("sector") or "unknown", []).append(p)
        by_cutoff.setdefault(p["cutoff"], []).append(p)
    slim = lambda preds: {k: _summarise(preds, horizon_days).get(k)
                          for k in ("n", "directional_accuracy_excl_holds", "bucket_mae", "rank_ic")}
    return {
        "by_sector": {sector: slim(preds) for sector, preds in sorted(by_sector.items())},
        "by_cutoff": {cutoff: slim(preds) for cutoff, preds in sorted(by_cutoff.items())},
    }


# --------------------------------------------------------------------------- #
# Calibration fitting (held-out) and Phase1-vs-Phase2 comparison
# --------------------------------------------------------------------------- #

def fit_thresholds(predictions: list[dict], persist: bool = True) -> list[tuple[float, int]] | None:
    """Choose raw_score cut points at the 20/40/60/80th percentiles, so buckets are
    evenly populated and ordered by raw_score. Persisted to cache/calibration.json,
    which scoring.py then loads. Fit this on a FIT fold and evaluate on a held-out
    fold - fitting and reporting on the same data overstates skill."""
    raws = sorted(p["raw_score"] for p in predictions if p.get("raw_score") is not None)
    if len(raws) < 5:
        return None

    def percentile(sorted_values, pct):
        idx = min(len(sorted_values) - 1, max(0, int(round(pct / 100 * (len(sorted_values) - 1)))))
        return sorted_values[idx]

    thresholds = [
        (round(percentile(raws, 80), 4), 5),
        (round(percentile(raws, 60), 4), 4),
        (round(percentile(raws, 40), 4), 3),
        (round(percentile(raws, 20), 4), 2),
    ]
    if persist:
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CALIBRATION_FILE.open("w") as calibration_file:
            json.dump({"bucket_thresholds": [[t, b] for t, b in thresholds], "n": len(raws)}, calibration_file, indent=2)
        scoring.reload_thresholds()
    return thresholds


def fit_and_evaluate(tickers: list[str] | None = None, fit_cutoffs: list[str] | None = None,
                     eval_cutoffs: list[str] | None = None, horizon_days: int = HORIZON_DAYS) -> dict:
    """Fit bucket thresholds on the FIT cutoffs, then evaluate on the held-out EVAL
    cutoffs (a forward-in-time test). The eval summary is the honest headline."""
    fit_cutoffs = fit_cutoffs or DEFAULT_FIT_CUTOFFS
    eval_cutoffs = eval_cutoffs or DEFAULT_EVAL_CUTOFFS
    fit_report = run_backtest(tickers, fit_cutoffs, horizon_days, label="fit")
    thresholds = fit_thresholds(fit_report["predictions"], persist=True)
    eval_report = run_backtest(tickers, eval_cutoffs, horizon_days, label="held_out_eval")
    return {
        "fitted_thresholds": thresholds,
        "fit_cutoffs": fit_cutoffs,
        "eval_cutoffs": eval_cutoffs,
        "fit_summary": fit_report["summary"],
        "held_out_summary": eval_report["summary"],
        "held_out_breakdowns": breakdowns(eval_report, horizon_days),
    }


def compare(tickers: list[str] | None = None, cutoffs: list[str] | None = None,
            horizon_days: int = HORIZON_DAYS) -> dict:
    """Phase-1-vs-Phase-2 head-to-head on the SAME (ticker, cutoff) grid, so you can
    see whether Nova beats the math baseline and at what token cost. Phase 2 needs
    Bedrock credentials; cached prices mean the only extra cost is Nova tokens."""
    from phase2 import score_ticker_v2  # local import: avoids requiring Bedrock for Phase 1 runs

    phase1 = run_backtest(tickers, cutoffs, horizon_days, scorer=score_ticker, score_key="score", label="phase1")
    # Gentler concurrency for the Nova leg to stay under Bedrock rate limits.
    phase2 = run_backtest(tickers, cutoffs, horizon_days, scorer=score_ticker_v2,
                          score_key="final_score", label="phase2", max_workers=4)
    return {"phase1": phase1["summary"], "phase2": phase2["summary"],
            "phase1_cost": phase1["summary"].get("cost"), "phase2_cost": phase2["summary"].get("cost")}


if __name__ == "__main__":
    report = run_backtest()
    summary = report["summary"]
    print(f"\n{'='*64}")
    print(f" BACKTEST ({report['label']})  |  {summary['n']} predictions  |  {summary.get('horizon_days')}-day horizon")
    print(f"{'='*64}")
    if summary["n"]:
        directional = summary["directional_accuracy_excl_holds"]
        baselines = summary["baselines"]
        if directional is not None:
            ci = summary["directional_accuracy_ci95"]
            print(f" Directional accuracy (decided calls): {directional*100:.0f}%  95% CI [{ci[0]*100:.0f}, {ci[1]*100:.0f}]"
                  f"  ({summary['decided_calls']} decided, {summary['pushes']} pushes, {summary['holds']} holds)")
            print(f"   vs always-Buy {(baselines['always_buy_directional_accuracy'] or 0)*100:.0f}%"
                  f"  vs random {(baselines['random_directional_accuracy'] or 0)*100:.0f}%")
        print(f" Rank-IC (within-cutoff, avg):      {summary['rank_ic']['mean']}  (cross-sectional skill; >0 is good)")
        print(f" Quantile spread (top-bottom alpha):{summary['quantile_spread'].get('top_minus_bottom')}")
        print(f" Corr(raw_score, actual alpha):     {summary['corr_raw_vs_alpha']}")
        print(f" Bucket MAE:                        {summary['bucket_mae']}  (vs always-Hold {baselines['always_hold_bucket_mae']})")
        cost = summary["cost"]
        print(f" Cost: ${cost['total_cost_usd']}  |  {cost['total_tokens']} tokens  |  mean latency {cost['mean_latency_sec']}s")
        print(f"{'-'*64}")
        print(" Calibration (actual alpha per predicted score):")
        print(f"   {'score':>5s} {'count':>6s} {'median':>10s} {'mean':>10s}")
        for bucket_score in range(5, 0, -1):
            entry = summary["calibration_by_score"].get(bucket_score)
            if entry:
                print(f"   {bucket_score:>5d} {entry['count']:>6d} {entry['median_actual_alpha']*100:>9.1f}% {entry['mean_actual_alpha']*100:>9.1f}%")
        for warning in summary["warnings"]:
            print(f" ! {warning}")
    print(f"{'='*64}\n")
