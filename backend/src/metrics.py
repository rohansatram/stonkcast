"""
Compute raw metric values for a ticker as of a cutoff date. Pure math, no LLM.
Every value is derived only from data knowable at the cutoff (see pointintime).

Any metric that can't be computed (insufficient history, missing statement,
non-positive denominator) comes back as None and is treated as neutral by the
scorer rather than guessed.
"""

import math
from datetime import datetime

import polars as pl

from pointintime import prices_asof, earnings_asof, statement_asof

TRADING_DAYS_30D = 21       # ~30 calendar days
TRADING_DAYS_52W = 252      # ~52 weeks
MOMENTUM_LOOKBACK = TRADING_DAYS_30D

# Medium-horizon momentum, aligned with the ~6-month forward-alpha target. We form
# it over ~6 months but SKIP the most recent month, because the 1-month window is
# dominated by short-term reversal rather than the 6-12 month momentum premium that
# actually persists over a 6-month horizon.
MOMENTUM_6M_LOOKBACK = 126   # ~6 months of trading days
MOMENTUM_6M_SKIP = TRADING_DAYS_30D
SURPRISE_WINDOW = 4          # average the last N earnings surprises (single quarter is noisy)


def _finite(value) -> float | None:
    """None for missing / NaN / inf, so bad data never silently poisons a score."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    numerator, denominator = _finite(numerator), _finite(denominator)
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _momentum(closing_prices: list[float], lookback: int = MOMENTUM_LOOKBACK, skip: int = 0) -> float | None:
    """Return over `lookback` trading days, optionally ending `skip` days before the
    last bar (skip>0 excludes the most recent, reversal-prone window)."""
    if len(closing_prices) <= lookback + skip:
        return None
    latest_price = _finite(closing_prices[-1 - skip])
    prior_price = _finite(closing_prices[-1 - skip - lookback])
    if latest_price is None or prior_price is None or prior_price == 0:
        return None
    return latest_price / prior_price - 1.0


def _relative_momentum(closing_prices: list[float], spy_prices: list[float],
                       lookback: int, skip: int = 0) -> float | None:
    """Stock momentum minus SPY momentum over the same window (alpha-aligned)."""
    stock = _momentum(closing_prices, lookback, skip)
    market = _momentum(spy_prices, lookback, skip) if spy_prices else None
    if stock is None or market is None:
        return None
    return stock - market


def _annualized_vol(closing_prices: list[float], window: int = TRADING_DAYS_52W) -> float | None:
    price_window = [_finite(price) for price in closing_prices[-(window + 1):]]
    if len(price_window) < 20:
        return None
    daily_returns = [
        price_window[i] / price_window[i - 1] - 1.0
        for i in range(1, len(price_window))
        if price_window[i] is not None and price_window[i - 1] not in (None, 0)
    ]
    if len(daily_returns) < 2:
        return None
    mean_return = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    return math.sqrt(variance) * math.sqrt(252)


def _eps_trend(reported_eps: list[float]) -> float | None:
    """TTM-over-TTM EPS growth (seasonality-neutral). Needs 8 quarters; falls
    back to same-quarter YoY (q[-1] vs q[-5]) if only 5-7 are available."""
    if len(reported_eps) >= 8:
        recent_ttm = sum(reported_eps[-4:])
        prior_ttm = sum(reported_eps[-8:-4])
        return (recent_ttm - prior_ttm) / abs(prior_ttm) if prior_ttm != 0 else None
    if len(reported_eps) >= 5 and reported_eps[-5] != 0:
        return (reported_eps[-1] - reported_eps[-5]) / abs(reported_eps[-5])
    return None


def _earnings_surprise(earnings: pl.DataFrame) -> float | None:
    """Average of the last few earnings surprises (a single quarter is noisy and
    decays fast over a 6-month horizon). Uses surprise_pct when present, otherwise
    derives it from reported vs estimated EPS, which yfinance also provides."""
    if earnings.is_empty():
        return None
    columns = earnings.columns
    reported = earnings["eps_reported"].to_list() if "eps_reported" in columns else []
    estimated = earnings["eps_estimate"].to_list() if "eps_estimate" in columns else []
    pct = earnings["surprise_pct"].to_list() if "surprise_pct" in columns else []

    surprises: list[float] = []
    for i in range(earnings.height):
        value = _finite(pct[i]) if i < len(pct) else None
        if value is None and i < len(reported) and i < len(estimated):
            rep, est = _finite(reported[i]), _finite(estimated[i])
            if rep is not None and est not in (None, 0):
                value = (rep - est) / abs(est) * 100.0
        if value is not None:
            surprises.append(value)

    window = surprises[-SURPRISE_WINDOW:]
    return sum(window) / len(window) if window else None


def _revenue_growth(financials_quarterly: pl.DataFrame, financials_annual: pl.DataFrame,
                    earnings: pl.DataFrame, cutoff: datetime) -> float | None:
    """Prefer fresh quarterly revenue growth (TTM-over-TTM with 8 quarters, else
    same-quarter YoY with 5-7, both seasonality-neutral), falling back to annual
    YoY only when too few quarters are public. Annual alone can be ~12 months stale."""
    quarterly = statement_asof(financials_quarterly, earnings, cutoff)
    if quarterly.height >= 5 and "total_revenue" in quarterly.columns:
        revenue = [_finite(v) for v in quarterly["total_revenue"].to_list()]
        if len(revenue) >= 8 and all(v is not None for v in revenue[-8:]):
            recent_ttm, prior_ttm = sum(revenue[-4:]), sum(revenue[-8:-4])
            if prior_ttm:
                return (recent_ttm - prior_ttm) / abs(prior_ttm)
        if revenue[-1] is not None and revenue[-5] not in (None, 0):
            return (revenue[-1] - revenue[-5]) / abs(revenue[-5])

    if financials_annual.height >= 2:
        latest = _finite(financials_annual["total_revenue"][-1])
        prior = _finite(financials_annual["total_revenue"][-2])
        if latest is not None and prior:
            return _safe_div(latest - prior, abs(prior))
    return None


def _fifty_two_week_position(prices: pl.DataFrame) -> float | None:
    price_window = prices.tail(TRADING_DAYS_52W)
    if price_window.height < 20:
        return None
    high = _finite(price_window["High"].max())
    low = _finite(price_window["Low"].min())
    last_price = _finite(price_window["Close"][-1])
    if high is None or low is None or last_price is None or high == low:
        return None
    return (last_price - low) / (high - low)


def compute_metrics(data: dict, spy_prices: pl.DataFrame, cutoff: datetime) -> dict | None:
    """
    Returns {metric_name: raw_value_or_None, ...} plus a few context fields.
    Returns None only if there is no usable price history at all.
    """
    prices = prices_asof(data["prices"], cutoff)
    if not prices.is_empty():
        prices = prices.filter(pl.col("Close").is_not_null() & pl.col("Close").is_finite())
    if prices.is_empty() or prices.height < 2:
        return None

    closing_prices = prices["Close"].to_list()
    last_close = closing_prices[-1]

    metrics: dict[str, float | None] = {}

    # --- price-derived ---
    metrics["momentum_30d"] = _momentum(closing_prices)
    metrics["fifty_two_week_position"] = _fifty_two_week_position(prices)
    metrics["volatility"] = _annualized_vol(closing_prices)

    spy_prices_pit = (
        prices_asof(spy_prices, cutoff) if spy_prices is not None and not spy_prices.is_empty() else pl.DataFrame()
    )
    spy_closes = spy_prices_pit["Close"].to_list() if not spy_prices_pit.is_empty() else []
    spy_momentum = _momentum(spy_closes) if spy_closes else None
    metrics["momentum_vs_spy"] = (
        metrics["momentum_30d"] - spy_momentum
        if metrics["momentum_30d"] is not None and spy_momentum is not None
        else None
    )
    # ~6-month relative momentum (skips the most recent month). This is the
    # horizon-aligned momentum signal; the 30-day terms are kept at low weight.
    metrics["momentum_6m_vs_spy"] = _relative_momentum(
        closing_prices, spy_closes, MOMENTUM_6M_LOOKBACK, MOMENTUM_6M_SKIP
    )

    # --- earnings-derived (announcement-dated) ---
    earnings = earnings_asof(data.get("earnings", pl.DataFrame()), cutoff)
    reported_eps = (
        [value for value in earnings["eps_reported"].to_list() if value is not None]
        if not earnings.is_empty() and "eps_reported" in earnings.columns
        else []
    )
    last_four_eps = reported_eps[-4:]
    trailing_eps = sum(last_four_eps) if len(last_four_eps) == 4 else None

    metrics["eps_trend"] = _eps_trend(reported_eps)

    metrics["earnings_surprise"] = _earnings_surprise(earnings)

    # P/E computed point-in-time: cutoff price / trailing EPS (never the live info value)
    metrics["pe_ratio"] = _safe_div(last_close, trailing_eps) if (trailing_eps and trailing_eps > 0) else None

    # --- financial-statement-derived (announcement-mapped) ---
    financials = statement_asof(data.get("financials_annual", pl.DataFrame()), earnings, cutoff)
    metrics["revenue_growth"] = _revenue_growth(
        data.get("financials_quarterly", pl.DataFrame()), financials, earnings, cutoff
    )

    if financials.height >= 1:
        metrics["ebitda_margin"] = _safe_div(financials["ebitda"][-1], financials["total_revenue"][-1])
    else:
        metrics["ebitda_margin"] = None

    balance_sheet = statement_asof(data.get("balance_annual", pl.DataFrame()), earnings, cutoff)
    if balance_sheet.height >= 1:
        # Negative equity makes D/E negative, which would wrongly read as low
        # leverage (bullish). Treat non-positive equity as undefined / neutral.
        equity = _finite(balance_sheet["stockholders_equity"][-1])
        metrics["debt_to_equity"] = (
            _safe_div(balance_sheet["total_debt"][-1], equity) if (equity is not None and equity > 0) else None
        )
    else:
        metrics["debt_to_equity"] = None

    # Final guard: coerce every metric to a finite float or None so no NaN/inf
    # can reach the z-score / weighted-sum stage.
    metrics = {name: _finite(value) for name, value in metrics.items()}
    metrics["_last_close"] = last_close
    metrics["_n_prices"] = prices.height
    metrics["_last_price_date"] = prices["Date"][-1]
    return metrics
