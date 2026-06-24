"""
Fetch FULL-history stock data for a ticker and normalise it into clean polars
structures. No cutoff filtering happens here - that is applied downstream
(see pointintime.py) so one fetch serves every cutoff date.

Use fetch_cached() for the rate-limit-friendly path; fetch() always hits yfinance.
"""

import yfinance as yf
import polars as pl
import pandas as pd
import numpy as np

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from cache import cached_fetch  # noqa: E402

PRICE_LOOKBACK_START = "2015-01-01"  # deep enough for early-cutoff backtests + 252d windows
EARNINGS_LIMIT = 28  # ~7 years of quarterly announcements
PRICE_FETCH_ATTEMPTS = 3  # yfinance is flaky; a cold judge-named ticker shouldn't die on a transient blip
PRICE_FETCH_BACKOFF_SEC = 1.0


def fetch_cached(ticker: str, refresh: bool = False, max_age=None) -> dict:
    """Cached full-history fetch (preferred). See cache.cached_fetch. `max_age`
    (a timedelta) re-fetches a blob older than it - the live run uses a short TTL."""
    return cached_fetch(ticker, fetch, refresh=refresh, max_age=max_age)


def fetch(ticker: str) -> dict:
    """Uncached full-history fetch straight from yfinance."""
    stock = yf.Ticker(ticker)
    return {
        "ticker": ticker.upper(),
        "prices": _fetch_prices(stock),
        "earnings": _fetch_earnings(stock),
        "financials_annual": _fetch_financials_annual(stock),
        "financials_quarterly": _fetch_financials_quarterly(stock),
        "balance_annual": _fetch_balance_annual(stock),
        "info": _fetch_info(stock),
    }


def _to_utc(date_series: pd.Series) -> pd.Series:
    if date_series.dt.tz is None:
        return date_series.dt.tz_localize("UTC")
    return date_series.dt.tz_convert("UTC")


def _fetch_prices(stock: yf.Ticker) -> pl.DataFrame:
    """
    Full daily OHLCV history, SPLIT-adjusted only (never dividend-adjusted).

    We deliberately avoid yfinance's auto_adjust=True: it also back-adjusts
    pre-cutoff bars for dividends paid AFTER the cutoff, which leaks future info
    into the P/E price level (the bias grows with backtest age). Split-only
    adjustment keeps the series internally consistent and on the same per-share
    basis as the (split-adjusted) reported EPS, while using raw price returns
    for momentum / volatility / 52-week position. Splits are scale-invariant in
    every ratio metric, so this is leak-free.
    """
    price_frame = pd.DataFrame()
    for attempt in range(PRICE_FETCH_ATTEMPTS):
        try:
            price_frame = stock.history(start=PRICE_LOOKBACK_START, auto_adjust=False, actions=True).reset_index()
        except Exception:
            price_frame = pd.DataFrame()
        if not price_frame.empty:
            break
        if attempt < PRICE_FETCH_ATTEMPTS - 1:
            time.sleep(PRICE_FETCH_BACKOFF_SEC * (attempt + 1))  # transient blip / rate limit -> back off and retry
    if price_frame.empty:
        return pl.DataFrame()
    price_frame["Date"] = _to_utc(price_frame["Date"])
    price_frame = price_frame.dropna(subset=["Close"])
    price_frame = price_frame[price_frame["Close"] > 0].reset_index(drop=True)
    if price_frame.empty:
        return pl.DataFrame()
    price_frame = _split_adjust(price_frame)
    columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    price_frame = price_frame[[column for column in columns_to_keep if column in price_frame.columns]]
    return pl.from_pandas(price_frame).sort("Date")


def _split_adjust(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust OHLC to the present share basis using the split history, so a
    split occurring mid-series doesn't appear as a phantom price jump. Rows are
    oldest-first; the factor for a bar is the product of all splits after it."""
    if "Stock Splits" not in price_frame.columns:
        return price_frame
    split_ratios = price_frame["Stock Splits"].fillna(0).to_numpy()
    num_rows = len(price_frame)
    split_factors = np.ones(num_rows)
    cumulative_factor = 1.0
    for row_index in range(num_rows - 1, -1, -1):
        split_factors[row_index] = cumulative_factor
        if split_ratios[row_index] and split_ratios[row_index] > 0:
            cumulative_factor *= split_ratios[row_index]
    for column in ("Open", "High", "Low", "Close"):
        if column in price_frame.columns:
            price_frame[column] = price_frame[column].to_numpy() / split_factors
    return price_frame


def _fetch_earnings(stock: yf.Ticker) -> pl.DataFrame:
    """
    Earnings keyed by true ANNOUNCEMENT date (not period end). This is the
    point-in-time anchor: a quarter's EPS only becomes usable once announced.
    Columns: announce_date (UTC), eps_estimate, eps_reported, surprise_pct.
    """
    try:
        earnings_frame = stock.get_earnings_dates(limit=EARNINGS_LIMIT)
    except Exception:
        return pl.DataFrame()
    if earnings_frame is None or earnings_frame.empty:
        return pl.DataFrame()

    earnings_frame = earnings_frame.reset_index()
    earnings_frame.columns = [str(column) for column in earnings_frame.columns]
    date_column = next((column for column in earnings_frame.columns if "Date" in column), earnings_frame.columns[0])
    earnings_frame = earnings_frame.rename(columns={
        date_column: "announce_date",
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_reported",
        "Surprise(%)": "surprise_pct",
    })
    earnings_frame["announce_date"] = _to_utc(pd.to_datetime(earnings_frame["announce_date"]))
    wanted_columns = ["announce_date", "eps_estimate", "eps_reported", "surprise_pct"]
    earnings_frame = earnings_frame[[column for column in wanted_columns if column in earnings_frame.columns]]
    return pl.from_pandas(earnings_frame).sort("announce_date")


def _extract_row(statement: pd.DataFrame, *candidate_names: str) -> pd.Series | None:
    for name in candidate_names:
        if name in statement.index:
            return statement.loc[name]
    return None


def _statement_to_polars(statement: pd.DataFrame, field_map: dict[str, tuple[str, ...]]) -> pl.DataFrame:
    """Turn a yfinance statement (metrics x period-end columns) into rows of
    {period_end, <field>...}, oldest first. Missing fields become null."""
    if statement is None or statement.empty:
        return pl.DataFrame()
    period_ends = pd.to_datetime(statement.columns, utc=True)
    rows = {"period_end": list(period_ends)}
    for field_name, candidate_names in field_map.items():
        source_row = _extract_row(statement, *candidate_names)
        rows[field_name] = [
            None if source_row is None else source_row.iloc[i] for i in range(len(statement.columns))
        ]
    return pl.from_pandas(pd.DataFrame(rows)).sort("period_end")


def _fetch_financials_annual(stock: yf.Ticker) -> pl.DataFrame:
    try:
        financials = stock.get_financials(freq="yearly")
    except Exception:
        return pl.DataFrame()
    return _statement_to_polars(financials, {
        "total_revenue": ("TotalRevenue", "OperatingRevenue"),
        "ebitda": ("EBITDA", "NormalizedEBITDA"),
    })


def _fetch_financials_quarterly(stock: yf.Ticker) -> pl.DataFrame:
    """Quarterly revenue, for a fresher (and seasonality-neutral) revenue-growth
    read than the annual statement. yfinance typically exposes only ~5 quarters."""
    try:
        financials = stock.get_financials(freq="quarterly")
    except Exception:
        return pl.DataFrame()
    return _statement_to_polars(financials, {
        "total_revenue": ("TotalRevenue", "OperatingRevenue"),
    })


def _fetch_balance_annual(stock: yf.Ticker) -> pl.DataFrame:
    try:
        balance_sheet = stock.get_balance_sheet(freq="yearly")
    except Exception:
        return pl.DataFrame()
    return _statement_to_polars(balance_sheet, {
        "total_debt": ("TotalDebt",),
        "stockholders_equity": ("StockholdersEquity", "CommonStockEquity", "TotalEquityGrossMinorityInterest"),
    })


def _fetch_info(stock: yf.Ticker) -> dict:
    """
    Sector / industry / name plus analyst consensus.

    NOT-POINT-IN-TIME NOTE: `sector` is the CURRENT (fetch-time) classification, not
    a cutoff snapshot. It only routes which peer basket a name is scored against, so
    the leakage is second-order (a reclassified name would have used different peers),
    but it is not strictly point-in-time and is disclosed as such downstream.

    LEAKAGE WARNING: recommendation_mean / recommendation_key are the CURRENT
    (fetch-time) analyst ratings, not a cutoff snapshot. yfinance has no
    historical value, so these bake in post-cutoff info. Display them for brief
    compliance, but EXCLUDE them from scoring and backtests.
    """
    try:
        raw_info = stock.info
    except Exception:
        raw_info = {}
    return {
        "sector": raw_info.get("sector"),
        "industry": raw_info.get("industry"),
        "long_name": raw_info.get("longName"),
        "recommendation_mean": raw_info.get("recommendationMean"),  # leaky: display-only
        "recommendation_key": raw_info.get("recommendationKey"),    # leaky: display-only
    }


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching FULL history for {ticker} ...")
    data = fetch_cached(ticker)
    print(f"\n=== {data['info']['long_name']} ({data['ticker']}) ===")
    print(f"Sector: {data['info']['sector']} | Industry: {data['info']['industry']}")
    print(f"Prices: {len(data['prices'])} rows", end="")
    if not data["prices"].is_empty():
        print(f" ({data['prices']['Date'][0].date()} -> {data['prices']['Date'][-1].date()})")
    print(f"Earnings announcements: {len(data['earnings'])}")
    print(f"Annual financials: {len(data['financials_annual'])} | Annual balance sheet: {len(data['balance_annual'])}")
