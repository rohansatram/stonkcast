import yfinance as yf
import polars as pl
import pandas as pd
from datetime import datetime, timezone

CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)
CUTOFF_PD = pd.Timestamp("2026-01-01", tz="UTC")  # yfinance returns pandas, so we need this for column filtering
PRICE_LOOKBACK_START = "2018-01-01"


def fetch(ticker: str) -> dict:
    stock = yf.Ticker(ticker)

    prices = _fetch_prices(stock)
    financials = _fetch_financials(stock)
    balance_sheet = _fetch_balance_sheet(stock)
    earnings = _fetch_earnings(stock)
    info = _fetch_info(stock)

    return {
        "ticker": ticker.upper(),
        "prices": prices,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "earnings": earnings,
        "info": info,
    }


def _fetch_prices(stock: yf.Ticker) -> pl.DataFrame:
    pandas_prices = stock.history(start=PRICE_LOOKBACK_START, end=CUTOFF_PD).reset_index()
    if pandas_prices["Date"].dt.tz is None:
        pandas_prices["Date"] = pandas_prices["Date"].dt.tz_localize("UTC")
    else:
        pandas_prices["Date"] = pandas_prices["Date"].dt.tz_convert("UTC")
    return pl.from_pandas(pandas_prices).filter(pl.col("Date") < CUTOFF)


def _fetch_financials(stock: yf.Ticker) -> pl.DataFrame:
    """
    Quarterly income statement (revenue, EBITDA, EPS etc.).
    yfinance returns this with metrics as rows and quarter dates as columns.
    We transpose it so each row is a quarter and each column is a metric - easier to work with.
    """
    pandas_financials = stock.get_financials(freq="quarterly")
    if pandas_financials is None or pandas_financials.empty:
        return pl.DataFrame()

    pandas_financials.columns = pd.to_datetime(pandas_financials.columns, utc=True)
    columns_before_cutoff = [col for col in pandas_financials.columns if col < CUTOFF_PD]
    pandas_financials = pandas_financials[columns_before_cutoff]

    transposed = pandas_financials.T.reset_index()
    transposed.columns = ["quarter"] + list(pandas_financials.index)
    transposed.columns = [str(col) for col in transposed.columns]
    return pl.from_pandas(transposed)


def _fetch_balance_sheet(stock: yf.Ticker) -> pl.DataFrame:
    """
    Quarterly balance sheet (debt, equity).
    Same transposition as financials - each row is a quarter.
    """
    pandas_balance_sheet = stock.get_balance_sheet(freq="quarterly")
    if pandas_balance_sheet is None or pandas_balance_sheet.empty:
        return pl.DataFrame()

    pandas_balance_sheet.columns = pd.to_datetime(pandas_balance_sheet.columns, utc=True)
    columns_before_cutoff = [col for col in pandas_balance_sheet.columns if col < CUTOFF_PD]
    pandas_balance_sheet = pandas_balance_sheet[columns_before_cutoff]

    transposed = pandas_balance_sheet.T.reset_index()
    transposed.columns = ["quarter"] + list(pandas_balance_sheet.index)
    transposed.columns = [str(col) for col in transposed.columns]
    return pl.from_pandas(transposed)


def _fetch_earnings(stock: yf.Ticker) -> pl.DataFrame:
    """Earnings history: actual EPS vs analyst estimate, used for the PEAD signal."""
    pandas_earnings = stock.get_earnings_history()
    if pandas_earnings is None or pandas_earnings.empty:
        return pl.DataFrame()

    pandas_earnings = pandas_earnings.reset_index()
    pandas_earnings["quarter"] = pd.to_datetime(pandas_earnings["quarter"], utc=True)
    return pl.from_pandas(pandas_earnings).filter(pl.col("quarter") < CUTOFF)


def _fetch_info(stock: yf.Ticker) -> dict:
    """
    Sector, industry, analyst consensus. Live data from yfinance -
    not snapshotted to the cutoff date, treat as approximately correct for Jan 2026.
    """
    raw_info = stock.info
    return {
        "sector": raw_info.get("sector"),
        "industry": raw_info.get("industry"),
        "recommendation_mean": raw_info.get("recommendationMean"),
        "recommendation_key": raw_info.get("recommendationKey"),
        "long_name": raw_info.get("longName"),
    }


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching data for {ticker} (cutoff: {CUTOFF.date()})...")
    data = fetch(ticker)

    print(f"\n=== {data['info']['long_name']} ({data['ticker']}) ===")
    print(f"Sector: {data['info']['sector']} | Industry: {data['info']['industry']}")
    print(f"Analyst consensus: {data['info']['recommendation_key']} ({data['info']['recommendation_mean']})")

    print(f"\nPrice history: {len(data['prices'])} trading days")
    print(f"  First: {data['prices']['Date'][0].date()}  Last: {data['prices']['Date'][-1].date()}")
    print(f"  Last close: ${data['prices']['Close'][-1]:.2f}")

    print(f"\nQuarterly financials: {len(data['financials'])} quarters")
    if not data['financials'].is_empty():
        print(f"  Quarters: {data['financials']['quarter'].to_list()}")

    print(f"\nBalance sheet: {len(data['balance_sheet'])} quarters")

    print(f"\nEarnings history: {len(data['earnings'])} entries")
    if not data['earnings'].is_empty():
        print(data['earnings'].tail(4))
