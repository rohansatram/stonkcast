"""Shared test setup: put src/ on the path and provide synthetic, offline data
builders so unit tests never touch the network."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _default_scoring_params():
    """Isolate scoring tests from any fitted cache/calibration.json or weights.json on
    disk: every test starts from the hand-set defaults unless it sets its own."""
    import scoring
    scoring.BUCKET_THRESHOLDS = list(scoring.DEFAULT_BUCKET_THRESHOLDS)
    scoring.WEIGHTS = dict(scoring.DEFAULT_WEIGHTS)
    yield


def _daily_prices(end: datetime, closes: list[float]) -> pl.DataFrame:
    """Ascending daily OHLCV ending on `end`, with High=Low=Close for simplicity."""
    n = len(closes)
    dates = [end - timedelta(days=(n - 1 - i)) for i in range(n)]
    return pl.DataFrame({
        "Date": dates,
        "Open": closes,
        "High": closes,
        "Low": closes,
        "Close": closes,
        "Volume": [1_000_000] * n,
    })


@pytest.fixture
def cutoff() -> datetime:
    return datetime(2025, 1, 1, tzinfo=UTC)


@pytest.fixture
def synth_prices() -> pl.DataFrame:
    # 60 ascending closes 100..159, ending 2024-12-31, plus one post-cutoff bar
    closes = [100.0 + i for i in range(60)]
    df = _daily_prices(datetime(2024, 12, 31, tzinfo=UTC), closes)
    post = pl.DataFrame({
        "Date": [datetime(2025, 1, 2, tzinfo=UTC)],
        "Open": [999.0], "High": [999.0], "Low": [999.0], "Close": [999.0], "Volume": [1],
    })
    return pl.concat([df, post]).sort("Date")


@pytest.fixture
def synth_spy() -> pl.DataFrame:
    return _daily_prices(datetime(2024, 12, 31, tzinfo=UTC), [400.0] * 60)


@pytest.fixture
def synth_earnings() -> pl.DataFrame:
    # 8 quarters before cutoff (EPS 1,1,1,1,2,2,2,2) + 1 announced AFTER cutoff
    announce = [
        datetime(2023, 2, 1, tzinfo=UTC), datetime(2023, 5, 1, tzinfo=UTC),
        datetime(2023, 8, 1, tzinfo=UTC), datetime(2023, 11, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 5, 1, tzinfo=UTC),
        datetime(2024, 8, 1, tzinfo=UTC), datetime(2024, 11, 1, tzinfo=UTC),
        datetime(2025, 2, 1, tzinfo=UTC),  # post-cutoff: must be excluded
    ]
    eps = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 99.0]
    return pl.DataFrame({
        "announce_date": announce,
        "eps_estimate": [e - 0.1 for e in eps],
        "eps_reported": eps,
        "surprise_pct": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 50.0],
    })


@pytest.fixture
def synth_financials() -> pl.DataFrame:
    return pl.DataFrame({
        "period_end": [datetime(2022, 12, 31, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)],
        "total_revenue": [100.0, 110.0],
        "ebitda": [30.0, 40.0],
    })


@pytest.fixture
def synth_balance() -> pl.DataFrame:
    return pl.DataFrame({
        "period_end": [datetime(2023, 12, 31, tzinfo=UTC)],
        "total_debt": [50.0],
        "stockholders_equity": [100.0],
    })


@pytest.fixture
def synth_data(synth_prices, synth_earnings, synth_financials, synth_balance) -> dict:
    return {
        "ticker": "TEST",
        "prices": synth_prices,
        "earnings": synth_earnings,
        "financials_annual": synth_financials,
        "balance_annual": synth_balance,
        "info": {"sector": "Technology", "industry": "X", "long_name": "Test Co",
                 "recommendation_mean": 2.0, "recommendation_key": "buy"},
    }
