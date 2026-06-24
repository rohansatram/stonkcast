"""Metric math + point-in-time correctness on synthetic data with hand-checkable
expected values."""

import math
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from metrics import compute_metrics, _eps_trend, _annualized_vol, _finite, _momentum, _earnings_surprise

UTC = timezone.utc


def test_compute_metrics_known_values(synth_data, synth_spy, cutoff):
    m = compute_metrics(synth_data, synth_spy, cutoff)
    # momentum_30d = 159/138 - 1 (closes[-1]/closes[-22])
    assert m["momentum_30d"] == pytest.approx(159 / 138 - 1, rel=1e-6)
    # 52-week position: High=Low=Close linear -> last is the max -> 1.0
    assert m["fifty_two_week_position"] == pytest.approx(1.0, rel=1e-6)
    # P/E = last_close(159) / trailing EPS(sum last 4 = 8)
    assert m["pe_ratio"] == pytest.approx(159 / 8, rel=1e-6)
    # EPS trend = TTM(8) vs prior TTM(4) -> +1.0
    assert m["eps_trend"] == pytest.approx(1.0, rel=1e-6)
    # revenue growth 100 -> 110
    assert m["revenue_growth"] == pytest.approx(0.10, rel=1e-6)
    # ebitda margin = 40/110
    assert m["ebitda_margin"] == pytest.approx(40 / 110, rel=1e-6)
    # D/E = 50/100
    assert m["debt_to_equity"] == pytest.approx(0.5, rel=1e-6)
    # earnings surprise = mean of the last 4 pre-cutoff surprises [1,1,1,5] = 2.0
    # (not the single latest, and never the post-cutoff 50.0)
    assert m["earnings_surprise"] == pytest.approx(2.0)
    # momentum vs flat SPY = stock momentum
    assert m["momentum_vs_spy"] == pytest.approx(m["momentum_30d"], rel=1e-6)


def test_metrics_are_deterministic(synth_data, synth_spy, cutoff):
    a = compute_metrics(synth_data, synth_spy, cutoff)
    b = compute_metrics(synth_data, synth_spy, cutoff)
    assert a == b


def test_post_cutoff_data_never_leaks(synth_data, synth_spy, cutoff):
    # trailing EPS uses [2,2,2,2]=8, so the post-cutoff 99.0 EPS must be absent
    m = compute_metrics(synth_data, synth_spy, cutoff)
    assert m["pe_ratio"] == pytest.approx(159 / 8, rel=1e-6)
    assert m["earnings_surprise"] != 50.0


def test_no_price_history_returns_none(synth_data, synth_spy):
    synth_data = {**synth_data, "prices": pl.DataFrame()}
    assert compute_metrics(synth_data, synth_spy, datetime(2025, 1, 1, tzinfo=UTC)) is None


def test_nan_revenue_yields_none(synth_data, synth_spy, cutoff):
    synth_data = {**synth_data, "financials_annual": pl.DataFrame({
        "period_end": [datetime(2022, 12, 31, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)],
        "total_revenue": [float("nan"), 110.0],
        "ebitda": [30.0, 40.0],
    })}
    m = compute_metrics(synth_data, synth_spy, cutoff)
    assert m["revenue_growth"] is None  # NaN prior year -> no false number


def test_eps_trend_ttm_neutralises_seasonality():
    # strongly seasonal (big Q1) but flat year-over-year -> ~0 trend
    seasonal = [4, 1, 1, 1, 4, 1, 1, 1]
    assert _eps_trend(seasonal) == pytest.approx(0.0)


def test_eps_trend_needs_enough_quarters():
    assert _eps_trend([1, 2, 3]) is None


def test_momentum_skip_excludes_recent_window():
    closes = [100.0 + i for i in range(160)]
    # 6-month momentum: 126-day return ending 21 days before the last bar
    assert _momentum(closes, 126, 21) == pytest.approx(closes[-22] / closes[-148] - 1)
    assert _momentum([100.0] * 100, 126, 21) is None  # not enough history (need >147)


def test_earnings_surprise_derived_from_estimate_when_pct_missing():
    e = pl.DataFrame({
        "eps_reported": [1.1, 1.2, 1.3, 1.4],
        "eps_estimate": [1.0, 1.0, 1.0, 1.0],
        "surprise_pct": [None, None, None, None],
    })
    # derived surprises (rep-est)/|est|*100 = [10,20,30,40]; mean of last 4 = 25
    assert _earnings_surprise(e) == pytest.approx(25.0)


def test_annualized_vol_zero_for_flat_series():
    assert _annualized_vol([100.0] * 60) == pytest.approx(0.0)


def test_finite_filters_nan_and_inf():
    assert _finite(float("nan")) is None
    assert _finite(float("inf")) is None
    assert _finite(None) is None
    assert _finite(3.5) == 3.5


def test_negative_equity_yields_no_debt_to_equity(synth_data, synth_spy, cutoff):
    synth_data = {**synth_data, "balance_annual": pl.DataFrame({
        "period_end": [datetime(2023, 12, 31, tzinfo=UTC)],
        "total_debt": [200.0],
        "stockholders_equity": [-50.0],  # distressed: negative equity
    })}
    m = compute_metrics(synth_data, synth_spy, cutoff)
    assert m["debt_to_equity"] is None  # must NOT become a bullish negative ratio


def test_null_close_bar_does_not_crash(synth_data, synth_spy, cutoff):
    closes = [None] + [100.0 + i for i in range(59)]  # a null bar at the start
    dates = [datetime(2024, 12, 31, tzinfo=UTC) - timedelta(days=59 - i) for i in range(60)]
    synth_data = {**synth_data, "prices": pl.DataFrame({
        "Date": dates, "Open": closes, "High": closes, "Low": closes,
        "Close": closes, "Volume": [1] * 60,
    })}
    m = compute_metrics(synth_data, synth_spy, cutoff)  # must not raise
    assert m is not None
    assert m["momentum_30d"] is not None
