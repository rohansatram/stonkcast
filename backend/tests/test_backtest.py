"""Backtest harness pure logic (offline)."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from backtest import (
    forward_alpha, outcome_bucket, _price_on_or_after, _pearson, _summarise,
    fit_thresholds, _wilson_interval, _rank_ic,
)

UTC = timezone.utc


def _series(start: datetime, closes: list[float]) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(len(closes))]
    return pl.DataFrame({"Date": dates, "Close": closes})


def test_price_on_or_after():
    p = _series(datetime(2025, 1, 1, tzinfo=UTC), [10.0, 11.0, 12.0])
    assert _price_on_or_after(p, datetime(2025, 1, 2, tzinfo=UTC)) == 11.0
    assert _price_on_or_after(p, datetime(2030, 1, 1, tzinfo=UTC)) is None


def test_forward_alpha_subtracts_market():
    # stock +20%, market +10% over the window -> alpha +10%
    stock = _series(datetime(2025, 1, 1, tzinfo=UTC), [100.0] + [None] * 9 + [120.0])
    spy = _series(datetime(2025, 1, 1, tzinfo=UTC), [100.0] + [None] * 9 + [110.0])
    stock = stock.drop_nulls(); spy = spy.drop_nulls()
    a = forward_alpha(stock, spy, datetime(2025, 1, 1, tzinfo=UTC), horizon_days=9)
    assert a == pytest.approx(0.20 - 0.10, rel=1e-6)


def test_forward_alpha_none_when_window_too_long():
    stock = _series(datetime(2025, 1, 1, tzinfo=UTC), [100.0, 101.0])
    spy = _series(datetime(2025, 1, 1, tzinfo=UTC), [100.0, 101.0])
    assert forward_alpha(stock, spy, datetime(2025, 1, 1, tzinfo=UTC), horizon_days=365) is None


def test_outcome_bucket_thresholds():
    assert outcome_bucket(0.15) == 5
    assert outcome_bucket(0.05) == 4
    assert outcome_bucket(0.0) == 3
    assert outcome_bucket(-0.05) == 2
    assert outcome_bucket(-0.15) == 1


def test_pearson_perfect_correlation():
    assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_summarise_counts_holds_and_accuracy():
    preds = [
        {"score": 5, "raw_score": 1.0, "actual_alpha": 0.2, "true_bucket": 5},  # correct bull
        {"score": 2, "raw_score": -0.5, "actual_alpha": -0.1, "true_bucket": 2},  # correct bear
        {"score": 4, "raw_score": 0.4, "actual_alpha": -0.1, "true_bucket": 2},  # wrong bull
        {"score": 3, "raw_score": 0.0, "actual_alpha": 0.5, "true_bucket": 5},   # hold (abstain)
    ]
    s = _summarise(preds, 182)
    assert s["n"] == 4
    assert s["holds"] == 1
    assert s["directional_calls"] == 3
    assert s["directional_accuracy_excl_holds"] == pytest.approx(2 / 3, rel=1e-3)


def test_push_within_dead_band_excluded_from_accuracy():
    preds = [
        {"score": 4, "raw_score": 0.5, "actual_alpha": 0.20, "true_bucket": 5},  # decided, correct
        {"score": 4, "raw_score": 0.4, "actual_alpha": 0.01, "true_bucket": 3},  # push: |alpha| within dead-band
    ]
    s = _summarise(preds, 182)
    assert s["pushes"] == 1
    assert s["decided_calls"] == 1
    assert s["directional_accuracy_excl_holds"] == pytest.approx(1.0)  # the +0.4% "Buy" isn't counted as a win


def test_baseline_strategies_and_cost_in_summary():
    preds = [
        {"score": 4, "actual_alpha": 0.2, "true_bucket": 5, "ticker": "A", "cutoff": "C", "tokens": 100, "cost_usd": 0.001},
        {"score": 2, "actual_alpha": -0.2, "true_bucket": 1, "ticker": "B", "cutoff": "C", "tokens": 100, "cost_usd": 0.001},
    ]
    s = _summarise(preds, 182)
    assert s["baselines"]["always_buy_directional_accuracy"] == pytest.approx(0.5)  # 1 of 2 decided beat the market
    assert s["baselines"]["always_hold_bucket_mae"] is not None
    assert s["cost"]["total_tokens"] == 200
    assert s["cost"]["total_cost_usd"] == pytest.approx(0.002)


def test_rank_ic_within_cutoff_is_one_when_order_matches():
    preds = [
        {"cutoff": "A", "raw_score": 0.1, "actual_alpha": 0.01, "score": 3, "true_bucket": 3},
        {"cutoff": "A", "raw_score": 0.5, "actual_alpha": 0.05, "score": 4, "true_bucket": 4},
        {"cutoff": "A", "raw_score": 0.9, "actual_alpha": 0.10, "score": 5, "true_bucket": 5},
    ]
    assert _rank_ic(preds)["mean"] == pytest.approx(1.0)


def test_fit_thresholds_orders_buckets_descending():
    preds = [{"raw_score": float(i)} for i in range(10)]
    thresholds = fit_thresholds(preds, persist=False)
    assert [bucket for _, bucket in thresholds] == [5, 4, 3, 2]
    cut_points = [t for t, _ in thresholds]
    assert cut_points == sorted(cut_points, reverse=True)


def test_wilson_interval_brackets_estimate():
    lo, hi = _wilson_interval(8, 10)
    assert 0 <= lo < 0.8 < hi <= 1
    assert _wilson_interval(0, 0) is None
