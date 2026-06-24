"""End-to-end checks via the real cache. Skipped automatically when the cache
isn't warm, so the unit suite stays fully offline."""

from datetime import datetime, timezone

import pytest

import cache
from phase1 import score_ticker, REQUIRED_METRICS

UTC = timezone.utc
_REQUIRED_CACHE = ["AAPL", "SPY", "MSFT", "NVDA", "ORCL", "AVGO", "CRM", "ADBE", "AMD", "CSCO", "INTC"]
pytestmark = pytest.mark.skipif(
    not all(cache.is_cached(t) for t in _REQUIRED_CACHE),
    reason="cache not warm (run `uv run python src/phase1.py AAPL` first)",
)


def test_score_ticker_shape():
    r = score_ticker("AAPL")
    assert r["score"] in {1, 2, 3, 4, 5}
    assert set(REQUIRED_METRICS).issubset(r["metrics"].keys())
    assert len(r["breakdown"]) == 10  # 10 weighted metrics (incl. 6m momentum + 52w position)
    assert r["tokens"] == 0  # phase 1 has no LLM
    assert "recommendation_mean" in r["analyst_consensus_display_only"]


def test_deterministic():
    a = score_ticker("AAPL")
    b = score_ticker("AAPL")
    a.pop("latency_sec"); b.pop("latency_sec")
    assert a == b


def test_cutoff_changes_result():
    now = score_ticker("AAPL", datetime(2026, 1, 1, tzinfo=UTC))
    past = score_ticker("AAPL", datetime(2023, 1, 1, tzinfo=UTC))
    # point-in-time: AAPL's P/E was much lower at end of 2022 than end of 2025
    assert past["metrics"]["pe_ratio"] < now["metrics"]["pe_ratio"]
    assert past["raw_score"] != now["raw_score"]


def test_score_is_in_range_for_several_tickers():
    for t in ["AAPL", "MSFT", "NVDA"]:
        r = score_ticker(t)
        assert 1 <= r["score"] <= 5
        assert r["coverage"] > 0.5  # most metrics should be available for mega-caps
