"""Baseline construction, focused on the cache-key correctness fix (the bug where
every ticker in a sector reused the first ticker's self-excluded baseline)."""

import polars as pl

import baselines
from baselines import sector_baseline


def _install_stub(monkeypatch):
    """Give each peer a distinct, deterministic metric value, no network."""
    baselines._baseline_cache.clear()
    monkeypatch.setattr(baselines, "fetch_cached", lambda t, refresh=False: {"ticker": t})

    def fake_metrics(data, spy, cutoff):
        # value derived from ticker so excluding different tickers shifts the mean
        seed = sum(ord(c) for c in data["ticker"])
        return {m: float(seed % 100) for m in baselines.METRIC_DIRECTION} | {"volatility": 1.0}

    monkeypatch.setattr(baselines, "compute_metrics", fake_metrics)


def test_exclude_is_part_of_cache_key(monkeypatch):
    _install_stub(monkeypatch)
    cutoff = __import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").timezone.utc)

    excl_aapl = sector_baseline("Technology", cutoff, pl.DataFrame(), exclude="AAPL")
    excl_msft = sector_baseline("Technology", cutoff, pl.DataFrame(), exclude="MSFT")

    # The bug: excl_msft would have returned the AAPL-excluded baseline from cache.
    # With the fix, excluding a different peer yields a different distribution.
    assert excl_aapl["momentum_vs_spy"]["mean"] != excl_msft["momentum_vs_spy"]["mean"]


def test_same_exclude_hits_cache(monkeypatch):
    _install_stub(monkeypatch)
    calls = {"n": 0}
    real = baselines.compute_metrics

    def counting(data, spy, cutoff):
        calls["n"] += 1
        return real(data, spy, cutoff)

    monkeypatch.setattr(baselines, "compute_metrics", counting)
    cutoff = __import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").timezone.utc)

    sector_baseline("Technology", cutoff, pl.DataFrame(), exclude="AAPL")
    after_first = calls["n"]
    sector_baseline("Technology", cutoff, pl.DataFrame(), exclude="AAPL")  # should be cached
    assert calls["n"] == after_first


def test_unknown_sector_returns_empty(monkeypatch):
    _install_stub(monkeypatch)
    cutoff = __import__("datetime").datetime(2025, 1, 1, tzinfo=__import__("datetime").timezone.utc)
    assert sector_baseline("Nonexistent", cutoff, pl.DataFrame()) == {}
