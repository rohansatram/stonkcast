"""Disk cache: hit/miss, refresh, TTL, corrupt-fallback, round-trip fidelity."""

import pickle
from datetime import datetime, timedelta, timezone

import polars as pl

import cache as cache_mod
from cache import cached_fetch, is_cached, clear_cache, fetched_at


def test_miss_then_hit_calls_fetch_once(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fetch_fn(ticker):
        calls["n"] += 1
        return {"ticker": ticker, "v": 42}

    cached_fetch("AAPL", fetch_fn)
    cached_fetch("AAPL", fetch_fn)  # served from disk
    assert calls["n"] == 1
    assert is_cached("AAPL")


def test_refresh_forces_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fetch_fn(ticker):
        calls["n"] += 1
        return {"v": calls["n"]}

    assert cached_fetch("X", fetch_fn)["v"] == 1
    assert cached_fetch("X", fetch_fn, refresh=True)["v"] == 2


def test_corrupt_cache_falls_back_to_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    (tmp_path / "Y.pkl").write_bytes(b"not a pickle")
    out = cached_fetch("Y", lambda t: {"ok": True})
    assert out == {"ok": True}


def test_round_trips_polars(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    cached_fetch("Z", lambda t: {"frame": df})
    loaded = cached_fetch("Z", lambda t: {"frame": pl.DataFrame()})  # should hit cache
    assert loaded["frame"].equals(df)


def test_clear_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    cached_fetch("A", lambda t: {"v": 1})
    clear_cache("A")
    assert not is_cached("A")


def test_max_age_refetches_when_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fetch_fn(ticker):
        calls["n"] += 1
        return {"v": calls["n"]}

    cached_fetch("S", fetch_fn)
    # Backdate the cached blob, then read it with a 1-day TTL -> must re-fetch.
    path = tmp_path / "S.pkl"
    record = pickle.loads(path.read_bytes())
    record["fetched_at"] = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    path.write_bytes(pickle.dumps(record))
    out = cached_fetch("S", fetch_fn, max_age=timedelta(days=1))
    assert calls["n"] == 2 and out["v"] == 2


def test_max_age_keeps_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fetch_fn(ticker):
        calls["n"] += 1
        return {"v": calls["n"]}

    cached_fetch("F", fetch_fn)
    cached_fetch("F", fetch_fn, max_age=timedelta(days=1))  # fresh -> cache hit
    assert calls["n"] == 1


def test_fetched_at_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    assert fetched_at("Q") is None
    cached_fetch("Q", lambda t: {"v": 1})
    assert fetched_at("Q") is not None
