"""Congress-trades parsing + point-in-time signal (offline, injected data)."""

from datetime import datetime, timezone

import pytest

from fetch.fetchCongressTrades import _parse_amount, _parse_date, _side, _normalize, congress_signal

UTC = timezone.utc


def test_parse_amount_uses_range_midpoint():
    assert _parse_amount("$1,001 - $15,000") == pytest.approx((1001 + 15000) / 2)
    assert _parse_amount("$1,000,001 - $5,000,000") == pytest.approx((1_000_001 + 5_000_000) / 2)
    assert _parse_amount("") is None


def test_parse_date_formats():
    assert _parse_date("01/15/2025") == datetime(2025, 1, 15, tzinfo=UTC)
    assert _parse_date("2025-01-15") == datetime(2025, 1, 15, tzinfo=UTC)
    assert _parse_date("garbage") is None


def test_side_normalisation():
    assert _side("purchase") == "buy"
    assert _side("Sale (Full)") == "sell"
    assert _side("sale_partial") == "sell"
    assert _side("exchange") is None


def test_normalize_drops_unusable_rows():
    assert _normalize({"ticker": "--", "type": "purchase", "disclosure_date": "01/01/2025"}) is None
    assert _normalize({"ticker": "NVDA", "type": "exchange", "disclosure_date": "01/01/2025"}) is None
    ok = _normalize({"ticker": "nvda", "type": "purchase", "disclosure_date": "01/01/2025",
                     "amount": "$1,001 - $15,000", "representative": "X"})
    assert ok["ticker"] == "NVDA" and ok["side"] == "buy"


def _tx(ticker, side, disclosure, amount=10000.0):
    return {"ticker": ticker, "side": side, "disclosure_date": datetime(*disclosure, tzinfo=UTC),
            "transaction_date": None, "amount_usd": amount, "member": "Member"}


def test_signal_is_point_in_time_and_windowed():
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)  # 90d window starts 2025-10-03
    txns = [
        _tx("NVDA", "buy", (2025, 12, 1)),    # in window, before cutoff -> counts
        _tx("NVDA", "buy", (2025, 11, 15)),   # counts
        _tx("NVDA", "sell", (2025, 12, 20)),  # counts
        _tx("NVDA", "buy", (2026, 2, 1)),     # disclosed AFTER cutoff -> excluded (no leak)
        _tx("NVDA", "buy", (2025, 6, 1)),     # before the window -> excluded
        _tx("AAPL", "buy", (2025, 12, 5)),    # different ticker -> excluded
    ]
    sig = congress_signal("NVDA", cutoff, window_days=90, transactions=txns)
    assert sig["purchases"] == 2
    assert sig["sales"] == 1
    assert sig["net_trades"] == 1
    assert sig["signal"] == "net buying"


def test_signal_none_when_no_trades():
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    sig = congress_signal("TSLA", cutoff, transactions=[])
    assert sig["signal"] == "none"
    assert sig["purchases"] == 0 and sig["sales"] == 0


def test_local_dataset_is_used_when_present(tmp_path, monkeypatch):
    import fetch.fetchCongressTrades as cg
    (tmp_path / "trades.csv").write_text(
        "ticker,type,transaction_date,disclosure_date,amount,representative\n"
        'NVDA,purchase,2025-11-01,2025-12-01,"$1,001 - $15,000",Jane Doe\n'
        'NVDA,sale,2025-11-10,2026-02-01,"$1,001 - $15,000",John Roe\n'  # disclosed after cutoff -> excluded
    )
    monkeypatch.setattr(cg, "LOCAL_DATA_DIR", tmp_path)
    cg._local_cache.update(loaded=False, data=None)
    try:
        sig = cg.congress_signal("NVDA", datetime(2026, 1, 1, tzinfo=UTC))
        assert sig["source"] == "local-dataset"
        assert sig["available"] is True
        assert sig["purchases"] == 1 and sig["sales"] == 0  # the post-cutoff sale is excluded
    finally:
        cg._local_cache.update(loaded=False, data=None)
