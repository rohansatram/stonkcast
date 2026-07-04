"""
US congressional stock trades (STOCK Act disclosures), point-in-time.

Members of Congress must publicly disclose personal stock trades within 45 days.
We expose a per-ticker signal: how many purchases vs sales were DISCLOSED before a
cutoff, in a trailing window. The gate is the DISCLOSURE (filing/report) date, not
the trade date, so it stays leak-free; the disclosure lag is real and respected.

Data source: a downloaded dataset in backend/data/ (CSV or JSON, Stock-Watcher /
Capitol-Trades schema). Generate one with dump_capitol_trades_playwright.py. If no
dataset is present, the signal is simply unavailable (neutral), no network needed.

    from fetch.fetchCongressTrades import congress_signal, congress_score
    sig = congress_signal("NVDA", datetime(2026, 1, 1, tzinfo=timezone.utc))
    value = congress_score(sig)   # [-1, 1] math signal, or None
"""

import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Drop a downloaded congress-trades dataset (CSV or JSON) in backend/data/ and it
# becomes the source. Absent that, the signal is unavailable (no network/keys).
LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_WINDOW_DAYS = 90
RECENT_SAMPLE = 5  # how many recent trades to surface for display / the UI


def _to_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_date(value) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(value).strip()[:10], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_amount(value) -> float | None:
    """Disclosed amounts are ranges like '$1,001 - $15,000'; use the midpoint."""
    numbers = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", str(value or "")) if n.replace(",", "").isdigit()]
    numbers = [n for n in numbers if n > 0]
    return sum(numbers) / len(numbers) if numbers else None


def _side(trade_type) -> str | None:
    """Normalise the many type spellings to 'buy' / 'sell' / None (exchange/other)."""
    text = str(trade_type or "").lower()
    if "purchase" in text or "buy" in text:
        return "buy"
    if "sale" in text or "sold" in text or "sell" in text:
        return "sell"
    return None


def _first(record: dict, *keys):
    """First present value among alternative key spellings (our schema or others')."""
    for key in keys:
        if record.get(key) not in (None, ""):
            return record[key]
    return None


def _normalize(record: dict) -> dict | None:
    ticker = str(_first(record, "ticker", "Ticker") or "").strip().upper()
    if not ticker or ticker in ("--", "N/A", "--."):
        return None
    side = _side(_first(record, "type", "Transaction"))
    disclosure = _parse_date(_first(record, "disclosure_date", "ReportDate", "Filed"))
    if side is None or disclosure is None:
        return None
    return {
        "ticker": ticker,
        "side": side,
        "disclosure_date": disclosure,
        "transaction_date": _parse_date(_first(record, "transaction_date", "TransactionDate")),
        "amount_usd": _parse_amount(_first(record, "amount", "Range", "Amount")),
        "member": _first(record, "representative", "senator", "Representative", "Name") or "Unknown",
    }


_local_cache: dict = {"loaded": False, "data": None}


def _load_local_all(refresh: bool = False) -> list[dict] | None:
    """All trades from any CSV/JSON dataset files in backend/data/, normalised.
    Returns None when the folder is absent/empty (so the signal is unavailable).
    Memoised in-process; pass refresh=True to re-read."""
    if not refresh and _local_cache["loaded"]:
        return _local_cache["data"]

    files = []
    if LOCAL_DATA_DIR.exists():
        files = sorted(LOCAL_DATA_DIR.glob("*.csv")) + sorted(LOCAL_DATA_DIR.glob("*.json"))

    rows: list[dict] = []
    for path in files:
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text())
                rows.extend(data if isinstance(data, list) else data.get("data", []))
            else:
                with path.open(newline="") as handle:
                    rows.extend(csv.DictReader(handle))
        except Exception:
            continue

    result = [n for n in (_normalize(r) for r in rows) if n is not None] if files else None
    _local_cache.update(loaded=True, data=result)
    return result


def congress_signal(ticker: str, cutoff: datetime, window_days: int = DEFAULT_WINDOW_DAYS,
                    transactions: list[dict] | None = None, refresh: bool = False) -> dict:
    """
    Point-in-time congressional-trading signal for `ticker`: purchases vs sales
    DISCLOSED in the [cutoff - window, cutoff) window. `transactions` (already
    normalised) can be injected for testing; otherwise it's loaded from the local
    dataset. With no dataset present, returns available=False with a neutral
    'none' signal so the agent degrades gracefully.
    """
    ticker = ticker.upper()
    cutoff = _to_utc(cutoff)
    window_start = cutoff - timedelta(days=window_days)

    available, error, source = True, None, "injected"
    if transactions is None:
        local = _load_local_all(refresh=refresh)
        if local is not None:  # a downloaded dataset is present -> use it
            transactions, source = [t for t in local if t["ticker"] == ticker], "local-dataset"
        else:
            transactions, available, error, source = [], False, "no congress dataset in backend/data/", "none"

    relevant = [
        t for t in transactions
        if t["ticker"] == ticker and window_start <= t["disclosure_date"] < cutoff
    ]
    purchases = [t for t in relevant if t["side"] == "buy"]
    sales = [t for t in relevant if t["side"] == "sell"]
    buy_usd = sum(t["amount_usd"] or 0 for t in purchases)
    sell_usd = sum(t["amount_usd"] or 0 for t in sales)

    if not relevant:
        signal = "none"
    elif len(purchases) > len(sales):
        signal = "net buying"
    elif len(sales) > len(purchases):
        signal = "net selling"
    else:
        signal = "mixed"

    recent = sorted(relevant, key=lambda t: t["disclosure_date"], reverse=True)[:RECENT_SAMPLE]
    return {
        "ticker": ticker,
        "available": available,
        "source": source,
        "error": error,
        "window_days": window_days,
        "as_of": cutoff.date().isoformat(),
        "purchases": len(purchases),
        "sales": len(sales),
        "net_trades": len(purchases) - len(sales),
        "buy_usd_est": round(buy_usd),
        "sell_usd_est": round(sell_usd),
        "net_usd_est": round(buy_usd - sell_usd),
        "signal": signal,
        "n_members": len({t["member"] for t in relevant}),
        "recent": [
            {
                "disclosed": t["disclosure_date"].date().isoformat(),
                "traded": t["transaction_date"].date().isoformat() if t["transaction_date"] else None,
                "side": t["side"],
                "member": t["member"],
                "amount_usd_est": round(t["amount_usd"]) if t["amount_usd"] else None,
            }
            for t in recent
        ],
    }


def congress_score(signal: dict | None) -> float | None:
    """
    Map a congress_signal() result to a [-1, 1] math signal (+ = net buying).

    Net-buying intensity = (buys - sells) / (buys + sells). Returns None when the
    data is unavailable or no trades fall in the window, so it contributes nothing
    to the score rather than dragging it toward neutral.
    """
    if not signal or not signal.get("available"):
        return None
    purchases, sales = signal.get("purchases", 0), signal.get("sales", 0)
    total = purchases + sales
    if total == 0:
        return None
    return (purchases - sales) / total


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    if len(sys.argv) > 2:
        cutoff = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)

    sig = congress_signal(ticker, cutoff)
    if not sig["available"]:
        print(f"Congress data unavailable: {sig['error']}")
        print("Generate a dataset with: uv run python src/fetch/dump_capitol_trades_playwright.py")
        raise SystemExit(1)
    print(f"\nCongress trades for {sig['ticker']} disclosed in the {sig['window_days']}d before {sig['as_of']}:")
    print(f"  {sig['purchases']} purchases / {sig['sales']} sales  ->  {sig['signal']}  "
          f"(net {sig['net_trades']:+d} trades, ${sig['net_usd_est']:+,} est) across {sig['n_members']} member(s)")
    print(f"  math signal = {congress_score(sig)}")
    for trade in sig["recent"]:
        amt = f"${trade['amount_usd_est']:,}" if trade["amount_usd_est"] else "n/a"
        print(f"    {trade['disclosed']}  {trade['side']:4s}  {amt:>12s}  {trade['member']}")
