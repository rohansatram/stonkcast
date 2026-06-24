"""
Point-in-time helpers. Everything here answers one question: "what was knowable
at `cutoff`?" If a number wasn't public yet, it must not reach the scorer.

- Prices: only bars strictly before the cutoff.
- Earnings: only quarters whose ANNOUNCEMENT date is before the cutoff.
- Financials / balance sheet: yfinance gives only period-end dates, so we map
  each period end to the earnings announcement that first followed it (when the
  10-K/10-Q became public), and keep it only if that announcement preceded the
  cutoff. Falls back to a conservative reporting lag if announcement dates are
  unavailable.
"""

import logging
from datetime import datetime, timedelta, timezone

import polars as pl

logger = logging.getLogger(__name__)

REPORTING_LAG_FALLBACK_DAYS = 90  # conservative (longest 10-K deadline); only used when announcement dates are missing


def to_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def prices_asof(prices: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    """Daily bars strictly before the cutoff (the cutoff day itself is unknown)."""
    if prices.is_empty():
        return prices
    return prices.filter(pl.col("Date") < to_utc(cutoff))


def earnings_asof(earnings: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    """Earnings rows announced strictly before the cutoff, oldest first."""
    if earnings.is_empty() or "announce_date" not in earnings.columns:
        return earnings
    return earnings.filter(pl.col("announce_date") < to_utc(cutoff)).sort("announce_date")


def announcement_for_period_end(period_end: datetime, announce_dates: list[datetime]) -> datetime:
    """The first earnings announcement on/after a period end = when that quarter
    went public. Falls back to period_end + reporting lag if none is found."""
    period_end_utc = to_utc(period_end)
    following_announcements = sorted(date for date in announce_dates if to_utc(date) >= period_end_utc)
    if following_announcements:
        return to_utc(following_announcements[0])
    # No announcement date matched: fall back to a conservative lag. This is mostly
    # safe (it tends to EXCLUDE a statement, not leak it), but it's a guess, so note it.
    logger.debug("announcement_for_period_end: no announcement on/after %s; using %d-day lag fallback",
                 period_end_utc.date().isoformat(), REPORTING_LAG_FALLBACK_DAYS)
    return period_end_utc + timedelta(days=REPORTING_LAG_FALLBACK_DAYS)


def statement_asof(statement: pl.DataFrame, earnings: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    """
    Filter a financials/balance-sheet frame (keyed by period_end) to rows that
    were public before the cutoff, using earnings announcement dates as the
    "became public" proxy. Oldest first.
    """
    if statement.is_empty() or "period_end" not in statement.columns:
        return statement
    cutoff = to_utc(cutoff)
    announce_dates = (
        [] if earnings.is_empty() else earnings["announce_date"].to_list()
    )
    keep_mask = [
        announcement_for_period_end(period_end, announce_dates) < cutoff
        for period_end in statement["period_end"].to_list()
    ]
    return statement.filter(pl.Series(keep_mask)).sort("period_end")
