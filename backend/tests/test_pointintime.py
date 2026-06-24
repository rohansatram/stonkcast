"""Point-in-time filtering: the no-leak guarantee."""

from datetime import datetime, timezone

import polars as pl

from pointintime import (
    prices_asof, earnings_asof, announcement_for_period_end, statement_asof,
    REPORTING_LAG_FALLBACK_DAYS,
)

UTC = timezone.utc


def test_prices_asof_excludes_cutoff_day_and_after(synth_prices, cutoff):
    out = prices_asof(synth_prices, cutoff)
    assert out["Date"].max() < cutoff
    assert 999.0 not in out["Close"].to_list()  # the post-cutoff bar is gone


def test_earnings_asof_excludes_future_announcements(synth_earnings, cutoff):
    out = earnings_asof(synth_earnings, cutoff)
    assert out.height == 8  # the 9th (2025-02-01) is dropped
    assert 99.0 not in out["eps_reported"].to_list()


def test_announcement_for_period_end_picks_first_following():
    announce = [datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 5, 1, tzinfo=UTC)]
    got = announcement_for_period_end(datetime(2023, 12, 31, tzinfo=UTC), announce)
    assert got == datetime(2024, 2, 1, tzinfo=UTC)


def test_announcement_for_period_end_falls_back_to_lag():
    got = announcement_for_period_end(datetime(2024, 12, 31, tzinfo=UTC), [])  # no announcements
    expected_min = datetime(2024, 12, 31, tzinfo=UTC)
    assert (got - expected_min).days == REPORTING_LAG_FALLBACK_DAYS


def test_statement_asof_drops_not_yet_announced(synth_financials, synth_earnings):
    # cutoff between the 2023-12-31 period end and its 2024-02-01 announcement:
    # the 2023 year is NOT public yet, only 2022 is.
    cutoff = datetime(2024, 1, 15, tzinfo=UTC)
    out = statement_asof(synth_financials, synth_earnings, cutoff)
    assert out.height == 1
    assert out["period_end"][0] == datetime(2022, 12, 31, tzinfo=UTC)


def test_statement_asof_keeps_announced(synth_financials, synth_earnings):
    cutoff = datetime(2025, 1, 1, tzinfo=UTC)
    out = statement_asof(synth_financials, synth_earnings, cutoff)
    assert out.height == 2  # both years public by 2025


def test_empty_frames_are_safe():
    empty = pl.DataFrame()
    c = datetime(2025, 1, 1, tzinfo=UTC)
    assert prices_asof(empty, c).is_empty()
    assert earnings_asof(empty, c).is_empty()
    assert statement_asof(empty, empty, c).is_empty()
