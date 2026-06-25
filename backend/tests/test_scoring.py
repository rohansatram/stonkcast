"""Scoring: z-scores, metric direction, bucketing, renormalisation, confidence."""

import pytest

from scoring import score, _bucket, _zscore, Z_CLIP, METRIC_FAMILY, DIRECT_WEIGHTS
from baselines import METRIC_DIRECTION

UNIT_BASELINE = {m: {"mean": 0.0, "std": 1.0} for m in METRIC_DIRECTION}
UNIT_BASELINE["volatility"] = {"mean": 0.0, "std": 1.0}


def _metrics(**overrides):
    base = {m: None for m in METRIC_DIRECTION}
    base["volatility"] = 0.0
    base.update(overrides)
    return base


def test_neutral_metrics_score_hold():
    m = _metrics(**{k: 0.0 for k in METRIC_DIRECTION})
    r = score(m, UNIT_BASELINE)
    assert r["score"] == 3
    assert r["label"] == "Hold"
    assert r["raw_score"] == pytest.approx(0.0)


def test_all_bullish_scores_strong_buy():
    # +1 metrics at +2 sigma, -1 metrics (pe, d/e) at -2 sigma -> every signal +2
    overrides = {m: (2.0 if d > 0 else -2.0) for m, d in METRIC_DIRECTION.items()}
    r = score(_metrics(**overrides), UNIT_BASELINE)
    assert r["score"] == 5
    assert r["raw_score"] == pytest.approx(2.0)
    assert r["coverage"] == pytest.approx(1.0)


def test_low_pe_is_bullish_direction():
    # a cheap stock (low P/E vs sector) should read bullish
    m = _metrics(pe_ratio=-2.0)  # 2 sigma below mean
    r = score(m, UNIT_BASELINE)
    bd = next(b for b in r["breakdown"] if b["metric"] == "pe_ratio")
    assert bd["signal"] > 0  # below-average P/E -> positive signal


def test_missing_metrics_renormalise():
    # only one metric present -> score reflects it, not diluted toward 0
    m = _metrics(momentum_vs_spy=2.0)
    r = score(m, UNIT_BASELINE)
    assert r["raw_score"] == pytest.approx(2.0)
    assert r["coverage"] == pytest.approx(0.07)  # momentum_vs_spy weight


def test_high_volatility_pulls_extreme_toward_hold():
    overrides = {m: (2.0 if d > 0 else -2.0) for m, d in METRIC_DIRECTION.items()}
    m = _metrics(**overrides)
    m["volatility"] = 2.0  # 2 sigma more volatile than sector
    r = score(m, UNIT_BASELINE)
    assert r["confidence"] == "low"
    assert r["score"] == 4  # pulled down from 5


def test_bucket_thresholds():
    assert _bucket(1.5) == 5
    assert _bucket(0.5) == 4
    assert _bucket(0.0) == 3
    assert _bucket(-0.5) == 2
    assert _bucket(-1.5) == 1


def test_zscore_is_clipped():
    assert _zscore(100.0, 0.0, 1.0) == Z_CLIP
    assert _zscore(-100.0, 0.0, 1.0) == -Z_CLIP


def test_low_coverage_abstains_to_hold():
    # only one metric present (weight 0.07 < 0.4 threshold) -> Hold, even if bullish
    m = _metrics(momentum_vs_spy=2.0)
    r = score(m, UNIT_BASELINE)
    assert r["score"] == 3
    assert r["confidence"] == "low"


def test_empty_baseline_is_safe():
    # no sector baseline -> nothing to normalise -> neutral hold, no crash
    r = score(_metrics(momentum_vs_spy=0.5), {})
    assert r["score"] == 3
    assert r["coverage"] == pytest.approx(0.0)


def test_extreme_call_needs_multiple_families():
    # only price-family metrics present, all strongly bullish -> would bucket 5,
    # but a single data family can't justify an extreme call, so it's pulled to 4.
    price_only = {m: (2.0 if METRIC_FAMILY.get(m) == "price" else None) for m in METRIC_DIRECTION}
    price_only["volatility"] = 0.0
    r = score(price_only, UNIT_BASELINE)
    assert r["families"] == ["price"]
    assert r["score"] == 4


def test_low_volatility_gives_high_confidence():
    overrides = {m: (2.0 if d > 0 else -2.0) for m, d in METRIC_DIRECTION.items()}
    m = _metrics(**overrides)
    m["volatility"] = -1.5  # 1.5 sigma calmer than sector (symmetric high-confidence cut)
    r = score(m, UNIT_BASELINE)
    assert r["confidence"] == "high"


def test_thin_peer_baseline_caps_confidence():
    overrides = {m: (2.0 if d > 0 else -2.0) for m, d in METRIC_DIRECTION.items()}
    m = _metrics(**overrides)
    m["volatility"] = -1.5  # would otherwise read "high"
    thin = {metric: {"mean": 0.0, "std": 1.0, "n": 4} for metric in METRIC_DIRECTION}
    thin["volatility"] = {"mean": 0.0, "std": 1.0, "n": 4}
    r = score(m, thin)
    assert r["median_peer_count"] == 4
    assert r["confidence"] == "medium"  # downgraded: median peer count 4 < 6


def test_congress_direct_signal_contributes_without_baseline():
    # congress is a direct signal: used as-is, no sector baseline needed.
    result = score({"congress": 1.0}, {})
    entry = next(e for e in result["breakdown"] if e["metric"] == "congress")
    assert entry["signal"] == pytest.approx(1.0)          # net buying -> bullish
    assert entry["z"] is None                              # not z-scored
    assert result["raw_score"] == pytest.approx(1.0)       # only signal present
    assert result["coverage"] == pytest.approx(DIRECT_WEIGHTS["congress"])


def test_congress_signal_is_clamped():
    entry = next(e for e in score({"congress": 9.0}, {})["breakdown"] if e["metric"] == "congress")
    assert entry["signal"] == pytest.approx(1.0)           # clamped into [-1, 1]


def test_congress_absent_is_neutral():
    result = score({"congress": None}, UNIT_BASELINE)
    entry = next(e for e in result["breakdown"] if e["metric"] == "congress")
    assert entry["contribution"] == 0.0                    # no data -> no effect
