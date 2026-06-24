"""
Turn raw metrics + a sector baseline into a 1-5 score.

Pipeline: z-score each metric vs its sector -> apply direction (+/-) so higher
always means more bullish -> weighted sum -> bucket to 1-5. Volatility is not a
return signal; it sets confidence and, when a name is far more volatile than its
peers, pulls an extreme score one notch toward 3 (we don't bluff on noisy names).

Weights and bucket thresholds are hand-set starting points, to be tuned against
the backtest later. They live here as plain constants on purpose.
"""

import json
import statistics
from pathlib import Path

from baselines import METRIC_DIRECTION

# Default weights (sum to 1.0). These are STARTING POINTS, not fitted values; the
# horizon-aligned 6-month momentum carries the most, the reversal-prone 1-month
# terms the least. Tune against the backtest (backtest.fit_thresholds calibrates
# the buckets; weights are the next tuning target).
WEIGHTS: dict[str, float] = {
    "momentum_6m_vs_spy": 0.18,
    "momentum_vs_spy": 0.07,
    "momentum_30d": 0.05,
    "fifty_two_week_position": 0.10,
    "revenue_growth": 0.12,
    "eps_trend": 0.13,
    "earnings_surprise": 0.09,
    "ebitda_margin": 0.08,
    "pe_ratio": 0.08,
    "debt_to_equity": 0.10,
}

# Which independent data family each metric draws on. An extreme (1 or 5) call
# should rest on more than one family, not a single clipped z (see score()).
METRIC_FAMILY: dict[str, str] = {
    "momentum_6m_vs_spy": "price", "momentum_vs_spy": "price",
    "momentum_30d": "price", "fifty_two_week_position": "price",
    "eps_trend": "earnings", "earnings_surprise": "earnings", "pe_ratio": "earnings",
    "revenue_growth": "statement", "ebitda_margin": "statement", "debt_to_equity": "statement",
}

Z_CLIP = 3.0  # clip z-scores so one wild peer can't dominate
MIN_COVERAGE_FOR_CALL = 0.4   # below this, too little data to make a directional call -> Hold
MIN_FAMILIES_FOR_EXTREME = 2  # an extreme 1/5 call needs evidence from >= this many families
MIN_PEERS_FOR_HIGH_CONF = 6   # below this median peer count, don't claim "high" confidence

# raw_score thresholds -> 1-5 bucket (raw_score is a weighted mean of clipped z-scores).
# These are hand-set defaults; backtest.fit_thresholds can overwrite them with cut
# points fitted on a held-out fold, persisted to cache/calibration.json.
DEFAULT_BUCKET_THRESHOLDS = [
    (1.00, 5),
    (0.30, 4),
    (-0.30, 3),
    (-1.00, 2),
]  # below the last -> 1

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "cache" / "calibration.json"


def _load_bucket_thresholds() -> list[tuple[float, int]]:
    """Fitted thresholds from cache/calibration.json if present, else the defaults."""
    try:
        with CALIBRATION_FILE.open() as calibration_file:
            pairs = json.load(calibration_file).get("bucket_thresholds")
        if pairs:
            return [(float(threshold), int(bucket)) for threshold, bucket in pairs]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return list(DEFAULT_BUCKET_THRESHOLDS)


BUCKET_THRESHOLDS = _load_bucket_thresholds()


def reload_thresholds() -> list[tuple[float, int]]:
    """Re-read cache/calibration.json into the module global (call after fitting)."""
    global BUCKET_THRESHOLDS
    BUCKET_THRESHOLDS = _load_bucket_thresholds()
    return BUCKET_THRESHOLDS


SCORE_LABELS = {1: "Strong Sell", 2: "Sell", 3: "Hold", 4: "Buy", 5: "Strong Buy"}


def _zscore(value: float, mean: float, std: float) -> float:
    z_score = (value - mean) / std
    return max(-Z_CLIP, min(Z_CLIP, z_score))


def _bucket(raw_score: float) -> int:
    for threshold, bucket in BUCKET_THRESHOLDS:
        if raw_score >= threshold:
            return bucket
    return 1


def score(metrics: dict, baseline: dict) -> dict:
    """
    Combine raw `metrics` against a sector `baseline` into a scored result.

    Returns a fully inspectable dict: the 1-5 score + label, the continuous
    raw_score behind it, confidence, and a per-metric breakdown (raw value,
    sector mean, z-score, directional signal, weight, contribution).
    """
    breakdown = []
    weighted_sum = 0.0
    used_weight = 0.0
    families_present: set[str] = set()
    peer_counts: list[int] = []

    for metric, weight in WEIGHTS.items():
        raw_value = metrics.get(metric)
        sector_stats = baseline.get(metric)
        breakdown_entry = {
            "metric": metric,
            "raw": raw_value,
            "sector_mean": sector_stats["mean"] if sector_stats else None,
            "sector_std": sector_stats["std"] if sector_stats else None,
            "sector_n": sector_stats.get("n") if sector_stats else None,
            "z": None,
            "signal": None,
            "weight": weight,
            "contribution": 0.0,
        }
        if raw_value is not None and sector_stats is not None:
            z_score = _zscore(raw_value, sector_stats["mean"], sector_stats["std"])
            signal = z_score * METRIC_DIRECTION[metric]  # higher signal = more bullish
            contribution = signal * weight
            breakdown_entry.update({"z": z_score, "signal": signal, "contribution": contribution})
            weighted_sum += contribution
            used_weight += weight
            families_present.add(METRIC_FAMILY.get(metric, metric))
            if sector_stats.get("n") is not None:
                peer_counts.append(sector_stats["n"])

        breakdown.append(breakdown_entry)

    # Renormalise by the weight actually used, so missing metrics don't drag the
    # score toward 0 (a stock with only half its metrics still gets a fair read).
    raw_score = weighted_sum / used_weight if used_weight > 0 else 0.0

    confidence, volatility_z = _confidence(metrics, baseline)
    bucket = _apply_confidence(_bucket(raw_score), volatility_z)

    # An extreme 1/5 call should not rest on a single data family (e.g. only price
    # momentum, with no fundamentals): pull it one notch in toward 2/4.
    if len(families_present) < MIN_FAMILIES_FOR_EXTREME:
        if bucket == 5:
            bucket = 4
        elif bucket == 1:
            bucket = 2

    # Thin peer baskets make the z-scores (and so the score) noisy; don't advertise
    # "high" confidence when the baseline rests on too few peers.
    median_peers = statistics.median(peer_counts) if peer_counts else None
    if confidence == "high" and median_peers is not None and median_peers < MIN_PEERS_FOR_HIGH_CONF:
        confidence = "medium"

    # Too few metrics to stand behind a directional call (e.g. a just-listed
    # ticker with no price/earnings history). Abstain with Hold instead of
    # guessing, so we don't take a confident bet on near-zero information.
    if used_weight < MIN_COVERAGE_FOR_CALL:
        bucket = 3
        confidence = "low"

    return {
        "score": bucket,
        "label": SCORE_LABELS[bucket],
        "raw_score": round(raw_score, 4),
        "confidence": confidence,
        "coverage": round(used_weight, 3),  # fraction of weight backed by real data
        "families": sorted(families_present),
        "median_peer_count": median_peers,
        "breakdown": breakdown,
    }


def _confidence(metrics: dict, baseline: dict) -> tuple[str, float | None]:
    """Confidence from how volatile the name is vs its sector (medium if unknown)."""
    volatility = metrics.get("volatility")
    vol_baseline = baseline.get("volatility")
    volatility_z = None
    if volatility is not None and vol_baseline and vol_baseline.get("std"):
        volatility_z = (volatility - vol_baseline["mean"]) / vol_baseline["std"]

    if volatility_z is None:
        return "medium", None
    # Symmetric cut points: unusually volatile vs sector -> low, unusually calm -> high.
    # (Confidence is a display/label field; the only place it shapes the bucket is the
    # high-volatility pull toward Hold in _apply_confidence.)
    if volatility_z > 1.0:
        return "low", volatility_z
    if volatility_z < -1.0:
        return "high", volatility_z
    return "medium", volatility_z


def _apply_confidence(bucket: int, volatility_z: float | None) -> int:
    """Pull extreme calls one notch toward Hold when the name is unusually volatile."""
    if volatility_z is not None and volatility_z > 1.0:
        if bucket == 5:
            return 4
        if bucket == 1:
            return 2
    return bucket
