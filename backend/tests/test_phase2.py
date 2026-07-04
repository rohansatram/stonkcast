"""Phase 2 prompt building and response parsing (offline, no Nova calls)."""
import pytest

from phase2 import (
    build_user_prompt, _parse_nova_json, _clamp_score, _derive_adjustment, _derive_confidence,
    build_bull_prompt, build_bear_prompt, build_judge_prompt, _combine_usage,
)


def _phase1_result():
    return {
        "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology",
        "cutoff": "2026-01-01", "score": 2, "label": "Sell", "raw_score": -0.33,
        "confidence": "high",
        "breakdown": [
            {"metric": "momentum_vs_spy", "raw": -0.042, "signal": -0.57, "weight": 0.20},
            {"metric": "pe_ratio", "raw": 36.4, "signal": 0.19, "weight": 0.10},
            {"metric": "congress", "raw": 1.0, "signal": 0.06, "weight": 0.06},
        ],
        "congress": {
            "available": True, "signal": "net buying", "window_days": 90,
            "purchases": 3, "sales": 1, "net_trades": 2, "n_members": 2,
            "recent": [{"member": "Jane Doe", "side": "buy", "amount_usd_est": 50000, "disclosed": "2025-12-01"}],
        },
    }


def _filing():
    return {"filing": {"type": "10-K", "filed_date": "2025-10-31",
                       "risk_excerpt": "Item 1A. Risk Factors. Supply chain concentration in Asia ...",
                       "mdna_excerpt": "Item 7. MD&A. Revenue grew 24% with expanding margins ..."}}


def test_user_prompt_contains_key_context():
    prompt = build_user_prompt(_phase1_result(), _filing())
    assert "AAPL" in prompt
    assert "2026-01-01" in prompt
    assert "momentum_vs_spy" in prompt
    assert "Risk Factors" in prompt
    assert "MD&A" in prompt
    assert "Revenue grew 24%" in prompt  # the bull-case section is included
    assert "2/5" in prompt


def test_user_prompt_handles_missing_mdna():
    filing = {"filing": {"type": "10-Q", "filed_date": "2025-11-01",
                         "risk_excerpt": "some risks", "mdna_excerpt": ""}}
    prompt = build_user_prompt(_phase1_result(), filing)
    assert "not reliably available in this filing" in prompt


def test_user_prompt_handles_missing_filing():
    prompt = build_user_prompt(_phase1_result(), {"filing": None})
    assert "none available" in prompt


def test_congress_in_both_breakdown_and_prompt_block():
    # Congress is used in BOTH places: a quant metric (breakdown) AND a rich prompt block.
    prompt = build_user_prompt(_phase1_result(), _filing())
    assert "congress" in prompt                 # the quant signal line in the breakdown
    assert "CONGRESSIONAL TRADES" in prompt      # the rich qualitative block for Nova
    assert "3 buys" in prompt
    assert "Jane Doe" in prompt                  # recent trade detail


def test_congress_block_withheld_when_anonymized():
    # member names could hint identity, so the leakage probe withholds the block
    prompt = build_user_prompt(_phase1_result(), _filing(), anonymize=True)
    assert "CONGRESSIONAL TRADES" not in prompt


def test_user_prompt_anonymize_withholds_identity():
    prompt = build_user_prompt(_phase1_result(), _filing(), anonymize=True)
    assert "AAPL" not in prompt
    assert "Apple" not in prompt
    assert "withheld" in prompt


def test_user_prompt_labels_low_confidence_extract():
    filing = {"filing": {"type": "10-K", "filed_date": "2025-10-31",
                         "risk_excerpt": "vague risk text", "risk_quality": "fallback",
                         "mdna_excerpt": "real mdna", "mdna_quality": "section"}}
    prompt = build_user_prompt(_phase1_result(), filing)
    assert "LOW-CONFIDENCE" in prompt


def test_user_prompt_notes_when_no_section_extracted():
    filing = {"filing": {"type": "10-Q", "filed_date": "2025-11-01",
                         "risk_excerpt": "", "risk_quality": "empty",
                         "mdna_excerpt": "", "mdna_quality": "empty"}}
    prompt = build_user_prompt(_phase1_result(), filing)
    assert "lean toward CONFIRMING" in prompt


def test_parse_clean_json():
    text = '{"final_score": 4, "confidence": "high", "adjustment": "raise", "risk_flags": ["fx"], "rationale": "ok"}'
    parsed = _parse_nova_json(text, fallback_score=2)
    assert parsed["final_score"] == 4
    assert parsed["adjustment"] == "raise"


def test_parse_json_with_surrounding_prose():
    text = 'Here is my answer:\n{"final_score": 3, "confidence": "medium", "adjustment": "confirm", "risk_flags": [], "rationale": "x"}\nThanks!'
    parsed = _parse_nova_json(text, fallback_score=2)
    assert parsed["final_score"] == 3


def test_parse_garbage_falls_back():
    parsed = _parse_nova_json("the model rambled with no json", fallback_score=2)
    assert parsed["final_score"] == 2
    assert parsed["parse_error"] is True


def test_final_score_is_clamped():
    parsed = _parse_nova_json('{"final_score": 9}', fallback_score=2)
    assert parsed["final_score"] == 5  # clamped into 1-5


def test_clamp_score_handles_bad_values():
    assert _clamp_score(None, 3) == 3
    assert _clamp_score("buy", 3) == 3
    assert _clamp_score(0, 3) == 1


def test_adjustment_is_derived_from_scores():
    assert _derive_adjustment(3, 2) == "raise"   # the AAPL case that was mislabelled
    assert _derive_adjustment(4, 5) == "lower"
    assert _derive_adjustment(2, 2) == "confirm"


def test_confidence_is_derived_from_agreement_and_coverage():
    assert _derive_confidence(3, 3, coverage=1.0) == "high"    # agree, full data
    assert _derive_confidence(5, 2, coverage=1.0) == "low"     # strong disagreement
    assert _derive_confidence(3, 3, coverage=0.5) == "low"     # thin data
    assert _derive_confidence(3, 2, coverage=1.0) == "medium"  # mild adjustment


# --- multi-agent panel (bull / bear / judge) ---

def test_bull_prompt_uses_mdna_and_quant():
    prompt = build_bull_prompt(_phase1_result(), _filing())
    assert "Revenue grew 24%" in prompt        # MD&A (bull evidence)
    assert "momentum_vs_spy" in prompt          # quant breakdown
    assert "BULL SCORE" in prompt
    assert "Supply chain concentration" not in prompt  # bear gets Risk Factors, not bull


def test_bear_prompt_uses_risk_and_quant():
    prompt = build_bear_prompt(_phase1_result(), _filing())
    assert "Supply chain concentration" in prompt   # Risk Factors (bear evidence)
    assert "momentum_vs_spy" in prompt
    assert "BEAR SCORE" in prompt
    assert "Revenue grew 24%" not in prompt          # bull gets MD&A, not bear


def test_judge_prompt_includes_both_arguments():
    prompt = build_judge_prompt(_phase1_result(), "BULL: strong demand. BULL SCORE: 4",
                                "BEAR: regulatory risk. BEAR SCORE: 2")
    assert "BULL ARGUMENT" in prompt and "BEAR ARGUMENT" in prompt
    assert "strong demand" in prompt and "regulatory risk" in prompt


def test_combine_usage_sums_calls():
    u = lambda i, o, c: {"input_tokens": i, "output_tokens": o, "total_tokens": i + o, "cost_usd": c, "model": "m"}
    combined = _combine_usage([u(100, 10, 0.001), u(120, 12, 0.0012), u(80, 8, 0.0008)])
    assert combined["input_tokens"] == 300
    assert combined["total_tokens"] == 330
    assert combined["calls"] == 3
    assert combined["cost_usd"] == pytest.approx(0.003)
