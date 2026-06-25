"""Phase 2 prompt building and response parsing (offline, no Nova calls)."""

from phase2 import build_user_prompt, _parse_nova_json, _clamp_score, _derive_adjustment, _derive_confidence


def _phase1_result():
    return {
        "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology",
        "cutoff": "2026-01-01", "score": 2, "label": "Sell", "raw_score": -0.33,
        "confidence": "high",
        "breakdown": [
            {"metric": "momentum_vs_spy", "raw": -0.042, "signal": -0.57, "weight": 0.20},
            {"metric": "pe_ratio", "raw": 36.4, "signal": 0.19, "weight": 0.10},
        ],
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


def _congress(signal="net buying"):
    return {"available": True, "signal": signal, "window_days": 90,
            "purchases": 3, "sales": 0, "n_members": 2}


def test_prompt_includes_congress_when_available():
    prompt = build_user_prompt(_phase1_result(), _filing(), congress=_congress())
    assert "CONGRESSIONAL TRADES" in prompt
    assert "3 buys" in prompt


def test_prompt_omits_congress_when_anonymized():
    # member names could hint at a famous trade, so the leakage probe withholds it
    prompt = build_user_prompt(_phase1_result(), _filing(), congress=_congress(), anonymize=True)
    assert "CONGRESSIONAL TRADES" not in prompt


def test_prompt_omits_congress_when_unavailable():
    prompt = build_user_prompt(_phase1_result(), _filing(), congress={"available": False})
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
