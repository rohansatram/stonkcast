"""
Phase 2: layer Amazon Nova reasoning on top of the Phase 1 math score.

Nova receives the deterministic quant score + per-metric breakdown and a
point-in-time SEC filing excerpt (Risk Factors / MD&A), then decides whether to
confirm, raise, or lower the score, flags qualitative risks, and explains it in
plain English. It must reason only from the provided text and not use knowledge
after the cutoff (the contest's whole premise).

Two modes, both via the same function:
- demo  : score_ticker_v2(ticker, past_cutoff)   -> graded by the backtest harness
- live  : score_ticker_v2(ticker, today)          -> the "real deal", current data

Returns a JSON-serialisable dict carrying the final score, the Nova reasoning,
and per-request token counts + cost (for the UI's token counter).
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from phase1 import score_ticker, DEFAULT_CUTOFF
from fetch.fetchSECFilings import fetch_filing
from nova import converse, DEFAULT_MODEL

SYSTEM_PROMPT = (
    "You are a disciplined equity risk analyst. You are given (1) a quantitative "
    "1-5 score produced by a math model from point-in-time market data, with a "
    "per-metric breakdown, and (2) an excerpt from the company's most recent SEC "
    "filing as of a knowledge cutoff date.\n\n"
    "You get two filing sections: MD&A (management's discussion of actual results "
    "and outlook, a basis to RAISE when results/guidance are strong) and Risk "
    "Factors (downside the company discloses, a basis to LOWER).\n\n"
    "Your job: weigh the bull case (MD&A) against the bear case (Risk Factors), then "
    "decide whether to CONFIRM, RAISE, or LOWER the quantitative score, producing a "
    "final 1-5 score (1=strong sell, 3=hold, 5=strong buy) that is RELATIVE to the "
    "S&P 500 (i.e. will the stock beat or lag the market over ~6 months).\n\n"
    "Hard rules:\n"
    "- Use ONLY the information provided. Do NOT use any knowledge of events after "
    "the cutoff date.\n"
    "- BE DECISIVE. When MD&A results or guidance clearly lean positive or negative, "
    "commit to a Buy/Sell (4 or 2), or Strong (5 or 1) when the lean is strong. "
    "Reserve Hold (3) for genuinely balanced or unclear cases. Every company lists "
    "risks, so the mere presence of Risk Factors is NOT a reason to retreat to neutral.\n"
    "- Risk Factors are largely boilerplate; weight specific MD&A results/guidance more.\n"
    "- risk_flags must be SPECIFIC to THIS company and drawn from the provided text "
    "(a named product, customer, geography, regulation, or dependency). Do NOT list "
    "generic risks like 'competition', 'supply chain', or 'macroeconomic conditions'.\n"
    "- Respond with ONLY a JSON object, no prose before or after, in exactly this shape:\n"
    '{"final_score": <1-5 int>, "risk_flags": [<specific short strings>], '
    '"rationale": "<2-4 sentences>"}'
)


def _section_line(label: str, text: str, quality: str) -> list[str]:
    """Render one filing section, labelled by extraction quality so Nova knows how
    much to trust it (a keyword-fallback slice may be mislabelled; empty is empty)."""
    if quality == "section" and text:
        return ["", f"--- {label} (extracted section) ---", text]
    if quality == "fallback" and text:
        return ["", f"--- {label} (LOW-CONFIDENCE extract; may be mislabelled, weight lightly) ---", text]
    return ["", f"--- {label} ---", "(not reliably available in this filing)"]


def build_user_prompt(phase1_result: dict, filing: dict, anonymize: bool = False) -> str:
    """Compose the user message: the quant breakdown + the filing excerpt.

    `anonymize` withholds the ticker/name (used by the backtest leakage probe, so a
    past cutoff can't be answered from the model's memory of a famous name's outcome)."""
    ticker_line = (
        "TICKER: (withheld for blind evaluation)" if anonymize
        else f"TICKER: {phase1_result['ticker']} ({phase1_result.get('name')})"
    )
    lines = [
        ticker_line,
        f"SECTOR: {phase1_result.get('sector')}",
        f"KNOWLEDGE CUTOFF: {phase1_result['cutoff']}",
        "",
        f"QUANTITATIVE SCORE (math model): {phase1_result['score']}/5 "
        f"({phase1_result['label']}), raw={phase1_result['raw_score']}, "
        f"confidence={phase1_result['confidence']}",
        "Per-metric signal (sector-relative z-score x direction; + is bullish):",
    ]
    for entry in phase1_result["breakdown"]:
        signal = "n/a" if entry["signal"] is None else f"{entry['signal']:+.2f}"
        raw = "n/a" if entry["raw"] is None else f"{entry['raw']:.3f}"
        lines.append(f"  - {entry['metric']}: signal={signal}, raw={raw}, weight={entry['weight']}")

    filing_block = filing.get("filing")
    if filing_block:
        lines.append("")
        lines.append(f"SEC FILING: {filing_block['type']} filed {filing_block['filed_date']}")
        mdna_quality = filing_block.get("mdna_quality", "section" if filing_block.get("mdna_excerpt") else "empty")
        risk_quality = filing_block.get("risk_quality", "section" if filing_block.get("risk_excerpt") else "empty")
        lines += _section_line("MD&A excerpt (results & outlook)", filing_block.get("mdna_excerpt"), mdna_quality)
        lines += _section_line("Risk Factors excerpt", filing_block.get("risk_excerpt"), risk_quality)
        if mdna_quality != "section" and risk_quality != "section":
            lines += ["", "NOTE: no filing section was reliably extracted. Do NOT invent qualitative "
                          "views; lean toward CONFIRMING the quantitative score."]
    else:
        lines += ["", "SEC FILING: none available before cutoff."]

    lines += ["", "Return the JSON object now."]
    return "\n".join(lines)


def _parse_nova_json(reply_text: str, fallback_score: int) -> dict:
    """Robustly pull the JSON object out of Nova's reply. Falls back to the quant
    score (a safe Hold-toward default) if parsing fails."""
    try:
        start = reply_text.index("{")
        end = reply_text.rindex("}") + 1
        parsed = json.loads(reply_text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {
            "final_score": fallback_score,
            "confidence": "low",
            "adjustment": "confirm",
            "risk_flags": [],
            "rationale": "Nova response could not be parsed; falling back to the quant score.",
            "parse_error": True,
        }
    parsed["final_score"] = _clamp_score(parsed.get("final_score"), fallback_score)
    return parsed


def _clamp_score(value, fallback: int) -> int:
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return fallback


def _derive_adjustment(final_score: int, quant_score: int) -> str:
    """Adjustment from the actual scores (the model's self-label is unreliable)."""
    if final_score > quant_score:
        return "raise"
    if final_score < quant_score:
        return "lower"
    return "confirm"


def _derive_confidence(final_score: int, quant_score: int, coverage: float) -> str:
    """Confidence from how much the two methods agree and how complete the data is.
    Nova's own self-reported confidence is uniformly 'medium', so we ignore it."""
    gap = abs(final_score - quant_score)
    if coverage < 0.6 or gap >= 2:
        return "low"   # thin data, or the two methods strongly disagree
    if gap == 0 and coverage >= 0.9:
        return "high"  # math and LLM agree on near-complete data
    return "medium"


def score_ticker_v2(ticker: str, cutoff: datetime = DEFAULT_CUTOFF,
                    model_id: str = DEFAULT_MODEL, refresh: bool = False, on_stage=None,
                    anonymize: bool = False) -> dict:
    """
    Phase 1 score + Nova reasoning over the point-in-time filing.

    on_stage(message) is an optional callback invoked at each step, so a UI can
    poll progress ("Computing quant score...", "Reasoning with Nova...").
    `anonymize` withholds the ticker/name from Nova (backtest leakage probe).
    """
    started = time.time()

    def emit(message: str):
        if on_stage:
            on_stage(message)

    # The quant score and the SEC filing fetch are independent, so run them
    # concurrently (the filing's network time hides under Phase 1's).
    emit("Computing quant score and fetching SEC filing...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        phase1_future = pool.submit(score_ticker, ticker, cutoff, refresh=refresh)
        filing_future = pool.submit(fetch_filing, ticker, cutoff, refresh=refresh)
        phase1_result = phase1_future.result()
        filing = filing_future.result()
    if "error" in phase1_result:
        return phase1_result
    filing_block = filing.get("filing") or {}

    emit("Reasoning with Amazon Nova...")
    user_prompt = build_user_prompt(phase1_result, filing, anonymize=anonymize)
    reply_text, usage = converse(user_prompt, system=SYSTEM_PROMPT, model_id=model_id)
    nova_result = _parse_nova_json(reply_text, fallback_score=phase1_result["score"])

    quant_score = phase1_result["score"]
    final_score = nova_result["final_score"]
    parse_error = bool(nova_result.get("parse_error"))
    # Derive the adjustment and confidence from the actual scores; the model's
    # self-reported labels are unreliable (it said "confirm" while changing the
    # number, and returns a flat "medium" confidence every time). On a parse
    # failure we fell back to the quant score, so don't present that as Nova
    # confidently CONFIRMING - flag it as unavailable and low confidence.
    if parse_error:
        adjustment, confidence = "unavailable", "low"
    else:
        adjustment = _derive_adjustment(final_score, quant_score)
        confidence = _derive_confidence(final_score, quant_score, phase1_result["coverage"])

    return {
        "ticker": phase1_result["ticker"],
        "name": phase1_result.get("name"),
        "sector": phase1_result.get("sector"),
        "cutoff": phase1_result["cutoff"],
        "final_score": final_score,
        "final_label": {1: "Strong Sell", 2: "Sell", 3: "Hold", 4: "Buy", 5: "Strong Buy"}[final_score],
        "quant_score": quant_score,
        "adjustment": adjustment,
        "confidence": confidence,
        "parse_error": parse_error,
        "anonymized": anonymize,
        "risk_flags": nova_result.get("risk_flags", []),
        "rationale": nova_result.get("rationale"),
        "filing_used": filing_block.get("type"),
        "filing_quality": {"mdna": filing_block.get("mdna_quality"), "risk": filing_block.get("risk_quality")},
        "phase1": phase1_result,
        "tokens": usage["total_tokens"],
        "token_usage": usage,            # input/output/total + model, for the UI counter
        "cost_usd": usage["cost_usd"],
        "latency_sec": round(time.time() - started, 3),
    }


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    cutoff = DEFAULT_CUTOFF
    if len(sys.argv) > 2:
        cutoff = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=DEFAULT_CUTOFF.tzinfo)

    result = score_ticker_v2(ticker, cutoff)
    if "error" in result:
        print(f"{result['ticker']}: {result['error']}")
        raise SystemExit(1)

    print(f"\n{'='*60}")
    print(f" {result['name']} ({result['ticker']})  |  {result['sector']}  |  cutoff {result['cutoff']}")
    print(f"{'='*60}")
    print(f" FINAL: {result['final_score']}/5  {result['final_label']}   "
          f"(quant was {result['quant_score']}, Nova chose to {result['adjustment']})")
    print(f" Confidence: {result['confidence']}  |  filing: {result['filing_used']}")
    print(f" Risk flags: {', '.join(result['risk_flags']) or 'none'}")
    print(f"\n Rationale: {result['rationale']}")
    print(f"{'-'*60}")
    usage = result["token_usage"]
    print(f" Tokens: {usage['input_tokens']} in + {usage['output_tokens']} out = {usage['total_tokens']}")
    print(f" Cost: ${result['cost_usd']}  |  Latency: {result['latency_sec']}s  |  Model: {usage['model']}")
    print(f"{'='*60}\n")
