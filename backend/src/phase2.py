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
from nova import converse, converse_stream, DEFAULT_MODEL

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
    "- CONGRESSIONAL TRADES may be shown (and are also reflected in the quant 'congress' "
    "signal). Treat net buying as a mild bullish tilt and net selling as mild bearish, but "
    "weight them LIGHTLY (disclosures lag the trade by up to 45 days) and never let them "
    "override clear fundamentals.\n"
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


def _quant_block(phase1_result: dict, anonymize: bool = False) -> list[str]:
    """Ticker context + the quant score + per-metric breakdown (shared by every prompt).
    `anonymize` withholds the ticker/name (backtest leakage probe)."""
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
    return lines


def _filing_section(filing: dict, which: str) -> list[str]:
    """Render one filing section ('mdna' or 'risk') with its quality label."""
    filing_block = filing.get("filing")
    if not filing_block:
        return ["", "SEC FILING: none available before cutoff."]
    key, label = {
        "mdna": ("mdna_excerpt", "MD&A excerpt (results & outlook)"),
        "risk": ("risk_excerpt", "Risk Factors excerpt"),
    }[which]
    quality = filing_block.get(f"{which}_quality", "section" if filing_block.get(key) else "empty")
    header = [f"SEC FILING: {filing_block['type']} filed {filing_block['filed_date']}"]
    return header + _section_line(label, filing_block.get(key), quality)


def _congress_block(phase1_result: dict, anonymize: bool = False) -> list[str]:
    """Rich congress context (also reflected in the quant 'congress' signal).
    Withheld under anonymize: member names hint at identity."""
    congress = phase1_result.get("congress")
    if not congress or not congress.get("available") or anonymize:
        return []
    if congress.get("signal") == "none":
        return ["", f"CONGRESSIONAL TRADES: none disclosed in the {congress.get('window_days')}d before cutoff."]
    lines = ["", f"CONGRESSIONAL TRADES (disclosed in the {congress.get('window_days')}d before cutoff; "
                 f"also in the quant 'congress' signal): {congress.get('purchases')} buys, "
                 f"{congress.get('sales')} sales by {congress.get('n_members')} member(s) "
                 f"-> {congress.get('signal')}"]
    for trade in (congress.get("recent") or [])[:5]:
        amount = f"~${trade['amount_usd_est']:,}" if trade.get("amount_usd_est") else "n/a"
        lines.append(f"  - {trade.get('member')}: {trade.get('side')} {amount} (disclosed {trade.get('disclosed')})")
    return lines


def build_user_prompt(phase1_result: dict, filing: dict, anonymize: bool = False) -> str:
    """Single-call prompt: quant breakdown + both filing sections + congress."""
    lines = _quant_block(phase1_result, anonymize)
    filing_block = filing.get("filing")
    if filing_block:
        lines += _filing_section(filing, "mdna") + _filing_section(filing, "risk")[1:]  # drop dup SEC FILING header
        mdna_q = filing_block.get("mdna_quality", "section" if filing_block.get("mdna_excerpt") else "empty")
        risk_q = filing_block.get("risk_quality", "section" if filing_block.get("risk_excerpt") else "empty")
        if mdna_q != "section" and risk_q != "section":
            lines += ["", "NOTE: no filing section was reliably extracted. Do NOT invent qualitative "
                          "views; lean toward CONFIRMING the quantitative score."]
    else:
        lines += ["", "SEC FILING: none available before cutoff."]
    lines += _congress_block(phase1_result, anonymize)
    lines += ["", "Return the JSON object now."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Multi-agent panel: a bull and a bear argue, then a judge decides.
# --------------------------------------------------------------------------- #

BULL_SYSTEM = (
    "You are a disciplined buy-side equity analyst making the strongest possible bullish case. "
    "Argue why this stock is likely to OUTPERFORM the S&P 500 over the next ~6 months. "
    "Use ONLY the provided quant signals, the MD&A excerpt, and any congressional buying. "
    "Build a coherent investment thesis rather than summarizing the inputs. Prioritize "
    "catalysts, accelerating trends, improving fundamentals, valuation support, momentum, "
    "and management guidance. Cite specific metrics or statements whenever possible. "
    "Do NOT use any knowledge or events after the cutoff date. Do NOT speculate beyond "
    "what is reasonably implied by the provided evidence. Write 3-5 concise sentences. "
    "The score should reflect how compelling the bullish evidence is, not simply that you "
    "are taking the bullish side. End with a line exactly: 'BULL SCORE: N' where N is "
    "an integer from 1-5."
)

BEAR_SYSTEM = (
    "You are a disciplined short-selling equity analyst making the strongest possible bearish case. "
    "Argue why this stock is likely to UNDERPERFORM the S&P 500 over the next ~6 months. "
    "Use ONLY the provided quant signals, the Risk Factors excerpt, and any congressional selling. "
    "Build a coherent investment thesis rather than summarizing the inputs. Prioritize "
    "deteriorating fundamentals, execution risks, weakening momentum, valuation concerns, "
    "competitive threats, and management risks. Cite specific metrics or statements whenever "
    "possible. Do NOT use any knowledge or events after the cutoff date. Do NOT speculate "
    "beyond what is reasonably implied by the provided evidence. Write 3-5 concise sentences. "
    "The score should reflect how compelling the bearish evidence is, not simply that you "
    "are taking the bearish side. End with a line exactly: 'BEAR SCORE: N' where N is "
    "an integer from 1-5."
)

JUDGE_SYSTEM = (
    "You are the head of an investment committee. You are given a quantitative 1-5 score "
    "with its breakdown, a BULL argument, a BEAR argument, and optional congressional-trade "
    "context. Weigh them and decide a FINAL 1-5 score (1=strong sell, 3=hold, 5=strong buy) "
    "RELATIVE to the S&P 500 over ~6 months.\n\n"
    "Hard rules:\n"
    "- BE DECISIVE: when one side is clearly stronger, commit to a Buy/Sell (4/2) or Strong "
    "(5/1). Reserve Hold (3) for genuinely balanced cases. Do not strongly favour something based on one factor. be balanced.\n"
    "- risk_flags must be SPECIFIC to THIS company (named product, customer, geography, "
    "regulation, dependency), not generic ('competition', 'supply chain', 'macro').\n"
    "- Respond with ONLY a JSON object, no prose before or after, in exactly this shape:\n"
    '{"final_score": <1-5 int>, "risk_flags": [<specific short strings>], '
    '"rationale": "<2-4 sentences explaining which side won and why>"}'
)


def build_bull_prompt(phase1_result: dict, filing: dict, anonymize: bool = False) -> str:
    lines = _quant_block(phase1_result, anonymize)
    lines += _filing_section(filing, "mdna")
    lines += _congress_block(phase1_result, anonymize)
    lines += ["", "Make the BULL case now. End with 'BULL SCORE: N'."]
    return "\n".join(lines)


def build_bear_prompt(phase1_result: dict, filing: dict, anonymize: bool = False) -> str:
    lines = _quant_block(phase1_result, anonymize)
    lines += _filing_section(filing, "risk")
    lines += _congress_block(phase1_result, anonymize)
    lines += ["", "Make the BEAR case now. End with 'BEAR SCORE: N'."]
    return "\n".join(lines)


def build_judge_prompt(phase1_result: dict, bull_case: str, bear_case: str, anonymize: bool = False) -> str:
    lines = _quant_block(phase1_result, anonymize)
    lines += ["", "--- BULL ARGUMENT ---", bull_case, "", "--- BEAR ARGUMENT ---", bear_case]
    lines += _congress_block(phase1_result, anonymize)
    lines += ["", "Return the JSON object now."]
    return "\n".join(lines)


def _combine_usage(usages: list[dict]) -> dict:
    """Sum token usage + cost across the panel's calls (model assumed identical)."""
    return {
        "input_tokens": sum(u["input_tokens"] for u in usages),
        "output_tokens": sum(u["output_tokens"] for u in usages),
        "total_tokens": sum(u["total_tokens"] for u in usages),
        "cost_usd": round(sum(u["cost_usd"] for u in usages), 6),
        "model": usages[0]["model"] if usages else None,
        "calls": len(usages),
    }


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
                    anonymize: bool = False, panel: bool = True, on_token=None) -> dict:
    """
    Phase 1 score + Nova reasoning over the point-in-time filing.

    panel=True (default): a bull and a bear analyst argue (in parallel), then a judge
    decides the final score (3 Nova calls). panel=False: a single analyst call.

    on_stage(message) reports progress; on_token(agent, chunk) streams the bull/bear
    arguments token-by-token (agent is "bull" or "bear") for live UI rendering.
    `anonymize` withholds the ticker/name from Nova (backtest probe).
    """
    started = time.time()

    def emit(message: str):
        if on_stage:
            on_stage(message)

    # Quant score (which now includes the congress signal) and the SEC filing are
    # independent, so fetch them concurrently (network time hides under Phase 1's).
    emit("Computing quant score, fetching filing...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        phase1_future = pool.submit(score_ticker, ticker, cutoff, refresh=refresh)
        filing_future = pool.submit(fetch_filing, ticker, cutoff, refresh=refresh)
        phase1_result = phase1_future.result()
        filing = filing_future.result()
    if "error" in phase1_result:
        return phase1_result
    filing_block = filing.get("filing") or {}

    bull_case = bear_case = None
    if panel:
        # Bull and bear argue independently (run concurrently, streamed), then the judge decides.
        emit("Bull and bear analysts arguing...")

        def stream_agent(prompt, system_prompt, tag):
            token_cb = (lambda chunk: on_token(tag, chunk)) if on_token else None
            return converse_stream(prompt, system=system_prompt, model_id=model_id, on_token=token_cb)

        with ThreadPoolExecutor(max_workers=2) as pool:
            bull_future = pool.submit(stream_agent, build_bull_prompt(phase1_result, filing, anonymize),
                                      BULL_SYSTEM, "bull")
            bear_future = pool.submit(stream_agent, build_bear_prompt(phase1_result, filing, anonymize),
                                      BEAR_SYSTEM, "bear")
            bull_case, bull_usage = bull_future.result()
            bear_case, bear_usage = bear_future.result()
        emit("Judge weighing the arguments...")
        judge_prompt = build_judge_prompt(phase1_result, bull_case, bear_case, anonymize)
        reply_text, judge_usage = converse(judge_prompt, system=JUDGE_SYSTEM, model_id=model_id)
        usage = _combine_usage([bull_usage, bear_usage, judge_usage])
    else:
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
        "mode": "panel" if panel else "single",
        "bull_case": bull_case,
        "bear_case": bear_case,
        "risk_flags": nova_result.get("risk_flags", []),
        "rationale": nova_result.get("rationale"),
        "filing_used": filing_block.get("type"),
        "filing_quality": {"mdna": filing_block.get("mdna_quality"), "risk": filing_block.get("risk_quality")},
        "congress": phase1_result.get("congress"),  # computed in Phase 1, now part of the quant score
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
    if result.get("bull_case"):
        print(f" BULL:\n   {result['bull_case'].strip()}")
        print(f"\n BEAR:\n   {result['bear_case'].strip()}")
        print(f"{'-'*60}")
    print(f" FINAL: {result['final_score']}/5  {result['final_label']}   "
          f"(quant was {result['quant_score']}, judge chose to {result['adjustment']})")
    print(f" Confidence: {result['confidence']}  |  filing: {result['filing_used']}  |  mode: {result['mode']}")
    print(f" Risk flags: {', '.join(result['risk_flags']) or 'none'}")
    print(f"\n Rationale: {result['rationale']}")
    print(f"{'-'*60}")
    usage = result["token_usage"]
    print(f" Tokens: {usage['input_tokens']} in + {usage['output_tokens']} out = {usage['total_tokens']}")
    print(f" Cost: ${result['cost_usd']}  |  Latency: {result['latency_sec']}s  |  Model: {usage['model']}")
    print(f"{'='*60}\n")
