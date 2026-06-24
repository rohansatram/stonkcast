"""
Fetch the latest SEC 10-K/10-Q filed before a cutoff date, point-in-time.

Filings are selected by SEC filing date (genuinely point-in-time), and text
extraction strips the inline-XBRL noise that dominates raw 10-K HTML, then pulls
the readable prose and a Risk-Factors excerpt for the LLM to reason over.

Caching: the ticker->CIK map, each ticker's submission index, and each filing's
extracted text are all disk-cached (the text is keyed by accession number, so it
serves every cutoff once fetched).
"""

import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parent.parent))
from cache import cached_fetch  # noqa: E402

DEFAULT_CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_HEADERS = {"User-Agent": "stonkcast research@stonkcast.com"}
FILING_TYPES = ("10-K", "10-Q")

MAX_TEXT_CHARS = 250_000     # full readable text cap (kept large so we can locate later sections)
RISK_EXCERPT_CHARS = 8_000   # Risk Factors excerpt handed to the LLM (cost control)
MDNA_EXCERPT_CHARS = 8_000   # MD&A excerpt handed to the LLM
MIN_SECTION_CHARS = 1_500    # a real section is long; a TOC entry / cross-reference is not

# SEC asks clients to stay under ~10 req/s and to back off on throttling. We
# serialise EDGAR GETs with a minimum interval and retry on 403/429 so a backtest
# sweep or a cold judge-named ticker doesn't get blocked mid-run.
EDGAR_MIN_INTERVAL_SEC = 0.15
EDGAR_MAX_ATTEMPTS = 3
_edgar_lock = threading.Lock()
_last_edgar_request = [0.0]


def _edgar_get(url: str) -> requests.Response:
    for attempt in range(EDGAR_MAX_ATTEMPTS):
        with _edgar_lock:
            wait = EDGAR_MIN_INTERVAL_SEC - (time.monotonic() - _last_edgar_request[0])
            if wait > 0:
                time.sleep(wait)
            _last_edgar_request[0] = time.monotonic()
        response = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
        if response.status_code in (403, 429) and attempt < EDGAR_MAX_ATTEMPTS - 1:
            time.sleep(1.0 * (attempt + 1))  # throttled -> back off and retry
            continue
        response.raise_for_status()
        return response
    response.raise_for_status()
    return response


def fetch_filing(ticker: str, cutoff: datetime = DEFAULT_CUTOFF, refresh: bool = False) -> dict:
    """Latest 10-K/10-Q filed before the cutoff, with extracted text. Returns a
    dict with `error` set if nothing usable is found."""
    cik = _resolve_cik(ticker, refresh=refresh)
    if cik is None:
        return {"ticker": ticker.upper(), "filing": None, "error": "CIK not found"}

    latest_filing = _latest_filing_before(cik, cutoff, FILING_TYPES, refresh=refresh)
    annual_filing = _latest_filing_before(cik, cutoff, ("10-K",), refresh=refresh)
    if latest_filing is None and annual_filing is None:
        return {"ticker": ticker.upper(), "cik": cik, "filing": None, "error": "no 10-K/10-Q before cutoff"}

    primary = latest_filing or annual_filing
    latest_text = _text_of(latest_filing, refresh) if latest_filing else ""
    if annual_filing and (not latest_filing or annual_filing["accession"] != latest_filing["accession"]):
        annual_text = _text_of(annual_filing, refresh)
    else:
        annual_text = latest_text

    # Risk Factors: prefer the annual 10-K (a 10-Q's are usually just "no material
    # changes"). MD&A: prefer the most recent filing (latest quarter's results),
    # falling back to the 10-K. Excerpts are computed from cached text at read time.
    risk_excerpt, risk_quality = _best_excerpt(_risk_excerpt(annual_text), _risk_excerpt(latest_text))
    mdna_excerpt, mdna_quality = _best_excerpt(_mdna_excerpt(latest_text), _mdna_excerpt(annual_text))

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "filing": {
            "type": primary["type"],
            "filed_date": primary["filed_date"],
            "document_url": primary["document_url"],
            "qualitative_source": annual_filing["filed_date"] if annual_filing else primary["filed_date"],
            "risk_excerpt": risk_excerpt,
            "mdna_excerpt": mdna_excerpt,
            "risk_quality": risk_quality,   # section | fallback | empty
            "mdna_quality": mdna_quality,   # section | empty
        },
    }


def _text_of(filing: dict, refresh: bool) -> str:
    return _filing_text_cached(filing["document_url"], filing["accession"], refresh=refresh)["text"]


def _resolve_cik(ticker: str, refresh: bool = False) -> str | None:
    data = cached_fetch("SEC_COMPANY_TICKERS", _fetch_company_tickers, refresh=refresh)
    return data.get(ticker.upper())


def _fetch_company_tickers(_key: str) -> dict:
    resp = _edgar_get(COMPANY_TICKERS_URL)
    return {e["ticker"].upper(): str(e["cik_str"]) for e in resp.json().values()}


def _latest_filing_before(cik: str, cutoff: datetime, allowed_forms: tuple[str, ...],
                          refresh: bool = False) -> dict | None:
    submissions = cached_fetch(f"SEC_SUB_{cik}", lambda _k: _fetch_submissions(cik), refresh=refresh)
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    for form, filed_str, accession, doc in zip(forms, dates, accessions, docs):
        if form not in allowed_forms:
            continue
        filed = datetime.fromisoformat(filed_str).replace(tzinfo=timezone.utc)
        if filed >= cutoff:
            continue
        acc_nodash = accession.replace("-", "")
        return {
            "type": form,
            "filed_date": filed_str,
            "accession": accession,
            "document_url": EDGAR_FILING_URL.format(cik=cik, acc=acc_nodash, doc=doc),
        }
    return None


def _fetch_submissions(cik: str) -> dict:
    url = EDGAR_SUBMISSIONS_URL.format(cik=str(cik).zfill(10))
    return _edgar_get(url).json()


def _filing_text_cached(document_url: str, accession: str, refresh: bool = False) -> dict:
    return cached_fetch(f"SEC_DOC_{accession}", lambda _k: _fetch_filing_text(document_url), refresh=refresh)


def _fetch_filing_text(document_url: str) -> dict:
    resp = _edgar_get(document_url)
    text = _extract_readable(resp.text)
    return {"text": text[:MAX_TEXT_CHARS]}


def _extract_readable(html: str) -> str:
    """Strip inline-XBRL and hidden elements, return collapsed readable prose."""
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all(["script", "style"]):
        el.decompose()
    for el in soup.find_all(lambda t: t.name and t.name.startswith("ix:")):  # inline XBRL
        el.decompose()
    for el in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "").lower()):
        el.decompose()
    body = soup.find("body") or soup
    return re.sub(r"\s+", " ", body.get_text(separator=" ", strip=True)).strip()


_QUALITY_RANK = {"section": 2, "fallback": 1, "empty": 0}


def _section_excerpt(text: str, item_tokens: tuple[str, ...], required_phrase: str,
                     end_tokens: tuple[str, ...], max_chars: int) -> tuple[str, str]:
    """
    Slice a named filing section (e.g. Item 1A Risk Factors, Item 7 MD&A).

    The first occurrence of an item header is the table-of-contents entry
    (immediately followed by a page number), and cross-references mention the
    item without being the section. So we keep only occurrences where the
    section's defining phrase follows closely, then pick the one with the most
    content before the next section header.

    Returns (text, quality): ("...","section") on a real hit, ("","empty") otherwise.
    """
    lower = text.lower()

    starts = []
    for item_token in item_tokens:
        cursor = 0
        while True:
            idx = lower.find(item_token, cursor)
            if idx == -1:
                break
            if required_phrase in lower[idx: idx + 40]:
                starts.append(idx)
            cursor = idx + len(item_token)

    def section_length(start: int) -> int:
        next_headers = [lower.find(token, start + 10) for token in end_tokens]
        next_headers = [pos for pos in next_headers if pos != -1]
        end = min(next_headers) if next_headers else len(text)
        return end - start

    if starts:
        best_start = max(starts, key=section_length)
        if section_length(best_start) >= MIN_SECTION_CHARS:
            return text[best_start: best_start + max_chars], "section"
    return "", "empty"


def _risk_excerpt(text: str) -> tuple[str, str]:
    """Item 1A Risk Factors. Falls back to the last 'risk factors' mention, but
    NEVER to the cover page: a blind slice handed over as 'Risk Factors' would make
    the LLM reason confidently over noise. Returns (text, quality)."""
    excerpt, quality = _section_excerpt(
        text, item_tokens=("item 1a",), required_phrase="risk factors",
        end_tokens=("item 1b", "item 2.", "item 6."), max_chars=RISK_EXCERPT_CHARS,
    )
    if quality == "section":
        return excerpt, "section"
    idx = text.lower().rfind("risk factors")
    if idx != -1:
        return text[idx: idx + RISK_EXCERPT_CHARS], "fallback"
    return "", "empty"


def _mdna_excerpt(text: str) -> tuple[str, str]:
    """Management's Discussion & Analysis (Item 7 in a 10-K, Item 2 in a 10-Q).
    Returns ("","empty") if not found (some thin 10-Qs omit a full MD&A)."""
    return _section_excerpt(
        text, item_tokens=("item 7.", "item 2."), required_phrase="management",
        end_tokens=("item 7a", "item 8.", "item 3.", "item 4."), max_chars=MDNA_EXCERPT_CHARS,
    )


def _best_excerpt(*candidates: tuple[str, str]) -> tuple[str, str]:
    """Pick the highest-quality (text, quality) candidate (section > fallback > empty)."""
    best = ("", "empty")
    for text, quality in candidates:
        if _QUALITY_RANK[quality] > _QUALITY_RANK[best[1]]:
            best = (text, quality)
    return best


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    cutoff = DEFAULT_CUTOFF
    if len(sys.argv) > 2:
        cutoff = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
    print(f"Fetching SEC filing for {ticker} (cutoff {cutoff.date()}) ...")
    result = fetch_filing(ticker, cutoff)
    if result.get("error"):
        print("Error:", result["error"])
    else:
        filing = result["filing"]
        print(f"Latest filing: {filing['type']} filed {filing['filed_date']}")
        print(f"Qualitative source (10-K): {filing['qualitative_source']}")
        print(f"MD&A excerpt: {len(filing['mdna_excerpt'])} chars [{filing['mdna_quality']}] | "
              f"Risk excerpt: {len(filing['risk_excerpt'])} chars [{filing['risk_quality']}]")
        print("\n--- MD&A (first 400 chars) ---")
        print(filing["mdna_excerpt"][:400])
        print("\n--- Risk Factors (first 400 chars) ---")
        print(filing["risk_excerpt"][:400])
