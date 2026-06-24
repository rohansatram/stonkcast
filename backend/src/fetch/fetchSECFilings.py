import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone

CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/{path}"
EDGAR_HEADERS = {"User-Agent": "stonkcast research@stonkcast.com"}

FILING_TYPES = ["10-K", "10-Q"]


def fetch_sec_filings(ticker: str) -> dict:
    cik = _resolve_cik(ticker)
    if cik is None:
        return {"ticker": ticker.upper(), "cik": None, "filing": None, "error": "CIK not found"}

    latest_filing = _fetch_latest_filing_before_cutoff(cik)
    if latest_filing is None:
        return {"ticker": ticker.upper(), "cik": cik, "filing": None, "error": "No 10-K or 10-Q found before cutoff"}

    filing_text = _fetch_filing_text(latest_filing["document_url"])

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "filing": {
            "type": latest_filing["type"],
            "filed_date": latest_filing["filed_date"],
            "document_url": latest_filing["document_url"],
            "text": filing_text,
        },
    }


def _resolve_cik(ticker: str) -> str | None:
    """Convert a ticker symbol to its SEC CIK number."""
    search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&enddt=2026-01-01&forms=10-K"
    ticker_lookup_url = f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&search_text=&action=getcompany&output=atom"

    company_tickers_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(company_tickers_url, headers=EDGAR_HEADERS)
    response.raise_for_status()

    all_companies = response.json()
    ticker_upper = ticker.upper()

    for entry in all_companies.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"])  # raw integer string, e.g. "320193"

    return None


def _fetch_latest_filing_before_cutoff(cik: str) -> dict | None:
    """Find the most recent 10-K or 10-Q filed before the cutoff date."""
    url = EDGAR_SUBMISSIONS_URL.format(cik=str(cik).zfill(10))
    response = requests.get(url, headers=EDGAR_HEADERS)
    response.raise_for_status()

    submissions = response.json()
    recent_filings = submissions.get("filings", {}).get("recent", {})

    filing_types = recent_filings.get("form", [])
    filing_dates = recent_filings.get("filingDate", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    primary_documents = recent_filings.get("primaryDocument", [])

    for filing_type, filed_date_str, accession_number, primary_document in zip(
        filing_types, filing_dates, accession_numbers, primary_documents
    ):
        if filing_type not in FILING_TYPES:
            continue

        filed_date = datetime.fromisoformat(filed_date_str).replace(tzinfo=timezone.utc)
        if filed_date >= CUTOFF:
            continue

        accession_path = accession_number.replace("-", "")
        document_url = EDGAR_FILING_URL.format(
            path=f"data/{cik}/{accession_path}/{primary_document}"
        )

        return {
            "type": filing_type,
            "filed_date": filed_date_str,
            "document_url": document_url,
        }

    return None


def _fetch_filing_text(document_url: str) -> str:
    """Download the filing document, strip HTML tags, and return plain text capped at 50k characters."""
    response = requests.get(document_url, headers=EDGAR_HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    body = soup.find("body") or soup
    plain_text = body.get_text(separator=" ", strip=True)
    return plain_text[:50_000]


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching SEC filings for {ticker} (cutoff: {CUTOFF.date()})...")

    data = fetch_sec_filings(ticker)

    if data.get("error"):
        print(f"Error: {data['error']}")
    else:
        filing = data["filing"]
        print(f"\nCIK: {data['cik']}")
        print(f"Latest filing: {filing['type']} filed on {filing['filed_date']}")
        print(f"Document URL: {filing['document_url']}")
        print(f"\nFirst 500 chars of filing text:")
        print(filing["text"][:500])
