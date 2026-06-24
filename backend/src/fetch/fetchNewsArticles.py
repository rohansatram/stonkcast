import requests
from datetime import datetime, timezone, timedelta
import time

CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)
NEWS_LOOKBACK_DAYS = 90  # fetch the 90 days leading up to the cutoff
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_ARTICLES = 25
GDELT_RATE_LIMIT_SECONDS = 5  # GDELT enforces 1 request per 5 seconds


def fetch_news_articles(ticker: str, company_name: str | None = None) -> dict:
    """
    Fetches news articles about a ticker from GDELT for the 90 days before the cutoff.
    company_name is optional but improves results - pass it from fetchStockData's info dict.
    """
    start_date = CUTOFF - timedelta(days=NEWS_LOOKBACK_DAYS)
    query = f"{company_name} {ticker}" if company_name else ticker

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": MAX_ARTICLES,
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": CUTOFF.strftime("%Y%m%d%H%M%S"),
        "format": "json",
        "sort": "DateDesc",
    }

    time.sleep(GDELT_RATE_LIMIT_SECONDS)
    response = requests.get(GDELT_URL, params=params, timeout=20)

    if response.status_code == 429:
        print("Warning: GDELT rate limit hit - returning empty news. Try again in a few minutes.")
        return {"ticker": ticker.upper(), "articles": []}

    response.raise_for_status()
    raw_articles = response.json().get("articles") or []
    articles = [_parse_article(raw) for raw in raw_articles]
    articles = [a for a in articles if a is not None]

    return {
        "ticker": ticker.upper(),
        "articles": articles,
    }


def _parse_article(raw: dict) -> dict | None:
    seen_date_str = raw.get("seendate", "")
    if not seen_date_str:
        return None

    try:
        # GDELT seendate format: "20251218T071500Z"
        published_at = datetime.strptime(seen_date_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None

    return {
        "title": raw.get("title", "").strip(),
        "url": raw.get("url", "").strip(),
        "domain": raw.get("domain", "").strip(),
        "language": raw.get("language", "").strip(),
        "published_at": published_at,
    }


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    company_name = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Fetching GDELT news for {ticker} (cutoff: {CUTOFF.date()}, lookback: {NEWS_LOOKBACK_DAYS} days)...")
    print("Waiting for GDELT rate limit...")
    data = fetch_news_articles(ticker, company_name)

    print(f"\nFound {len(data['articles'])} articles\n")
    for article in data["articles"]:
        print(f"[{article['published_at'].strftime('%Y-%m-%d')}] {article['domain']}: {article['title']}")
