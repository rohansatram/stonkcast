import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time
from curl_cffi import requests

CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
MAX_ARTICLES = 20
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_news_articles(ticker: str) -> dict:
    url = YAHOO_RSS_URL.format(ticker=ticker.upper())
    response = _fetch_with_retry(url)
    articles = _parse_rss(response.text)
    articles_before_cutoff = [a for a in articles if a["published_at"] < CUTOFF][:MAX_ARTICLES]

    return {
        "ticker": ticker.upper(),
        "articles": articles_before_cutoff,
    }


def _fetch_with_retry(url: str, max_attempts: int = 3) -> requests.Response:
    last_response = None
    for attempt in range(max_attempts):
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10, impersonate="chrome")
        if response.status_code != 429:
            response.raise_for_status()
            return response
        last_response = response
        time.sleep(2 ** attempt)

    last_response.raise_for_status()
    raise RuntimeError(f"Failed after {max_attempts} attempts: {url}")


def _parse_rss(rss_xml: str) -> list[dict]:
    root = ET.fromstring(rss_xml)
    channel = root.find("channel")
    if channel is None:
        return []

    articles = []
    for item in channel.findall("item"):
        title = item.findtext("title", default="").strip()
        description = item.findtext("description", default="").strip()
        link = item.findtext("link", default="").strip()
        pub_date_str = item.findtext("pubDate", default="")

        if not pub_date_str:
            continue

        try:
            published_at = parsedate_to_datetime(pub_date_str).astimezone(timezone.utc)
        except Exception:
            continue

        articles.append({
            "title": title,
            "description": description,
            "link": link,
            "published_at": published_at,
        })

    return articles


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching news for {ticker} (cutoff: {CUTOFF.date()})...")
    data = fetch_news_articles(ticker)

    print(f"\nFound {len(data['articles'])} articles before cutoff\n")
    for article in data["articles"]:
        print(f"[{article['published_at'].strftime('%Y-%m-%d')}] {article['title']}")
        if article["description"]:
            print(f"  {article['description'][:120]}...")
        print()
