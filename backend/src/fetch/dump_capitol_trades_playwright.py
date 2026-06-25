"""
Capitol Trades dumper via Playwright DOM scraping -> backend/data/capitol_trades.json
(which fetchCongressTrades.py then reads, offline + point-in-time).

The site renders the trades table server-side (no client API to replay, the bff
endpoint is Cloudflare-blocked and CORS-locked), so we read the rendered table.
Disclosure date is derived as (traded date + 'filed after' days), which is exact,
avoiding the relative "Yesterday" text in the PUBLISHED column.

Setup (one time, on your machine):
    uv add playwright
    uv run playwright install chromium

Run:
    uv run python src/fetch/dump_capitol_trades_playwright.py        # until pages run out
    uv run python src/fetch/dump_capitol_trades_playwright.py 50     # first 50 pages (~600 trades)
"""

import asyncio
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from playwright.async_api import async_playwright

OUTPUT = Path(__file__).resolve().parents[2] / "data" / "capitol_trades.json"
SAFETY_PAGE_CAP = 3000   # ~36k trades; high enough to page back well past Oct 2025

HEADLESS = False
PAGE_PAUSE_MS = 700
CHECKPOINT_EVERY = 25  # save progress every N pages so a long run survives an interrupt

# Extract one page's rows straight from the table DOM.
ROW_EXTRACTOR = """() => [...document.querySelectorAll('table tbody tr')].map(tr => {
    const text = (el) => el ? el.innerText.trim() : '';
    const cells = tr.children;
    const detail = tr.querySelector('a[href*="/trades/"]');
    return {
        member: text(tr.querySelector('.politician-name')),
        ticker: text(tr.querySelector('.issuer-ticker')),
        type: text(tr.querySelector('.tx-type')),
        traded: text(cells[3]),          // e.g. "29 May\\n2026"
        filed_after: text(cells[4]),     // e.g. "days\\n25"
        size: text(tr.querySelector('.trade-size')),  // e.g. "1M-5M"
        id: detail ? detail.getAttribute('href') : null,
    };
})"""


def _amount(size_text: str) -> float | None:
    """'1M-5M' / '15K-50K' / '250K' -> midpoint in dollars."""
    if not size_text or "N/A" in size_text:
        return None
    values = []
    for part in re.split(r"[-–—]", size_text):
        match = re.search(r"([\d.]+)\s*([KMB]?)", part.strip().upper())
        if not match:
            continue
        number = float(match.group(1))
        number *= {"K": 1e3, "M": 1e6, "B": 1e9}.get(match.group(2), 1)
        values.append(number)
    return sum(values) / len(values) if values else None


def _flatten_dom(row: dict) -> dict | None:
    ticker = (row.get("ticker") or "").split(":")[0].strip().upper()
    if not ticker:
        return None
    try:
        traded = datetime.strptime(" ".join(row["traded"].split()), "%d %b %Y")
    except (ValueError, KeyError):
        return None
    gap_match = re.search(r"\d+", row.get("filed_after") or "")
    disclosure = traded + timedelta(days=int(gap_match.group())) if gap_match else traded
    return {
        "ticker": ticker,
        "type": (row.get("type") or "").lower(),
        "transaction_date": traded.strftime("%Y-%m-%d"),
        "disclosure_date": disclosure.strftime("%Y-%m-%d"),
        "amount": _amount(row.get("size")),
        "representative": row.get("member") or "Unknown",
    }


async def dump(max_pages: int, since: str | None = None) -> None:
    """Scrape newest-first. Stops at max_pages, or once disclosures fall before
    `since` (YYYY-MM-DD) - so you fetch exactly the date range you need."""
    rows: list[dict] = []
    seen_ids: set[str] = set()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=HEADLESS)
        page = await (await browser.new_context()).new_page()

        for page_num in range(1, min(max_pages, SAFETY_PAGE_CAP) + 1):
            await page.goto(f"https://www.capitoltrades.com/trades?page={page_num}",
                            wait_until="domcontentloaded")
            try:
                await page.wait_for_selector("table tbody tr", timeout=20000)
            except Exception:
                break
            await page.wait_for_timeout(PAGE_PAUSE_MS)

            raw_rows = await page.evaluate(ROW_EXTRACTOR)
            new = [r for r in raw_rows if r.get("id") and r["id"] not in seen_ids]
            if not new:  # no fresh rows -> we've run off the end (or pagination looped)
                break

            reached_floor = False
            for raw in new:
                seen_ids.add(raw["id"])
                flat = _flatten_dom(raw)
                if not flat:
                    continue
                if since and flat["disclosure_date"] < since:  # sorted newest-first -> we can stop
                    reached_floor = True
                    continue
                rows.append(flat)
            print(f"page {page_num}: +{len(new)} (total kept {len(rows)})")
            if page_num % CHECKPOINT_EVERY == 0 and rows:
                _save(rows)
                print(f"  checkpoint saved ({len(rows)} trades)")
            if reached_floor:
                print(f"reached disclosures before {since}; stopping.")
                break

        await browser.close()

    if not rows:
        print("No rows scraped. Open the page and confirm the table rendered.")
        return
    _save(rows)
    print(f"\nWrote {len(rows)} trades to {OUTPUT}")
    print("The agent will use this automatically (fetchCongressTrades reads backend/data/).")


def _save(rows: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=0))


if __name__ == "__main__":
    # Args (any order): a YYYY-MM-DD date = stop once disclosures fall before it;
    # an integer = max pages. e.g. `... 2025-10-01`  or  `... 50`  or  `... 2025-10-01 800`.
    max_pages, since = SAFETY_PAGE_CAP, None
    for arg in sys.argv[1:]:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", arg):
            since = arg
        elif arg.isdigit():
            max_pages = int(arg)
    print(f"Scraping newest-first" + (f" back to {since}" if since else f" (up to {max_pages} pages)") + " ...")
    asyncio.run(dump(max_pages, since))
