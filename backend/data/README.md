# Congress trades dataset (drop a file here)

Download a congressional-trades dataset and place the CSV or JSON file in this
folder. `fetch/fetchCongressTrades.py` auto-detects any `*.csv` / `*.json` here
and uses it as the source (no API key, no network).

**Where to get one (free):**
- Kaggle: search "senate stock watcher" / "house stock watcher" / "congress trading"
- data.world: "House Stock Watcher" and "Senate Stock Watcher" datasets

**Required columns** (Stock-Watcher schema, the loader is tolerant of variants):
`ticker`, `type` (purchase/sale), `transaction_date`, `disclosure_date`, `amount`, `representative`/`senator`

**Freshness matters:** the signal is gated on `disclosure_date`. A snapshot that
ends in 2023 powers historical demos but returns "none" for a Jan 2026 cutoff.
Use the most recent dump you can find.
