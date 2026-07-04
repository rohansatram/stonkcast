# stonkcast backend

A stock predictor for the Stock Risk Radar hackathon. It turns public financial
data into a 1-5 buy/sell score, and is built to be **validated by rewinding the
clock**: run it as if it were an earlier date, then compare its calls to what
actually happened. The hard rule everywhere is **no look-ahead**: a score may use
only data that was publicly knowable strictly before its cutoff date.

## Phases

- **Phase 1 (done):** fetch + pure-math scorer. No LLM. Deterministic, fully
  cached, point-in-time. This is the standalone baseline.
- **Phase 2 (done):** feed the data + Phase 1 output + a point-in-time SEC filing
  excerpt (MD&A + Risk Factors) to Amazon Nova, which confirms/raises/lowers the
  score and flags company-specific risks. Must beat Phase 1 to justify itself —
  use `backtest.compare()` to check.

## Run it

```bash
uv run python src/phase1.py AAPL              # score at the default Jan 2026 cutoff
uv run python src/phase1.py NVDA 2023-01-01   # score at an earlier (backtest) cutoff
uv run python src/backtest.py                 # grade the scorer vs actual forward alpha
uv run pytest -q                              # 79 tests
```

First run for a sector is slow (~30s: it fetches the ticker, SPY, and ~10 sector
peers). Everything is cached to `cache/` after that, so later runs are ~1-2s.

`backtest.fit_and_evaluate()` fits the bucket thresholds on the earlier cutoffs and
reports the held-out latest cutoff (forward-in-time, not in-sample), persisting the
fit to `cache/calibration.json` (which `scoring.py` then loads). `backtest.compare()`
runs Phase 1 vs Phase 2 on the same grid, with token/cost/latency totals.

## How the score is built

`score_ticker(ticker, cutoff)` →

1. **Fetch** full history (prices, earnings, annual + quarterly financials, balance
   sheet) via yfinance, disk-cached by ticker. SPY and sector peers too.
2. **Point-in-time filter** (`pointintime.py`): prices before the cutoff; earnings
   filtered by true *announcement* date; financials/balance sheet mapped to the
   announcement that made them public.
3. **Metrics** (`metrics.py`): the required metrics + volatility, averaged earnings
   surprise, 1-month and ~6-month (skip-a-month) momentum-vs-SPY, and 52-week
   position. Pure arithmetic; bad/missing data → `None`.
4. **Sector baseline** (`baselines.py`): the same metrics for a basket of sector
   peers, as of the cutoff, summarised with a robust **median + scaled-MAD** center/
   spread (steady at small peer counts) to z-score against; per-metric peer counts
   are surfaced.
5. **Score** (`scoring.py`): z-score each metric vs sector, apply direction so
   higher = more bullish, weighted-sum, bucket to 1-5. High volatility lowers
   confidence and pulls extreme calls toward Hold; an extreme 1/5 needs evidence
   from ≥2 data families; thin peer baskets cap confidence at "medium".

Output is fully inspectable: the score, the continuous `raw_score` behind it,
confidence, families used, and a per-metric breakdown (raw value, sector mean +
peer count, z, weight, contribution).

## No-leakage guarantees (the whole point)

- Prices are **split-adjusted only, never dividend-adjusted**, so post-cutoff
  dividends can't contaminate the P/E price level (`auto_adjust=True` would).
- Earnings use the **announcement date**, not the period end (the Dec-quarter
  announced in late January is correctly excluded from a Jan-1 cutoff).
- Financials/balance sheet become "known" only at the matching earnings
  announcement (90-day conservative fallback if unavailable, logged when it fires).
- `recommendation_mean` / `recommendation_key` are live analyst values with no
  history, so they are **display-only and excluded from scoring**.
- P/E, 52-week position, and trailing EPS are computed point-in-time, never read
  from the live `info` fields.
- `sector` is the one **not-point-in-time** input (a live yfinance classification);
  it only routes the peer basket, and is disclosed as such in the output.

## Known limits / future work

- `sector` and the peer baskets / backtest universe are **today's survivors**, so
  the headline hit-rate is optimistic vs a universe including delisted names. The
  backtest prints this caveat and per-sector / per-cutoff breakdowns.
- Phase 2's "use only pre-cutoff info" is a prompt rule, not an enforceable control;
  Nova may recall a famous name's outcome. Grade Phase 2 with `anonymize=True`
  (the leakage probe) and treat the anonymized number as the honest one.
- Revenue growth prefers fresh quarterly statements but falls back to **annual**
  when yfinance exposes too few quarters, so it can still be stale mid-cycle.
- Weights in `scoring.py` are hand-set **defaults**, not fitted; bucket thresholds
  can be fitted via `backtest.fit_and_evaluate()`. Tuning the weights against the
  backtest is the next step.
