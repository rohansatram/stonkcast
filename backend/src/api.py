"""
HTTP API for the UI. Scoring runs in a background thread and reports progress
stages, so the frontend can poll for messages ("Computing quant score...",
"Reasoning with Nova...") and then the final result.

Two modes:
- live : score with today's data (the "real deal"); no outcome to measure yet.
- demo : score at a past start date and measure the ACTUAL alpha to an end date,
         so you can see whether the agent called it.

Run:  uv run python src/api.py     (serves the frontend + API at :8000)
"""

import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from phase2 import score_ticker_v2
from backtest import forward_alpha, outcome_bucket, _price_on_or_after
from fetch.fetchStockData import fetch_cached

UTC = timezone.utc
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
LIVE_MAX_AGE = timedelta(hours=12)  # in live mode, refresh the target/benchmark if the cache is older than this

app = FastAPI(title="stonkcast")
_jobs: dict[str, dict] = {}  # job_id -> {status, stage, result, error}


@app.middleware("http")
async def no_store_frontend(request, call_next):
    """Stop the browser caching the frontend during dev, so edits to the HTML/JS/CSS
    always take effect on refresh (a stale app.js was hiding UI fixes)."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store"
    return response


class ScoreRequest(BaseModel):
    ticker: str
    mode: str                       # "live" | "demo"
    start_date: str | None = None   # demo: the knowledge cutoff
    end_date: str | None = None     # demo: when to measure the outcome


def _run_job(job_id: str, request: ScoreRequest) -> None:
    job = _jobs[job_id]
    try:
        if request.mode == "live":
            cutoff = datetime.now(UTC)
            # Make "live" provably live: refresh the target and benchmark if their
            # cached blobs are stale. Peers/SPY otherwise stay cached to avoid rate
            # limits, but a day-old close behind a "today" score would be a contradiction.
            job.update(stage="Refreshing live market data...")
            fetch_cached(request.ticker, max_age=LIVE_MAX_AGE)
            fetch_cached("SPY", max_age=LIVE_MAX_AGE)
        else:
            cutoff = datetime.fromisoformat(request.start_date).replace(tzinfo=UTC)

        result = score_ticker_v2(
            request.ticker, cutoff, on_stage=lambda message: job.update(stage=message)
        )
        if "error" in result:
            job.update(status="error", error=result["error"])
            return

        if request.mode == "demo":
            job.update(stage="Measuring actual performance...")
            result["outcome"] = _measure_outcome(request, cutoff, result["final_score"])

        job.update(status="done", result=result)
    except Exception as exc:  # surface the failure to the UI rather than hanging
        job.update(status="error", error=f"{type(exc).__name__}: {exc}")


def _measure_outcome(request: ScoreRequest, cutoff: datetime, final_score: int) -> dict:
    """Actual stock vs market performance from cutoff to end_date (demo mode)."""
    end_date = datetime.fromisoformat(request.end_date).replace(tzinfo=UTC)
    horizon_days = (end_date - cutoff).days
    prices = fetch_cached(request.ticker)["prices"]
    spy_prices = fetch_cached("SPY")["prices"]

    alpha = forward_alpha(prices, spy_prices, cutoff, horizon_days)
    if alpha is None:
        return {"available": False, "note": "outcome window extends beyond available data"}

    stock_return = _return(prices, cutoff, end_date)
    market_return = _return(spy_prices, cutoff, end_date)
    direction = "bullish" if final_score >= 4 else "bearish" if final_score <= 2 else "neutral"
    hit = None if direction == "neutral" else (
        (direction == "bullish" and alpha > 0) or (direction == "bearish" and alpha < 0)
    )
    return {
        "available": True,
        "end_date": request.end_date,
        "stock_return": stock_return,
        "market_return": market_return,
        "alpha": round(alpha, 4),
        "true_bucket": outcome_bucket(alpha),
        "predicted_direction": direction,
        "hit": hit,
    }


def _return(prices, start: datetime, end: datetime) -> float | None:
    entry = _price_on_or_after(prices, start)
    exit_price = _price_on_or_after(prices, end)
    if entry in (None, 0) or exit_price is None:
        return None
    return round(exit_price / entry - 1.0, 4)


@app.post("/api/score")
def start_score(request: ScoreRequest) -> dict:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "stage": "Starting...", "result": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, request), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str) -> dict:
    return _jobs.get(job_id, {"status": "error", "error": "unknown job id"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
