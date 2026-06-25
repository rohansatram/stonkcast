"use strict";

const POLL_INTERVAL_MS = 700;
const SCORE_LABELS = { 1: "Strong Sell", 2: "Sell", 3: "Hold", 4: "Buy", 5: "Strong Buy" };

// Meme loading messages (cycled while a run is in flight).
const MEME_LOADING = [
  "consulting the stonks man… 📈",
  "asking Congress what they bought… 🏛️",
  "reading SEC filings so you don't have to… 📄",
  "summoning Amazon Nova… 🔮",
  "doing finance… 💸",
  "trust me bro modeling… 🧠",
  "stonks? or not stonks?",
];
let memeTimer = null;

const el = (id) => document.getElementById(id);

function startMemeLoader() {
  let i = 0;
  el("status-text").textContent = MEME_LOADING[0];
  memeTimer = setInterval(() => {
    i = (i + 1) % MEME_LOADING.length;
    el("status-text").textContent = MEME_LOADING[i];
  }, 1300);
}

function stopMemeLoader() {
  if (memeTimer) { clearInterval(memeTimer); memeTimer = null; }
}

let activePoll = null;  // so switching tabs cancels an in-flight poll

// --- Tabs ---
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    el("panel-" + tab.dataset.tab).classList.add("active");
    resetView();  // clear any prior run state and inputs when switching tabs
  });
});

function resetView() {
  if (activePoll) { clearInterval(activePoll); activePoll = null; }
  stopMemeLoader();
  ["status", "error", "result"].forEach((id) => { el(id).hidden = true; });
  el("status-text").textContent = "";
  ["demo-ticker", "live-ticker"].forEach((id) => { el(id).value = ""; });
}

// --- Forms ---
el("form-demo").addEventListener("submit", (event) => {
  event.preventDefault();
  run({
    ticker: el("demo-ticker").value.trim().toUpperCase(),
    mode: "demo",
    start_date: el("demo-start").value,
    end_date: el("demo-end").value,
  });
});

el("form-live").addEventListener("submit", (event) => {
  event.preventDefault();
  run({ ticker: el("live-ticker").value.trim().toUpperCase(), mode: "live" });
});

// --- Run + poll ---
async function run(request) {
  if (!request.ticker) return;  // don't start a run without a ticker

  showOnly("status");
  startMemeLoader();

  let jobId;
  try {
    const response = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    jobId = (await response.json()).job_id;
  } catch (err) {
    return showError("Could not reach the backend. Is the server running?");
  }
  poll(jobId);
}

function poll(jobId) {
  activePoll = setInterval(async () => {
    let job;
    try {
      job = await (await fetch("/api/status/" + jobId)).json();
    } catch (err) {
      clearInterval(activePoll);
      stopMemeLoader();
      return showError("Lost connection to the backend.");
    }

    if (job.status === "running") {
      // meme loader owns the status text while running
    } else if (job.status === "done") {
      clearInterval(activePoll);
      stopMemeLoader();
      renderResult(job.result);
    } else {
      clearInterval(activePoll);
      stopMemeLoader();
      showError(job.error || "Something went wrong.");
    }
  }, POLL_INTERVAL_MS);
}

// --- Rendering ---
function renderResult(result) {
  const score = result.final_score;
  const badge = el("score-badge");
  badge.className = "score-badge score-" + score;
  el("score-num").textContent = score;
  el("score-label").textContent = SCORE_LABELS[score];

  el("result-title").textContent = `${result.name || result.ticker} (${result.ticker})`;
  el("result-sub").textContent = `${result.sector || ""} · cutoff ${result.cutoff}`;

  const adjustText = {
    raise: `Nova raised it from ${result.quant_score}`,
    lower: `Nova lowered it from ${result.quant_score}`,
    confirm: `Nova confirmed the quant score`,
  }[result.adjustment];
  el("result-chips").innerHTML = chip(adjustText) + chip("confidence: " + result.confidence);

  el("rationale").textContent = result.rationale || "";
  el("risk-flags").innerHTML = (result.risk_flags || [])
    .map((flag) => `<span class="flag">${escapeHtml(flag)}</span>`)
    .join("");

  renderOutcome(result.outcome);
  renderCongress(result.congress);

  const usage = result.token_usage || {};
  el("m-tokens").textContent = result.tokens ?? 0;
  el("m-tokens-detail").textContent = usage.input_tokens != null
    ? `(${usage.input_tokens} in / ${usage.output_tokens} out)` : "";
  el("m-cost").textContent = (result.cost_usd ?? 0).toFixed(5);
  el("m-latency").textContent = result.latency_sec ?? 0;
  el("m-model").textContent = usage.model || "";

  showOnly("result");
}

function renderCongress(congress) {
  const block = el("congress");
  // Only show it when there's actually congressional activity to report.
  if (!congress || !congress.available || congress.signal === "none"
      || (congress.purchases === 0 && congress.sales === 0)) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  el("congress-summary").textContent =
    `${congress.signal} — ${congress.purchases} buys / ${congress.sales} sales disclosed before the cutoff`;
  el("congress-recent").innerHTML = (congress.recent || [])
    .map((trade) => {
      const amount = trade.amount_usd_est ? ` ~$${Number(trade.amount_usd_est).toLocaleString()}` : "";
      return `<span class="flag congress-${trade.side}">${escapeHtml(trade.member)} · ${trade.side}${amount}</span>`;
    })
    .join("");
}

function renderOutcome(outcome) {
  const block = el("outcome");
  if (!outcome) { block.hidden = true; return; }
  block.hidden = false;

  if (!outcome.available) {
    el("o-pred").textContent = "—";
    el("o-stock").textContent = el("o-market").textContent = el("o-alpha").textContent = "n/a";
    setVerdict("na", outcome.note || "outcome not available yet");
    return;
  }

  el("o-pred").textContent = outcome.predicted_direction;
  setReturn("o-stock", outcome.stock_return);
  setReturn("o-market", outcome.market_return);
  setReturn("o-alpha", outcome.alpha);

  if (outcome.hit === null) {
    setVerdict("na", "sat it out 🤷");
  } else if (outcome.hit) {
    setVerdict("hit", "called it 😎");
  } else {
    setVerdict("miss", "tuff 💀 (not financial advice)");
  }
}

function setReturn(id, value) {
  const node = el(id);
  if (value == null) { node.textContent = "n/a"; node.className = "v"; return; }
  node.textContent = (value >= 0 ? "+" : "") + (value * 100).toFixed(1) + "%";
  node.className = "v " + (value >= 0 ? "pos" : "neg");
}

function setVerdict(kind, text) {
  const node = el("o-verdict");
  node.className = "verdict " + kind;
  node.textContent = text;
}

function chip(text) { return text ? `<span class="chip">${escapeHtml(text)}</span>` : ""; }

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function showOnly(sectionId) {
  ["status", "error", "result"].forEach((id) => { el(id).hidden = id !== sectionId; });
}

function showError(message) {
  el("error").textContent = message;
  showOnly("error");
}
