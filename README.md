# MarketLens

> **"Markets don't move in a vacuum."**
> An end-to-end stock analysis pipeline with anomaly detection, Transformer + TFT forecasting, FinBERT sentiment, and GPT-4o analyst reports — served as an interactive web application.

---

## What it does

MarketLens runs a 5-module pipeline on any ticker and date range, then presents the results as an interactive web dashboard.

| Module | What it does | Technology |
|--------|-------------|------------|
| **Module 1** | Fetches historical prices and news events | yFinance + Finnhub API |
| **Module 2** | Detects anomalous trading days using 8 detectors (ZScore, Bollinger, Volume, RSI, MACD, Gap, Intraday, Consecutive) with a 2-layer funnel | Custom detector ensemble |
| **Module 3** | FinBERT news sentiment + Transformer and TFT price forecasting | FinBERT · PyTorch Transformer · Temporal Fusion Transformer |
| **Module 4** | Generates a structured bullet-point analyst report | GPT-4o (OpenAI) |
| **Module 5** | Visualizes all outputs as interactive charts | Recharts (React) |

---

## Web dashboard

The dashboard loads progressively in three stages:

- **Stage 1 — loads automatically** on page open: price chart with MA20/MA60/S&P500 comparison, anomaly detection chart with expandable event list, FinBERT sentiment score
- **Stage 2 — on demand (Run Forecast button)**: Transformer and TFT actual-vs-predicted charts side by side, with directional accuracy and MAE; first run trains the models (~3 min each on CPU after warm_up), subsequent runs are instant from disk cache
- **Stage 3 — on demand (Generate Report button)**: live market metrics from Yahoo Finance (P/E, beta, VIX, analyst rating) + GPT-4o analyst report

---

## Project structure

```
MarketLens-Showcase/
├── models.py                    # Shared data contracts (PricePoint, AnomalyPoint, …)
├── module1_data_fetcher.py      # yFinance + Finnhub fetchers with CSV cache
├── module2_anomaly_detector.py  # 8-detector funnel anomaly detection
├── module3_sentiment_lstm.py    # FinBERT sentiment + Transformer/TFT forecasters
├── module4_claude_report.py     # GPT-4o report builder
├── module5_visualizer.py        # Matplotlib chart generator (CLI use)
├── main_pipeline.py             # CLI entry point — runs all 5 modules end-to-end
├── warm_up.py                   # Daily data refresh script (run before frontend)
│
├── app/
│   ├── backend/
│   │   ├── api.py               # FastAPI backend (4 endpoints)
│   │   └── requirements.txt     # Python dependencies
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx                       # Root — control panel + layout
│       │   ├── data/api.js                   # API client
│       │   └── components/
│       │       ├── Chart1Price.jsx           # Module 1: price + MA + S&P500 + volume
│       │       ├── Chart2Anomaly.jsx         # Module 2: anomaly scatter + event list
│       │       ├── Chart3Forecast.jsx        # Module 3: Transformer + TFT + sentiment
│       │       └── Chart4Report.jsx          # Module 4: market metrics + AI report
│       ├── index.html
│       ├── package.json
│       └── vite.config.js
```

---

## Setup

> **All commands below are run from the project root (`MarketLens-Showcase/`) unless a `cd` step says otherwise.**

### Prerequisites

- Python 3.11+
- Node.js 18+
- API keys (see below)

### 1. Clone the repo

```bash
git clone <repo-url>
cd MarketLens-Showcase
```

### 2. Create a `.env` file in the project root

```env
# Required for Module 4 AI report
OPENAI_API_KEY=your_openai_key_here

# Required for Module 1 news fetching (free tier works)
FINNHUB_API_KEY=your_finnhub_key_here
```

> **Both keys are optional for basic use.**
> - Without `FINNHUB_API_KEY`: price + anomaly analysis still works; news context and FinBERT sentiment are unavailable. A warning banner is shown in the dashboard.
> - Without `OPENAI_API_KEY`: Stages 1 and 2 work fully; the "Generate Report" button returns an error.
>
> Get a free Finnhub key at [finnhub.io](https://finnhub.io) · Get an OpenAI key at [platform.openai.com](https://platform.openai.com)

### 3. Install backend dependencies

```bash
cd app/backend
pip install -r requirements.txt
cd ../..
```

> **Note:** `torch` and `transformers` are large packages (~2 GB total). The app degrades gracefully if they are not installed — Stage 2 forecast and FinBERT sentiment will be unavailable, but Stage 1 and Stage 3 continue to work.

### 4. Install frontend dependencies

```bash
cd app/frontend
npm install
cd ../..
```

---

## Running the system

### Daily workflow

```
Every day (once):        python warm_up.py
Then open frontend:      two terminals (backend + frontend)
```

### Step 1 — Run warm_up (once per day)

`warm_up.py` refreshes all data caches so the frontend loads instantly. It does four things per ticker:

1. **Incremental price fetch** — appends only new trading days (no full re-download)
2. **Incremental news fetch** — appends only new events from Finnhub
3. **Full FinBERT sentiment** — scores all news events with no cap; saves result to `data_cache/{TICKER}_sentiment.json`
4. **Forecast** — trains Transformer + TFT; prompts you to confirm if a cache already exists

```bash
# Default: refreshes META
python warm_up.py

# Multiple tickers
python warm_up.py META AAPL TSLA
```

Expected output:
```
MarketLens warm_up — 2026-04-17
Tickers: META

────────────────────────────────────────────────────
  META
────────────────────────────────────────────────────
  [Prices] Up to date (last: 2026-04-16)
  [News] Fetching 2026-04-16 → 2026-04-17 (2 days, ~10s) …
  [News] +5 new events appended.
  [Sentiment] Running full FinBERT on 14149 events (885 batches, ~2m 48s estimated) …
  [Sentiment] neutral (+0.005) — done in 2m 50s, saved to META_sentiment.json
  [Forecast] Cache already exists for META (2021-01-01 ~ 2026-04-16).
  Refresh? [y/N]
```

> **warm_up is not required before every page refresh.** Run it once in the morning. The frontend reads from cache all day and remains fast. Re-run it only when you want fresh data.

---

### Step 2 — Start the backend

```bash
cd app/backend
uvicorn api:app --reload --port 8000
```

You should see FinBERT loading at startup (~10–30 s first time):
```
[Module 3] Loading FinBERT...
[Module 3] FinBERT ready.
INFO: Application startup complete.
```

> **Common mistakes:**
> - Running from the project root instead of `app/backend/` will fail with "Could not import module api". Always `cd app/backend` first.
> - Port already in use: `lsof -ti :8000 | xargs kill -9`

---

### Step 3 — Start the frontend

Open a second terminal:

```bash
cd app/frontend
npm run dev
```

Then open **http://localhost:5173** in your browser (Vite may use **5174** if 5173 is already in use).

The dashboard loads the default ticker (META, 2021 – today) automatically. Change the ticker and date range in the control panel and click **Analyze** to explore other stocks.

> **If the UI looks stale after code changes**, do a hard refresh: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows). If that doesn't help, restart the Vite dev server.

---

## Performance

### warm_up.py (run once per day)

| Step | First run (cold cache) | Subsequent runs |
|------|------------------------|-----------------|
| Price fetch | ~5–10 s (yFinance download) | ~5 s (delta only) or instant |
| News fetch | ~5 min (5-year history, Finnhub rate-limited) | ~10–30 s (delta only) |
| FinBERT sentiment | **~3 min** (14 k events, 885 batches × 0.19 s on CPU) | Instant (JSON cache hit) |
| Transformer forecast | **~3 min** (reuses pre-scored sentiment + 300 epochs) | Prompt to skip (instant) |
| TFT forecast | **~3 min** (reuses pre-scored sentiment + 50 epochs) | Prompt to skip (instant) |
| **Total** | **~9 min** | **~1–2 min** |

> All estimates are printed to the terminal before each step starts. Actual elapsed time is printed when each step finishes.

### Frontend (after warm_up)

| Operation | After warm_up | Without warm_up (cold) |
|-----------|--------------|------------------------|
| Stage 1 — price + anomaly | ~1–2 s (cache) | ~5–10 s (yFinance download) |
| Stage 1 — FinBERT sentiment | Instant (JSON cache) | ~15–30 s (capped at 150 events) |
| Stage 2 — Forecast | Instant (JSON cache) | ~2–4 min (trains from scratch) |
| Stage 3 — GPT-4o report | ~5–10 s (live API) | same |
| Market metrics | ~2–3 s (live yFinance) | same |

All caches are stored in `data_cache/` (excluded from git).

---

## Running the CLI pipeline

To run all 5 modules from the command line and generate PNG charts:

```bash
python main_pipeline.py
```

Edit the bottom of `main_pipeline.py` to change the ticker and date range:

```python
run_pipeline(
    ticker = "META",
    start  = date(2021, 1, 1),
    end    = date(2026, 4, 15),
)
```

---

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/analyze/{ticker}?start=&end=` | Stage 1: prices, anomalies, FinBERT sentiment |
| `GET /api/forecast/{ticker}?start=&end=` | Stage 2: Transformer + TFT forecast (disk-cached) |
| `GET /api/report/{ticker}?start=&end=` | Stage 3: GPT-4o analyst report |
| `GET /api/market-info/{ticker}` | Live market metrics from Yahoo Finance (P/E, beta, VIX, …) |
| `GET /health` | Health check |

All `end` parameters default to today's date if omitted.
