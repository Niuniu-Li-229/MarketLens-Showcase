# MarketLens

> **"Markets don't move in a vacuum."**
> An end-to-end stock analysis pipeline with anomaly detection, Transformer forecasting, and Claude AI reports — served as an interactive web application.

---

## What it does

MarketLens runs a 5-module pipeline on any ticker and date range, then presents the results as an interactive web dashboard.

| Module | What it does | Technology |
|--------|-------------|------------|
| **Module 1** | Fetches historical prices and news events | yFinance + Finnhub API |
| **Module 2** | Detects anomalous trading days using 8 detectors (ZScore, Bollinger, Volume, RSI, MACD, Gap, Intraday, Consecutive) with a 2-layer funnel | Custom detector ensemble |
| **Module 3** | Sentiment analysis on news + Transformer price forecasting | Mock sentiment · PyTorch Transformer |
| **Module 4** | Generates a three-paragraph analyst report | Claude claude-sonnet-4-6 (Anthropic) |
| **Module 5** | Visualizes all outputs as interactive charts | Recharts (React) |

---

## Web dashboard

The dashboard loads progressively in three stages so you see results as quickly as possible:

- **Stage 1 — loads automatically** on page open: price chart with MA20/MA60/S&P500 comparison, anomaly detection chart with expandable event list
- **Stage 2 — on demand (Run Forecast button)**: Transformer actual-vs-predicted chart with directional accuracy and MAE; first run trains the model (~2–4 min), all subsequent runs are instant from disk cache
- **Stage 3 — on demand (Generate Report button)**: live market metrics (P/E, beta, VIX, analyst rating) + Claude AI analyst report

---

## Project structure

```
MarketLens-Showcase/
├── models.py                    # Shared data contracts (PricePoint, AnomalyPoint, …)
├── module1_data_fetcher.py      # yFinance + Finnhub fetchers with CSV cache
├── module2_anomaly_detector.py  # 8-detector funnel anomaly detection
├── module3_sentiment_lstm.py    # Sentiment analyser + LSTM/Transformer forecasters
├── module4_claude_report.py     # Claude AI report builder
├── module5_visualizer.py        # Matplotlib chart generator (CLI use)
├── main_pipeline.py             # CLI entry point — runs all 5 modules
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
│       │       ├── Chart3Forecast.jsx        # Module 3: forecast chart + sentiment
│       │       └── Chart4Report.jsx          # Module 4: market metrics + AI report
│       ├── index.html
│       ├── package.json
│       └── vite.config.js
```

---

## Setup

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

```bash
# Required for Module 4 (AI report)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Required for Module 1 news fetching (free tier works)
FINNHUB_API_KEY=your_finnhub_key_here

# Optional — only needed if running main_pipeline.py directly
OPENAI_API_KEY=your_openai_key_here
```

> **Both keys are optional for basic use.**
> - Without `FINNHUB_API_KEY`: the app runs price + anomaly analysis only (no news context). An alert is shown in the dashboard.
> - Without `ANTHROPIC_API_KEY`: Stages 1 and 2 work fully; the "Generate Report" button in Stage 3 will return an error.
>
> Get a free Finnhub key at [finnhub.io](https://finnhub.io) · Get an Anthropic key at [console.anthropic.com](https://console.anthropic.com)

### 3. Install backend dependencies

```bash
cd app/backend
pip install -r requirements.txt
```

> **Note:** `torch` and `transformers` are large packages (~2 GB). If you only want to run Stage 1 and Stage 3 (price/anomaly/report, no Transformer forecast), you can skip them — the app degrades gracefully.

### 4. Install frontend dependencies

```bash
cd app/frontend
npm install
```

---

## Running the web app

Open **two terminals**:

**Terminal 1 — Backend**
```bash
cd app/backend
uvicorn api:app --reload --port 8000
```

You should see:
```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

**Terminal 2 — Frontend**
```bash
cd app/frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

The dashboard loads META (2021–2026) automatically. Change the ticker and date range in the control panel and click **Analyze** to explore other stocks.

---

## Performance notes

| Operation | First run | Subsequent runs |
|-----------|-----------|-----------------|
| Price + anomaly (Stage 1) | ~5–10 s (yFinance download) | ~2–3 s (CSV cache) |
| News fetch | ~5 min for 5-year range (Finnhub rate limit) | Instant (CSV cache) |
| Transformer forecast (Stage 2) | ~2–4 min (model trains from scratch) | Instant (JSON cache) |
| Claude report (Stage 3) | ~5–10 s (live API call) | N/A (not cached) |

All caches are stored in `data_cache/` (excluded from git). Once a ticker + date range has been run once, all subsequent loads are fast.

---

## Running the CLI pipeline

To run all 5 modules from the command line (generates PNG charts):

```bash
# From the project root
python main_pipeline.py
```

Edit the last lines of `main_pipeline.py` to change the ticker and date range:

```python
run_pipeline(
    ticker = "META",
    start  = date(2021, 1, 1),
    end    = date(2026, 4, 5),
)
```

---

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/analyze/{ticker}?start=&end=` | Stage 1: prices, anomalies, sentiment |
| `GET /api/forecast/{ticker}?start=&end=` | Stage 2: Transformer forecast (cached) |
| `GET /api/report/{ticker}?start=&end=` | Stage 3: Claude AI report |
| `GET /api/market-info/{ticker}` | Live market metrics (P/E, beta, VIX, …) |
| `GET /health` | Health check |
