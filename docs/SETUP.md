# Setup & local development

> All commands assume you are in the repo root unless a `cd` step says otherwise.

## Prerequisites

- Python **3.11+**
- Node.js **18+**
- (Optional) OpenAI and Finnhub API keys — see notes below

## 1. Clone and enter the repo

```bash
git clone https://github.com/Niuniu-Li-229/MarketLens-Showcase.git
cd MarketLens-Showcase
```

## 2. Configure environment variables

Create a `.env` file in the repo root:

```env
OPENAI_API_KEY=your_openai_key_here
FINNHUB_API_KEY=your_finnhub_key_here
```

Both keys are **optional** — the app degrades gracefully:
- Without `FINNHUB_API_KEY`: prices and anomalies still work; news context and FinBERT sentiment are unavailable.
- Without `OPENAI_API_KEY`: Stages 1 and 2 work fully; the "Generate Report" button returns a local fallback report.

Free Finnhub key: [finnhub.io](https://finnhub.io) · OpenAI key: [platform.openai.com](https://platform.openai.com)

## 3. Install backend dependencies

```bash
cd app/backend
pip install -r requirements.txt
cd ../..
```

> `torch` and `transformers` are large (~2 GB combined). The app degrades if they're missing — Stage 1 still works, Stage 2 forecast and FinBERT sentiment do not.

## 4. Install frontend dependencies

```bash
cd app/frontend
npm install
cd ../..
```

## 5. Warm the cache (once per day)

```bash
python scripts/warm_up.py            # default: META
python scripts/warm_up.py META AAPL  # multiple tickers
```

This does an incremental price + news fetch, scores all news with FinBERT, and trains the Transformer + TFT once. After this runs, the dashboard loads from cache and is fast all day.

## 6. Start the backend (terminal 1)

```bash
cd app/backend
uvicorn api:app --reload --port 8000
```

> Must be run from `app/backend/` — running from the repo root fails with `Could not import module api`.

## 7. Start the frontend (terminal 2)

```bash
cd app/frontend
npm run dev
```

Open **http://localhost:5173** in your browser. The dashboard auto-loads META (2021 → today). Change the ticker / date range and click **Analyze** to explore other stocks.

## Optional: run the CLI pipeline

```bash
python scripts/main_pipeline.py
```

Edit the bottom of [`../scripts/main_pipeline.py`](../scripts/main_pipeline.py) to change ticker and date range. Generates PNG charts in the current working directory.

## Reproduce the validation results

```bash
python scripts/walk_forward_validation.py
# Saves META_validation_5Y_PRICE.png and prints the comparison table
```
