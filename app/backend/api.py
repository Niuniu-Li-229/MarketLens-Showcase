"""
api.py — FastAPI backend wrapping Modules 1–4.

Run:
    cd Showcase/app/backend
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000

Modules are imported from the parent Showcase directory via sys.path insertion.
Swap mock → real by changing only the import aliases at the top.
"""

import os
import sys
from datetime import datetime, date

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Resolve parent Showcase directory so modules 1-4 can be imported
_SHOWCASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(_SHOWCASE_DIR))

from models import AnalysisResult, AnomalyPoint, MarketEvent, PricePoint  # noqa: E402
from module1_data_fetcher import MockDataFetcher as DataFetcher            # noqa: E402
from module2_anomaly_detector import ThresholdDetector, FunnelDetector    # noqa: E402
from module3_sentiment_lstm import (                                        # noqa: E402
    MockSentimentAnalyzer as SentimentAnalyzer,
    MockForecaster as Forecaster,
)
from module4_claude_report import StandardReportBuilder, ReportGenerator  # noqa: E402

app = FastAPI(title="MarketLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Serialisers ────────────────────────────────────────────────────────────────

def _ser_price(p: PricePoint) -> dict:
    return {
        "date": str(p.date),
        "open": p.open,
        "high": p.high,
        "low": p.low,
        "close": p.close,
        "volume": p.volume,
    }


def _ser_event(e: MarketEvent) -> dict:
    return {
        "date": str(e.date),
        "title": e.title,
        "description": e.description,
        "source": e.source,
        "event_type": e.event_type.value,
    }


def _ser_anomaly(a: AnomalyPoint) -> dict:
    return {
        "date": str(a.date),
        "percent_change": a.percent_change,
        "is_gain": a.is_gain(),
        "comment": a.comment,
        "price_point": _ser_price(a.price_point),
        "related_events": [_ser_event(e) for e in a.related_events],
    }


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date '{s}'. Expected YYYY-MM-DD.",
        )


def _run_modules_1_3(
    ticker: str,
    start: date,
    end: date,
    threshold: float,
) -> tuple:
    """Run Modules 1–3 and return raw pipeline outputs."""
    fetcher = DataFetcher()
    detector = FunnelDetector([ThresholdDetector(threshold=threshold)], min_triggers=1)
    sentiment = SentimentAnalyzer()
    forecaster = Forecaster()

    prices = fetcher.fetch_prices(ticker, start, end)
    events = fetcher.fetch_news(ticker, start, end)
    anomalies = detector.detect(prices, events)
    sentiment_score, sentiment_label = sentiment.analyze(events)
    predicted_price = forecaster.predict(prices, anomalies)

    total_return = (
        (prices[-1].close - prices[0].open) / prices[0].open * 100.0
        if prices else 0.0
    )
    return prices, events, anomalies, sentiment_score, sentiment_label, predicted_price, total_return


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/analyze/{ticker}")
async def analyze(
    ticker: str,
    start: str = Query(default="2025-09-02", description="Start date YYYY-MM-DD"),
    end: str = Query(default="2025-09-05", description="End date YYYY-MM-DD"),
    threshold: float = Query(
        default=0.5,
        description="Anomaly detection threshold (% open-to-close change). "
                    "Lower = more sensitive. MockDataFetcher moves are <1%, so 0.5 shows results.",
    ),
):
    """
    Run Modules 1–3 and return structured analysis data.

    Modules run:
      1 (Data Fetcher)     → prices + news events
      2 (Anomaly Detector) → anomaly list with related events
      3 (Sentiment/LSTM)   → sentiment score/label + predicted price
    """
    try:
        start_d = _parse_date(start)
        end_d = _parse_date(end)
        prices, events, anomalies, s_score, s_label, pred_price, total_ret = (
            _run_modules_1_3(ticker.upper(), start_d, end_d, threshold)
        )
        return {
            "ticker": ticker.upper(),
            "start_date": start,
            "end_date": end,
            "total_return": round(total_ret, 4),
            "prices": [_ser_price(p) for p in prices],
            "events": [_ser_event(e) for e in events],
            "anomalies": [_ser_anomaly(a) for a in anomalies],
            "predicted_price": pred_price,
            "sentiment_score": s_score,
            "sentiment_label": s_label,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/report/{ticker}")
async def generate_report(
    ticker: str,
    start: str = Query(default="2025-09-02"),
    end: str = Query(default="2025-09-05"),
    threshold: float = Query(default=0.5),
):
    """
    Run the full pipeline (Modules 1–4) and return a Claude-generated report.

    Requires ANTHROPIC_API_KEY in the environment.
    This endpoint is intentionally separate because Module 4 makes a live API
    call and may take several seconds.
    """
    try:
        start_d = _parse_date(start)
        end_d = _parse_date(end)
        prices, events, anomalies, s_score, s_label, pred_price, total_ret = (
            _run_modules_1_3(ticker.upper(), start_d, end_d, threshold)
        )

        result = AnalysisResult(
            ticker=ticker.upper(),
            start_date=start_d,
            end_date=end_d,
            total_return=total_ret,
            anomalies=anomalies,
            predicted_price=pred_price,
            sentiment_score=s_score,
            sentiment_label=s_label,
        )

        generator = ReportGenerator(builder=StandardReportBuilder())
        report_text = generator.generate(result)
        return {"ticker": ticker.upper(), "report": report_text}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok", "modules": ["module1", "module2", "module3", "module4"]}
