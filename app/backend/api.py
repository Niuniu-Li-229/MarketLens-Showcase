"""
api.py — FastAPI backend for MarketLens (v2).

Endpoints:
  GET /api/analyze/{ticker}     Stage 1: prices + anomalies + sentiment (fast, cached)
  GET /api/forecast/{ticker}    Stage 2: Transformer forecast (slow, disk-cached)
  GET /api/report/{ticker}      Stage 3: Claude AI report (on-demand)
  GET /api/market-info/{ticker} Live market metrics from yfinance

Run:
  cd app/backend
  uvicorn api:app --reload --port 8000
"""

import os
import sys
import json
import asyncio
from datetime import datetime, date, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

_SHOWCASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _SHOWCASE_DIR)

from models import AnalysisResult, AnomalyPoint, MarketEvent, PricePoint
from module1_data_fetcher import YFinancePriceFetcher, FinnhubNewsFetcher, DataCache
from module2_anomaly_detector import (
    FunnelDetector,
    ZScoreDetector, BollingerDetector, VolumeDetector,
    RSIDetector, MACDDetector,
    GapDetector, IntradayRangeDetector, ConsecutiveMoveDetector,
)
from module3_sentiment_lstm import (
    FinBERTAnalyzer,
    MockSentimentAnalyzer,
    MockForecaster,
    TransformerForecaster,
    TFTForecaster,
)

# Load FinBERT once at startup; fall back to mock if transformers not installed.
try:
    _sentiment_analyzer = FinBERTAnalyzer()
except Exception as _e:
    import warnings
    warnings.warn(f"[api] FinBERT unavailable ({_e}); falling back to MockSentimentAnalyzer.")
    _sentiment_analyzer = MockSentimentAnalyzer()
from module4_claude_report import StandardReportBuilder, ReportGenerator

app = FastAPI(title="MarketLens API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE_DIR = Path(_SHOWCASE_DIR) / "data_cache"
_CACHE_DIR.mkdir(exist_ok=True)


# ── Detector (fixed defaults, no UI threshold) ─────────────────────────────────

def _make_detector() -> FunnelDetector:
    return FunnelDetector([
        ZScoreDetector(),
        BollingerDetector(),
        VolumeDetector(),
        RSIDetector(),
        MACDDetector(),
        GapDetector(),
        IntradayRangeDetector(),
        ConsecutiveMoveDetector(),
    ], min_triggers=2)


# ── Forecast disk cache ────────────────────────────────────────────────────────

def _forecast_cache_path(ticker: str, start: date, end: date) -> Path:
    return _CACHE_DIR / f"{ticker.upper()}_{start}_{end}_forecast.json"

def _load_forecast_cache(ticker: str, start: date, end: date) -> dict | None:
    path = _forecast_cache_path(ticker, start, end)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def _save_forecast_cache(ticker: str, start: date, end: date, data: dict) -> None:
    try:
        with open(_forecast_cache_path(ticker, start, end), "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[Cache] Failed to save forecast: {e}")

def _load_sentiment_cache(ticker: str, end: date) -> tuple[float, str] | None:
    """Return (score, label) if a warm_up sentiment cache exists and covers end."""
    path = _CACHE_DIR / f"{ticker.upper()}_sentiment.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            sc = json.load(f)
        if sc.get("end", "") >= str(end):
            print(f"[Sentiment] Cache hit: {sc['label']} ({sc['score']:+.3f})")
            return sc["score"], sc["label"]
    except Exception:
        pass
    return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date '{s}'. Expected YYYY-MM-DD.")

def _rolling_mean(arr: list[float], w: int) -> list[float | None]:
    result = []
    for i in range(len(arr)):
        if i < w - 1:
            result.append(None)
        else:
            result.append(round(sum(arr[i - w + 1:i + 1]) / w, 2))
    return result

def _spy_cache_path() -> Path:
    return _CACHE_DIR / "SPY_prices.json"

def _fetch_spy(start: date, end: date) -> list[dict]:
    """Fetch S&P500 closes, disk-cached to avoid a yfinance round-trip every request."""
    cache_path = _spy_cache_path()

    # Load from cache and filter to requested range
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                all_spy = json.load(f)
            filtered = [r for r in all_spy if start.isoformat() <= r["date"] <= end.isoformat()]
            if filtered:
                print(f"[SPY] Cache hit: {len(filtered)} days")
                return filtered
        except Exception:
            pass

    # Cache miss — download and save
    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download(
            "^GSPC",
            start=str(start),
            end=str(end + timedelta(days=1)),
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return []
        rows = [
            {"date": row.Index.date().isoformat(), "close": round(float(row.Close), 2)}
            for row in df.itertuples()
        ]
        # Merge with existing cache before saving
        existing = {}
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    for r in json.load(f):
                        existing[r["date"]] = r["close"]
            except Exception:
                pass
        for r in rows:
            existing[r["date"]] = r["close"]
        merged = [{"date": d, "close": c} for d, c in sorted(existing.items())]
        with open(cache_path, "w") as f:
            json.dump(merged, f)
        print(f"[SPY] Fetched and cached {len(rows)} days")
        return rows
    except Exception as e:
        print(f"[SPY] Fetch failed: {e}")
        return []


# ── Serialisers ────────────────────────────────────────────────────────────────

def _ser_price(p: PricePoint, ma20=None, ma60=None) -> dict:
    return {
        "date": str(p.date),
        "open": p.open, "high": p.high, "low": p.low,
        "close": p.close, "volume": p.volume,
        "ma20": ma20, "ma60": ma60,
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

def _fetch_prices_and_news(ticker: str, start: date, end: date):
    """Shared M1 fetch used by multiple endpoints. Returns (prices, events, news_available).

    News strategy — cache-first to avoid slow Finnhub re-fetching:
      1. If CSV cache exists for ticker, filter by date range and return immediately.
      2. Otherwise fetch from Finnhub (slow on first run), then cache.
      3. If FINNHUB_API_KEY is missing or fetch fails, fall back gracefully.
    """
    # Check price cache — re-fetch if stale (last cached date > 3 days before end)
    _cache = DataCache()
    cached_prices = _cache.load_prices(ticker)
    if cached_prices and cached_prices[-1].date >= end - timedelta(days=3):
        prices = [p for p in cached_prices if start <= p.date <= end]
        print(f"[Prices] Cache hit: {len(prices)} days for {ticker}")
    else:
        if cached_prices:
            print(f"[Prices] Cache stale (last: {cached_prices[-1].date}, end: {end}), re-fetching…")
        prices = YFinancePriceFetcher().fetch_prices(ticker, start, end)

    if not prices:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker} ({start}~{end})")

    news_available = True
    # Check CSV cache first — avoids the ~5-min Finnhub re-fetch on every request
    _cache = DataCache()
    cached_news = _cache.load_news(ticker)
    if cached_news:
        events = [e for e in cached_news if start <= e.date <= end]
        print(f"[News] Cache hit: {len(events)} events for {ticker} {start}~{end}")
    else:
        try:
            news_fetcher = FinnhubNewsFetcher()
            events = news_fetcher.fetch_news(ticker, start, end)
        except Exception as e:
            print(f"[News] Falling back — no events: {e}")
            news_available = False
            events = []

    return prices, events, news_available


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/analyze/{ticker}")
async def analyze(
    ticker: str,
    start: str = Query(default="2021-01-01", description="Start date YYYY-MM-DD"),
    end:   str | None = Query(default=None,  description="End date YYYY-MM-DD (defaults to today)"),
):
    """
    Stage 1 — fast path.
    Runs Module 1 (prices + news), Module 2 (anomaly detection),
    Module 3 sentiment. Also returns MA20/MA60 and S&P500 for Chart 1.
    News falls back gracefully if FINNHUB_API_KEY is not set.
    """
    try:
        start_d = _parse_date(start)
        end_d   = _parse_date(end) if end else date.today()
        t = ticker.upper()

        prices, events, news_available = _fetch_prices_and_news(t, start_d, end_d)

        detector = _make_detector()
        anomalies = detector.detect(prices, events, ticker=t)

        # Prefer warm_up full-FinBERT cache; fall back to capped live inference
        last_trading_day = prices[-1].date if prices else end_d
        cached_sent = _load_sentiment_cache(t, last_trading_day)
        if cached_sent:
            s_score, s_label = cached_sent
        else:
            s_score, s_label = _sentiment_analyzer.analyze(events)

        closes = [p.close for p in prices]
        ma20   = _rolling_mean(closes, 20)
        ma60   = _rolling_mean(closes, 60)
        spy    = _fetch_spy(start_d, end_d)

        total_return = (
            (prices[-1].close - prices[0].open) / prices[0].open * 100.0
            if prices else 0.0
        )

        return {
            "ticker":          t,
            "start_date":      start,
            "end_date":        end,
            "total_return":    round(total_return, 4),
            "prices":          [_ser_price(p, ma20[i], ma60[i]) for i, p in enumerate(prices)],
            "spy_prices":      spy,
            "events":          [_ser_event(e) for e in events],
            "anomalies":       [_ser_anomaly(a) for a in anomalies],
            "sentiment_score": s_score,
            "sentiment_label": s_label,
            "news_available":  news_available,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/forecast/{ticker}")
async def forecast(
    ticker: str,
    start: str      = Query(default="2021-01-01"),
    end:   str|None = Query(default=None),
):
    """
    Stage 2 — slow path (30–60 s for multi-year range).
    Runs Transformer + TFT and caches to disk keyed by ticker + last trading day.
    Subsequent calls for the same range are instant.
    """
    try:
        start_d = _parse_date(start)
        end_d   = _parse_date(end) if end else date.today()
        t = ticker.upper()

        # Load prices first so we can snap the cache key to the last trading day
        prices, events, _ = _fetch_prices_and_news(t, start_d, end_d)
        last_trading_day  = prices[-1].date if prices else end_d

        cached = _load_forecast_cache(t, start_d, last_trading_day)
        if cached:
            print(f"[Forecast] Cache hit: {t} {start_d}~{last_trading_day}")
            return cached

        detector  = _make_detector()
        anomalies = detector.detect(prices, events, ticker=t)

        loop = asyncio.get_event_loop()

        # Transformer
        tf     = TransformerForecaster()
        tf_res = await loop.run_in_executor(None, lambda: tf.predict(prices, anomalies, ticker=t))

        # TFT
        tft     = TFTForecaster()
        tft_res = await loop.run_in_executor(None, lambda: tft.predict(prices, anomalies, ticker=t))

        def _to_list(arr):
            if arr is None: return []
            return [round(float(x), 4) if x is not None else None for x in arr]

        data = {
            "ticker":           t,
            # Transformer
            "model_name":       tf_res.model_name,
            "day5_price":       tf_res.day5_price,
            "forecast_5d":      tf_res.forecast_5d,
            "actual":           _to_list(tf_res.actual),
            "predicted":        _to_list(tf_res.predicted),
            "test_dates":       [str(d) for d in tf_res.test_dates],
            "dir_accuracy":     round(tf_res.dir_accuracy, 4),
            "mae":              round(tf_res.mae, 2),
            "sector_name":      tf_res.sector_name,
            # TFT
            "tft_model_name":   tft_res.model_name,
            "tft_day5_price":   tft_res.day5_price,
            "tft_actual":       _to_list(tft_res.actual),
            "tft_predicted":    _to_list(tft_res.predicted),
            "tft_test_dates":   [str(d) for d in tft_res.test_dates],
            "tft_dir_accuracy": round(tft_res.dir_accuracy, 4),
            "tft_mae":          round(tft_res.mae, 2),
        }
        _save_forecast_cache(t, start_d, last_trading_day, data)
        return data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/report/{ticker}")
async def generate_report(
    ticker: str,
    start: str      = Query(default="2021-01-01"),
    end:   str|None = Query(default=None),
):
    """
    Stage 3 — GPT-4o report (on-demand, requires OPENAI_API_KEY).
    Uses cached Transformer day5 price if available, otherwise mock.
    """
    try:
        start_d = _parse_date(start)
        end_d   = _parse_date(end) if end else date.today()
        t = ticker.upper()

        prices, events, _ = _fetch_prices_and_news(t, start_d, end_d)
        detector  = _make_detector()
        anomalies = detector.detect(prices, events, ticker=t)

        last_trading_day = prices[-1].date if prices else end_d
        cached_sent  = _load_sentiment_cache(t, last_trading_day)
        s_score, s_label = cached_sent if cached_sent else _sentiment_analyzer.analyze(events)
        cached_fc    = _load_forecast_cache(t, start_d, last_trading_day)
        pred_price   = (
            cached_fc.get("tft_day5_price") or cached_fc.get("day5_price") if cached_fc
            else MockForecaster().predict(prices, anomalies).day5_price
        )

        total_return = (
            (prices[-1].close - prices[0].open) / prices[0].open * 100.0
            if prices else 0.0
        )

        result = AnalysisResult(
            ticker          = t,
            start_date      = start_d,
            end_date        = end_d,
            total_return    = total_return,
            anomalies       = anomalies,
            predicted_price = pred_price,
            sentiment_score = s_score,
            sentiment_label = s_label,
        )

        generator   = ReportGenerator(builder=StandardReportBuilder())
        report_text = generator.generate(result)
        return {"ticker": t, "report": report_text}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/market-info/{ticker}")
async def market_info(ticker: str):
    """
    Live market metrics from yfinance:
    P/E ratio, market cap, 52-week position, beta, analyst rating/target,
    VIX, and 30-day relative performance vs S&P500.
    """
    try:
        import yfinance as yf
        t = ticker.upper()

        info        = yf.Ticker(t).info
        pe_ratio    = info.get("trailingPE")
        mkt_cap_raw = info.get("marketCap")
        w52_high    = info.get("fiftyTwoWeekHigh")
        w52_low     = info.get("fiftyTwoWeekLow")
        current     = info.get("currentPrice") or info.get("regularMarketPrice")
        beta        = info.get("beta")
        rec         = (info.get("recommendationKey") or "n/a").upper()
        target      = info.get("targetMeanPrice")

        w52_pos = upside = None
        if w52_high and w52_low and current and (w52_high - w52_low) > 0:
            w52_pos = round((current - w52_low) / (w52_high - w52_low) * 100, 1)
        if target and current:
            upside = round((target - current) / current * 100, 1)

        cap_label = "n/a"
        if mkt_cap_raw:
            cap_label = (
                f"${mkt_cap_raw / 1e12:.1f}T" if mkt_cap_raw >= 1e12 else
                f"${mkt_cap_raw / 1e9:.0f}B"  if mkt_cap_raw >= 1e9  else
                f"${mkt_cap_raw / 1e6:.0f}M"
            )

        vix = vix_label = None
        try:
            vix_hist = yf.Ticker("^VIX").history(period="5d")
            if not vix_hist.empty:
                vix = round(float(vix_hist["Close"].iloc[-1]), 1)
                vix_label = "Low fear" if vix < 15 else "Moderate" if vix < 25 else "High fear"
        except Exception:
            pass

        rel_perf = None
        try:
            hs = yf.Ticker(t).history(period="30d")["Close"]
            hm = yf.Ticker("SPY").history(period="30d")["Close"]
            if len(hs) > 1 and len(hm) > 1:
                rel_perf = round(
                    float((hs.iloc[-1] - hs.iloc[0]) / hs.iloc[0] * 100
                          - (hm.iloc[-1] - hm.iloc[0]) / hm.iloc[0] * 100), 1
                )
        except Exception:
            pass

        return {
            "ticker":          t,
            "pe_ratio":        round(pe_ratio, 1) if pe_ratio else None,
            "market_cap":      cap_label,
            "week52_position": w52_pos,
            "beta":            round(beta, 2) if beta else None,
            "analyst_rating":  rec,
            "analyst_target":  round(target, 2) if target else None,
            "upside":          upside,
            "vix":             vix,
            "vix_label":       vix_label,
            "rel_perf_30d":    rel_perf,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
