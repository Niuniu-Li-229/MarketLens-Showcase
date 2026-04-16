"""
warm_up.py — Daily data refresh for MarketLens.

Run once per day before opening the frontend:
    python warm_up.py              # defaults to META
    python warm_up.py META AAPL TSLA

Steps per ticker:
    1. Incremental price fetch  — appends new trading days to CSV cache
    2. Incremental news fetch   — appends new events to CSV cache (Finnhub)
    3. Full FinBERT sentiment   — scores ALL events (no cap), saves to cache
    4. Transformer + TFT        — prompts if forecast cache already exists
"""

import sys
import json
import time
import math
import argparse
from datetime import date, timedelta
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models import PricePoint
from module1_data_fetcher import YFinancePriceFetcher, FinnhubNewsFetcher, DataCache
from module2_anomaly_detector import (
    FunnelDetector,
    ZScoreDetector, BollingerDetector, VolumeDetector,
    RSIDetector, MACDDetector,
    GapDetector, IntradayRangeDetector, ConsecutiveMoveDetector,
)
from module3_sentiment_lstm import FinBERTAnalyzer, TransformerForecaster, TFTForecaster

CACHE_DIR       = ROOT / "data_cache"
START_DATE      = date(2021, 1, 1)
DEFAULT_TICKERS = ["META"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_detector() -> FunnelDetector:
    return FunnelDetector([
        ZScoreDetector(), BollingerDetector(), VolumeDetector(),
        RSIDetector(), MACDDetector(),
        GapDetector(), IntradayRangeDetector(), ConsecutiveMoveDetector(),
    ], min_triggers=2)


def _to_list(arr) -> list:
    if arr is None:
        return []
    return [round(float(x), 4) if x is not None else None for x in arr]


def _fmt(seconds: float) -> str:
    """Format elapsed seconds as '4m 12s' or '45s'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ── Step 1: Incremental price fetch ───────────────────────────────────────────

def refresh_prices(ticker: str) -> list[PricePoint]:
    cache  = DataCache()
    cached = cache.load_prices(ticker)
    today  = date.today()

    if cached and cached[-1].date >= today - timedelta(days=3):
        print(f"  [Prices] Up to date (last: {cached[-1].date})")
        return cached

    fetch_start = (cached[-1].date + timedelta(days=1)) if cached else START_DATE
    print(f"  [Prices] Fetching {fetch_start} → {today} …")

    # YFinancePriceFetcher.save_prices() auto-merges by date key
    new_prices = YFinancePriceFetcher().fetch_prices(ticker, fetch_start, today)
    if not new_prices:
        print(f"  [Prices] No new data returned.")
        return cached or []

    print(f"  [Prices] +{len(new_prices)} new days appended.")
    # Reload to get full merged list
    return cache.load_prices(ticker)


# ── Step 2: Incremental news fetch ────────────────────────────────────────────

def refresh_news(ticker: str) -> list:
    cache  = DataCache()
    cached = cache.load_news(ticker)
    today  = date.today()

    last_date   = max(e.date for e in cached) if cached else START_DATE - timedelta(days=1)
    fetch_start = last_date + timedelta(days=1)

    if fetch_start > today:
        print(f"  [News] Up to date (last: {last_date})")
        return cached

    delta_days = (today - fetch_start).days + 1
    est_news   = max(10, math.ceil(delta_days / 7) * 2)   # ~2s per weekly window
    print(f"  [News] Fetching {fetch_start} → {today} ({delta_days} days, ~{est_news}s) …")
    t0 = time.time()
    try:
        # FinnhubNewsFetcher saves new events; DataCache.save_news auto-merges
        new_events = FinnhubNewsFetcher().fetch_news(ticker, fetch_start, today)
        print(f"  [News] +{len(new_events)} new events appended. ({_fmt(time.time()-t0)})")
    except Exception as e:
        print(f"  [News] Fetch failed ({e}). Using existing cache.")

    # Reload to get full merged list
    return cache.load_news(ticker)


# ── Step 3: Full FinBERT sentiment ────────────────────────────────────────────

def run_sentiment(ticker: str, events: list, last_date: date) -> tuple[float, str]:
    n_batches = math.ceil(len(events) / 16)
    est_secs  = math.ceil(n_batches * 0.4)   # ~0.4s per batch on CPU
    print(f"  [Sentiment] Running full FinBERT on {len(events)} events "
          f"({n_batches} batches, ~{_fmt(est_secs)} estimated) …")
    t0           = time.time()
    analyzer     = FinBERTAnalyzer()
    score, label = analyzer.analyze(events, max_events=None)
    elapsed      = time.time() - t0

    out = {
        "ticker":      ticker,
        "start":       str(START_DATE),
        "end":         str(last_date),
        "score":       score,
        "label":       label,
        "event_count": len(events),
        "updated_at":  str(date.today()),
    }
    path = CACHE_DIR / f"{ticker}_sentiment.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  [Sentiment] {label} ({score:+.3f}) — done in {_fmt(elapsed)}, saved to {path.name}")
    return score, label


# ── Step 4: Forecast (Transformer + TFT) ─────────────────────────────────────

def run_forecast(ticker: str, prices: list, events: list) -> None:
    last_trading_day = prices[-1].date
    cache_path = CACHE_DIR / f"{ticker}_{START_DATE}_{last_trading_day}_forecast.json"

    if cache_path.exists():
        ans = input(
            f"\n  [Forecast] Cache already exists for {ticker} "
            f"({START_DATE} ~ {last_trading_day}).\n"
            f"  Refresh? [y/N] "
        ).strip().lower()
        if ans != "y":
            print(f"  [Forecast] Skipped.")
            return

    detector  = _make_detector()
    anomalies = detector.detect(prices, events, ticker=ticker)
    print(f"  [Anomalies] {len(anomalies)} detected.")

    print(f"  [Forecast] Training Transformer (~1-3 min) …")
    t0     = time.time()
    tf_res = TransformerForecaster().predict(prices, anomalies, ticker=ticker)
    print(f"  [Forecast] Transformer done in {_fmt(time.time()-t0)} — day5: ${tf_res.day5_price:.2f}")

    print(f"  [Forecast] Training TFT (~1-3 min) …")
    t0      = time.time()
    tft_res = TFTForecaster().predict(prices, anomalies, ticker=ticker)
    print(f"  [Forecast] TFT done in {_fmt(time.time()-t0)} — day5: ${tft_res.day5_price:.2f}")

    data = {
        "ticker":           ticker,
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
    cache_path.write_text(json.dumps(data, indent=2))
    print(f"  [Forecast] Saved → {cache_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def warm_up(tickers: list[str]) -> None:
    wall_start = time.time()
    print(f"\nMarketLens warm_up — {date.today()}")
    print(f"Tickers: {', '.join(tickers)}\n")

    for ticker in tickers:
        ticker     = ticker.upper()
        t_start    = time.time()
        print(f"{'─' * 52}")
        print(f"  {ticker}")
        print(f"{'─' * 52}")

        prices = refresh_prices(ticker)
        if not prices:
            print(f"  [!] No price data for {ticker} — skipping.\n")
            continue

        events = refresh_news(ticker)
        run_sentiment(ticker, events, prices[-1].date)
        run_forecast(ticker, prices, events)

        print(f"  [{ticker}] Done in {_fmt(time.time()-t_start)}.\n")

    print(f"warm_up complete — total {_fmt(time.time()-wall_start)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketLens daily warm-up")
    parser.add_argument(
        "tickers", nargs="*", default=DEFAULT_TICKERS,
        help="Ticker symbols to refresh (default: META)",
    )
    args = parser.parse_args()
    warm_up(args.tickers)
