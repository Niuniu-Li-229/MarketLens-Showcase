"""
main_pipeline.py — Pipeline entry point.
Compares Transformer vs TFT using the same 8 anomaly-derived features.
"""

from datetime import date
from models import AnalysisResult

from module1_data_fetcher     import YFinancePriceFetcher   as PriceFetcher
from module1_data_fetcher     import FinnhubNewsFetcher     as NewsFetcher
from module2_anomaly_detector import (
    FunnelDetector,
    ZScoreDetector, BollingerDetector, VolumeDetector,
    RSIDetector, MACDDetector,
    GapDetector, IntradayRangeDetector, ConsecutiveMoveDetector,
)
from module3_sentiment_lstm   import FinBERTAnalyzer        as SentimentAnalyzer
from module3_sentiment_lstm   import TransformerForecaster, TFTForecaster
from module4_claude_report    import StandardReportBuilder, ReportGenerator
from module5_visualizer       import generate_all_charts


def build_pipeline():
    price_fetcher = PriceFetcher()
    news_fetcher  = NewsFetcher()
    detector      = FunnelDetector([
        ZScoreDetector(),
        BollingerDetector(),
        VolumeDetector(),
        RSIDetector(),
        MACDDetector(),
        GapDetector(),
        IntradayRangeDetector(),
        ConsecutiveMoveDetector(),
    ], min_triggers=2)
    sentiment   = SentimentAnalyzer()
    transformer = TransformerForecaster()
    tft         = TFTForecaster()
    generator   = ReportGenerator(builder=StandardReportBuilder())
    return price_fetcher, news_fetcher, detector, sentiment, transformer, tft, generator


def run_pipeline(ticker: str, start: date, end: date) -> str:
    (price_fetcher, news_fetcher,
     detector, sentiment, transformer, tft, generator) = build_pipeline()

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────
    print(f"\n[1] Fetching data for {ticker}...")
    prices = price_fetcher.fetch_prices(ticker, start, end)
    events = news_fetcher.fetch_news(ticker, start, end)
    print(f"    {len(prices)} price points, {len(events)} news events.")

    if not prices:
        raise ValueError(f"No price data for {ticker} ({start}~{end}).")

    # ── Step 2: Anomaly detection ─────────────────────────────────────────────
    print(f"\n[2] Detecting anomalies across {len(prices)} trading days...")
    anomalies = detector.detect(prices, events, ticker=ticker)
    print(f"    Found {len(anomalies)} anomalies.")

    # ── Step 3: Sentiment ─────────────────────────────────────────────────────
    print(f"\n[3] Analysing sentiment...")
    sentiment_score, sentiment_label = sentiment.analyze(events)
    print(f"    Sentiment: {sentiment_label} ({sentiment_score:+.2f})")

    # ── Step 3: Transformer (same features as anomaly detectors) ─────────────
    print(f"\n[3] Training Transformer (9 features incl. sentiment, seq=20)...")
    tf_result = transformer.predict(prices, anomalies, ticker=ticker, events=events)

    # ── Step 3: TFT (same features, encoder-decoder architecture) ────────────
    print(f"\n[3] Training TFT (9 features incl. sentiment, seq=20)...")
    tft_result = tft.predict(prices, anomalies, ticker=ticker, events=events)

    # ── Print comparison ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Model Comparison — {ticker}")
    print(f"{'─'*50}")
    print(f"  {'Model':<35} {'MAE':>8}  {'Dir Acc':>8}")
    print(f"  {'─'*50}")
    print(f"  {'Transformer (9f, seq=20)':<35} "
          f"${tf_result.mae:>7.2f}  {tf_result.dir_accuracy:>7.1%}")
    print(f"  {'TFT (9f, seq=20)':<35} "
          f"${tft_result.mae:>7.2f}  {tft_result.dir_accuracy:>7.1%}")
    print(f"{'─'*50}")
    winner = "TFT" if tft_result.mae < tf_result.mae else "Transformer"
    improvement = abs(tf_result.mae - tft_result.mae) / max(tf_result.mae, 1e-9) * 100
    print(f"  Winner by MAE: {winner} ({improvement:.1f}% lower MAE)\n")

    # Use TFT Day-5 as the headline predicted price (better model)
    predicted_price = float(tft_result.day5_price)

    total_return = (
        (prices[-1].close - prices[0].open) / prices[0].open * 100.0
        if prices else 0.0
    )

    result = AnalysisResult(
        ticker          = ticker,
        start_date      = start,
        end_date        = end,
        total_return    = total_return,
        anomalies       = anomalies,
        predicted_price = predicted_price,
        sentiment_score = sentiment_score,
        sentiment_label = sentiment_label,
    )
    print(f"    {result}")

    # ── Step 4: AI report ─────────────────────────────────────────────────────
    print(f"\n[4] Generating AI report...")
    report = generator.generate(result)
    print(f"\n{'─'*60}\n{report}\n{'─'*60}")

    # ── Step 5: Charts (Transformer = lstm_result slot, TFT = tf_result slot) ─
    print(f"\n[5] Generating poster charts...")
    generate_all_charts(
        ticker      = ticker,
        prices      = prices,
        anomalies   = anomalies,
        result      = result,
        report      = report,
        lstm_result = tf_result,    # left panel  → Transformer
        tf_result   = tft_result,   # right panel → TFT
    )

    return report


if __name__ == "__main__":
    run_pipeline(
        ticker = "META",
        start  = date(2021, 1, 1),
        end    = date(2026, 4, 15),
    )