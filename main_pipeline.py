"""
main_pipeline.py — Pipeline entry point.
Runs both LSTM (baseline) and Transformer (improved) from Module 3.
"""

from datetime import date
from models import AnalysisResult

from module1_data_fetcher     import YFinanceNewsFetcher      as DataFetcher
from module2_anomaly_detector import FunnelDetector
from module2_anomaly_detector import ZScoreDetector, BollingerDetector
from module2_anomaly_detector import IQRDetector, VolumeDetector
from module3_sentiment_lstm   import FinBERTAnalyzer          as SentimentAnalyzer
from module3_sentiment_lstm   import LSTMForecaster, TransformerForecaster
from module4_claude_report    import StandardReportBuilder, ReportGenerator
from module5_visualizer       import generate_all_charts
from known_events             import enrich_anomalies_with_known_events


def build_pipeline():
    fetcher   = DataFetcher()
    detector  = FunnelDetector(
        detectors = [
            ZScoreDetector(window=20, z_threshold=2.0),
            BollingerDetector(window=20, k=2.0),
            IQRDetector(window=20, multiplier=1.5),
            VolumeDetector(window=20, multiplier=2.0),
        ],
        min_triggers = 2,
    )
    sentiment   = SentimentAnalyzer()
    lstm        = LSTMForecaster()
    transformer = TransformerForecaster()
    generator   = ReportGenerator(builder=StandardReportBuilder())
    return fetcher, detector, sentiment, lstm, transformer, generator


def run_pipeline(ticker: str, start: date, end: date) -> str:
    fetcher, detector, sentiment, lstm, transformer, generator = build_pipeline()

    print(f"\n{'='*52}")
    print(f"  Running pipeline for {ticker}")
    print(f"  Period: {start} → {end}")
    print(f"{'='*52}")

    # Step 1: Fetch data
    print(f"[1] Fetching data for {ticker}...")
    prices = fetcher.fetch_prices(ticker, start, end)
    events = fetcher.fetch_news(ticker, start, end)

    # Step 2: Detect anomalies
    print(f"[2] Detecting anomalies across {len(prices)} trading days...")
    anomalies = detector.detect(prices, events)
    anomalies = enrich_anomalies_with_known_events(anomalies, ticker)
    print(f"    Found {len(anomalies)} anomalies.")

    # Step 3: Sentiment + both models
    print(f"[3] Analysing sentiment...")
    sentiment_score, sentiment_label = sentiment.analyze(events)

    print(f"[3] Training LSTM (baseline)...")
    lstm_result = lstm.predict(prices, anomalies, ticker)

    print(f"[3] Training Transformer (improved)...")
    tf_result = transformer.predict(prices, anomalies, ticker)

    total_return = (
        (prices[-1].close - prices[0].open) / prices[0].open * 100.0
        if prices else 0.0
    )

    # Use Transformer Day 5 price as the primary prediction
    result = AnalysisResult(
        ticker          = ticker,
        start_date      = start,
        end_date        = end,
        total_return    = total_return,
        anomalies       = anomalies,
        predicted_price = tf_result.day5_price,
        sentiment_score = sentiment_score,
        sentiment_label = sentiment_label,
    )
    print(f"    {result}")

    # Step 4: AI report
    print(f"[4] Generating AI report...")
    report = generator.generate(result)
    print("\n--- REPORT ---\n")
    print(report)

    # Step 5: Charts — pass both model results to visualizer
    print(f"\n[5] Generating poster charts...")
    generate_all_charts(
        ticker    = ticker,
        prices    = prices,
        anomalies = anomalies,
        result    = result,
        report    = report,
        lstm_result = lstm_result,
        tf_result   = tf_result,
    )

    return report


if __name__ == "__main__":
    run_pipeline(
        ticker = "TSLA",
        start  = date(2018, 1, 1),
        end    = date(2026, 4, 4),
    )