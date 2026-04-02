"""
main.py — Pipeline entry point.

Design: All concrete implementations are injected at the top of this file.
To swap mock → real: change the class names here only.
No other file needs to change.
"""

from datetime import date
from models import AnalysisResult

# ── Swap implementations here — nowhere else ──────────────────────────────────
from module1_data_fetcher    import MockDataFetcher          as DataFetcher
from module2_anomaly_detector import ThresholdDetector, FunnelDetector
from module3_sentiment_lstm  import MockSentimentAnalyzer    as SentimentAnalyzer
from module3_sentiment_lstm  import MockForecaster           as Forecaster
from module4_claude_report   import StandardReportBuilder, ReportGenerator
# ─────────────────────────────────────────────────────────────────────────────


def build_pipeline():
    """
    Assemble the pipeline from its components.
    All wiring happens here — modules are unaware of each other.
    """
    fetcher   = DataFetcher()
    detector  = FunnelDetector([ThresholdDetector()], min_triggers=1)
    sentiment = SentimentAnalyzer()
    forecaster = Forecaster()
    generator = ReportGenerator(builder=StandardReportBuilder())
    return fetcher, detector, sentiment, forecaster, generator


def run_pipeline(ticker: str, start: date, end: date) -> str:
    fetcher, detector, sentiment, forecaster, generator = build_pipeline()

    print(f"[1] Fetching data for {ticker}...")
    prices = fetcher.fetch_prices(ticker, start, end)
    events = fetcher.fetch_news(ticker, start, end)

    print(f"[2] Detecting anomalies across {len(prices)} trading days...")
    anomalies = detector.detect(prices, events)
    print(f"    Found {len(anomalies)} anomalies.")

    print(f"[3] Analysing sentiment and forecasting price...")
    sentiment_score, sentiment_label = sentiment.analyze(events)
    predicted_price = forecaster.predict(prices, anomalies)

    total_return = (
        (prices[-1].close - prices[0].open) / prices[0].open * 100.0
        if prices else 0.0
    )

    result = AnalysisResult(
        ticker=ticker,
        start_date=start,
        end_date=end,
        total_return=total_return,
        anomalies=anomalies,
        predicted_price=predicted_price,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
    )
    print(f"    {result}")

    print(f"[4] Generating Claude report...")
    return generator.generate(result)


if __name__ == "__main__":
    report = run_pipeline(
        ticker="NVDA",
        start=date(2025, 9, 2),
        end=date(2025, 9, 5),
    )
    print("\n--- REPORT ---\n")
    print(report)
