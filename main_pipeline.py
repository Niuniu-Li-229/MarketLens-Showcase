"""
main_pipeline.py — Pipeline entry point.

切换模式：只改顶部 import 区域，其他代码不动。

Mock 模式（默认，无需 API key，快速测试）：
  - PriceFetcher  = MockPriceFetcher
  - NewsFetcher   = MockNewsFetcher
  - Forecaster    = MockForecaster

真实模式（需要 API keys）：
  - PriceFetcher  = YFinancePriceFetcher
  - NewsFetcher   = FinnhubNewsFetcher
  - Forecaster    = LSTMForecaster + TransformerForecaster
"""

from datetime import date
from models import AnalysisResult

# ── 当前模式：真实数据 ─────────────────────────────────────────────────────────
from module1_data_fetcher     import YFinancePriceFetcher   as PriceFetcher
from module1_data_fetcher     import (
    FinnhubNewsFetcher, AlphaVantageNewsFetcher,
    YFinanceEventsFetcher, KnownEventsFetcher, CompositeNewsFetcher,
)
from module2_anomaly_detector import (
    FunnelDetector,
    ZScoreDetector, BollingerDetector, VolumeDetector,
    RSIDetector, MACDDetector,
    GapDetector, IntradayRangeDetector, ConsecutiveMoveDetector,
)
from module3_sentiment_lstm   import MockSentimentAnalyzer  as SentimentAnalyzer
from module3_sentiment_lstm   import LSTMForecaster, TransformerForecaster
from module4_claude_report    import StandardReportBuilder, ReportGenerator
from module5_visualizer       import generate_all_charts
# ─────────────────────────────────────────────────────────────────────────────

# ── Mock 模式（注释掉上面，取消注释下面）────────────────────────────────────────
# from module1_data_fetcher     import MockPriceFetcher       as PriceFetcher
# from module1_data_fetcher     import MockNewsFetcher        as NewsFetcher
# from module2_anomaly_detector import FunnelDetector, ZScoreDetector, BollingerDetector
# from module2_anomaly_detector import VolumeDetector, RSIDetector, MACDDetector
# from module2_anomaly_detector import GapDetector, IntradayRangeDetector, ConsecutiveMoveDetector
# from module3_sentiment_lstm   import MockSentimentAnalyzer  as SentimentAnalyzer
# from module3_sentiment_lstm   import MockForecaster         as LSTMForecaster
# from module3_sentiment_lstm   import MockForecaster         as TransformerForecaster
# from module4_claude_report    import StandardReportBuilder, ReportGenerator
# from module5_visualizer       import generate_all_charts
# ─────────────────────────────────────────────────────────────────────────────


def build_pipeline():
    price_fetcher = PriceFetcher()
    # Composite news: Alpha Vantage + Finnhub + YFinance + Known (curated)
    fetchers = [YFinanceEventsFetcher(), KnownEventsFetcher()]
    try:
        fetchers.insert(0, AlphaVantageNewsFetcher())
    except Exception:
        pass  # Alpha Vantage key not set
    try:
        fetchers.insert(0, FinnhubNewsFetcher())
    except Exception as e:
        print(f"[News] Finnhub unavailable ({e}), using YFinance + curated events")
    news_fetcher = CompositeNewsFetcher(fetchers)
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
    sentiment  = SentimentAnalyzer()
    lstm       = LSTMForecaster()
    tf         = TransformerForecaster()
    generator  = ReportGenerator(builder=StandardReportBuilder())
    return price_fetcher, news_fetcher, detector, sentiment, lstm, tf, generator


def run_pipeline(ticker: str, start: date, end: date) -> str:
    (price_fetcher, news_fetcher,
     detector, sentiment, lstm, tf, generator) = build_pipeline()

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────
    print(f"\n[1] Fetching data for {ticker}...")
    prices = price_fetcher.fetch_prices(ticker, start, end)
    events = news_fetcher.fetch_news(ticker, start, end)
    print(f"    {len(prices)} price points, {len(events)} news events.")

    if not prices:
        raise ValueError(f"No price data for {ticker} ({start} ~ {end}). "
                         "Check ticker symbol and date range.")

    # ── Step 2: Anomaly detection ─────────────────────────────────────────────
    print(f"\n[2] Detecting anomalies across {len(prices)} trading days...")
    anomalies = detector.detect(prices, events, ticker=ticker)
    print(f"    Found {len(anomalies)} anomalies.")

    # ── Step 3: Sentiment + forecasting ──────────────────────────────────────
    print(f"\n[3] Analysing sentiment...")
    sentiment_score, sentiment_label = sentiment.analyze(events)
    print(f"    Sentiment: {sentiment_label} ({sentiment_score:+.2f})")

    print(f"\n[3] Training LSTM forecaster...")
    lstm_result = lstm.predict(prices, anomalies, ticker=ticker)

    print(f"\n[3] Training Transformer forecaster...")
    tf_result = tf.predict(prices, anomalies, ticker=ticker)

    # Use Transformer Day-5 prediction as the headline predicted price
    predicted_price = float(tf_result.day5_price)
    print(f"    Transformer Day-5 prediction: ${predicted_price:.2f}")

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
    print(f"\n    {result}")

    # ── Step 4: AI report ─────────────────────────────────────────────────────
    print(f"\n[4] Generating AI report...")
    report = generator.generate(result)
    print(f"\n{'─'*60}\n{report}\n{'─'*60}")

    # ── Step 5: Charts ────────────────────────────────────────────────────────
    print(f"\n[5] Generating poster charts...")
    generate_all_charts(
        ticker      = ticker,
        prices      = prices,
        anomalies   = anomalies,
        result      = result,
        report      = report,
        lstm_result = lstm_result,
        tf_result   = tf_result,
    )

    return report


if __name__ == "__main__":
    run_pipeline(
        ticker = "META",
        start  = date(2021, 1, 1),
        end    = date(2026, 4, 5),
    )