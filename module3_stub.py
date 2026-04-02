"""
Module 3 — Sentiment Analysis + LSTM Forecast
Owner: Person 3

Design: Sentiment and forecasting are separated into their own base classes.
Each concern is independently swappable — e.g. replace MockSentimentAnalyzer
with FinBERTAnalyzer without touching the forecasting code at all.

        SentimentAnalyzer (abstract)
        ├── MockSentimentAnalyzer    ← used now
        └── FinBERTAnalyzer          ← TODO

        PriceForecaster (abstract)
        ├── MockForecaster           ← used now
        └── LSTMForecaster           ← TODO
"""

from abc import ABC, abstractmethod
from models import MarketEvent, AnomalyPoint, PricePoint


# ── Sentiment ─────────────────────────────────────────────────────────────────

class SentimentAnalyzer(ABC):
    """
    Abstract interface for sentiment analysis.
    Returns (score, label): score ∈ [-1.0, +1.0], label ∈ {bullish, neutral, bearish}.
    """

    @abstractmethod
    def analyze(self, events: list[MarketEvent]) -> tuple[float, str]:
        ...


class MockSentimentAnalyzer(SentimentAnalyzer):
    """Heuristic mock based on EventType. Replace with FinBERT for real scores."""

    def analyze(self, events: list[MarketEvent]) -> tuple[float, str]:
        from models import EventType
        if not events:
            return 0.0, "neutral"

        bullish = {EventType.EARNINGS, EventType.ANALYST, EventType.PRODUCT}
        bearish = {EventType.REGULATORY, EventType.MACRO}

        score = sum(
            0.3 if e.event_type in bullish else -0.3 if e.event_type in bearish else 0.0
            for e in events
        )
        score = max(-1.0, min(1.0, score))
        label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
        return round(score, 2), label


class FinBERTAnalyzer(SentimentAnalyzer):
    """
    TODO: Real sentiment using ProsusAI/finbert.
    pip install transformers torch
    """

    def __init__(self):
        # self.pipe = pipeline("text-classification", model="ProsusAI/finbert")
        raise NotImplementedError("FinBERT not yet integrated.")

    def analyze(self, events: list[MarketEvent]) -> tuple[float, str]:
        # scores = [self.pipe(e.description)[0] for e in events]
        # ...aggregate and normalize...
        raise NotImplementedError


# ── Forecasting ───────────────────────────────────────────────────────────────

class PriceForecaster(ABC):
    """
    Abstract interface for price forecasting.
    Takes full price history + anomalies as features.
    Returns predicted next close price.
    """

    @abstractmethod
    def predict(
        self,
        prices: list[PricePoint],
        anomalies: list[AnomalyPoint],
    ) -> float:
        ...


class MockForecaster(PriceForecaster):
    """Nudges last close based on most recent anomaly direction."""

    def predict(self, prices: list[PricePoint], anomalies: list[AnomalyPoint]) -> float:
        if not prices:
            return 0.0
        last_close = prices[-1].close
        if not anomalies:
            return round(last_close, 2)
        nudge = 0.01 if anomalies[-1].is_gain() else -0.01
        return round(last_close * (1 + nudge), 2)


class LSTMForecaster(PriceForecaster):
    """
    TODO: Load a pre-trained LSTM model and predict from a feature vector
    built from price history + anomaly flags.

    pip install tensorflow numpy
    Train and save: model.save(f"saved_models/{ticker}_lstm.h5")
    """

    def __init__(self, model_path: str):
        # import tensorflow as tf
        # self.model = tf.keras.models.load_model(model_path)
        raise NotImplementedError("LSTM model not yet trained.")

    def predict(self, prices: list[PricePoint], anomalies: list[AnomalyPoint]) -> float:
        # features = self._build_features(prices, anomalies)
        # return float(self.model.predict(features)[0][0])
        raise NotImplementedError

    def _build_features(
        self,
        prices: list[PricePoint],
        anomalies: list[AnomalyPoint],
    ):
        """
        TODO: Build feature matrix — shape (sequence_len, n_features).
        Suggested features per timestep:
          - open_to_close_change
          - volume (normalized)
          - is_anomaly flag (1/0)
          - sentiment_score of nearby events
        """
        raise NotImplementedError
