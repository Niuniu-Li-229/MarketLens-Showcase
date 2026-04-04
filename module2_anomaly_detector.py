"""
Module 2 — Anomaly Detector
Owner: Person 2

Design: Each detection algorithm is its own class implementing AnomalyDetector.
The FunnelDetector composes multiple detectors and requires a minimum number
of them to agree before flagging a day — this is the 4-layer funnel.

To add a new algorithm: add a new subclass. Never modify existing ones.

        AnomalyDetector (abstract)
        ├── ThresholdDetector     ← simple mock, used now
        ├── ZScoreDetector        ← TODO
        ├── BollingerDetector     ← TODO
        ├── IQRDetector           ← TODO
        ├── VolumeDetector        ← TODO
        └── FunnelDetector        ← composes the above, main entry point
"""

from abc import ABC, abstractmethod
from datetime import date, timedelta
from models import PricePoint, MarketEvent, AnomalyPoint, EventType


class AnomalyDetector(ABC):
    """
    Abstract interface for a single anomaly detection algorithm.
    Each subclass implements one layer of the funnel.
    """

    @abstractmethod
    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        """Return True if this price point is anomalous by this detector's criteria."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name — used in logging and poster diagrams."""
        ...


class ThresholdDetector(AnomalyDetector):
    """Simple percent-change threshold. Used as mock until real layers are built."""

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"Threshold(>{self.threshold}%)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        return abs(price.open_to_close_change()) >= self.threshold


class ZScoreDetector(AnomalyDetector):
    """Flag days where |z-score of daily return| > threshold (default 2.0)."""

    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold

    @property
    def name(self) -> str:
        return f"ZScore(>{self.z_threshold}σ)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        import numpy as np
        returns = [p.open_to_close_change() for p in all_prices]
        mean, std = np.mean(returns), np.std(returns)
        if std == 0:
            return False
        return abs((price.open_to_close_change() - mean) / std) > self.z_threshold


class BollingerDetector(AnomalyDetector):
    """Flag days where close breaks outside Bollinger Bands (window=20, k=2)."""

    def __init__(self, window: int = 20, k: float = 2.0):
        self.window = window
        self.k = k

    @property
    def name(self) -> str:
        return f"Bollinger(w={self.window}, k={self.k})"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        import numpy as np
        idx = next((i for i, p in enumerate(all_prices) if p.date == price.date), None)
        if idx is None or idx < self.window - 1:
            return False
        window_closes = [p.close for p in all_prices[idx - self.window + 1: idx + 1]]
        mid = np.mean(window_closes)
        std = np.std(window_closes)
        if std == 0:
            return False
        return price.close < mid - self.k * std or price.close > mid + self.k * std


class IQRDetector(AnomalyDetector):
    """Flag days where daily return falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""

    @property
    def name(self) -> str:
        return "IQR(1.5)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        import numpy as np
        returns = [p.open_to_close_change() for p in all_prices]
        q1, q3 = np.percentile(returns, 25), np.percentile(returns, 75)
        iqr = q3 - q1
        return price.open_to_close_change() < q1 - 1.5 * iqr or price.open_to_close_change() > q3 + 1.5 * iqr


class VolumeDetector(AnomalyDetector):
    """Flag days where volume > (multiplier × rolling average volume)."""

    def __init__(self, window: int = 20, multiplier: float = 2.0):
        self.window = window
        self.multiplier = multiplier

    @property
    def name(self) -> str:
        return f"Volume(>{self.multiplier}x avg)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        import numpy as np
        idx = next((i for i, p in enumerate(all_prices) if p.date == price.date), None)
        if idx is None or idx < self.window:
            return False
        avg_volume = np.mean([p.volume for p in all_prices[idx - self.window: idx]])
        if avg_volume == 0:
            return False
        return price.volume > self.multiplier * avg_volume


class FunnelDetector:
    """
    Composes multiple AnomalyDetectors and flags a day as anomalous
    only if at least `min_triggers` detectors agree.
    This is the main entry point for Module 2.

    Usage (mock):
        detector = FunnelDetector([ThresholdDetector()], min_triggers=1)

    Usage (full 4-layer funnel):
        detector = FunnelDetector([
            ZScoreDetector(),
            BollingerDetector(),
            IQRDetector(),
            VolumeDetector(),
        ], min_triggers=2)
    """

    def __init__(self, detectors: list[AnomalyDetector], min_triggers: int = 1):
        self.detectors    = detectors
        self.min_triggers = min_triggers

    def detect(
        self,
        prices: list[PricePoint],
        events: list[MarketEvent],
        ticker: str | None = None,
        pre_days: int = 3,
        post_days: int = 1,
    ) -> list[AnomalyPoint]:
        prices = sorted(prices, key=lambda p: p.date)
        anomalies = []
        for price in prices:
            triggered = [d for d in self.detectors if d.is_anomaly(price, prices)]
            if len(triggered) >= self.min_triggers:
                nearby  = _find_nearby_events(price.date, events, pre_days, post_days, ticker)
                comment = _build_comment(price, triggered, nearby)
                anomalies.append(AnomalyPoint(
                    price_point=price,
                    percent_change=price.open_to_close_change(),
                    related_events=nearby,
                    comment=comment,
                ))
        return anomalies


# ── helpers ───────────────────────────────────────────────────────────────────

# Lower number = shown first in related_events list passed to module4.
_EVENT_PRIORITY: dict[EventType, int] = {
    EventType.EARNINGS:   1,
    EventType.REGULATORY: 2,
    EventType.PRODUCT:    3,
    EventType.ANALYST:    4,
    EventType.PERSONNEL:  5,
    EventType.MACRO:      6,
    EventType.OTHER:      7,
}

# Ticker → keywords that must appear in title/description to count as related.
_TICKER_KEYWORDS: dict[str, list[str]] = {
    "NVDA":  ["nvidia", "nvda", "jensen huang", "h100", "h20", "blackwell"],
    "AAPL":  ["apple", "aapl", "tim cook", "iphone", "ipad", "ios", "mac"],
    "AMZN":  ["amazon", "amzn", "aws", "andy jassy"],
    "GOOGL": ["google", "googl", "alphabet", "sundar pichai", "gemini", "youtube"],
    "META":  ["meta", "facebook", "instagram", "whatsapp", "zuckerberg", "llama"],
    "TSLA":  ["tesla", "tsla", "elon musk", "cybertruck", "optimus"],
}


def _relevance_score(event: MarketEvent, anomaly_date: date) -> float:
    """
    Lower score = more relevant = shown first.

    Score = type_base + distance_penalty
      type_base      : EventType priority (EARNINGS=1 … OTHER=7)
      distance_penalty: pre-event  days × 0.5  (causal — news before anomaly)
                        post-event days × 0.8  (reaction — news after anomaly)

    Examples (pre_days=3, post_days=1):
      EARNINGS  day 0  → 1 + 0.0 = 1.0  ← best possible
      EARNINGS  day -1 → 1 + 0.5 = 1.5
      REGULATORY day 0 → 2 + 0.0 = 2.0
      EARNINGS  day -3 → 1 + 1.5 = 2.5
      OTHER     day +1 → 7 + 0.8 = 7.8  ← least relevant
    """
    days_before = (anomaly_date - event.date).days   # positive = before anomaly
    type_base   = _EVENT_PRIORITY.get(event.event_type, 7)
    if days_before >= 0:                              # pre-event (causal)
        distance_penalty = days_before * 0.5
    else:                                             # post-event (reaction)
        distance_penalty = abs(days_before) * 0.8
    return type_base + distance_penalty


def _find_nearby_events(
    anomaly_date: date,
    events: list[MarketEvent],
    pre_days: int,
    post_days: int,
    ticker: str | None = None,
    top_n: int = 10,
) -> list[MarketEvent]:
    # Asymmetric window: wider look-back (cause) than look-forward (reaction).
    nearby = [
        e for e in events
        if -pre_days <= (e.date - anomaly_date).days <= post_days
    ]
    # Ticker keyword filter: keep events that mention the company.
    if ticker:
        kws = _TICKER_KEYWORDS.get(ticker.upper(), [])
        if kws:
            text_match = [
                e for e in nearby
                if any(k in (e.title + " " + e.description).lower() for k in kws)
            ]
            # Fall back to unfiltered list if nothing survives (e.g. unknown ticker).
            if text_match:
                nearby = text_match
    # Sort by composite relevance: EventType priority + time-distance penalty.
    nearby.sort(key=lambda e: _relevance_score(e, anomaly_date))
    return nearby[:top_n]


def _build_comment(
    price: PricePoint,
    triggered: list[AnomalyDetector],
    events: list[MarketEvent],
) -> str:
    direction   = "surged" if price.open_to_close_change() > 0 else "dropped"
    layer_names = ", ".join(d.name for d in triggered)
    sources     = ", ".join(e.title for e in events) if events else "no related news"
    return (
        f"Price {direction} {abs(price.open_to_close_change()):.2f}% on {price.date}. "
        f"Triggered by: {layer_names}. Related events: {sources}."
    )
