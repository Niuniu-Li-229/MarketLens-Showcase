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
from models import PricePoint, MarketEvent, AnomalyPoint


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
    """TODO: Flag days where |z-score of daily return| > threshold (default 2.0)."""

    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold

    @property
    def name(self) -> str:
        return f"ZScore(>{self.z_threshold}σ)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        # import numpy as np
        # returns = [p.open_to_close_change() for p in all_prices]
        # mean, std = np.mean(returns), np.std(returns)
        # if std == 0: return False
        # return abs((price.open_to_close_change() - mean) / std) > self.z_threshold
        raise NotImplementedError


class BollingerDetector(AnomalyDetector):
    """TODO: Flag days where close breaks outside Bollinger Bands (window=20, k=2)."""

    def __init__(self, window: int = 20, k: float = 2.0):
        self.window = window
        self.k = k

    @property
    def name(self) -> str:
        return f"Bollinger(w={self.window}, k={self.k})"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        raise NotImplementedError


class IQRDetector(AnomalyDetector):
    """TODO: Flag days where daily return falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""

    @property
    def name(self) -> str:
        return "IQR(1.5)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        raise NotImplementedError


class VolumeDetector(AnomalyDetector):
    """TODO: Flag days where volume > (multiplier × rolling average volume)."""

    def __init__(self, window: int = 20, multiplier: float = 2.0):
        self.window = window
        self.multiplier = multiplier

    @property
    def name(self) -> str:
        return f"Volume(>{self.multiplier}x avg)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        raise NotImplementedError


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
        window_days: int = 2,
    ) -> list[AnomalyPoint]:
        anomalies = []
        for price in prices:
            triggered = [d for d in self.detectors if d.is_anomaly(price, prices)]
            if len(triggered) >= self.min_triggers:
                nearby  = _find_nearby_events(price.date, events, window_days)
                comment = _build_comment(price, triggered, nearby)
                anomalies.append(AnomalyPoint(
                    price_point=price,
                    percent_change=price.open_to_close_change(),
                    related_events=nearby,
                    comment=comment,
                ))
        return anomalies


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_nearby_events(
    anomaly_date: date,
    events: list[MarketEvent],
    window_days: int,
) -> list[MarketEvent]:
    return [e for e in events if abs((e.date - anomaly_date).days) <= window_days]


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
