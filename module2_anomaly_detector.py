"""
Module 2 — Anomaly Detector
4-layer funnel: ZScore + Bollinger + IQR + Volume.
Works for any ticker.
"""

from abc import ABC, abstractmethod
from datetime import date
import numpy as np
from models import PricePoint, MarketEvent, AnomalyPoint


class AnomalyDetector(ABC):
    @abstractmethod
    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool: ...
    @property
    @abstractmethod
    def name(self) -> str: ...


class ThresholdDetector(AnomalyDetector):
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
    @property
    def name(self): return f"Threshold(>{self.threshold}%)"
    def is_anomaly(self, price, all_prices):
        return abs(price.open_to_close_change()) >= self.threshold


class ZScoreDetector(AnomalyDetector):
    def __init__(self, window: int = 20, z_threshold: float = 2.0):
        self.window = window
        self.z_threshold = z_threshold
    @property
    def name(self): return f"ZScore(w={self.window},>{self.z_threshold}σ)"
    def is_anomaly(self, price, all_prices):
        idx = _idx(price, all_prices)
        if idx is None or idx < self.window: return False
        returns = np.array([p.open_to_close_change()
                            for p in all_prices[idx-self.window:idx]])
        mean, std = returns.mean(), returns.std()
        if std == 0: return False
        return bool(abs((price.open_to_close_change() - mean) / std) > self.z_threshold)


class BollingerDetector(AnomalyDetector):
    def __init__(self, window: int = 20, k: float = 2.0):
        self.window = window; self.k = k
    @property
    def name(self): return f"Bollinger(w={self.window},k={self.k})"
    def is_anomaly(self, price, all_prices):
        idx = _idx(price, all_prices)
        if idx is None or idx < self.window: return False
        closes = np.array([p.close for p in all_prices[idx-self.window:idx]])
        mean, std = closes.mean(), closes.std()
        return bool(price.close > mean + self.k*std or
                    price.close < mean - self.k*std)


class IQRDetector(AnomalyDetector):
    def __init__(self, window: int = 20, multiplier: float = 1.5):
        self.window = window; self.multiplier = multiplier
    @property
    def name(self): return f"IQR(w={self.window},x{self.multiplier})"
    def is_anomaly(self, price, all_prices):
        idx = _idx(price, all_prices)
        if idx is None or idx < self.window: return False
        returns = np.array([p.open_to_close_change()
                            for p in all_prices[idx-self.window:idx]])
        q1, q3 = np.percentile(returns, 25), np.percentile(returns, 75)
        iqr = q3 - q1
        ret = price.open_to_close_change()
        return bool(ret < q1 - self.multiplier*iqr or
                    ret > q3 + self.multiplier*iqr)


class VolumeDetector(AnomalyDetector):
    def __init__(self, window: int = 20, multiplier: float = 2.0):
        self.window = window; self.multiplier = multiplier
    @property
    def name(self): return f"Volume(w={self.window},>{self.multiplier}x)"
    def is_anomaly(self, price, all_prices):
        idx = _idx(price, all_prices)
        if idx is None or idx < self.window: return False
        avg = np.mean([p.volume for p in all_prices[idx-self.window:idx]])
        return bool(price.volume > self.multiplier * avg)


class FunnelDetector:
    """
    Flags a day as anomalous only if ≥ min_triggers detectors agree.

    Recommended:
        FunnelDetector([
            ZScoreDetector(), BollingerDetector(),
            IQRDetector(),    VolumeDetector(),
        ], min_triggers=2)
    """
    def __init__(self, detectors: list[AnomalyDetector], min_triggers: int = 2):
        self.detectors    = detectors
        self.min_triggers = min_triggers

    def detect(self, prices: list[PricePoint],
               events: list[MarketEvent],
               window_days: int = 2) -> list[AnomalyPoint]:
        anomalies = []
        for price in prices:
            triggered = [d for d in self.detectors
                         if d.is_anomaly(price, prices)]
            if len(triggered) >= self.min_triggers:
                nearby  = [e for e in events
                           if abs((e.date - price.date).days) <= window_days]
                comment = _build_comment(price, triggered, nearby)
                anomalies.append(AnomalyPoint(
                    price_point    = price,
                    percent_change = price.open_to_close_change(),
                    related_events = nearby,
                    comment        = comment,
                ))
        return anomalies


# ── helpers ───────────────────────────────────────────────────────────────────

def _idx(price: PricePoint, all_prices: list[PricePoint]) -> int | None:
    for i, p in enumerate(all_prices):
        if p.date == price.date:
            return i
    return None

def _build_comment(price, triggered, events) -> str:
    direction   = "surged" if price.open_to_close_change() > 0 else "dropped"
    layer_names = ", ".join(d.name for d in triggered)
    sources     = ", ".join(e.title for e in events) if events else "no related news"
    return (f"Price {direction} {abs(price.open_to_close_change()):.2f}% "
            f"on {price.date}. Triggered by: {layer_names}. "
            f"Related events: {sources}.")