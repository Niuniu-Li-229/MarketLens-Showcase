"""
models.py — Shared data contract for the stock analysis pipeline.
All modules import from this file. Do not modify field names without
notifying the full team.

Changes from v1:
  - Added __post_init__ validation (mirrors Java IllegalArgumentException guards)
  - Fixed EarningsEvent event_type override — now uses field(init=False)
  - Added __str__ helpers for clean logging
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class EventType(Enum):
    EARNINGS   = "EARNINGS"
    ANALYST    = "ANALYST"
    REGULATORY = "REGULATORY"
    MACRO      = "MACRO"
    PRODUCT    = "PRODUCT"
    OTHER      = "OTHER"


@dataclass
class PricePoint:
    date:   date
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError("High must be >= low.")
        if any(v <= 0 for v in [self.open, self.high, self.low, self.close]):
            raise ValueError("OHLC prices must be positive.")
        if self.volume < 0:
            raise ValueError("Volume must be non-negative.")

    def daily_range(self) -> float:
        return self.high - self.low

    def open_to_close_change(self) -> float:
        return (self.close - self.open) / self.open * 100.0

    def __str__(self):
        return (f"[Price] {self.date}  close={self.close:.2f}"
                f"  range={self.daily_range():.2f}  vol={self.volume:,}")


@dataclass
class MarketEvent:
    date:       date
    title:      str
    description: str
    source:     str
    event_type: EventType

    def __post_init__(self):
        if not self.title or not self.title.strip():
            raise ValueError("Title must not be blank.")
        if not self.source or not self.source.strip():
            raise ValueError("Source must not be blank.")

    def __str__(self):
        return f"[{self.event_type.value}] {self.title} — {self.source} ({self.date})"


@dataclass
class EarningsEvent(MarketEvent):
    """Specialized MarketEvent carrying earnings-specific data."""
    reported_eps:      float = 0.0
    beat_expectations: bool  = False
    # event_type is fixed — not settable by caller
    event_type: EventType = field(init=False, default=EventType.EARNINGS)

    def __str__(self):
        beat = "BEAT" if self.beat_expectations else "MISSED"
        return f"[EARNINGS] {self.title} — EPS: {self.reported_eps:.2f} ({beat}) ({self.date})"


@dataclass
class AnomalyPoint:
    price_point:    PricePoint
    percent_change: float
    related_events: list[MarketEvent] = field(default_factory=list)
    comment:        str = ""

    def is_gain(self) -> bool:
        return self.percent_change > 0

    @property
    def date(self) -> date:
        return self.price_point.date

    def __str__(self):
        return (f"AnomalyPoint[date={self.date}"
                f"  change={self.percent_change:+.2f}%"
                f"  events={len(self.related_events)}]")


@dataclass
class AnalysisResult:
    """
    Complete result of analyzing a stock over a date range.
    This is the object passed into Module 4 (Claude prompt builder).
    """
    ticker:       str
    start_date:   date
    end_date:     date
    total_return: float
    anomalies:    list[AnomalyPoint] = field(default_factory=list)
    summary:      str = ""

    # Populated by Module 3 — optional until LSTM/sentiment is built
    predicted_price: Optional[float] = None
    sentiment_score: Optional[float] = None  # -1.0 (bearish) to +1.0 (bullish)
    sentiment_label: Optional[str]   = None  # "bullish" | "bearish" | "neutral"

    def __post_init__(self):
        if not self.ticker or not self.ticker.strip():
            raise ValueError("Ticker must not be blank.")
        if self.end_date < self.start_date:
            raise ValueError("End date must not be before start date.")
        self.ticker = self.ticker.upper()

    def anomaly_count(self) -> int:
        return len(self.anomalies)

    def __str__(self):
        return (f"AnalysisResult[ticker={self.ticker}"
                f"  period={self.start_date}→{self.end_date}"
                f"  return={self.total_return:+.2f}%"
                f"  anomalies={self.anomaly_count()}]")
