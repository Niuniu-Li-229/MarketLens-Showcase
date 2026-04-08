"""
Module 2 — Anomaly Detector
Owner: Person 2

Design: Each detection algorithm is its own class implementing AnomalyDetector.
FunnelDetector uses a two-tier strategy:

  Tier 1 — Price layer (fast path):
    If |close-to-close %| >= price_threshold (default 5%), flag immediately.

  Tier 2 — Funnel (slow path):
    If Tier 1 does not trigger, run all detectors. Flag if >= min_triggers agree.

To add a new algorithm: add a new subclass. Never modify existing ones.

        AnomalyDetector (abstract)
        ├── ThresholdDetector          ← simple mock
        ├── ZScoreDetector             ← close-to-close return z-score
        ├── BollingerDetector          ← close outside Bollinger Bands
        ├── VolumeDetector             ← volume spike
        ├── RSIDetector                ← RSI overbought / oversold
        ├── MACDDetector               ← MACD / signal-line crossover
        ├── GapDetector                ← opening gap from prev close
        ├── IntradayRangeDetector      ← intraday high-low range spike
        ├── ConsecutiveMoveDetector    ← n consecutive same-direction moves
        └── FunnelDetector             ← composes the above, main entry point
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
        idx = next((i for i, p in enumerate(all_prices) if p.date == price.date), None)
        if idx is None or idx == 0:
            return False
        returns = [
            all_prices[i].close_to_close_change(all_prices[i - 1].close)
            for i in range(1, len(all_prices))
        ]
        today_return = price.close_to_close_change(all_prices[idx - 1].close)
        mean, std = np.mean(returns), np.std(returns)
        if std == 0:
            return False
        return abs((today_return - mean) / std) > self.z_threshold


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


class RSIDetector(AnomalyDetector):
    """
    Flag days where RSI is overbought (> overbought) or oversold (< oversold).
    Uses standard Wilder smoothing over `period` close-to-close moves.
    Default thresholds: RSI > 70 or RSI < 30.
    """

    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        self.period     = period
        self.overbought = overbought
        self.oversold   = oversold

    @property
    def name(self) -> str:
        return f"RSI(p={self.period}, ob={self.overbought}, os={self.oversold})"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        idx = next((i for i, p in enumerate(all_prices) if p.date == price.date), None)
        if idx is None or idx < self.period:
            return False
        closes = [p.close for p in all_prices[idx - self.period: idx + 1]]
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains  = sum(d for d in deltas if d > 0)
        losses = sum(abs(d) for d in deltas if d < 0)
        avg_gain = gains  / self.period
        avg_loss = losses / self.period
        if avg_loss == 0:
            return avg_gain > 0   # RSI = 100, overbought
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi > self.overbought or rsi < self.oversold



class MACDDetector(AnomalyDetector):
    """
    Flag days where the MACD line crosses the signal line (momentum reversal).
    MACD  = EMA(fast) - EMA(slow)
    Signal = EMA(signal_period) of MACD
    A crossover (sign change in MACD - Signal) indicates a trend shift.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})"

    @staticmethod
    def _ema(values: list[float], period: int) -> list[float]:
        k   = 2.0 / (period + 1)
        ema = [values[0]]
        for v in values[1:]:
            ema.append(v * k + ema[-1] * (1 - k))
        return ema

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        idx = next((i for i, p in enumerate(all_prices) if p.date == price.date), None)
        # Need slow + signal + 1 days minimum to compute two consecutive diff values.
        if idx is None or idx < self.slow + self.signal:
            return False
        closes    = [p.close for p in all_prices[: idx + 1]]
        ema_fast  = self._ema(closes, self.fast)
        ema_slow  = self._ema(closes, self.slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        # Align signal computation to where slow EMA has stabilised.
        sig_line  = self._ema(macd_line[self.slow - 1:], self.signal)
        macd_trim = macd_line[self.slow - 1:]
        if len(sig_line) < 2:
            return False
        diff_now  = macd_trim[-1] - sig_line[-1]
        diff_prev = macd_trim[-2] - sig_line[-2]
        # Crossover = sign changed between yesterday and today.
        return (diff_now > 0) != (diff_prev > 0)


class GapDetector(AnomalyDetector):
    """
    Flag days where the opening gap from the previous close exceeds a threshold.
    Gap = |today.open - yesterday.close| / yesterday.close × 100.
    Captures pre-market news effects (earnings releases, overnight announcements).
    Default threshold: 2.0%.
    """

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"Gap(>{self.threshold}%)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        idx = next((i for i, p in enumerate(all_prices) if p.date == price.date), None)
        if idx is None or idx == 0:
            return False
        prev_close = all_prices[idx - 1].close
        gap_pct = abs(price.open - prev_close) / prev_close * 100.0
        return gap_pct >= self.threshold


class IntradayRangeDetector(AnomalyDetector):
    """
    Flag days where intraday range (high - low) / close is unusually large
    compared to the rolling average. Captures days with extreme intraday
    volatility that may not show up in open-to-close or close-to-close returns.
    Default: today's range > 2× rolling 20-day average.
    """

    def __init__(self, window: int = 20, multiplier: float = 2.0):
        self.window     = window
        self.multiplier = multiplier

    @property
    def name(self) -> str:
        return f"IntradayRange(w={self.window}, >{self.multiplier}x)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        idx = next((i for i, p in enumerate(all_prices) if p.date == price.date), None)
        if idx is None or idx < self.window:
            return False
        avg_range = sum(
            (p.high - p.low) / p.close
            for p in all_prices[idx - self.window: idx]
        ) / self.window
        if avg_range == 0:
            return False
        today_range = (price.high - price.low) / price.close
        return today_range > self.multiplier * avg_range


class ConsecutiveMoveDetector(AnomalyDetector):
    """
    Flag the last day of a run where price has moved in the same direction
    for n consecutive days, each move >= min_pct.
    Detects sustained trend anomalies that single-day detectors miss.
    Default: 3 consecutive days each moving >= 1.0% in the same direction.
    """

    def __init__(self, n: int = 3, min_pct: float = 1.0):
        self.n       = n
        self.min_pct = min_pct

    @property
    def name(self) -> str:
        return f"ConsecutiveMove(n={self.n}, >{self.min_pct}%/day)"

    def is_anomaly(self, price: PricePoint, all_prices: list[PricePoint]) -> bool:
        idx = next((i for i, p in enumerate(all_prices) if p.date == price.date), None)
        if idx is None or idx < self.n:
            return False
        changes = [
            all_prices[i].close_to_close_change(all_prices[i - 1].close)
            for i in range(idx - self.n + 1, idx + 1)
        ]
        if not all(abs(c) >= self.min_pct for c in changes):
            return False
        return all(c > 0 for c in changes) or all(c < 0 for c in changes)


class FunnelDetector:
    """
    Composes multiple AnomalyDetectors using a two-tier strategy.
    This is the main entry point for Module 2.

    Tier 1 — Price layer (fast path):
        If |close-to-close %| >= price_threshold, flag immediately.

    Tier 2 — Funnel (slow path):
        If Tier 1 does not trigger, run all detectors. Flag only if
        >= min_triggers detectors agree (default: 2).

    Usage (mock):
        detector = FunnelDetector([ThresholdDetector()], min_triggers=1)

    Usage (full 7-layer funnel):
        detector = FunnelDetector([
            ZScoreDetector(), BollingerDetector(), IQRDetector(),
            VolumeDetector(), RSIDetector(), ATRDetector(), MACDDetector(),
        ], min_triggers=2)
    """

    def __init__(self, detectors: list[AnomalyDetector], min_triggers: int = 2):
        self.detectors    = detectors
        self.min_triggers = min_triggers

    def detect(
        self,
        prices: list[PricePoint],
        events: list[MarketEvent],
        ticker: str | None = None,
        pre_days: int = 3,
        post_days: int = 1,
        price_threshold: float = 5.0,
    ) -> list[AnomalyPoint]:
        """
        Two-tier anomaly detection:

        Tier 1 — Price layer (fast path):
            If |close-to-close %| >= price_threshold, flag immediately.
            No funnel required — raw price movement is sufficient evidence.

        Tier 2 — Funnel (slow path):
            If the price move is not large enough on its own, fall through to
            the funnel. Flag only if >= self.min_triggers detectors agree.
        """
        prices = sorted(prices, key=lambda p: p.date)
        anomalies = []
        for i, price in enumerate(prices):
            if i == 0:
                continue  # need a previous close to compute close-to-close change
            pct_change = price.close_to_close_change(prices[i - 1].close)

            # ── Tier 1: direct price-level anomaly ────────────────────────────
            if abs(pct_change) >= price_threshold:
                nearby  = _find_nearby_events(price.date, events, pre_days, post_days, ticker)
                comment = _build_comment(price, pct_change, [], nearby)
                anomalies.append(AnomalyPoint(
                    price_point=price,
                    percent_change=pct_change,
                    related_events=nearby,
                    comment=comment,
                ))
                continue

            # ── Tier 2: funnel fallback (min_triggers detectors must agree) ───
            triggered = [d for d in self.detectors if d.is_anomaly(price, prices)]
            if len(triggered) >= self.min_triggers:
                nearby  = _find_nearby_events(price.date, events, pre_days, post_days, ticker)
                comment = _build_comment(price, pct_change, triggered, nearby)
                anomalies.append(AnomalyPoint(
                    price_point=price,
                    percent_change=pct_change,
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
    pct_change: float,
    triggered: list[AnomalyDetector],
    events: list[MarketEvent],
) -> str:
    direction   = "surged" if pct_change > 0 else "dropped"
    layer_names = "PriceLayer" if not triggered else ", ".join(d.name for d in triggered)
    sources     = ", ".join(e.title for e in events) if events else "no related news"
    return (
        f"Price {direction} {abs(pct_change):.2f}% on {price.date} (close-to-close). "
        f"Triggered by: {layer_names}. Related events: {sources}."
    )
