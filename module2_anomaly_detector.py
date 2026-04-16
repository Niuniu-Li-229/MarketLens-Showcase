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
                nearby  = _find_nearby_events(price.date, events, pre_days, post_days,
                                              ticker, pct_change=pct_change)
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
                nearby  = _find_nearby_events(price.date, events, pre_days, post_days,
                                              ticker, pct_change=pct_change)
                comment = _build_comment(price, pct_change, triggered, nearby)
                anomalies.append(AnomalyPoint(
                    price_point=price,
                    percent_change=pct_change,
                    related_events=nearby,
                    comment=comment,
                ))
        return anomalies


# ══════════════════════════════════════════════════════════════════════════════
# News–Anomaly Matching Pipeline
#
# Step 0 — Time window: keep events within [anomaly - pre_days, anomaly + post_days]
# Stage 1 — Market Layer: classify each candidate as Firm / Sector / Market / Noise
# Stage 2 — (reserved) Embedding similarity for events without relevance scores
# Stage 3 — Composite score: weighted sum of relevance, importance, direction, recency
#           Apply minimum score threshold, return top_n sorted by score descending
# ══════════════════════════════════════════════════════════════════════════════


# ── Lookup tables ─────────────────────────────────────────────────────────────

# Ticker → keywords. If ANY keyword appears in title+description → firm layer.
_TICKER_KEYWORDS: dict[str, list[str]] = {
    "NVDA":  ["nvidia", "nvda", "jensen huang", "h100", "h20", "blackwell",
              "geforce", "cuda", "tensorrt"],
    "AAPL":  ["apple", "aapl", "tim cook", "iphone", "ipad", "ios", "mac",
              "app store", "apple intelligence", "vision pro", "airpods"],
    "AMZN":  ["amazon", "amzn", "aws", "andy jassy",
              "prime video", "alexa", "kindle"],
    "GOOGL": ["google", "googl", "alphabet", "sundar pichai", "gemini", "youtube",
              "waymo", "deepmind", "android", "chrome", "pixel"],
    "META":  ["meta", "facebook", "instagram", "whatsapp", "zuckerberg", "llama",
              "threads", "ray-ban", "reels", "oculus", "quest 3", "quest 2",
              "reality labs", "meta ai", "meta platforms", "messenger",
              "horizon worlds"],
    "TSLA":  ["tesla", "tsla", "elon musk", "cybertruck", "optimus",
              "supercharger", "autopilot", "full self-driving", "fsd"],
}

# Sector-level keywords: tech industry terms that are not about a specific firm.
_SECTOR_KEYWORDS: list[str] = [
    # tech industry grouping
    "big tech", "magnificent seven", "magnificent 7", "mag 7", "faang",
    "mega cap", "mega-cap",
    "tech stocks", "tech sector", "tech giant", "tech earnings",
    "tech rally", "tech rout", "tech selloff", "tech layoffs",
    "silicon valley",
    # social media / advertising
    "social media", "social network",
    "digital advertising", "online advertising", "ad market",
    "ad spend", "ad spending", "digital ad", "ad tech",
    "content moderation",
    # AI / compute / hardware
    "ai stocks", "ai trade", "ai boom", "ai bubble",
    "semiconductor", "chip ", "gpu ",
    "data center", "cloud computing",
    "open source", "open-source",
    "vr headset", "mixed reality", "ar glasses",
    # regulation
    "big tech regulation", "privacy",
]

# Market-level keywords: broad macro/index events that affect ALL stocks.
_MARKET_KEYWORDS: list[str] = [
    "nasdaq", "s&p 500", "s&p500", "dow jones",
    "wall street", "stock market",
    "bull market", "bear market",
    "market rally", "market selloff", "market crash",
    "risk-on", "risk-off", "risk on", "risk off",
]

# EventTypes that imply sector-level relevance even without keyword matches.
# If Module 1 already classified an event as one of these types, we don't need
# to re-check sector keywords — the EventType itself is the signal.
_SECTOR_EVENTTYPES: set[EventType] = {
    EventType.EARNINGS, EventType.LEGAL, EventType.REGULATORY,
    EventType.PRODUCT, EventType.AI_TECH, EventType.ANALYST,
    EventType.PERSONNEL,
}

# EventType → importance weight (0.0 – 1.0).
# Reflects how strongly this type of event typically drives price anomalies.
_EVENT_IMPORTANCE: dict[EventType, float] = {
    EventType.EARNINGS:   1.00,
    EventType.LEGAL:      0.85,
    EventType.REGULATORY: 0.80,
    EventType.PERSONNEL:  0.75,
    EventType.PRODUCT:    0.70,
    EventType.AI_TECH:    0.65,
    EventType.ANALYST:    0.60,
    EventType.MACRO:      0.50,
    EventType.OTHER:      0.15,
}

# Alpha Vantage sentiment scores range ≈ -0.35 to +0.35.
# Divide by this to normalize to [-1, +1] before computing direction match.
_AV_SENTIMENT_SCALE = 0.35

# Minimum composite score to keep an event. Calibrated via forensic analysis:
# true mismatches scored 0.39–0.52, correct matches scored 0.55+.
MIN_SCORE_THRESHOLD = 0.45


# ── Stage 1: Market Layer ─────────────────────────────────────────────────────

def _has_sector_mention(text: str) -> bool:
    """Check if text matches any sector keyword."""
    return any(k in text for k in _SECTOR_KEYWORDS)


def _classify_market_layer(
    event: MarketEvent,
    ticker: str | None,
) -> tuple[str, float]:
    """
    Classify an event into one of four market layers.

    Three keyword lists, each serving a distinct role:
      _TICKER_KEYWORDS         → firm  (1.0)  "Is this about the company?"
      _TICKER_SECTOR_KEYWORDS  → sector (0.6)  "Is this about the company's industry?"
      + _COMMON_SECTOR_KEYWORDS
      _MARKET_KEYWORDS         → market (0.3)  "Is this about the broad market?"

    Plus: EventType from Module 1 is reused as a signal (no duplicate keywords).
    If Module 1 already classified an event as EARNINGS/LEGAL/etc., that implies
    sector-level relevance — no need to re-match keywords.

    Decision chain (short-circuit, first match wins):

      Priority  Condition                                 → Layer    Weight
      ────────  ────────────────────────────────────────  ────────  ──────
      1         source = yfinance                         → firm     1.0
      2         ticker keyword hit in text                → firm     1.0
      3         AV rel ≥ 0.7, no ticker kw, +sector kw   → sector   0.6
      4         AV rel ≥ 0.7, no ticker kw, -sector kw   → sector   0.5
      5         AV rel 0.3–0.7, +sector keyword          → sector   0.6
      6         AV rel 0.3–0.7, -sector keyword          → sector   0.4
      7         AV rel < 0.3                              → noise    0.0
      8         no AV, +sector keyword                   → sector   0.6
      9         no AV, +market keyword                   → market   0.3
      10        no AV, EventType in _SECTOR_EVENTTYPES   → sector   0.6
      11        no AV, EventType = MACRO                 → market   0.3
      12        no AV, EventType = OTHER                 → noise    0.0

    Returns: (layer_name, weight)
    """
    text = (event.title + " " + event.description).lower()

    # ① Ticker keyword match — ground truth, highest priority
    has_ticker_mention = False
    if ticker:
        kws = _TICKER_KEYWORDS.get(ticker.upper(), [])
        if kws and any(k in text for k in kws):
            has_ticker_mention = True

    # ② yfinance source → always firm
    if event.source == "yfinance":
        return "firm", 1.0

    # ③ Ticker keyword hit → firm
    if has_ticker_mention:
        return "firm", 1.0

    # ④ AV relevance cross-check (never blindly trust AV for firm)
    has_sector = _has_sector_mention(text)

    if event.relevance_score is not None:
        if event.relevance_score >= 0.7:
            # AV says high relevance but no ticker keyword.
            # Downgrade to sector — AV relevance can be wrong.
            return ("sector", 0.6) if has_sector else ("sector", 0.5)
        if event.relevance_score >= 0.3:
            return ("sector", 0.6) if has_sector else ("sector", 0.4)
        return "noise", 0.0  # AV says low relevance

    # ⑤ No AV data — fall back to keywords + EventType
    if has_sector:
        return "sector", 0.6
    if any(k in text for k in _MARKET_KEYWORDS):
        return "market", 0.3
    # Reuse Module 1's EventType: typed events imply sector relevance
    if event.event_type in _SECTOR_EVENTTYPES:
        return "sector", 0.6
    if event.event_type == EventType.MACRO:
        return "market", 0.3
    # OTHER with no keyword matches → noise
    return "noise", 0.0


# ── Stage 3: Composite Scoring ────────────────────────────────────────────────

def _recency_score(event_date: date, anomaly_date: date) -> float:
    """
    Time decay: same-day = 1.0, decays toward 0 at window edges.
    Pre-event (causal) decays slower than post-event (reaction).

      delta  direction   score
      ─────  ─────────   ─────
        0    same day    1.0
       +1    1d before   0.8
       +2    2d before   0.6
       +3    3d before   0.4
       -1    1d after    0.5
    """
    delta = (anomaly_date - event_date).days  # positive = event is before anomaly
    if delta == 0:
        return 1.0
    if delta > 0:
        return max(0.0, 1.0 - delta * 0.2)
    # Post-event: starts lower (0.5) and decays faster
    return max(0.0, 0.5 - abs(delta) * 0.3)


def _direction_match_score(sentiment: float | None, pct_change: float) -> float:
    """
    Measures how well the event's sentiment explains the anomaly direction.

      sentiment   anomaly    result
      ─────────   ───────    ──────
      bearish     drop       → 1.0  (strong causal agreement)
      bullish     surge      → 1.0
      None        any        → 0.5  (neutral, no data)
      bullish     drop       → 0.0  (contradiction)
      bearish     surge      → 0.0

    Alpha Vantage sentiment is ≈ -0.35 to +0.35. We normalize to [-1, +1]
    before computing agreement so the score uses the full 0–1 range.
    """
    if sentiment is None:
        return 0.5

    # Normalize AV scale (-0.35..+0.35) → (-1..+1)
    normalized = max(-1.0, min(1.0, sentiment / _AV_SENTIMENT_SCALE))
    anomaly_dir = 1.0 if pct_change > 0 else -1.0

    agreement = normalized * anomaly_dir  # positive when same direction
    return max(0.0, min(1.0, 0.5 + agreement * 0.5))


def _composite_score(
    event: MarketEvent,
    anomaly_date: date,
    pct_change: float,
    market_weight: float,
) -> float:
    """
    Final relevance score for one (event, anomaly) pair. Higher = more relevant.

    score = 0.40 × relevance        (market_weight from Stage 1)
          + 0.30 × event_importance  (fixed weight per EventType)
          + 0.15 × direction_match   (sentiment ↔ anomaly direction agreement)
          + 0.15 × recency           (time decay from event to anomaly)

    Weight rationale:
      - relevance (0.40): dominant signal — is this event about the right company?
      - importance (0.30): EARNINGS > LEGAL > PRODUCT > OTHER, hard prior
      - direction (0.15): sentiment agreement is a tiebreaker, not a primary signal.
        Events without sentiment get 0.5 (neutral), so the max swing is ±0.075,
        which cannot override a full EventType tier difference (0.30 × 0.15 = 0.045).
      - recency (0.15): same-day events beat 3-day-old events

    All four components are in [0, 1], so score ∈ [0, 1].
    """
    relevance  = market_weight
    importance = _EVENT_IMPORTANCE.get(event.event_type, 0.15)
    direction  = _direction_match_score(event.sentiment_score, pct_change)
    recency    = _recency_score(event.date, anomaly_date)

    return (0.40 * relevance
          + 0.30 * importance
          + 0.15 * direction
          + 0.15 * recency)


# ── Main matching function ────────────────────────────────────────────────────

def _find_nearby_events(
    anomaly_date: date,
    events: list[MarketEvent],
    pre_days: int,
    post_days: int,
    ticker: str | None = None,
    top_n: int = 10,
    pct_change: float = 0.0,
) -> list[MarketEvent]:
    """
    News–anomaly matching pipeline.

    Step 0 — Time window: [anomaly_date - pre_days, anomaly_date + post_days]
    Stage 1 — Market Layer: Firm / Sector / Market / Noise → drop noise
    Stage 2 — (reserved for embedding similarity)
    Stage 3 — Composite score → drop below MIN_SCORE_THRESHOLD → sort desc → top_n
    """
    # Step 0: time window
    nearby = [
        e for e in events
        if -pre_days <= (e.date - anomaly_date).days <= post_days
    ]

    # Stage 1: classify and drop noise
    candidates: list[tuple[MarketEvent, float]] = []
    for e in nearby:
        layer, weight = _classify_market_layer(e, ticker)
        if layer != "noise":
            candidates.append((e, weight))

    # Stage 3: score, threshold, sort
    scored = [
        (e, _composite_score(e, anomaly_date, pct_change, w))
        for e, w in candidates
    ]
    scored = [(e, s) for e, s in scored if s >= MIN_SCORE_THRESHOLD]
    scored.sort(key=lambda x: -x[1])

    return [e for e, _ in scored[:top_n]]


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
