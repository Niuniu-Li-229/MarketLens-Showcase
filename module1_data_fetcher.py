"""
Module 1 — Data Fetcher

        PriceFetcher (abstract)
        ├── MockPriceFetcher        ← testing / demo
        └── YFinancePriceFetcher    ← real: yfinance + CSV cache

        NewsFetcher (abstract)
        ├── MockNewsFetcher         ← testing / demo
        └── FinnhubNewsFetcher      ← real: Finnhub API + CSV cache
"""

import csv
import os
import time
import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path

from models import PricePoint, MarketEvent, EarningsEvent, EventType

logger   = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).parent / "data_cache"


# ── CSV cache ─────────────────────────────────────────────────────────────────

class DataCache:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    # prices ──────────────────────────────────────────────────────────────────

    def _prices_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}_prices.csv"

    def load_prices(self, ticker: str) -> list[PricePoint]:
        path = self._prices_path(ticker)
        if not path.exists():
            return []
        points: list[PricePoint] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    points.append(PricePoint(
                        date   = date.fromisoformat(row["date"]),
                        open   = float(row["open"]),
                        high   = float(row["high"]),
                        low    = float(row["low"]),
                        close  = float(row["close"]),
                        volume = int(row["volume"]),
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping cached price row: %s", e)
        return points

    def save_prices(self, ticker: str, new_points: list[PricePoint]) -> None:
        existing = {p.date: p for p in self.load_prices(ticker)}
        for p in new_points:
            existing[p.date] = p
        merged = sorted(existing.values(), key=lambda p: p.date)
        with open(self._prices_path(ticker), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])
            for p in merged:
                writer.writerow([p.date, p.open, p.high, p.low, p.close, p.volume])
        logger.info("Cached %d price rows for %s", len(merged), ticker)

    # news ────────────────────────────────────────────────────────────────────

    def _news_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}_news.csv"

    def load_news(self, ticker: str) -> list[MarketEvent]:
        path = self._news_path(ticker)
        if not path.exists():
            return []
        events: list[MarketEvent] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    event_type = EventType(row["event_type"])
                    if event_type == EventType.EARNINGS and "reported_eps" in row:
                        events.append(EarningsEvent(
                            date             = date.fromisoformat(row["date"]),
                            title            = row["title"],
                            description      = row["description"],
                            source           = row["source"],
                            reported_eps     = float(row["reported_eps"] or 0.0),
                            beat_expectations= row["beat_expectations"] == "True",
                        ))
                    else:
                        events.append(MarketEvent(
                            date       = date.fromisoformat(row["date"]),
                            title      = row["title"],
                            description= row["description"],
                            source     = row["source"],
                            event_type = event_type,
                        ))
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping cached news row: %s", e)
        return events

    def save_news(self, ticker: str, new_events: list[MarketEvent]) -> None:
        existing = {(e.date, e.title): e for e in self.load_news(ticker)}
        for e in new_events:
            existing[(e.date, e.title)] = e
        merged = sorted(existing.values(), key=lambda e: e.date)
        with open(self._news_path(ticker), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "title", "description", "source",
                             "event_type", "reported_eps", "beat_expectations"])
            for e in merged:
                if isinstance(e, EarningsEvent):
                    writer.writerow([e.date, e.title, e.description, e.source,
                                     e.event_type.value, e.reported_eps, e.beat_expectations])
                else:
                    writer.writerow([e.date, e.title, e.description, e.source,
                                     e.event_type.value, "", ""])
        logger.info("Cached %d news rows for %s", len(merged), ticker)


# ── Abstract interfaces ───────────────────────────────────────────────────────

class PriceFetcher(ABC):
    @abstractmethod
    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]: ...

class NewsFetcher(ABC):
    @abstractmethod
    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]: ...


# ── Mock implementations ──────────────────────────────────────────────────────

class MockPriceFetcher(PriceFetcher):
    """
    Returns cached CSV data if available, otherwise falls back to
    hardcoded NVDA sample prices (4 days in Sept 2025).
    Compatible with module2's close_to_close_change (needs ≥2 rows).
    """
    _FALLBACK = [
        (date(2025, 9, 2), 170.00, 172.38, 167.22, 170.78, 231_160_000),
        (date(2025, 9, 3), 171.06, 172.41, 168.88, 170.62, 164_420_000),
        (date(2025, 9, 4), 170.57, 171.84, 169.41, 171.66, 141_670_000),
        (date(2025, 9, 5), 168.03, 169.03, 164.07, 167.02, 224_440_000),
    ]

    def __init__(self):
        self._cache = DataCache()

    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]:
        cached = self._cache.load_prices(ticker)
        if cached:
            return [p for p in cached if start <= p.date <= end]
        return [PricePoint(*row) for row in self._FALLBACK if start <= row[0] <= end]


class MockNewsFetcher(NewsFetcher):
    """
    Returns cached CSV data if available, otherwise falls back to
    hardcoded NVDA sample events.
    """
    _FALLBACK = [
        (date(2025, 9, 2),
         "Analysts raise NVIDIA price targets ahead of earnings",
         "Wall Street analysts raised targets citing strong data center demand.",
         "Bloomberg", EventType.ANALYST),
        (date(2025, 9, 3),
         "NVIDIA reports record Q3 earnings, beats estimates",
         "Record revenue driven by surging demand for H100 and B100 AI chips.",
         "Reuters", EventType.EARNINGS),
        (date(2025, 9, 5),
         "Broad market sell-off on Fed rate concerns",
         "Investors rotated out of tech stocks amid rates fears.",
         "WSJ", EventType.MACRO),
    ]

    def __init__(self):
        self._cache = DataCache()

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        cached = self._cache.load_news(ticker)
        if cached:
            return [e for e in cached if start <= e.date <= end]
        return [MarketEvent(*row) for row in self._FALLBACK]


# ── Real: yfinance price fetcher ──────────────────────────────────────────────

class YFinancePriceFetcher(PriceFetcher):
    """
    Fetches historical OHLCV prices via yfinance with automatic CSV caching.
    Handles yfinance v0.2+ MultiIndex columns automatically.
    """

    def __init__(self):
        self._cache = DataCache()

    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]:
        import yfinance as yf
        import pandas as pd

        df = yf.download(
            ticker,
            start       = start,
            end         = end + timedelta(days=1),  # yfinance end is exclusive
            auto_adjust = True,
            progress    = False,
        )

        if df.empty:
            logger.warning("yfinance returned no data for %s (%s~%s)", ticker, start, end)
            return []

        # Flatten MultiIndex columns (yfinance v0.2+)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        points: list[PricePoint] = []
        for row in df.itertuples():
            try:
                points.append(PricePoint(
                    date   = row.Index.date(),
                    open   = round(float(row.Open),   2),
                    high   = round(float(row.High),   2),
                    low    = round(float(row.Low),    2),
                    close  = round(float(row.Close),  2),
                    volume = int(row.Volume),
                ))
            except (ValueError, TypeError) as e:
                logger.warning("Skipping dirty row %s: %s", row.Index, e)

        if points:
            self._cache.save_prices(ticker, points)

        logger.info("Fetched %d price points for %s", len(points), ticker)
        return points


# ── Event-type classifier (shared by FinnhubNewsFetcher) ─────────────────────

_EVENT_KEYWORDS: dict[EventType, list[str]] = {
    EventType.EARNINGS: [
        "earnings", "revenue", "quarterly results", "profit", "income",
        "beat estimates", "miss estimates", "beat expectations", "guidance",
        "gross margin", "net income", "quarterly", "fiscal year",
        "q1 ", "q2 ", "q3 ", "q4 ",
    ],
    EventType.ANALYST: [
        "analyst", "upgrade", "downgrade", "price target", "rating",
        "overweight", "underweight", "outperform", "underperform",
        "initiates coverage", "raises target", "cuts target",
    ],
    EventType.REGULATORY: [
        "sec", "regulation", "fda", "antitrust", "lawsuit", "compliance",
        "export control", "export ban", "chip ban", "sanction", "chips act",
    ],
    EventType.MACRO: [
        "fed", "interest rate", "inflation", "recession", "gdp",
        "unemployment", "tariff", "trade war", "rate hike", "rate cut",
        "treasury yield", "sell-off",
    ],
    EventType.PRODUCT: [
        "launch", "product", "release", "unveil", "announce", "partnership",
        "acquisition", "acquires", "merger", "funding round",
    ],
    EventType.PERSONNEL: [
        "ceo", "cfo", "coo", "cto", "appointed", "resigns", "resignation",
        "steps down", "named as", "new chief", "new president",
    ],
}

_NOISE_TITLE_PATTERNS = [
    "stock market today:",
    "these stocks moved the most",
    "stocks moving the most today",
    "most active stocks",
    "dow jones futures:",
]


def _classify_event(headline: str, category: str) -> EventType:
    text = f"{headline} {category}".lower()
    for event_type, keywords in _EVENT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return event_type
    return EventType.OTHER


# ── Real: Finnhub news fetcher ────────────────────────────────────────────────

class FinnhubNewsFetcher(NewsFetcher):
    """
    Fetches company news via Finnhub API with adaptive weekly windowing
    and automatic CSV caching.

    Note: Finnhub free tier only supports ~1 year of historical news.
    For historical analysis beyond 1 year, use MockNewsFetcher or
    add a KnownEventsFetcher instead.
    """

    INITIAL_WINDOW_DAYS = 7
    FINNHUB_MAX_PER_REQ = 250
    API_SLEEP_SEC       = 1.1   # free tier: 60 req/min

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Finnhub API key required. "
                "Set FINNHUB_API_KEY in your .env file."
            )
        self._cache = DataCache()

    def _fetch_recursive(self, client, ticker: str,
                          w_start: date, w_end: date) -> list[dict]:
        import finnhub
        time.sleep(self.API_SLEEP_SEC)
        try:
            raw = client.company_news(ticker, _from=str(w_start), to=str(w_end))
        except finnhub.FinnhubAPIException as e:
            logger.error("Finnhub API error %s (%s~%s): %s", ticker, w_start, w_end, e)
            return []
        except Exception as e:
            logger.error("Finnhub fetch error %s (%s~%s): %s", ticker, w_start, w_end, e)
            return []

        if len(raw) < self.FINNHUB_MAX_PER_REQ:
            return raw

        if w_start == w_end:
            logger.warning("Cap hit on single day %s for %s", w_start, ticker)
            return raw

        mid   = w_start + (w_end - w_start) // 2
        left  = self._fetch_recursive(client, ticker, w_start, mid)
        right = self._fetch_recursive(client, ticker, mid + timedelta(days=1), w_end)
        return left + right

    @staticmethod
    def _parse_article(article: dict, fallback_date: date) -> MarketEvent | None:
        headline    = (article.get("headline") or "").strip()
        source      = (article.get("source")   or "").strip()
        description = (article.get("summary")  or "").strip()
        if not headline or not source or not description:
            return None
        if any(p in headline.lower() for p in _NOISE_TITLE_PATTERNS):
            return None
        ts           = article.get("datetime", 0)
        article_date = date.fromtimestamp(ts) if ts else fallback_date
        event_type   = _classify_event(headline, article.get("category", ""))
        return MarketEvent(
            date        = article_date,
            title       = headline,
            description = description[:500],
            source      = source,
            event_type  = event_type,
        )

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        import finnhub
        client = finnhub.Client(api_key=self.api_key)

        windows: list[tuple[date, date]] = []
        w_start = start
        while w_start <= end:
            w_end = min(w_start + timedelta(days=self.INITIAL_WINDOW_DAYS - 1), end)
            windows.append((w_start, w_end))
            w_start = w_end + timedelta(days=1)

        logger.info("Fetching news for %s (%s~%s) in %d windows",
                    ticker, start, end, len(windows))

        raw_all: list[dict] = []
        for ws, we in windows:
            raw_all.extend(self._fetch_recursive(client, ticker, ws, we))

        seen: set[tuple[date, str]] = set()
        events: list[MarketEvent]   = []
        for article in raw_all:
            ev = self._parse_article(article, fallback_date=start)
            if ev is None:
                continue
            key = (ev.date, ev.title)
            if key in seen:
                continue
            seen.add(key)
            events.append(ev)

        events.sort(key=lambda e: e.date)
        if events:
            self._cache.save_news(ticker, events)

        logger.info("Fetched %d unique news events for %s", len(events), ticker)
        return events
