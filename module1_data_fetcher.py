"""
Module 1 — Data Fetcher
Owner: Person 1

Design: Two abstract base classes — PriceFetcher and NewsFetcher —
separate the concerns of price data vs. news retrieval.
Concrete classes implement specific sources.

        PriceFetcher (abstract)
        ├── MockPriceFetcher        ← used for testing
        └── YFinancePriceFetcher    ← real: yfinance

        NewsFetcher (abstract)
        ├── MockNewsFetcher         ← used for testing
        └── FinnhubNewsFetcher      ← real: Finnhub API
"""

import csv
import os
import time
import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path

from models import PricePoint, MarketEvent, EventType

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "data_cache"

# ── CSV Cache ─────────────────────────────────────────────────────────────────


class DataCache:
    """
    Read/write price and news data as CSV files under data_cache/.
    File layout:
        data_cache/{TICKER}_prices.csv
        data_cache/{TICKER}_news.csv

    On each run, new rows are merged with existing data (deduplicated by date
    for prices, by date+title for news), so the cache grows over time.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    # ── prices ────────────────────────────────────────────────────────────

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
                        date=date.fromisoformat(row["date"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping cached price row: %s", e)
        return points

    def save_prices(self, ticker: str, new_points: list[PricePoint]) -> None:
        existing = {p.date: p for p in self.load_prices(ticker)}
        for p in new_points:
            existing[p.date] = p  # newer data overwrites same date
        merged = sorted(existing.values(), key=lambda p: p.date)

        path = self._prices_path(ticker)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])
            for p in merged:
                writer.writerow([p.date, p.open, p.high, p.low, p.close, p.volume])
        logger.info("Cached %d price rows for %s → %s", len(merged), ticker, path)

    # ── news ──────────────────────────────────────────────────────────────

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
                    events.append(MarketEvent(
                        date=date.fromisoformat(row["date"]),
                        title=row["title"],
                        description=row["description"],
                        source=row["source"],
                        event_type=EventType(row["event_type"]),
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping cached news row: %s", e)
        return events

    def save_news(self, ticker: str, new_events: list[MarketEvent]) -> None:
        existing = {(e.date, e.title): e for e in self.load_news(ticker)}
        for e in new_events:
            existing[(e.date, e.title)] = e
        merged = sorted(existing.values(), key=lambda e: e.date)

        path = self._news_path(ticker)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "title", "description", "source", "event_type"])
            for e in merged:
                writer.writerow([e.date, e.title, e.description, e.source, e.event_type.value])
        logger.info("Cached %d news rows for %s → %s", len(merged), ticker, path)


# ── Abstract interfaces ──────────────────────────────────────────────────────


class PriceFetcher(ABC):
    @abstractmethod
    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]:
        """Fetch OHLCV price data for the given ticker and date range."""
        ...


class NewsFetcher(ABC):
    @abstractmethod
    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        """Fetch relevant news/market events for the given ticker and date range."""
        ...


# ── Mock implementations (testing / demo) ────────────────────────────────────


class MockPriceFetcher(PriceFetcher):
    """
    Returns cached data if available for the ticker, otherwise falls back
    to hardcoded NVDA sample prices.
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
        return [PricePoint(*row) for row in self._FALLBACK]


class MockNewsFetcher(NewsFetcher):
    """
    Returns cached data if available for the ticker, otherwise falls back
    to hardcoded NVDA sample events.
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


# ── Real implementations ─────────────────────────────────────────────────────


class YFinancePriceFetcher(PriceFetcher):
    """Fetch historical OHLCV prices via yfinance, with automatic CSV caching."""

    def __init__(self):
        self._cache = DataCache()

    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]:
        import yfinance as yf

        # yfinance end is exclusive — add 1 day to include the end date
        df = yf.download(
            ticker,
            start=start,
            end=end + timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            logger.warning("yfinance returned no data for %s (%s ~ %s)", ticker, start, end)
            return []

        # yfinance v0.2+ returns MultiIndex columns even for a single ticker
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df = df.droplevel("Ticker", axis=1)

        points: list[PricePoint] = []
        for row in df.itertuples():
            try:
                point = PricePoint(
                    date=row.Index.date(),
                    open=round(float(row.Open), 2),
                    high=round(float(row.High), 2),
                    low=round(float(row.Low), 2),
                    close=round(float(row.Close), 2),
                    volume=int(row.Volume),
                )
                points.append(point)
            except (ValueError, TypeError) as e:
                logger.warning("Skipping dirty row %s: %s", row.Index, e)

        if points:
            self._cache.save_prices(ticker, points)

        logger.info("Fetched %d price points for %s", len(points), ticker)
        return points


# ── Finnhub event-type classifier ────────────────────────────────────────────

_EVENT_KEYWORDS: dict[EventType, list[str]] = {
    EventType.EARNINGS:   ["earnings", "revenue", "eps", "quarterly results", "profit", "income"],
    EventType.ANALYST:    ["analyst", "upgrade", "downgrade", "price target", "rating", "coverage"],
    EventType.REGULATORY: ["sec", "regulation", "fda", "antitrust", "lawsuit", "compliance", "fine"],
    EventType.MACRO:      ["fed", "interest rate", "inflation", "recession", "gdp", "unemployment", "tariff"],
    EventType.PRODUCT:    ["launch", "product", "release", "unveil", "announce", "partnership", "deal"],
}


def _classify_event(headline: str, category: str) -> EventType:
    """Map a Finnhub headline + category to an EventType via keyword matching."""
    text = f"{headline} {category}".lower()
    for event_type, keywords in _EVENT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return event_type
    return EventType.OTHER


class FinnhubNewsFetcher(NewsFetcher):
    """
    Fetch company news via the Finnhub API (finnhub-python SDK), with automatic CSV caching.

    Finnhub caps each request at ~250 articles. The fetcher first splits the
    date range into weekly windows, then adaptively bisects any window that
    hits the 250 cap — so only news-dense periods incur extra API calls.
    """

    INITIAL_WINDOW_DAYS = 7   # first-pass split granularity
    FINNHUB_MAX_PER_REQ = 250
    API_SLEEP_SEC = 1.1       # Finnhub free tier: 60 req/min → ~1 req/sec safe

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Finnhub API key is required. "
                "Pass it to the constructor or set FINNHUB_API_KEY env var."
            )
        self._cache = DataCache()

    # ── internal: adaptive recursive fetch ───────────────────────────────

    def _fetch_recursive(self, client, ticker: str, w_start: date, w_end: date) -> list[dict]:
        """
        Fetch raw articles for one window. If the result hits the 250 cap
        and the window is wider than 1 day, bisect and recurse into each half.
        """
        import finnhub
        time.sleep(self.API_SLEEP_SEC)
        try:
            raw = client.company_news(ticker, _from=str(w_start), to=str(w_end))
        except finnhub.FinnhubAPIException as e:
            logger.error("Finnhub API error for %s (%s~%s): %s", ticker, w_start, w_end, e)
            return []
        except Exception as e:
            logger.error("Unexpected error fetching news for %s (%s~%s): %s", ticker, w_start, w_end, e)
            return []

        if len(raw) < self.FINNHUB_MAX_PER_REQ:
            return raw

        # Hit the cap — try to bisect
        if w_start == w_end:
            logger.warning(
                "Finnhub returned %d articles for %s on single day %s — cannot split further",
                len(raw), ticker, w_start,
            )
            return raw

        mid = w_start + (w_end - w_start) // 2
        logger.info(
            "Window %s~%s hit %d cap, bisecting at %s",
            w_start, w_end, len(raw), mid,
        )
        left = self._fetch_recursive(client, ticker, w_start, mid)
        right = self._fetch_recursive(client, ticker, mid + timedelta(days=1), w_end)
        return left + right

    # ── internal: parse one raw article → MarketEvent ────────────────────

    @staticmethod
    def _parse_article(article: dict, fallback_date: date) -> MarketEvent | None:
        headline = (article.get("headline") or "").strip()
        source = (article.get("source") or "").strip()
        description = (article.get("summary") or "").strip()
        if not headline or not source or not description:
            return None

        ts = article.get("datetime", 0)
        article_date = date.fromtimestamp(ts) if ts else fallback_date
        event_type = _classify_event(headline, article.get("category", ""))

        return MarketEvent(
            date=article_date,
            title=headline,
            description=description[:500],
            source=source,
            event_type=event_type,
        )

    # ── public interface ─────────────────────────────────────────────────

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        import finnhub
        client = finnhub.Client(api_key=self.api_key)

        # First pass: split into weekly windows
        windows: list[tuple[date, date]] = []
        w_start = start
        while w_start <= end:
            w_end = min(w_start + timedelta(days=self.INITIAL_WINDOW_DAYS - 1), end)
            windows.append((w_start, w_end))
            w_start = w_end + timedelta(days=1)

        logger.info(
            "Fetching news for %s (%s ~ %s) in %d initial window(s)...",
            ticker, start, end, len(windows),
        )

        # Collect raw articles — each window may recursively bisect
        raw_all: list[dict] = []
        for ws, we in windows:
            raw_all.extend(self._fetch_recursive(client, ticker, ws, we))

        # Parse & deduplicate by (date, title)
        seen: set[tuple[date, str]] = set()
        events: list[MarketEvent] = []
        for article in raw_all:
            event = self._parse_article(article, fallback_date=start)
            if event is None:
                continue
            key = (event.date, event.title)
            if key in seen:
                continue
            seen.add(key)
            events.append(event)

        events.sort(key=lambda e: e.date)

        if events:
            self._cache.save_news(ticker, events)

        logger.info("Fetched %d unique news events for %s", len(events), ticker)
        return events
