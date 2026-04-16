"""
Module 1 — Data Fetcher

        PriceFetcher (abstract)
        ├── MockPriceFetcher        ← testing / demo
        └── YFinancePriceFetcher    ← real: yfinance + CSV cache

        NewsFetcher (abstract)
        ├── MockNewsFetcher         ← testing / demo
        ├── FinnhubNewsFetcher      ← real: Finnhub API + CSV cache
        ├── AlphaVantageNewsFetcher ← real: Alpha Vantage NEWS_SENTIMENT API
        ├── YFinanceEventsFetcher   ← real: yfinance earnings/splits/dividends
        ├── KnownEventsFetcher      ← curated major historical events
        ├── NewsDataFetcher         ← real: NewsData.io historical news API
        ├── SECFetcher              ← real: SEC EDGAR 8-K filings (no API key)
        ├── NewsApiFetcher          ← real: Guardian + NYT APIs
        └── CompositeNewsFetcher    ← merges multiple fetchers
"""

import csv
import json
import os
import time
import logging
import requests
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

    _NEWS_COLUMNS = [
        "date", "title", "description", "source", "event_type",
        "reported_eps", "beat_expectations",
        "url", "sentiment_score", "relevance_score",
    ]

    def load_news(self, ticker: str) -> list[MarketEvent]:
        path = self._news_path(ticker)
        if not path.exists():
            return []
        events: list[MarketEvent] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    event_type = EventType(row["event_type"])
                    # Parse optional fields (backward-compatible with old CSVs)
                    url = row.get("url") or None
                    sent = row.get("sentiment_score")
                    sentiment = float(sent) if sent else None
                    rel = row.get("relevance_score")
                    relevance = float(rel) if rel else None

                    if event_type == EventType.EARNINGS and "reported_eps" in row:
                        ev = EarningsEvent(
                            date             = date.fromisoformat(row["date"]),
                            title            = row["title"],
                            description      = row["description"],
                            source           = row["source"],
                            reported_eps     = float(row["reported_eps"] or 0.0),
                            beat_expectations= row["beat_expectations"] == "True",
                        )
                    else:
                        ev = MarketEvent(
                            date       = date.fromisoformat(row["date"]),
                            title      = row["title"],
                            description= row["description"],
                            source     = row["source"],
                            event_type = event_type,
                        )
                    ev.url = url
                    ev.sentiment_score = sentiment
                    ev.relevance_score = relevance
                    events.append(ev)
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
            writer.writerow(self._NEWS_COLUMNS)
            for e in merged:
                if isinstance(e, EarningsEvent):
                    writer.writerow([
                        e.date, e.title, e.description, e.source,
                        e.event_type.value, e.reported_eps, e.beat_expectations,
                        e.url or "", e.sentiment_score or "", e.relevance_score or "",
                    ])
                else:
                    writer.writerow([
                        e.date, e.title, e.description, e.source,
                        e.event_type.value, "", "",
                        e.url or "", e.sentiment_score or "", e.relevance_score or "",
                    ])
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
        # expanded: financial reporting
        "eps", "earnings per share", "operating income", "ebitda",
        "revenue growth", "top line", "bottom line", "reported results",
        "earnings call", "earnings report", "earnings surprise",
        "sales growth", "profit margin", "operating margin",
        "fiscal quarter", "annual results", "full-year results",
        "home run results", "blowout quarter", "record quarter",
        "beats wall street", "tops estimates", "falls short",
        "revenue miss", "profit warning", "results top",
    ],
    EventType.ANALYST: [
        "analyst", "upgrade", "downgrade", "price target", "rating",
        "overweight", "underweight", "outperform", "underperform",
        "initiates coverage", "raises target", "cuts target",
        # expanded: wall street actions
        "raises pt", "cuts pt", "lowers pt", "boosts pt",
        "raises price target", "cuts price target", "lowers price target",
        "buy rating", "sell rating", "hold rating", "neutral rating",
        "strong buy", "market perform", "sector perform",
        "top pick", "top stock", "best stock", "favorite stock",
        "conviction list", "focus list",
        "bull case", "bear case", "base case",
        "reiterates", "maintains rating", "reaffirms",
        "wall street", "broker", "brokerage",
        "morgan stanley", "goldman sachs", "jpmorgan", "bofa",
        "bank of america", "citigroup", "citi ", "barclays",
        "wells fargo", "ubs ", "deutsche bank", "credit suisse",
        "jefferies", "piper sandler", "bernstein", "mizuho",
        "needham", "wedbush", "rbc ", "td cowen", "cowen",
        "oppenheimer", "stifel", "canaccord", "loop capital",
        "keybanc", "evercore", "wolfe research", "truist",
        "raymond james", "susquehanna", "rosenblatt",
    ],
    EventType.LEGAL: [
        # lawsuits, court proceedings, settlements
        "lawsuit", "sued", "sues", "suing", "litigation",
        "court ruling", "court order", "court case", "courtroom",
        "trial", "verdict", "settlement", "settles",
        "judge", "jury", "plaintiff", "defendant",
        "injunction", "damages", "class action", "class-action",
        "patent infringement", "patent dispute", "ip dispute",
        "copyright", "trademark", "trade secret",
        "appeals court", "supreme court", "federal court",
        "ftc case", "doj case",
        "consent decree", "plea", "indictment",
        "legal action", "legal battle", "legal challenge",
        "break up", "break-up", "breakup", "divestiture",
        "forced to sell", "force sale", "must sell",
        # expanded: investigations, law firms
        "law firm investigates", "investigates claims",
        "on behalf of investors", "securities fraud",
        "legal siege", "legal threat",
    ],
    EventType.REGULATORY: [
        "sec ", "regulation", "fda", "compliance",
        "export control", "export ban", "chip ban", "sanction", "chips act",
        # expanded: government & regulatory actions
        "antitrust", "monopoly", "anti-competitive",
        "ftc ", "doj ", "eu commission", "european commission",
        "data privacy", "gdpr", "privacy law", "privacy regulation",
        "content moderation", "section 230",
        "congressional hearing", "senate hearing", "subpoena",
        "regulatory scrutiny", "regulatory probe", "regulatory review",
        "fine ", "fined", "penalty", "penalized",
        "ban ", "banned", "bans ", "blocking",
        "censorship", "restrict", "restriction",
        "investigation", "investigated", "probe",
        "digital markets act", "digital services act",
        "competition authority", "watchdog",
        "tax probe", "tax ruling", "digital tax",
        # expanded: testimony, whistleblower
        "whistleblower", "testifies", "testimony",
        "congressional", "senate ",
    ],
    EventType.MACRO: [
        "fed ", "interest rate", "inflation", "recession", "gdp",
        "unemployment", "tariff", "trade war", "rate hike", "rate cut",
        "treasury yield", "sell-off",
        # expanded: broad market & economic events
        "federal reserve", "fomc", "monetary policy",
        "bond yield", "yield curve", "inverted yield",
        "jobs report", "nonfarm payroll", "consumer confidence",
        "cpi ", "ppi ", "consumer price", "producer price",
        "economic slowdown", "economic growth", "soft landing",
        "trade tension", "trade deal", "trade policy",
        "tariff exemption", "tariff war", "reciprocal tariff",
        "geopolitical", "war ", "conflict",
        "oil price", "crude oil", "opec",
        "china trade", "china economy", "china tariff",
        "market crash", "market correction", "bear market",
        "market rally", "broad market", "risk-off", "risk off",
        "global sell-off", "global selloff", "market selloff",
        "market rout", "tech rout", "nasdaq drop", "s&p drop",
        "debt ceiling", "government shutdown",
        "supply chain", "chip shortage",
        "currency", "dollar strength", "dollar weakness",
    ],
    EventType.AI_TECH: [
        # AI strategy, model releases, data centers, chips
        "artificial intelligence", "ai model", "ai training",
        "large language model", "llm ", "generative ai", "gen ai",
        "machine learning", "deep learning", "neural network",
        "chatbot", "chatgpt", "copilot",
        "ai agent", "ai assistant", "ai tool",
        "ai chip", "ai gpu", "ai accelerator",
        "data center", "datacenter", "hyperscale",
        "ai infrastructure", "compute capacity",
        "ai investment", "ai spending", "ai capex",
        "ai strategy", "ai roadmap", "ai pivot",
        "open source ai", "open-source ai",
        "ai safety", "ai regulation", "ai governance",
        "ai revenue", "ai monetization",
        "training data", "ai training data",
        "superintelligence", "agi ",
        # specific model/product names
        "llama ", "llama-", "gpt-", "gemini ai",
        "stable diffusion", "midjourney",
        "transformer model", "foundation model",
        "ai supercomputer", "gpu cluster",
        "nvidia h100", "nvidia b100", "nvidia h20",
        "tensor processing", "tpu ",
        # expanded: AI industry terms
        "ai arms race", "ai push", "ai race",
        "ai recruiting", "ai talent", "ai team",
        "ai hype", "ai boom", "ai bubble",
        "ai stock", "ai trade", "ai winner",
        "ai disruption", "ai replace",
    ],
    EventType.PRODUCT: [
        "launch", "release", "unveil", "announce", "partnership",
        "acquisition", "acquires", "merger", "funding round",
        # expanded: products, features, deals
        "product", "new feature", "introduces", "rolls out",
        "debuts", "ships", "available now", "coming soon",
        "beta test", "early access", "preview",
        "app update", "software update", "platform update",
        "smart glasses", "ray-ban", "headset", "vr ", "ar ",
        "quest ", "oculus", "reality labs",
        "metaverse", "virtual reality", "augmented reality",
        "mixed reality", "spatial computing",
        "instagram feature", "whatsapp feature", "facebook feature",
        "threads app", "reels ", "stories ",
        "ads platform", "advertising", "ad revenue",
        "new service", "new platform", "new tool",
        "deal ", "signed deal", "contract",
        "strategic investment", "invests in", "investment in",
        "joint venture", "collaboration", "alliance",
        "subscriber", "monthly active user", "daily active user",
        "user growth", "users milestone", "billion users",
        "expansion", "expands into", "enters market",
        "supply agreement", "licensing deal", "content deal",
        "hardware", "device", "wearable",
        # expanded: service events, rebranding
        "outage", "service disruption", "went down",
        "rebrand", "rebrands", "renamed", "rename",
        "pivot", "pivoting", "new name",
    ],
    EventType.PERSONNEL: [
        "ceo", "cfo", "coo", "cto", "appointed", "resigns", "resignation",
        "steps down", "named as", "new chief", "new president",
        # expanded: executive and board changes
        "executive", "board member", "board of directors",
        "director joins", "director leaves", "new director",
        "chairman", "vice president", "vp ",
        "hire", "hired", "hiring", "fires", "fired", "firing",
        "layoff", "layoffs", "job cuts", "cuts jobs",
        "restructuring", "reorganization", "reorg",
        "workforce reduction", "headcount",
        "founder", "co-founder", "succession",
        "interim ceo", "interim cfo",
        "management shakeup", "leadership change",
        # expanded: talent moves
        "poach", "poaches", "poached",
        "recruit", "recruiting blitz",
        "leaves for", "departs for",
    ],
}

_NOISE_TITLE_PATTERNS = [
    # market roundups & generic lists
    "stock market today:",
    "these stocks moved the most",
    "stocks moving the most today",
    "most active stocks",
    "dow jones futures:",
    "top 10 trending stocks on wallstreetbets",
    "trending stocks:",
    "stocks to watch this week",
    "stocks to watch today",
    "morning squawk",
    "market wrap",
    "market minute",
    "stocks making the biggest moves",
    # generic personal finance / non-actionable
    "retirement account",
    "my mom wants to spend",
    "my dad wants to spend",
    "interview preparation guide",
    "interview kickstart",
    "young banker holding",
    "portfolio sees red",
    # clickbait stock predictions with no substance
    "stock could be the perfect pick",
    "the best stocks to buy with",
    "2 stocks to buy and hold",
    "3 stocks to buy and hold",
    # other company focus (no ticker relevance)
    "why alibaba stock",
    "why nvidia stock",
    "amd: a tale of",
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


# ── Real: Alpha Vantage NEWS_SENTIMENT fetcher ───────────────────────────────

class AlphaVantageNewsFetcher(NewsFetcher):
    """
    Fetches news with built-in sentiment & relevance scores from Alpha Vantage.

    Advantages over Finnhub:
      - Returns article URLs
      - Built-in per-ticker sentiment scores (-1.0 to 1.0)
      - Built-in per-ticker relevance scores (0.0 to 1.0)
      - Covers longer historical range

    Free tier limits: 25 requests/day, 5 requests/minute.
    Each request returns up to 200 articles (with limit=200).
    Uses monthly windowing to paginate over long date ranges.
    """

    API_URL       = "https://www.alphavantage.co/query"
    MAX_PER_REQ   = 200       # max articles per request
    API_SLEEP_SEC = 12.5      # 5 req/min → 12s between calls
    WINDOW_DAYS   = 30        # monthly windows for pagination

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required. "
                "Set ALPHA_VANTAGE_API_KEY in your .env file. "
                "Get a free key at https://www.alphavantage.co/support/#api-key"
            )
        self._cache = DataCache()

    def _fetch_window(self, ticker: str, w_start: date, w_end: date) -> list[dict]:
        """Fetch one time window from the API."""
        params = {
            "function":  "NEWS_SENTIMENT",
            "tickers":   ticker,
            "time_from": w_start.strftime("%Y%m%dT0000"),
            "time_to":   w_end.strftime("%Y%m%dT2359"),
            "limit":     str(self.MAX_PER_REQ),
            "sort":      "EARLIEST",
            "apikey":    self.api_key,
        }

        time.sleep(self.API_SLEEP_SEC)
        try:
            resp = requests.get(self.API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Alpha Vantage fetch error %s (%s~%s): %s",
                         ticker, w_start, w_end, e)
            return []

        # Check for API error messages
        if "Information" in data or "Error Message" in data:
            msg = data.get("Information") or data.get("Error Message")
            logger.warning("Alpha Vantage API message: %s", msg)
            return []

        return data.get("feed", [])

    def _parse_article(self, article: dict, ticker: str) -> MarketEvent | None:
        """Parse one Alpha Vantage feed item into a MarketEvent."""
        title       = (article.get("title") or "").strip()
        summary     = (article.get("summary") or "").strip()
        source      = (article.get("source") or "").strip()
        url         = (article.get("url") or "").strip()
        time_pub    = article.get("time_published", "")

        if not title or not source:
            return None
        if any(p in title.lower() for p in _NOISE_TITLE_PATTERNS):
            return None

        # Parse date from "YYYYMMDDTHHMMSS" format
        try:
            article_date = date(int(time_pub[:4]), int(time_pub[4:6]), int(time_pub[6:8]))
        except (ValueError, IndexError):
            return None

        # Extract per-ticker sentiment and relevance scores
        sentiment = None
        relevance = None
        for ts in article.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                try:
                    sentiment = float(ts.get("ticker_sentiment_score", 0))
                except (ValueError, TypeError):
                    pass
                try:
                    relevance = float(ts.get("relevance_score", 0))
                except (ValueError, TypeError):
                    pass
                break

        # Fall back to overall sentiment if no per-ticker score
        if sentiment is None:
            try:
                sentiment = float(article.get("overall_sentiment_score", 0))
            except (ValueError, TypeError):
                pass

        event_type = _classify_event(title, "")

        ev = MarketEvent(
            date        = article_date,
            title       = title,
            description = summary[:500],
            source      = source,
            event_type  = event_type,
            url         = url or None,
            sentiment_score = sentiment,
            relevance_score = relevance,
        )
        return ev

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        """
        Fetch news across the full date range using monthly windows.
        Deduplicates by (date, title) and auto-caches to CSV.
        """
        windows: list[tuple[date, date]] = []
        w_start = start
        while w_start <= end:
            w_end = min(w_start + timedelta(days=self.WINDOW_DAYS - 1), end)
            windows.append((w_start, w_end))
            w_start = w_end + timedelta(days=1)

        logger.info("Alpha Vantage: fetching %s (%s~%s) in %d windows",
                     ticker, start, end, len(windows))

        seen: set[tuple[date, str]] = set()
        events: list[MarketEvent] = []

        for i, (ws, we) in enumerate(windows):
            raw = self._fetch_window(ticker, ws, we)
            batch_count = 0
            for article in raw:
                ev = self._parse_article(article, ticker)
                if ev is None:
                    continue
                key = (ev.date, ev.title)
                if key in seen:
                    continue
                seen.add(key)
                events.append(ev)
                batch_count += 1
            logger.info("  Window %d/%d (%s~%s): %d articles, %d new events",
                         i + 1, len(windows), ws, we, len(raw), batch_count)

            # Check if we hit rate limit (API returns empty or error)
            if not raw and i < len(windows) - 1:
                logger.warning("Alpha Vantage: empty response, possible rate limit. "
                               "Stopping at window %d/%d. Resume later.",
                               i + 1, len(windows))
                break

        events.sort(key=lambda e: e.date)
        if events:
            self._cache.save_news(ticker, events)

        logger.info("Alpha Vantage: %d unique events for %s", len(events), ticker)
        return events


# ── Historical data: yfinance earnings/splits/dividends ─────────────────────

class YFinanceEventsFetcher(NewsFetcher):
    """
    Extracts historical events from yfinance: earnings dates, stock splits,
    and significant dividends. Covers the full history of a ticker — fills
    the gap left by Finnhub's ~1-year news limit.
    """

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        events: list[MarketEvent] = []

        # 1. Earnings dates (most important for anomaly matching)
        try:
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                for ts, row in earnings.iterrows():
                    try:
                        d = ts.date() if hasattr(ts, 'date') else ts
                        if not (start <= d <= end):
                            continue
                        eps_est = row.get("EPS Estimate", None)
                        eps_act = row.get("Reported EPS", None)
                        surprise = row.get("Surprise(%)", None)

                        # Build descriptive title
                        parts = [f"{ticker} earnings report"]
                        if eps_act is not None and not _is_nan(eps_act):
                            parts.append(f"EPS: ${eps_act:.2f}")
                        if eps_est is not None and not _is_nan(eps_est):
                            parts.append(f"(est: ${eps_est:.2f})")
                        if surprise is not None and not _is_nan(surprise):
                            beat = "beat" if surprise > 0 else "missed"
                            parts.append(f"— {beat} by {abs(surprise):.1f}%")

                        title = " ".join(parts)

                        desc = f"Quarterly earnings report for {ticker}."
                        if eps_act is not None and eps_est is not None:
                            if not _is_nan(eps_act) and not _is_nan(eps_est):
                                desc += f" Reported EPS ${eps_act:.2f} vs estimate ${eps_est:.2f}."

                        if eps_act is not None and not _is_nan(eps_act):
                            beat_exp = (surprise is not None and not _is_nan(surprise)
                                        and surprise > 0)
                            events.append(EarningsEvent(
                                date=d, title=title, description=desc,
                                source="yfinance",
                                reported_eps=float(eps_act),
                                beat_expectations=beat_exp,
                            ))
                        else:
                            events.append(MarketEvent(
                                date=d, title=title, description=desc,
                                source="yfinance",
                                event_type=EventType.EARNINGS,
                            ))
                    except Exception as e:
                        logger.debug("Skipping earnings row: %s", e)
        except Exception as e:
            logger.warning("Could not fetch earnings dates for %s: %s", ticker, e)

        # 2. Stock splits
        try:
            splits = stock.splits
            if splits is not None and not splits.empty:
                for ts, ratio in splits.items():
                    try:
                        d = ts.date() if hasattr(ts, 'date') else ts
                        if not (start <= d <= end) or ratio == 0:
                            continue
                        title = f"{ticker} stock split {ratio:.0f}:1"
                        desc = f"{ticker} executed a {ratio:.0f}-for-1 stock split."
                        events.append(MarketEvent(
                            date=d, title=title, description=desc,
                            source="yfinance", event_type=EventType.PRODUCT,
                        ))
                    except Exception as e:
                        logger.debug("Skipping split row: %s", e)
        except Exception as e:
            logger.warning("Could not fetch splits for %s: %s", ticker, e)

        # 3. Significant dividends (only include if amount is notable)
        try:
            dividends = stock.dividends
            if dividends is not None and not dividends.empty:
                for ts, amount in dividends.items():
                    try:
                        d = ts.date() if hasattr(ts, 'date') else ts
                        if not (start <= d <= end) or amount <= 0:
                            continue
                        title = f"{ticker} dividend payment ${amount:.2f}/share"
                        desc = f"{ticker} paid a dividend of ${amount:.2f} per share."
                        events.append(MarketEvent(
                            date=d, title=title, description=desc,
                            source="yfinance", event_type=EventType.EARNINGS,
                        ))
                    except Exception as e:
                        logger.debug("Skipping dividend row: %s", e)
        except Exception as e:
            logger.warning("Could not fetch dividends for %s: %s", ticker, e)

        events.sort(key=lambda e: e.date)
        logger.info("YFinanceEvents: %d events for %s (%s~%s)",
                     len(events), ticker, start, end)
        return events


def _is_nan(v) -> bool:
    """Check if a value is NaN (works for float and numpy)."""
    try:
        return v != v  # NaN != NaN is True
    except (TypeError, ValueError):
        return False


# ── Known historical events for popular tickers ─────────────────────────────

# Manually curated major events that Finnhub misses due to its 1-year limit.
# These are high-impact events known to drive large price moves.
_KNOWN_EVENTS: dict[str, list[tuple[date, str, str, EventType]]] = {
    "META": [
        # 2021
        (date(2021, 1, 6), "Facebook bans Trump following Capitol riot",
         "Facebook indefinitely suspends Donald Trump's account after the Jan 6 Capitol breach.",
         EventType.REGULATORY),
        (date(2021, 1, 27), "Facebook Q4 2020 earnings beat, warns of iOS headwinds",
         "Q4 revenue $28.1B beat estimates. Company warned Apple's iOS 14 privacy changes would hurt ad targeting.",
         EventType.EARNINGS),
        (date(2021, 4, 28), "Facebook Q1 2021 earnings crush estimates, revenue up 48%",
         "Q1 revenue $26.2B vs $23.7B expected. DAU 1.88B. Strong digital ad market recovery.",
         EventType.EARNINGS),
        (date(2021, 6, 28), "FTC files antitrust complaint against Facebook",
         "The FTC refiled its antitrust lawsuit alleging Facebook maintains an illegal monopoly in social networking.",
         EventType.LEGAL),
        (date(2021, 7, 28), "Facebook Q2 2021 earnings beat but warns of growth slowdown",
         "Q2 revenue $29.1B beat estimates. Warned of slower growth as pandemic digital surge normalizes.",
         EventType.EARNINGS),
        (date(2021, 10, 4), "Facebook, Instagram, WhatsApp suffer massive global outage",
         "All Facebook services went down for ~6 hours due to a BGP routing configuration error. Stock dropped 4.9%.",
         EventType.PRODUCT),
        (date(2021, 10, 5), "Facebook whistleblower Frances Haugen testifies before Senate",
         "Former employee testified that Facebook prioritized profits over user safety, especially for teens.",
         EventType.REGULATORY),
        (date(2021, 10, 25), "Facebook Q3 2021 earnings beat, revenue $29B",
         "Beat estimates but warned of Apple ATT impact. Reality Labs lost $2.6B.",
         EventType.EARNINGS),
        (date(2021, 10, 28), "Facebook rebrands to Meta Platforms",
         "CEO Zuckerberg announced the company rebrand to Meta, pivoting focus to the metaverse.",
         EventType.PRODUCT),
        # 2022
        (date(2022, 2, 2), "Meta Q4 2021 earnings miss, first daily user decline",
         "Facebook reported first-ever decline in daily active users. Stock plunged 26% after hours.",
         EventType.EARNINGS),
        (date(2022, 4, 27), "Meta Q1 2022 earnings beat lowered expectations",
         "Revenue $27.9B slightly beat. User growth returned. Stock surged 18% after hours.",
         EventType.EARNINGS),
        (date(2022, 7, 27), "Meta Q2 2022: first ever revenue decline",
         "Revenue fell 1% to $28.8B, first-ever year-over-year decline. Digital ad market weakened.",
         EventType.EARNINGS),
        (date(2022, 10, 26), "Meta Q3 2022 earnings disappoint, Reality Labs burns $3.7B",
         "Revenue down 4% to $27.7B. Reality Labs lost $3.7B. Guided higher capex for AI/metaverse.",
         EventType.EARNINGS),
        (date(2022, 11, 9), "Meta announces massive layoffs — 11,000 employees cut",
         "Zuckerberg announced cutting 13% of workforce, the company's first mass layoff.",
         EventType.PERSONNEL),
        # 2023
        (date(2023, 2, 1), "Meta Q4 2022 earnings: 'Year of Efficiency' announced",
         "Revenue $32.2B beat estimates. Zuckerberg declared 2023 the 'Year of Efficiency'. Stock surged 23%.",
         EventType.EARNINGS),
        (date(2023, 3, 14), "Meta announces second round of layoffs — 10,000 jobs cut",
         "Meta cut another 10,000 roles as part of its efficiency drive. Flattened management layers.",
         EventType.PERSONNEL),
        (date(2023, 4, 26), "Meta Q1 2023 earnings blow past estimates",
         "Revenue $28.6B vs $27.7B expected. User engagement up across all apps. Efficiency gains visible.",
         EventType.EARNINGS),
        (date(2023, 7, 5), "Meta launches Threads, gaining 100M users in 5 days",
         "Threads launched as a Twitter competitor, hitting 100 million sign-ups in under a week.",
         EventType.PRODUCT),
        (date(2023, 7, 18), "Meta releases Llama 2 as open-source AI model",
         "Meta released Llama 2, making a powerful large language model freely available for research and commercial use.",
         EventType.AI_TECH),
        (date(2023, 7, 26), "Meta Q2 2023 earnings blow past estimates, stock surges",
         "Revenue $32.0B vs $31.1B expected. Net income more than doubled YoY. Stock up 10% after hours.",
         EventType.EARNINGS),
        (date(2023, 9, 27), "Meta unveils Quest 3 VR headset and AI-powered smart glasses",
         "Meta launched Quest 3 ($499) and AI-powered Ray-Ban smart glasses with Meta AI assistant.",
         EventType.PRODUCT),
        (date(2023, 10, 25), "Meta Q3 2023 earnings crush estimates, AI drives ad growth",
         "Revenue $34.1B vs $33.6B expected. AI-powered ad targeting significantly improved ROAS.",
         EventType.EARNINGS),
        # 2024
        (date(2024, 2, 1), "Meta Q4 2023 earnings blowout, announces first-ever dividend",
         "Revenue $40.1B vs $39.2B expected. Announced $0.50/share quarterly dividend and $50B buyback. Stock surged 20%.",
         EventType.EARNINGS),
        (date(2024, 4, 18), "Meta releases Llama 3, most capable open-source LLM",
         "Meta released Llama 3 in 8B and 70B parameter versions, claiming state-of-the-art open-source performance.",
         EventType.AI_TECH),
        (date(2024, 4, 24), "Meta Q1 2024 earnings beat but capex guidance spooks investors",
         "Revenue $36.5B beat estimates. But raised 2024 capex guidance to $35-40B for AI infrastructure. Stock fell 15%.",
         EventType.EARNINGS),
        (date(2024, 7, 31), "Meta Q2 2024 earnings strong, $37-40.5B Q3 revenue guide",
         "Revenue $39.1B vs $38.3B expected. Strong Reels monetization. Guided Q3 revenue $38.5-41B.",
         EventType.EARNINGS),
        (date(2024, 9, 25), "Meta unveils Orion AR glasses prototype and Quest 3S",
         "Meta showed off Orion, the most advanced AR glasses prototype, and launched the $299 Quest 3S.",
         EventType.PRODUCT),
        (date(2024, 10, 30), "Meta Q3 2024 earnings beat, AI capex continues to surge",
         "Revenue $40.6B vs $40.3B expected. Raised 2024 capex to $38-40B. Reality Labs lost $4.4B.",
         EventType.EARNINGS),
    ],
    "NVDA": [
        (date(2021, 4, 12), "NVIDIA announces Grace CPU for data centers",
         "NVIDIA unveiled Grace, its first data center CPU, designed for AI and HPC workloads.",
         EventType.PRODUCT),
        (date(2022, 9, 1), "US orders NVIDIA to halt A100/H100 chip sales to China",
         "US government restricted export of NVIDIA's top AI chips to China, citing national security.",
         EventType.REGULATORY),
        (date(2023, 5, 24), "NVIDIA Q1 FY2024 earnings: data center revenue explodes",
         "Revenue guided $11B for Q2, 50% above consensus. Data center demand for H100 chips surging.",
         EventType.EARNINGS),
        (date(2023, 5, 30), "NVIDIA briefly hits $1 trillion market cap",
         "NVIDIA became the first chipmaker to reach a $1 trillion valuation, driven by AI demand.",
         EventType.PRODUCT),
        (date(2024, 3, 18), "NVIDIA unveils Blackwell B200 GPU at GTC 2024",
         "Jensen Huang presented the B200 GPU with 20 petaflops of FP4 AI performance at GTC.",
         EventType.AI_TECH),
        (date(2024, 6, 10), "NVIDIA announces 10-for-1 stock split",
         "NVIDIA executed a 10:1 stock split, making shares more accessible to retail investors.",
         EventType.PRODUCT),
    ],
    "AAPL": [
        (date(2021, 4, 20), "Apple unveils AirTag, M1 iMac, and new iPad Pro",
         "Apple announced AirTag tracker, redesigned M1 iMac, and M1-powered iPad Pro at spring event.",
         EventType.PRODUCT),
        (date(2021, 9, 10), "Epic Games vs Apple: judge rules Apple not a monopoly",
         "Federal judge ruled Apple is not a monopoly but ordered changes to App Store anti-steering rules.",
         EventType.LEGAL),
        (date(2023, 6, 5), "Apple unveils Vision Pro mixed reality headset at WWDC",
         "Apple announced the $3,499 Vision Pro, its first major new product category since Apple Watch.",
         EventType.PRODUCT),
        (date(2024, 6, 10), "Apple announces Apple Intelligence AI features at WWDC 2024",
         "Apple revealed Apple Intelligence, a suite of on-device and cloud AI features powered by its own models.",
         EventType.AI_TECH),
    ],
    "GOOGL": [
        (date(2021, 10, 20), "DOJ antitrust suit against Google moves forward",
         "Federal judge allowed the DOJ's antitrust case against Google's search monopoly to proceed to trial.",
         EventType.LEGAL),
        (date(2023, 2, 6), "Google announces Bard AI chatbot, stock drops 8%",
         "Google rushed to announce Bard in response to ChatGPT. A demo error caused stock to drop 8%.",
         EventType.AI_TECH),
        (date(2023, 12, 6), "Google unveils Gemini, its most capable AI model",
         "Google launched Gemini AI model family, claiming superiority over GPT-4 in several benchmarks.",
         EventType.AI_TECH),
        (date(2024, 8, 5), "Federal judge rules Google is a monopolist in search",
         "Judge ruled Google illegally maintained its search monopoly through exclusive default deals worth $26B/year.",
         EventType.LEGAL),
    ],
    "TSLA": [
        (date(2021, 10, 25), "Hertz orders 100,000 Teslas, TSLA hits $1T market cap",
         "Hertz announced a massive 100,000 Tesla order. Tesla briefly crossed $1 trillion market cap.",
         EventType.PRODUCT),
        (date(2022, 4, 14), "Elon Musk launches $44B hostile bid for Twitter",
         "Musk offered to buy Twitter for $54.20/share, raising concerns about Tesla CEO distraction.",
         EventType.PERSONNEL),
        (date(2022, 10, 27), "Musk completes Twitter acquisition for $44B",
         "Musk closed the Twitter deal, becoming CEO. Tesla investors worried about leadership distraction.",
         EventType.PERSONNEL),
        (date(2024, 10, 10), "Tesla unveils Cybercab robotaxi and Robovan at We, Robot event",
         "Tesla showed off the Cybercab ($30K, no steering wheel) and a 20-seat Robovan concept.",
         EventType.PRODUCT),
    ],
    "AMZN": [
        (date(2022, 3, 9), "Amazon announces 20-for-1 stock split",
         "Amazon approved a 20:1 stock split and $10B buyback program.",
         EventType.PRODUCT),
        (date(2023, 1, 4), "Amazon announces 18,000 layoffs, largest in company history",
         "Andy Jassy announced the largest layoff in Amazon's history, primarily in corporate and tech roles.",
         EventType.PERSONNEL),
        (date(2023, 9, 25), "Amazon invests up to $4B in Anthropic",
         "Amazon announced a major investment in AI startup Anthropic, maker of Claude chatbot.",
         EventType.AI_TECH),
    ],
}


class KnownEventsFetcher(NewsFetcher):
    """
    Returns manually curated major historical events for popular tickers.
    These cover the pre-Finnhub period where API data is unavailable.
    """

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        known = _KNOWN_EVENTS.get(ticker.upper(), [])
        events = []
        for d, title, desc, etype in known:
            if start <= d <= end:
                if etype == EventType.EARNINGS:
                    events.append(EarningsEvent(
                        date=d, title=title, description=desc,
                        source="curated",
                    ))
                else:
                    events.append(MarketEvent(
                        date=d, title=title, description=desc,
                        source="curated", event_type=etype,
                    ))
        logger.info("KnownEvents: %d events for %s (%s~%s)",
                     len(events), ticker, start, end)
        return events


# ── Composite fetcher: merge multiple sources ────────────────────────────────

class CompositeNewsFetcher(NewsFetcher):
    """
    Combines multiple NewsFetcher sources into one deduplicated stream.
    Events are deduplicated by (date, title) and sorted chronologically.
    Automatically saves merged results to cache.
    """

    def __init__(self, fetchers: list[NewsFetcher]):
        self._fetchers = fetchers
        self._cache = DataCache()

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        seen: set[tuple[date, str]] = set()
        merged: list[MarketEvent] = []

        for fetcher in self._fetchers:
            try:
                events = fetcher.fetch_news(ticker, start, end)
                for ev in events:
                    key = (ev.date, ev.title)
                    if key not in seen:
                        seen.add(key)
                        merged.append(ev)
            except Exception as e:
                logger.warning("Fetcher %s failed for %s: %s",
                               type(fetcher).__name__, ticker, e)

        merged.sort(key=lambda e: e.date)
        if merged:
            self._cache.save_news(ticker, merged)

        logger.info("CompositeNews: %d total events for %s (%s~%s) from %d sources",
                     len(merged), ticker, start, end, len(self._fetchers))
        return merged


# ── Real: NewsData.io historical news fetcher ─────────────────────────────────

_NEWSDATA_ARCHIVE_URL = "https://newsdata.io/api/1/archive"
_NEWSDATA_NEWS_URL    = "https://newsdata.io/api/1/news"


class NewsDataFetcher(NewsFetcher):
    """
    Fetches news from NewsData.io with up to 5 years of history (plan-dependent).

    Plan history limits:
        Free         — no archive access (falls back to /news for recent articles)
        Basic        — 6 months
        Professional — 2 years
        Corporate    — 5 years

    API docs: https://newsdata.io/historical-news-api
    """

    PAGE_SIZE     = 50   # max for paid plans; free tier max is 10
    REQUEST_SLEEP = 1.0  # seconds between requests

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("NEWSDATA_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "NewsData API key required. "
                "Set NEWSDATA_API_KEY in your .env file. "
                "Get a key at https://newsdata.io"
            )
        self._cache = DataCache()

    @staticmethod
    def _parse_article(article: dict) -> MarketEvent | None:
        title       = (article.get("title")       or "").strip()
        source      = (article.get("source_name") or "").strip()
        description = (article.get("description") or article.get("content") or "").strip()

        if not title or not source or not description:
            return None
        if any(p in title.lower() for p in _NOISE_TITLE_PATTERNS):
            return None

        pub_date_str = article.get("pubDate") or ""
        try:
            article_date = date.fromisoformat(pub_date_str[:10])
        except (ValueError, TypeError):
            return None

        categories   = article.get("category") or []
        category_str = " ".join(categories) if isinstance(categories, list) else str(categories)
        event_type   = _classify_event(title, category_str)

        return MarketEvent(
            date        = article_date,
            title       = title,
            description = description[:500],
            source      = source,
            event_type  = event_type,
        )

    def _paginate(self, url: str, params: dict) -> list[MarketEvent]:
        """Fetch all pages from a newsdata.io endpoint and return parsed events."""
        seen:       set[tuple[date, str]] = set()
        events:     list[MarketEvent]     = []
        page_token: str | None            = None

        while True:
            if page_token:
                params["page"] = page_token
            elif "page" in params:
                del params["page"]

            time.sleep(self.REQUEST_SLEEP)

            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "success":
                msg = ""
                if isinstance(data.get("results"), dict):
                    msg = data["results"].get("message", "")
                raise RuntimeError(f"NewsData.io API error: {msg or data.get('status')}")

            for article in (data.get("results") or []):
                ev = self._parse_article(article)
                if ev is None:
                    continue
                key = (ev.date, ev.title)
                if key in seen:
                    continue
                seen.add(key)
                events.append(ev)

            page_token = data.get("nextPage")
            if not page_token:
                break

        return events

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        base_params: dict = {
            "apikey":   self.api_key,
            "q":        ticker,
            "language": "en",
            "size":     self.PAGE_SIZE,
        }

        # ── Try archive endpoint first (paid plans: 6 months – 5 years) ──────
        logger.info("Trying NewsData.io /archive for %s (%s~%s)", ticker, start, end)
        try:
            events = self._paginate(_NEWSDATA_ARCHIVE_URL, {
                **base_params,
                "from_date": str(start),
                "to_date":   str(end),
            })
            logger.info("Fetched %d events for %s via /archive", len(events), ticker)

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.warning(
                    "NewsData.io /archive returned 403 — your plan does not include "
                    "archive access (requires Basic plan or above). "
                    "Falling back to /news for recent articles only. "
                    "Upgrade at https://newsdata.io/pricing"
                )
                # ── Fallback: /news endpoint (free tier, recent articles only) ─
                try:
                    events = self._paginate(_NEWSDATA_NEWS_URL, {
                        **base_params,
                        "from_date": str(start),
                        "to_date":   str(end),
                    })
                    logger.info("Fetched %d events for %s via /news (fallback)", len(events), ticker)
                except Exception as e2:
                    logger.error("NewsData.io /news fallback also failed for %s: %s", ticker, e2)
                    return []
            else:
                logger.error("NewsData.io fetch error for %s: %s", ticker, e)
                return []

        except Exception as e:
            logger.error("NewsData.io fetch error for %s: %s", ticker, e)
            return []

        events.sort(key=lambda ev: ev.date)
        if events:
            self._cache.save_news(ticker, events)

        logger.info("Fetched %d unique news events for %s via NewsData.io", len(events), ticker)
        return events


# ── Real: SEC EDGAR 8-K fetcher ───────────────────────────────────────────────

_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SEC_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"

# Request headers required by SEC fair-access policy
_SEC_HEADERS = {
    "User-Agent": "MarketLens research@marketlens.local",
    "Accept-Encoding": "gzip, deflate",
}

_SEC_REQUEST_SLEEP = 0.12   # stay well under 10 req/sec


# 8-K Item → EventType mapping
_ITEM_EVENT_TYPE: dict[str, EventType] = {
    "1.01": EventType.PRODUCT,      # material agreement (partnership, contract)
    "1.02": EventType.PRODUCT,      # termination of material agreement
    "1.03": EventType.REGULATORY,   # bankruptcy / receivership
    "2.01": EventType.PRODUCT,      # acquisition / disposition of assets (M&A)
    "2.02": EventType.EARNINGS,     # results of operations / earnings
    "2.03": EventType.OTHER,        # off-balance-sheet obligations
    "2.04": EventType.REGULATORY,   # triggering event for obligations
    "2.05": EventType.PERSONNEL,    # departure of named executive officers
    "2.06": EventType.OTHER,        # material impairment
    "3.01": EventType.REGULATORY,   # notice of delisting
    "3.02": EventType.REGULATORY,   # unregistered sale of equity
    "3.03": EventType.REGULATORY,   # material modifications to rights of security holders
    "4.01": EventType.REGULATORY,   # changes in registrant's certifying accountant
    "4.02": EventType.REGULATORY,   # non-reliance on financial statements
    "5.01": EventType.PERSONNEL,    # changes in control
    "5.02": EventType.PERSONNEL,    # departure / appointment of directors/officers
    "5.03": EventType.REGULATORY,   # amendments to articles of incorporation
    "5.04": EventType.REGULATORY,   # temporary suspension of trading under employee plans
    "5.05": EventType.REGULATORY,   # amendment to code of ethics
    "5.06": EventType.REGULATORY,   # change in shell company status
    "5.07": EventType.OTHER,        # submission of matters to a vote
    "5.08": EventType.OTHER,        # shareholder director nominations
    "6.01": EventType.OTHER,
    "6.02": EventType.OTHER,
    "6.03": EventType.OTHER,
    "6.04": EventType.OTHER,
    "6.05": EventType.OTHER,
    "7.01": EventType.OTHER,        # Reg FD disclosure
    "8.01": EventType.OTHER,        # other events (catch-all)
    "9.01": EventType.OTHER,        # financial statements / exhibits (skip below)
}

# Items with no meaningful narrative — skip when building description
_SEC_SKIP_ITEMS = {"9.01"}

_cik_cache: dict[str, str] = {}


def _item_to_event_type(items_str: str) -> EventType:
    """
    items_str is the raw 'items' field from EDGAR, e.g. '2.02,9.01' or '5.02'.
    Return the highest-priority EventType found; fall back to OTHER.
    """
    priority_order = [
        EventType.EARNINGS, EventType.PERSONNEL, EventType.PRODUCT,
        EventType.REGULATORY, EventType.OTHER,
    ]
    found: set[EventType] = set()
    for item in items_str.replace(";", ",").split(","):
        item = item.strip()
        if item in _ITEM_EVENT_TYPE and item not in _SEC_SKIP_ITEMS:
            found.add(_ITEM_EVENT_TYPE[item])
    for et in priority_order:
        if et in found:
            return et
    return EventType.OTHER


def _item_to_title(items_str: str, company_name: str) -> str:
    """Human-readable title derived from Item codes."""
    _ITEM_LABEL: dict[str, str] = {
        "1.01": "Entry into material agreement",
        "1.02": "Termination of material agreement",
        "1.03": "Bankruptcy or receivership",
        "2.01": "Completion of acquisition or disposition",
        "2.02": "Results of operations / earnings release",
        "2.04": "Triggering events for obligations",
        "3.01": "Notice of delisting",
        "4.01": "Change in certifying accountant",
        "4.02": "Non-reliance on prior financial statements",
        "5.01": "Change in control",
        "5.02": "Change in directors or principal officers",
        "5.03": "Amendment to articles of incorporation",
        "5.05": "Amendment to code of ethics",
        "7.01": "Regulation FD disclosure",
        "8.01": "Other material event",
    }
    labels = []
    for item in items_str.replace(";", ",").split(","):
        item = item.strip()
        if item in _ITEM_LABEL:
            labels.append(_ITEM_LABEL[item])
    if labels:
        return f"{company_name}: {labels[0]}"
    return f"{company_name}: SEC 8-K filing"


def _lookup_cik(ticker: str) -> str | None:
    """Return zero-padded 10-digit CIK for a ticker, or None if not found."""
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    time.sleep(_SEC_REQUEST_SLEEP)
    try:
        resp = requests.get(_SEC_TICKERS_URL, headers=_SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _cik_cache[ticker] = cik
                return cik
    except Exception as e:
        logger.error("CIK lookup failed for %s: %s", ticker, e)
    return None


def _safe_date(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


class SECFetcher(NewsFetcher):
    """
    Fetches 8-K filings from SEC EDGAR for a given ticker.
    Uses reportDate as the event date (= when the event occurred, not when
    the filing was submitted), so event-anomaly linking stays accurate.
    No API key required. Rate limit: 10 req/sec (enforced via sleep).

    8-K Item → EventType mapping:
        1.01  Entry into material agreement       → PRODUCT   (partnerships, contracts)
        2.02  Results of operations (earnings)    → EARNINGS
        5.02  Director/officer departure/appoint  → PERSONNEL
        (see _ITEM_EVENT_TYPE for full mapping)
    """

    def __init__(self):
        self._cache = DataCache()

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        ticker = ticker.upper()

        cik = _lookup_cik(ticker)
        if not cik:
            logger.error("Could not resolve CIK for %s — no SEC events fetched", ticker)
            return []

        logger.info("Fetching SEC 8-K filings for %s (CIK %s) %s~%s", ticker, cik, start, end)

        time.sleep(_SEC_REQUEST_SLEEP)
        try:
            resp = requests.get(
                _SEC_SUBMISSIONS_URL.format(cik=cik),
                headers=_SEC_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("SEC submissions fetch failed for %s: %s", ticker, e)
            return []

        company_name = data.get("name", ticker)
        recent       = data.get("filings", {}).get("recent", {})

        forms        = recent.get("form",            [])
        report_dates = recent.get("reportDate",      [])
        filing_dates = recent.get("filingDate",      [])
        items_list   = recent.get("items",           [])
        accessions   = recent.get("accessionNumber", [])

        events: list[MarketEvent] = []
        seen:   set[str]          = set()

        for i, form in enumerate(forms):
            if form != "8-K":
                continue

            # Use reportDate; fall back to filingDate if missing
            raw_date = report_dates[i] if i < len(report_dates) and report_dates[i] else (
                filing_dates[i] if i < len(filing_dates) else None
            )
            if not raw_date:
                continue
            try:
                event_date = date.fromisoformat(raw_date)
            except ValueError:
                continue

            if not (start <= event_date <= end):
                continue

            items_str  = items_list[i] if i < len(items_list) else ""
            accession  = accessions[i] if i < len(accessions) else ""

            # Skip pure exhibit filings with no narrative content
            if not items_str or all(
                it.strip() in _SEC_SKIP_ITEMS
                for it in items_str.replace(";", ",").split(",")
                if it.strip()
            ):
                continue

            event_type  = _item_to_event_type(items_str)
            title       = _item_to_title(items_str, company_name)
            description = (
                f"SEC Form 8-K filed by {company_name} "
                f"(Items: {items_str}). "
                f"Accession: {accession.replace('-', '')}."
            )

            key = f"{event_date}|{items_str}|{accession}"
            if key in seen:
                continue
            seen.add(key)

            events.append(MarketEvent(
                date        = event_date,
                title       = title,
                description = description,
                source      = "SEC EDGAR",
                event_type  = event_type,
            ))

        events.sort(key=lambda e: e.date)

        # Also fetch additional filing pages if company has > 40 recent filings
        older_events = self._fetch_older_filings(
            cik, company_name, ticker, start, end, data
        )
        if older_events:
            combined = {(e.date, e.title): e for e in events}
            for e in older_events:
                combined.setdefault((e.date, e.title), e)
            events = sorted(combined.values(), key=lambda e: e.date)

        if events:
            self._cache.save_news(ticker, events)

        logger.info("Fetched %d SEC 8-K events for %s", len(events), ticker)
        return events

    def _fetch_older_filings(
        self,
        cik: str,
        company_name: str,
        ticker: str,
        start: date,
        end: date,
        data: dict,
    ) -> list[MarketEvent]:
        """Fetch additional filing pages for companies with deep history."""
        filing_files = data.get("filings", {}).get("files", [])
        if not filing_files:
            return []

        events: list[MarketEvent] = []
        seen:   set[str]          = set()

        for file_info in filing_files:
            name = file_info.get("name", "")
            if not name:
                continue

            time.sleep(_SEC_REQUEST_SLEEP)
            try:
                url  = f"https://data.sec.gov/submissions/{name}"
                resp = requests.get(url, headers=_SEC_HEADERS, timeout=20)
                resp.raise_for_status()
                page = resp.json()
            except Exception as e:
                logger.warning("Failed to fetch older filing page %s: %s", name, e)
                continue

            forms        = page.get("form",            [])
            report_dates = page.get("reportDate",      [])
            filing_dates = page.get("filingDate",      [])
            items_list   = page.get("items",           [])
            accessions   = page.get("accessionNumber", [])

            # Stop fetching pages once all filings are older than start
            page_dates = [
                date.fromisoformat(d) for d in report_dates if d and _safe_date(d)
            ]
            if page_dates and max(page_dates) < start:
                break

            for i, form in enumerate(forms):
                if form != "8-K":
                    continue

                raw_date = report_dates[i] if i < len(report_dates) and report_dates[i] else (
                    filing_dates[i] if i < len(filing_dates) else None
                )
                if not raw_date:
                    continue
                try:
                    event_date = date.fromisoformat(raw_date)
                except ValueError:
                    continue

                if not (start <= event_date <= end):
                    continue

                items_str = items_list[i] if i < len(items_list) else ""
                accession = accessions[i] if i < len(accessions) else ""

                if not items_str or all(
                    it.strip() in _SEC_SKIP_ITEMS
                    for it in items_str.replace(";", ",").split(",")
                    if it.strip()
                ):
                    continue

                event_type  = _item_to_event_type(items_str)
                title       = _item_to_title(items_str, company_name)
                description = (
                    f"SEC Form 8-K filed by {company_name} "
                    f"(Items: {items_str}). "
                    f"Accession: {accession.replace('-', '')}."
                )

                key = f"{event_date}|{items_str}|{accession}"
                if key in seen:
                    continue
                seen.add(key)

                events.append(MarketEvent(
                    date        = event_date,
                    title       = title,
                    description = description,
                    source      = "SEC EDGAR",
                    event_type  = event_type,
                ))

        return events


# ── Real: Guardian + NYT news fetcher ────────────────────────────────────────

_GUARDIAN_URL = "https://content.guardianapis.com/search"
_NYT_URL      = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

_REQUEST_SLEEP_NYT      = 6.5   # NYT: 10 req/min → 1 per 6s (with buffer)
_REQUEST_SLEEP_GUARDIAN = 0.1   # Guardian: 12 req/sec

# Ticker → search terms used as API query (broad, to maximise recall)
_TICKER_QUERY: dict[str, str] = {
    "META":  '"Meta" OR "Facebook" OR "Mark Zuckerberg" OR "Instagram" OR "WhatsApp"',
    "AAPL":  '"Apple" OR "Tim Cook" OR "iPhone" OR "iOS"',
    "NVDA":  '"Nvidia" OR "Jensen Huang" OR "H100" OR "Blackwell"',
    "AMZN":  '"Amazon" OR "AWS" OR "Andy Jassy"',
    "GOOGL": '"Google" OR "Alphabet" OR "Sundar Pichai" OR "Gemini"',
    "TSLA":  '"Tesla" OR "Elon Musk" OR "Cybertruck"',
    "MSFT":  '"Microsoft" OR "Satya Nadella" OR "Azure" OR "Copilot"',
}


def _query_for(ticker: str) -> str:
    return _TICKER_QUERY.get(ticker.upper(), f'"{ticker}"')


def _is_relevant(headline: str, description: str, ticker: str) -> bool:
    """Drop articles that don't mention the company by name.
    Uses _TICKER_KEYWORDS from module2 — the single source of truth.
    Falls back to True for tickers not in the list."""
    from module2_anomaly_detector import _TICKER_KEYWORDS  # lazy import avoids circular dep
    kws = _TICKER_KEYWORDS.get(ticker.upper())
    if not kws:
        return True
    text = f"{headline} {description}".lower()
    return any(kw in text for kw in kws)


def _parse_guardian_article(article: dict, ticker: str) -> MarketEvent | None:
    headline    = (article.get("webTitle") or "").strip()
    source      = "The Guardian"
    description = (article.get("fields", {}).get("trailText") or "").strip()

    if not headline or not description:
        return None
    if any(p in headline.lower() for p in _NOISE_TITLE_PATTERNS):
        return None
    if not _is_relevant(headline, description, ticker):
        return None

    pub_date_str = article.get("webPublicationDate", "")[:10]
    try:
        article_date = date.fromisoformat(pub_date_str)
    except ValueError:
        return None

    category   = article.get("sectionName", "")
    event_type = _classify_event(headline, category)

    return MarketEvent(
        date        = article_date,
        title       = headline,
        description = description[:500],
        source      = source,
        event_type  = event_type,
    )


def _parse_nyt_article(article: dict, ticker: str) -> MarketEvent | None:
    _headline   = article.get("headline") or {}
    headline    = (_headline.get("main") or _headline.get("print_headline") or "").strip()
    source      = "New York Times"
    description = (article.get("abstract") or article.get("snippet") or "").strip()

    if not headline or not description:
        return None
    if any(p in headline.lower() for p in _NOISE_TITLE_PATTERNS):
        return None
    if not _is_relevant(headline, description, ticker):
        return None

    pub_date_str = (article.get("pub_date") or "")[:10]
    try:
        article_date = date.fromisoformat(pub_date_str)
    except ValueError:
        return None

    category   = article.get("section_name", "")
    event_type = _classify_event(headline, category)

    return MarketEvent(
        date        = article_date,
        title       = headline,
        description = description[:500],
        source      = source,
        event_type  = event_type,
    )


def _yearly_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into ≤1-year chunks (NYT cap avoidance)."""
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(date(cur.year, 12, 31), end)
        chunks.append((cur, chunk_end))
        cur = date(cur.year + 1, 1, 1)
    return chunks


class NewsApiFetcher(NewsFetcher):
    """
    Fetches articles from The Guardian and New York Times for a given ticker + date range.
    Uses whichever keys are available; skips a source gracefully if its key
    is missing or returns an error.

    Rate limits:
        Guardian : 5,000 req/day, 12 req/sec
        NYT      : 4,000 req/day, 10 req/min (enforced via sleep)

    API keys (both free, no credit card):
        GUARDIAN_API_KEY: https://open-platform.theguardian.com/access/support-api/
        NYT_API_KEY:      https://developer.nytimes.com/get-started
    """

    def __init__(
        self,
        guardian_key: str | None = None,
        nyt_key:      str | None = None,
    ):
        self.guardian_key = guardian_key or os.environ.get("GUARDIAN_API_KEY", "")
        self.nyt_key      = nyt_key      or os.environ.get("NYT_API_KEY",      "")

        if not self.guardian_key and not self.nyt_key:
            raise ValueError(
                "At least one API key required.\n"
                "  GUARDIAN_API_KEY: https://open-platform.theguardian.com/access/support-api/\n"
                "  NYT_API_KEY:      https://developer.nytimes.com/get-started\n"
                "Both are free — no credit card required."
            )
        self._cache = DataCache()

    def _fetch_guardian(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        if not self.guardian_key:
            logger.info("Guardian key not set — skipping")
            return []

        query    = _query_for(ticker)
        events:  list[MarketEvent]     = []
        seen:    set[tuple[date, str]] = set()
        page     = 1
        total_pages = 1

        logger.info("Fetching Guardian articles for %s (%s~%s)", ticker, start, end)

        while page <= total_pages:
            time.sleep(_REQUEST_SLEEP_GUARDIAN)
            try:
                resp = requests.get(_GUARDIAN_URL, params={
                    "q":            query,
                    "from-date":    str(start),
                    "to-date":      str(end),
                    "order-by":     "oldest",
                    "page-size":    200,
                    "page":         page,
                    "show-fields":  "trailText",
                    "api-key":      self.guardian_key,
                }, timeout=20)
                resp.raise_for_status()
                data = resp.json().get("response", {})
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                if status == 401:
                    logger.error("Guardian API key invalid (401). Check GUARDIAN_API_KEY.")
                elif status == 429:
                    logger.warning("Guardian rate limit hit — stopping pagination")
                else:
                    logger.error("Guardian HTTP error %s: %s", status, e)
                break
            except Exception as e:
                logger.error("Guardian fetch error: %s", e)
                break

            total_pages = data.get("pages", 1)
            for article in data.get("results", []):
                ev = _parse_guardian_article(article, ticker)
                if ev is None:
                    continue
                key = (ev.date, ev.title)
                if key in seen:
                    continue
                seen.add(key)
                events.append(ev)

            page += 1

        logger.info("Guardian: %d articles for %s", len(events), ticker)
        return events

    def _fetch_nyt(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        if not self.nyt_key:
            logger.info("NYT key not set — skipping")
            return []

        query   = _query_for(ticker)
        events: list[MarketEvent]     = []
        seen:   set[tuple[date, str]] = set()

        # NYT caps at 100 pages (1,000 results) per query.
        # For multi-year ranges, split into yearly chunks to stay under the cap.
        chunks = _yearly_chunks(start, end)
        logger.info("Fetching NYT articles for %s in %d yearly chunks", ticker, len(chunks))

        for chunk_start, chunk_end in chunks:
            page = 0
            while True:
                time.sleep(_REQUEST_SLEEP_NYT)
                try:
                    resp = requests.get(_NYT_URL, params={
                        "q":          query,
                        "begin_date": chunk_start.strftime("%Y%m%d"),
                        "end_date":   chunk_end.strftime("%Y%m%d"),
                        "sort":       "oldest",
                        "page":       page,
                        "fl":         "headline,abstract,snippet,pub_date,section_name",
                        "api-key":    self.nyt_key,
                    }, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else "?"
                    if status == 401:
                        logger.error("NYT API key invalid (401). Check NYT_API_KEY.")
                    elif status == 429:
                        logger.warning("NYT rate limit hit — pausing 60s")
                        time.sleep(60)
                        continue
                    else:
                        logger.error("NYT HTTP error %s: %s", status, e)
                    break
                except Exception as e:
                    logger.error("NYT fetch error: %s", e)
                    break

                docs = data.get("response", {}).get("docs") or []

                for article in docs:
                    ev = _parse_nyt_article(article, ticker)
                    if ev is None:
                        continue
                    key = (ev.date, ev.title)
                    if key in seen:
                        continue
                    seen.add(key)
                    events.append(ev)

                # NYT doesn't reliably return meta.hits — paginate until empty page
                # or 100-page hard cap (1,000 results per chunk)
                if len(docs) < 10 or page >= 99:
                    break

                page += 1

        logger.info("NYT: %d articles for %s", len(events), ticker)
        return events

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        ticker = ticker.upper()

        guardian_events = self._fetch_guardian(ticker, start, end)
        nyt_events      = self._fetch_nyt(ticker, start, end)

        # Merge, deduplicate by (date, title)
        combined: dict[tuple[date, str], MarketEvent] = {}
        for ev in guardian_events + nyt_events:
            combined.setdefault((ev.date, ev.title), ev)

        events = sorted(combined.values(), key=lambda e: e.date)

        if events:
            self._cache.save_news(ticker, events)

        logger.info(
            "NewsApiFetcher total: %d events for %s (%d Guardian + %d NYT)",
            len(events), ticker, len(guardian_events), len(nyt_events),
        )
        return events