"""
Module 1 — Data Fetcher
Supports any ticker via yfinance + NewsAPI + yfinance built-in news.
"""

from abc import ABC, abstractmethod
from datetime import date, timedelta
import os
import requests
import datetime as dt
from dotenv import load_dotenv
import yfinance as yf
from models import PricePoint, MarketEvent, EventType

load_dotenv()


class DataFetcher(ABC):
    @abstractmethod
    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]: ...
    @abstractmethod
    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]: ...


class MockDataFetcher(DataFetcher):
    """Hardcoded NVDA sample data for quick testing."""

    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]:
        return [
            PricePoint(date(2025, 9, 2), 170.00, 172.38, 167.22, 170.78, 231_160_000),
            PricePoint(date(2025, 9, 3), 171.06, 172.41, 168.88, 170.62, 164_420_000),
            PricePoint(date(2025, 9, 4), 170.57, 171.84, 169.41, 171.66, 141_670_000),
            PricePoint(date(2025, 9, 5), 168.03, 169.03, 164.07, 167.02, 224_440_000),
        ]

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        return [
            MarketEvent(date(2025, 9, 2),
                "Analysts raise NVIDIA price targets ahead of earnings",
                "Wall Street analysts raised targets citing strong data center demand.",
                "Bloomberg", EventType.ANALYST),
            MarketEvent(date(2025, 9, 3),
                "NVIDIA reports record Q3 earnings, beats estimates",
                "Record revenue driven by surging demand for H100 and B100 AI chips.",
                "Reuters", EventType.EARNINGS),
            MarketEvent(date(2025, 9, 5),
                "Broad market sell-off on Fed rate concerns",
                "Investors rotated out of tech stocks amid rates fears.",
                "WSJ", EventType.MACRO),
        ]


class YFinanceNewsFetcher(DataFetcher):
    """
    Real fetcher: yfinance (prices) + yfinance news + NewsAPI.
    Works for any ticker without configuration.
    pip install yfinance requests python-dotenv
    """

    _NEWSAPI_BASE    = "https://newsapi.org/v2/everything"
    _MAX_ARTICLES    = 100
    _FREE_TIER_LIMIT = 29

    _KEYWORD_MAP = {
        EventType.EARNINGS:   ["earnings", "eps", "revenue", "profit",
                               "beat", "miss", "quarterly", "results", "guidance"],
        EventType.ANALYST:    ["analyst", "upgrade", "downgrade",
                               "price target", "rating", "overweight", "buy rating"],
        EventType.REGULATORY: ["sec", "fine", "lawsuit", "regulation",
                               "ban", "investigation", "antitrust", "penalty", "probe"],
        EventType.MACRO:      ["fed", "federal reserve", "interest rate",
                               "inflation", "gdp", "recession", "tariff", "cpi", "jobs"],
        EventType.PRODUCT:    ["launch", "product", "release", "announced",
                               "unveil", "chip", "model", "deal", "acquisition"],
    }

    def __init__(self, news_api_key: str | None = None):
        self._api_key = news_api_key or os.environ.get("NEWSAPI_KEY", "")

    # ── Prices ────────────────────────────────────────────────────────────────

    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]:
        df = yf.download(
            ticker,
            start             = str(start),
            end               = str(end + timedelta(days=1)),
            auto_adjust       = True,
            progress          = False,
            multi_level_index = False,
        )
        if df.empty:
            print(f"[Module 1] No price data for {ticker}.")
            return []

        prices = []
        for ts, row in df.iterrows():
            try:
                prices.append(PricePoint(
                    date   = ts.date(),
                    open   = float(row["Open"]),
                    high   = float(row["High"]),
                    low    = float(row["Low"]),
                    close  = float(row["Close"]),
                    volume = int(row["Volume"]),
                ))
            except (ValueError, KeyError):
                pass

        print(f"[Module 1] Fetched {len(prices)} price points for {ticker}.")
        return prices

    # ── News ──────────────────────────────────────────────────────────────────

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        events = []
        events.extend(self._fetch_yfinance_news(ticker))
        if self._api_key:
            events.extend(self._fetch_newsapi(ticker, start, end))

        # Deduplicate by title prefix
        seen, unique = set(), []
        for e in events:
            key = e.title.lower()[:60]
            if key not in seen:
                seen.add(key)
                unique.append(e)

        unique = [e for e in unique if start <= e.date <= end]
        print(f"[Module 1] {len(unique)} news events for {ticker}.")
        return unique

    def _fetch_yfinance_news(self, ticker: str) -> list[MarketEvent]:
        try:
            raw    = yf.Ticker(ticker).news or []
            events = []
            for item in raw:
                try:
                    content   = item.get("content", item)
                    title     = (content.get("title") or "").strip()
                    if not title:
                        continue
                    summary   = (content.get("summary") or
                                 content.get("description") or title).strip()
                    provider  = content.get("provider", {})
                    publisher = (provider.get("displayName") or
                                 provider.get("name") or "Yahoo Finance").strip()
                    raw_date  = (content.get("pubDate") or
                                 content.get("displayTime") or "")
                    pub_date  = (dt.date.fromisoformat(raw_date[:10])
                                 if raw_date else dt.date.today())
                    events.append(MarketEvent(
                        date        = pub_date,
                        title       = title,
                        description = summary,
                        source      = publisher,
                        event_type  = self._classify(title, summary),
                    ))
                except Exception:
                    continue
            return events
        except Exception as e:
            print(f"[Module 1] yfinance news failed: {e}")
            return []

    def _fetch_newsapi(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        earliest = date.today() - timedelta(days=self._FREE_TIER_LIMIT)
        clamped  = max(start, earliest)
        try:
            resp = requests.get(self._NEWSAPI_BASE, params={
                "q": ticker, "from": str(clamped), "to": str(end),
                "language": "en", "sortBy": "relevancy",
                "pageSize": self._MAX_ARTICLES, "apiKey": self._api_key,
            }, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
        except Exception as e:
            print(f"[Module 1] NewsAPI failed: {e}")
            return []

        events = []
        for a in articles:
            try:
                title  = (a.get("title") or "").strip()
                desc   = (a.get("description") or "").strip()
                source = (a.get("source", {}).get("name") or "").strip()
                raw_d  = a.get("publishedAt", "")[:10]
                if not title or not source:
                    continue
                events.append(MarketEvent(
                    date        = date.fromisoformat(raw_d),
                    title       = title,
                    description = desc or title,
                    source      = source,
                    event_type  = self._classify(title, desc),
                ))
            except Exception:
                continue
        return events

    def _classify(self, title: str, description: str) -> EventType:
        text = (title + " " + description).lower()
        for event_type, keywords in self._KEYWORD_MAP.items():
            if any(kw in text for kw in keywords):
                return event_type
        return EventType.OTHER