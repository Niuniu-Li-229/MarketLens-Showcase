"""
Module 1 — Data Fetcher
Owner: Person 1

Design: Abstract base class DataFetcher defines the interface.
Concrete classes implement specific sources.
To add a new data source (e.g. Alpha Vantage, SEC filings),
add a new subclass — never modify existing ones.

        DataFetcher (abstract)
        ├── MockDataFetcher       ← used now
        └── YFinanceNewsFetcher   ← TODO: real implementation
"""

from abc import ABC, abstractmethod
from datetime import date
from models import PricePoint, MarketEvent, EventType


class DataFetcher(ABC):
    """
    Abstract interface for all data fetchers.
    Any new data source must implement both methods.
    """

    @abstractmethod
    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]:
        """Fetch OHLCV price data for the given ticker and date range."""
        ...

    @abstractmethod
    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        """Fetch relevant news/market events for the given ticker and date range."""
        ...


class MockDataFetcher(DataFetcher):
    """Hardcoded NVDA sample data. Used for testing and poster demo."""

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
    TODO: Real implementation using yfinance + NewsAPI.

    pip install yfinance requests python-dotenv
    Add NEWSAPI_KEY to .env
    """

    def __init__(self, news_api_key: str):
        self.news_api_key = news_api_key

    def fetch_prices(self, ticker: str, start: date, end: date) -> list[PricePoint]:
        # import yfinance as yf
        # df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        # return [
        #     PricePoint(
        #         date=row.Index.date(),
        #         open=row.Open, high=row.High,
        #         low=row.Low,   close=row.Close,
        #         volume=int(row.Volume)
        #     ) for row in df.itertuples()
        # ]
        raise NotImplementedError

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        # import requests
        # resp = requests.get("https://newsapi.org/v2/everything", params={
        #     "q": ticker, "from": start, "to": end,
        #     "sortBy": "relevancy", "apiKey": self.news_api_key
        # })
        # return [_parse_article(a) for a in resp.json().get("articles", [])]
        raise NotImplementedError
