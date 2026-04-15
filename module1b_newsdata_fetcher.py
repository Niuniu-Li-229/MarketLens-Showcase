"""
Module 1b — NewsData.io Historical News Fetcher

Standalone alternative to FinnhubNewsFetcher in module1_data_fetcher.py.
Supports up to 5 years of history depending on plan (vs Finnhub's ~1 year).

Usage:
    from module1b_newsdata_fetcher import NewsDataFetcher
    fetcher = NewsDataFetcher()   # reads NEWSDATA_API_KEY from env
    events = fetcher.fetch_news("META", date(2021, 1, 1), date(2026, 4, 9))

Plan history limits:
    Free         — no archive access (falls back to /news for recent articles)
    Basic        — 6 months
    Professional — 2 years
    Corporate    — 5 years

API docs: https://newsdata.io/historical-news-api
"""

import logging
import os
import time
from datetime import date

import requests

from module1_data_fetcher import (
    DataCache,
    NewsFetcher,
    _NOISE_TITLE_PATTERNS,
    _classify_event,
)
from models import MarketEvent

logger = logging.getLogger(__name__)

_ARCHIVE_URL = "https://newsdata.io/api/1/archive"
_NEWS_URL    = "https://newsdata.io/api/1/news"


class NewsDataFetcher(NewsFetcher):
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
            events = self._paginate(_ARCHIVE_URL, {
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
                    events = self._paginate(_NEWS_URL, {
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


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    ticker = sys.argv[1] if len(sys.argv) > 1 else "META"
    start  = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2021, 1, 1)
    end    = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2026, 4, 9)

    logging.basicConfig(level=logging.INFO)
    fetcher = NewsDataFetcher()
    events  = fetcher.fetch_news(ticker, start, end)

    print(f"\n{len(events)} events for {ticker} ({start} ~ {end})\n")
    for ev in events[:10]:
        print(f"  {ev.date}  [{ev.event_type.value:12s}]  {ev.title[:80]}")
    if len(events) > 10:
        print(f"  … and {len(events) - 10} more")
