"""
Module 1d — NYT + Guardian News Fetcher

Fetches news articles from The Guardian and New York Times APIs.
Covers MACRO, ANALYST, PRODUCT (non-M&A), and REGULATORY events that
don't appear in SEC 8-K filings (e.g. tariffs, rate decisions, antitrust
investigations, product launches, analyst upgrades).

Complements module1c_sec_fetcher.py:
    SEC EDGAR  → EARNINGS, PERSONNEL, PRODUCT (M&A), REGULATORY (formal filings)
    This module → MACRO, ANALYST, PRODUCT (launches), REGULATORY (investigations)

API keys required (both free, no credit card):
    GUARDIAN_API_KEY : https://open-platform.theguardian.com/access/support-api/
    NYT_API_KEY      : https://developer.nytimes.com/get-started

Rate limits:
    Guardian : 5,000 req/day, 12 req/sec
    NYT      : 4,000 req/day, 10 req/min  (enforced via sleep)

Usage:
    from module1d_news_fetcher import NewsApiFetcher
    fetcher = NewsApiFetcher()
    events = fetcher.fetch_news("META", date(2021, 1, 1), date(2026, 4, 9))

Quick test:
    python module1d_news_fetcher.py META 2021-01-01 2026-04-09
"""

import logging
import os
import time
from datetime import date, timedelta

import requests

from models import EventType, MarketEvent
from module1_data_fetcher import (
    DataCache,
    NewsFetcher,
    _NOISE_TITLE_PATTERNS,
    _classify_event,
)
from module2_anomaly_detector import _TICKER_KEYWORDS

logger = logging.getLogger(__name__)

_GUARDIAN_URL = "https://content.guardianapis.com/search"
_NYT_URL      = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

REQUEST_SLEEP_NYT      = 6.5   # NYT: 10 req/min → 1 per 6s (with buffer)
REQUEST_SLEEP_GUARDIAN = 0.1   # Guardian: 12 req/sec


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
    kws = _TICKER_KEYWORDS.get(ticker.upper())
    if not kws:
        return True
    text = f"{headline} {description}".lower()
    return any(kw in text for kw in kws)


def _parse_guardian_article(article: dict, ticker: str) -> MarketEvent | None:
    # Field mapping mirrors FinnhubNewsFetcher._parse_article variable names
    headline    = (article.get("webTitle") or "").strip()
    source      = "The Guardian"
    description = (article.get("fields", {}).get("trailText") or "").strip()

    if not headline or not source or not description:
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

    # Pass (headline, category) to _classify_event — same as Finnhub
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
    # Field mapping mirrors FinnhubNewsFetcher._parse_article variable names
    _headline   = article.get("headline") or {}
    headline    = (_headline.get("main") or _headline.get("print_headline") or "").strip()
    source      = "New York Times"
    description = (article.get("abstract") or article.get("snippet") or "").strip()

    if not headline or not source or not description:
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

    # Pass (headline, category) to _classify_event — same as Finnhub
    category   = article.get("section_name", "")
    event_type = _classify_event(headline, category)

    return MarketEvent(
        date        = article_date,
        title       = headline,
        description = description[:500],
        source      = source,
        event_type  = event_type,
    )


class NewsApiFetcher(NewsFetcher):
    """
    Fetches articles from Guardian and NYT for a given ticker + date range.
    Uses whichever keys are available; skips a source gracefully if its key
    is missing or returns an error.
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

    # ── Guardian ──────────────────────────────────────────────────────────────

    def _fetch_guardian(
        self, ticker: str, start: date, end: date
    ) -> list[MarketEvent]:
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
            time.sleep(REQUEST_SLEEP_GUARDIAN)
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

    # ── NYT ───────────────────────────────────────────────────────────────────

    def _fetch_nyt(
        self, ticker: str, start: date, end: date
    ) -> list[MarketEvent]:
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
                time.sleep(REQUEST_SLEEP_NYT)
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

    # ── Public interface ──────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _yearly_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into ≤1-year chunks (NYT cap avoidance)."""
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(date(cur.year, 12, 31), end)
        chunks.append((cur, chunk_end))
        cur = date(cur.year + 1, 1, 1)
    return chunks


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    ticker = sys.argv[1] if len(sys.argv) > 1 else "META"
    start  = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2021, 1, 1)
    end    = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2026, 4, 9)

    logging.basicConfig(level=logging.INFO)
    fetcher = NewsApiFetcher()
    events  = fetcher.fetch_news(ticker, start, end)

    print(f"\n{len(events)} news events for {ticker} ({start} ~ {end})\n")
    by_type: dict[str, int] = {}
    for ev in events:
        by_type[ev.event_type.value] = by_type.get(ev.event_type.value, 0) + 1
    print(f"Breakdown: {by_type}\n")
    for ev in events[:15]:
        print(f"  {ev.date}  [{ev.event_type.value:12s}]  [{ev.source:15s}]  {ev.title[:70]}")
    if len(events) > 15:
        print(f"  … and {len(events) - 15} more")
