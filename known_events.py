"""
known_events.py — Historical event library for anomaly annotation.

Three layers of coverage for ANY ticker:
1. Ticker-specific hardcoded events (highest quality)
2. Auto-fetched events via Claude AI (for unknown tickers)
3. Universal macro events (always included as fallback)
"""

from datetime import date, timedelta
import os
import re
import anthropic
from models import MarketEvent, EventType


# ── Universal macro events (affect ALL stocks) ────────────────────────────────

MACRO_EVENTS: list[tuple] = [
    (date(2018, 12, 24), "S&P 500 falls 20% in worst December since Great Depression",
     "Broad market selloff on Fed rate hike fears.", "Bloomberg", EventType.MACRO),

    (date(2020, 3, 16), "COVID-19 pandemic triggers market circuit breakers",
     "US markets halted as pandemic panic selling accelerates.", "Reuters", EventType.MACRO),

    (date(2020, 3, 23), "Fed announces unlimited QE to stabilise markets",
     "Federal Reserve pledges unlimited bond buying.", "WSJ", EventType.MACRO),

    (date(2021, 1, 27), "Reddit WallStreetBets short squeeze causes market chaos",
     "GameStop and other meme stocks surge; hedge funds squeezed.", "CNBC", EventType.MACRO),

    (date(2022, 1, 26), "Fed signals aggressive rate hike path, markets drop",
     "Powell hints at faster tightening; tech stocks hit hard.", "Bloomberg", EventType.MACRO),

    (date(2022, 2, 24), "Russia invades Ukraine, global markets plunge",
     "Full-scale invasion triggers commodity shock and risk-off selloff.", "Reuters", EventType.MACRO),

    (date(2022, 6, 13), "S&P 500 enters bear market on inflation shock",
     "CPI hits 40-year high; Fed accelerates rate hikes.", "WSJ", EventType.MACRO),

    (date(2023, 3, 10), "Silicon Valley Bank collapses in largest US bank failure since 2008",
     "SVB failure triggers bank contagion fears; tech stocks hit.", "Bloomberg", EventType.MACRO),

    (date(2023, 11, 1), "Fed pauses rate hikes, markets rally strongly",
     "Powell signals peak rates; S&P 500 surges 5% in November.", "Reuters", EventType.MACRO),

    (date(2024, 8, 5), "Global markets crash on Japan carry trade unwind",
     "Nikkei falls 12%; VIX spikes to 65 as yen carry trades unwind.", "Bloomberg", EventType.MACRO),

    (date(2024, 11, 6), "Trump wins US presidential election, markets surge",
     "S&P 500 hits record high on pro-business policy expectations.", "Reuters", EventType.MACRO),

    (date(2025, 4, 3), "Trump announces sweeping global tariffs, markets crash",
     "Largest single-day tariff announcement since Smoot-Hawley.", "WSJ", EventType.MACRO),

    (date(2025, 4, 9), "Trump pauses tariffs for 90 days, markets surge 10%+",
     "S&P 500 records one of its best days ever on tariff pause.", "Bloomberg", EventType.MACRO),
]


# ── Ticker-specific events ────────────────────────────────────────────────────

TICKER_EVENTS: dict[str, list[tuple]] = {

    "TSLA": [
        (date(2018, 8, 7), "Elon Musk tweets 'funding secured' at $420",
         "Musk tweet about taking Tesla private triggers SEC probe.", "SEC", EventType.REGULATORY),
        (date(2020, 2, 4), "Tesla stock surges on record Q4 deliveries",
         "Tesla beats delivery estimates; stock up 20%+ in days.", "Reuters", EventType.EARNINGS),
        (date(2020, 3, 19), "Tesla halts production amid COVID-19",
         "Fremont factory shut under shelter-in-place orders.", "Bloomberg", EventType.MACRO),
        (date(2021, 11, 8), "Elon Musk sells $5B Tesla shares after Twitter poll",
         "Musk polled followers on selling 10% of his stake.", "WSJ", EventType.OTHER),
        (date(2022, 10, 27), "Tesla drops as Musk sells shares to fund Twitter",
         "Musk sold billions in Tesla stock to finance $44B Twitter deal.", "Reuters", EventType.OTHER),
        (date(2023, 1, 26), "Tesla beats Q4 despite aggressive price cuts",
         "Margins held despite 20% price reductions globally.", "CNBC", EventType.EARNINGS),
        (date(2024, 1, 24), "Tesla misses Q4 delivery estimates",
         "Deliveries missed forecasts; stock drops sharply.", "Reuters", EventType.EARNINGS),
        (date(2024, 4, 2), "Tesla cuts global workforce by 10%",
         "CEO Musk announces layoffs as competition intensifies.", "Bloomberg", EventType.OTHER),
        (date(2024, 7, 23), "Tesla Q2 profit surges, Musk announces robotaxi",
         "Earnings beat; autonomous vehicle date revealed.", "CNBC", EventType.EARNINGS),
        (date(2024, 10, 24), "Tesla Q3 beats estimates, Cybercab revealed",
         "Margins recover; robotaxi design unveiled at event.", "Reuters", EventType.EARNINGS),
    ],

    "NVDA": [
        (date(2022, 8, 8), "NVIDIA revenue warning on gaming slowdown",
         "Pre-announced below-consensus Q2 revenue.", "Bloomberg", EventType.EARNINGS),
        (date(2023, 5, 25), "NVIDIA raises guidance on AI chip demand explosion",
         "Blowout Q1 results on data center AI demand.", "CNBC", EventType.EARNINGS),
        (date(2024, 2, 21), "NVIDIA Q4 revenue triples on AI chip demand",
         "Record revenue from H100 GPU sales.", "Reuters", EventType.EARNINGS),
        (date(2024, 5, 22), "NVIDIA announces 10-for-1 stock split after earnings beat",
         "Split announced to improve share accessibility.", "Bloomberg", EventType.EARNINGS),
        (date(2024, 8, 28), "NVIDIA Q2 beats but guidance disappoints",
         "Results beat but forward guidance fell short.", "WSJ", EventType.EARNINGS),
        (date(2024, 11, 20), "NVIDIA Blackwell chip demand described as insane",
         "CEO Jensen Huang says demand far exceeds supply.", "CNBC", EventType.PRODUCT),
    ],

    "AAPL": [
        (date(2020, 3, 23), "Apple closes all retail stores globally",
         "458 stores closed outside China amid pandemic.", "Reuters", EventType.MACRO),
        (date(2021, 1, 27), "Apple records $111B quarter on iPhone 12 demand",
         "First time Apple crossed $100B quarterly revenue.", "CNBC", EventType.EARNINGS),
        (date(2023, 1, 3), "Apple cuts iPhone production orders amid weak demand",
         "Supply chain reports reduced orders to key suppliers.", "Bloomberg", EventType.PRODUCT),
        (date(2024, 8, 14), "China orders officials to stop using iPhones",
         "Beijing expands iPhone ban to state-owned enterprises.", "WSJ", EventType.REGULATORY),
        (date(2024, 11, 1), "Apple Q4 beats on record services revenue",
         "Services segment hits all-time high.", "Reuters", EventType.EARNINGS),
        (date(2025, 2, 28), "Apple announces $500B US investment plan",
         "Largest-ever commitment to US manufacturing and jobs.", "Bloomberg", EventType.PRODUCT),
    ],

    "MSFT": [
        (date(2023, 1, 23), "Microsoft announces 10,000 layoffs amid slowdown",
         "Headcount reduction across divisions.", "Reuters", EventType.OTHER),
        (date(2023, 2, 7), "Microsoft launches ChatGPT-powered Bing search",
         "New AI search product unveiled to take on Google.", "Bloomberg", EventType.PRODUCT),
        (date(2024, 1, 30), "Microsoft Azure growth reaccelerates on AI Copilot",
         "Cloud revenue beats expectations.", "CNBC", EventType.EARNINGS),
        (date(2024, 7, 30), "Microsoft Q4 beats, Azure AI revenue up 60% YoY",
         "AI-driven cloud demand continues to accelerate.", "Reuters", EventType.EARNINGS),
    ],

    "GOOGL": [
        (date(2022, 10, 25), "Alphabet Q3 misses on ad revenue slowdown",
         "YouTube and Search revenue disappoints.", "Bloomberg", EventType.EARNINGS),
        (date(2023, 2, 8), "Google Bard AI demo contains factual error",
         "ChatGPT rival makes mistake in promo video; stock drops.", "Reuters", EventType.PRODUCT),
        (date(2024, 1, 30), "Alphabet Q4 beats on cloud and search growth",
         "Google Cloud accelerates despite AI competition.", "CNBC", EventType.EARNINGS),
        (date(2025, 4, 9), "Google gains on tariff pause as tech rallies",
         "Broad tech rally on 90-day US tariff pause.", "Bloomberg", EventType.MACRO),
    ],

    "META": [
        (date(2022, 10, 26), "Meta reports worst earnings since IPO, down 24%",
         "Metaverse losses hit $9.4B; revenue misses badly.", "Reuters", EventType.EARNINGS),
        (date(2023, 2, 1), "Meta surges 23% on 'Year of Efficiency' plan",
         "Zuckerberg announces cost cuts; stock recovers sharply.", "Bloomberg", EventType.EARNINGS),
        (date(2023, 10, 26), "Meta Q3 beats, raises Q4 guidance strongly",
         "Ad revenue rebounds; AI investments paying off.", "CNBC", EventType.EARNINGS),
        (date(2024, 4, 25), "Meta drops 15% on heavy AI spending forecast",
         "Capex guidance raised sharply; investors spooked.", "WSJ", EventType.EARNINGS),
    ],

    "AMZN": [
        (date(2022, 10, 27), "Amazon Q3 misses badly, issues weak guidance",
         "Worst earnings day in company history; down 20%.", "Reuters", EventType.EARNINGS),
        (date(2023, 4, 27), "Amazon Q1 beats on AWS reacceleration",
         "Cloud growth picks up; stock surges 10%+.", "Bloomberg", EventType.EARNINGS),
        (date(2024, 2, 1), "Amazon Q4 beats across all segments",
         "AWS, advertising, and retail all exceed estimates.", "CNBC", EventType.EARNINGS),
    ],

    "JPM": [
        (date(2020, 3, 16), "JPMorgan builds $8.3B reserves for COVID losses",
         "Bank sets aside billions anticipating pandemic loan losses.", "Bloomberg", EventType.MACRO),
        (date(2022, 4, 13), "JPMorgan Q1 profit drops 42% on war provisions",
         "Results hurt by Ukraine war reserves.", "Reuters", EventType.EARNINGS),
        (date(2023, 5, 1), "JPMorgan acquires First Republic in FDIC deal",
         "JPM wins auction for failed First Republic.", "Bloomberg", EventType.REGULATORY),
        (date(2024, 10, 11), "JPMorgan Q3 beats on strong investment banking",
         "Record investment banking fees drive earnings above estimates.", "CNBC", EventType.EARNINGS),
    ],

    "XOM": [
        (date(2022, 7, 29), "ExxonMobil Q2 profit hits record $17.9B on oil surge",
         "Highest quarterly profit in company history.", "Reuters", EventType.EARNINGS),
        (date(2023, 10, 27), "ExxonMobil acquires Pioneer Natural for $60B",
         "Largest oil deal in decades; expands Permian basin.", "Bloomberg", EventType.PRODUCT),
        (date(2024, 8, 2), "ExxonMobil Q2 misses on lower oil prices",
         "Crude price decline hurts margins.", "WSJ", EventType.EARNINGS),
    ],

    "BAC": [
        (date(2023, 3, 13), "Bank of America drops on SVB contagion fears",
         "Regional bank crisis spreads to large caps.", "Bloomberg", EventType.REGULATORY),
        (date(2023, 10, 17), "Bank of America Q3 beats on net interest income",
         "Higher rates boost lending margins.", "CNBC", EventType.EARNINGS),
        (date(2024, 7, 16), "Bank of America Q2 beats, net interest income rises",
         "Results top estimates across business lines.", "Reuters", EventType.EARNINGS),
    ],
}


# ── Lookup ────────────────────────────────────────────────────────────────────

def get_all_events_for_ticker(ticker: str) -> list[MarketEvent]:
    """
    Return all known events for a ticker:
    1. Ticker-specific hardcoded events (if available)
    2. AI-generated events (for unknown tickers, via Claude)
    3. Universal macro events (always included)
    """
    ticker = ticker.upper()

    # Layer 1: hardcoded specific events
    specific = TICKER_EVENTS.get(ticker, [])

    # Layer 2: if no specific events, ask Claude to generate them
    if not specific:
        print(f"[KnownEvents] {ticker} not in library — "
              f"fetching events via AI...")
        specific = _fetch_events_via_ai(ticker)

    # Layer 3: always add macro events
    raw = specific + MACRO_EVENTS
    return [
        MarketEvent(date=d, title=title, description=desc,
                    source=src, event_type=etype)
        for d, title, desc, src, etype in raw
    ]


def _fetch_events_via_ai(ticker: str) -> list[tuple]:
    """
    Ask Claude to generate major historical events for any ticker.
    Returns list of (date, title, description, source, EventType) tuples.
    Falls back to empty list if API call fails.
    """
    try:
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY") or
                    os.environ.get("OPENAI_API_KEY", "")
        )

        prompt = f"""List the 8 most important stock-price-moving events for {ticker} 
from 2018 to 2026. Focus on earnings surprises, major news, regulatory events, 
and macroeconomic impacts specific to this company.

Respond ONLY with a JSON array. No explanation. Format:
[
  {{
    "date": "YYYY-MM-DD",
    "title": "Short headline (max 80 chars)",
    "description": "One sentence explanation",
    "source": "News source name",
    "type": "EARNINGS|ANALYST|REGULATORY|MACRO|PRODUCT|OTHER"
  }}
]"""

        message = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 1000,
            messages   = [{"role": "user", "content": prompt}]
        )

        import json
        raw_text = message.content[0].text.strip()
        # Strip markdown code fences if present
        raw_text = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        items    = json.loads(raw_text)

        type_map = {
            "EARNINGS":   EventType.EARNINGS,
            "ANALYST":    EventType.ANALYST,
            "REGULATORY": EventType.REGULATORY,
            "MACRO":      EventType.MACRO,
            "PRODUCT":    EventType.PRODUCT,
            "OTHER":      EventType.OTHER,
        }

        events = []
        for item in items:
            try:
                events.append((
                    date.fromisoformat(item["date"]),
                    item["title"],
                    item["description"],
                    item.get("source", "AI-generated"),
                    type_map.get(item.get("type", "OTHER"), EventType.OTHER),
                ))
            except Exception:
                continue

        print(f"[KnownEvents] AI generated {len(events)} events for {ticker}.")
        return events

    except Exception as e:
        print(f"[KnownEvents] AI fetch failed for {ticker}: {e} "
              f"— using macro events only.")
        return []


def enrich_anomalies_with_known_events(
    anomalies: list,
    ticker: str,
    window_days: int = 3,
) -> list:
    """
    For each anomaly with no linked events, search known events
    within ±window_days and attach matching ones.

    Works for ANY ticker:
    - Known tickers get specific + macro events
    - Unknown tickers get macro events only (covers crashes, Fed, geopolitics)
    """
    known = get_all_events_for_ticker(ticker)
    if not known:
        return anomalies

    enriched = 0
    for anomaly in anomalies:
        if anomaly.related_events:
            continue
        for event in known:
            if abs((event.date - anomaly.date).days) <= window_days:
                anomaly.related_events.append(event)
                enriched += 1

    ticker_specific = len(TICKER_EVENTS.get(ticker.upper(), []))
    print(f"[KnownEvents] {ticker}: {ticker_specific} specific + "
          f"{len(MACRO_EVENTS)} macro events → enriched {enriched} anomalies.")
    return anomalies