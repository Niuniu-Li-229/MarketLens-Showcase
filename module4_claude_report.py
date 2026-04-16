"""
Module 4 — AI Report Generator

        PromptBuilder (abstract)
        ├── StandardReportBuilder    ← 3-section analyst report
        └── RiskReportBuilder        ← downside / volatility focused

        ReportGenerator              ← calls OpenAI GPT-4o

pip install openai python-dotenv
"""

import os
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv
from models import AnalysisResult

load_dotenv()


# ── Abstract base ─────────────────────────────────────────────────────────────

class PromptBuilder(ABC):
    @abstractmethod
    def build(self, result: AnalysisResult) -> str: ...


# ── Standard report builder ───────────────────────────────────────────────────

class StandardReportBuilder(PromptBuilder):
    SYSTEM_PROMPT = (
        "You are a senior financial analyst. Write clear, direct reports. "
        "Avoid generic disclaimers. Be specific about events and their likely causes. "
        "Do NOT use markdown formatting (no **, no ##). "
        "Structure your response with exactly these section labels on their own lines: "
        "PERFORMANCE:, ANOMALIES:, OUTLOOK: "
        "Each section must have exactly 3 bullet lines starting with '• '. "
        "Each bullet must be a single sentence under 100 characters. No prose paragraphs."
    )

    def build(self, result: AnalysisResult) -> str:
        sections = [
            self._header(result),
            self._anomalies(result),
            self._intelligence(result),
            self._instruction(),
        ]
        return "\n\n".join(s for s in sections if s)

    def _header(self, r: AnalysisResult) -> str:
        return (
            f"Stock: {r.ticker}\n"
            f"Period: {r.start_date} to {r.end_date}\n"
            f"Total return: {r.total_return:+.2f}%\n"
            f"Anomalies detected: {r.anomaly_count()}"
        )

    def _anomalies(self, r: AnalysisResult) -> str:
        if not r.anomalies:
            return "No significant anomalies detected."
        lines = ["Anomalous trading days (top 10 by magnitude):"]
        top10 = sorted(r.anomalies,
                       key=lambda a: abs(a.percent_change),
                       reverse=True)[:10]
        for i, a in enumerate(sorted(top10, key=lambda a: a.date), 1):
            lines.append(f"\n  {i}. {a.date} — {a.percent_change:+.2f}%")
            lines.append(f"     {a.comment}")
            for e in a.related_events[:3]:
                lines.append(f"     [{e.event_type.value}] {e.title} ({e.source})")
        return "\n".join(lines)

    def _intelligence(self, r: AnalysisResult) -> str:
        parts = []
        if r.sentiment_label and r.sentiment_score is not None:
            parts.append(
                f"News sentiment: {r.sentiment_label} "
                f"(score: {r.sentiment_score:+.2f})"
            )
        if r.predicted_price is not None:
            price = float(r.predicted_price)
            parts.append(f"Transformer model predicted next close: ${price:.2f}")
        return "\n".join(parts) if parts else ""

    def _instruction(self) -> str:
        return (
            "Write a structured analyst report with exactly 3 sections.\n"
            "Each section: the label on its own line, then exactly 3 bullet lines "
            "starting with '• '. Each bullet is one sentence under 100 characters.\n\n"
            "PERFORMANCE: (3 bullets on overall return and key drivers)\n"
            "ANOMALIES: (3 bullets on the most significant anomaly events and their "
            "likely causes — reference the two-tier detection where relevant)\n"
            "OUTLOOK: (3 bullets on sentiment, model prediction, and forward-looking view)\n\n"
            "Be specific. Reference the events listed above. Plain text only. No markdown."
        )


# ── Risk-focused report builder ───────────────────────────────────────────────

class RiskReportBuilder(PromptBuilder):
    """Variant focused on downside risk, volatility, and tail events."""

    SYSTEM_PROMPT = (
        "You are a risk analyst specialising in equity volatility. "
        "Focus on downside scenarios, tail risks, and volatility drivers. "
        "Be specific and quantitative. Plain text only. No markdown."
    )

    def build(self, result: AnalysisResult) -> str:
        header = (
            f"Stock: {result.ticker}\n"
            f"Period: {result.start_date} to {result.end_date}\n"
            f"Total return: {result.total_return:+.2f}%\n"
            f"Anomalies detected: {result.anomaly_count()}\n"
        )
        losses = sorted(
            [a for a in result.anomalies if a.percent_change < 0],
            key=lambda a: a.percent_change,
        )[:5]
        loss_lines = "\n".join(
            f"  {a.date}: {a.percent_change:+.2f}%  —  {a.comment[:120]}"
            for a in losses
        ) or "  None recorded."

        intel = []
        if result.sentiment_label and result.sentiment_score is not None:
            intel.append(f"Current sentiment: {result.sentiment_label} "
                         f"({result.sentiment_score:+.2f})")
        if result.predicted_price is not None:
            intel.append(f"Predicted next close: ${float(result.predicted_price):.2f}")

        return (
            f"{header}\n"
            f"Largest drawdowns:\n{loss_lines}\n\n"
            + ("\n".join(intel) + "\n\n" if intel else "")
            + "Write a risk-focused report with exactly 3 sections:\n"
              "RISK PROFILE: (key volatility drivers and beta characteristics)\n"
              "DOWNSIDE EVENTS: (the worst anomalies and their root causes)\n"
              "RISK OUTLOOK: (forward-looking risk assessment and key watch items)\n"
              "Plain text only. No markdown. No bullet points."
        )


# ── Report generator ──────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Calls OpenAI API to generate a report from the AnalysisResult.
    Falls back to a local summary if the API call fails.

    Usage:
        generator = ReportGenerator(builder=StandardReportBuilder())
        report    = generator.generate(result)
    """

    def __init__(
        self,
        builder:    PromptBuilder = None,
        model:      str = "gpt-4o",
        max_tokens: int = 1024,
    ):
        self.builder    = builder or StandardReportBuilder()
        self.model      = model
        self.max_tokens = max_tokens

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Add it to your .env file: OPENAI_API_KEY=sk-..."
            )
        self._client = OpenAI(api_key=api_key)

    def generate(self, result: AnalysisResult) -> str:
        prompt     = self.builder.build(result)
        system_msg = getattr(self.builder, "SYSTEM_PROMPT",
                             StandardReportBuilder.SYSTEM_PROMPT)

        print(f"[Module 4] Calling {self.model}...")
        try:
            response = self._client.chat.completions.create(
                model      = self.model,
                max_tokens = self.max_tokens,
                messages   = [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt},
                ],
            )
            report = response.choices[0].message.content
            print(f"[Module 4] Report generated ({len(report)} chars).")
            return report

        except Exception as e:
            print(f"[Module 4] OpenAI call failed: {e}")
            print("[Module 4] Falling back to local summary.")
            return self._local_fallback(result)

    def _local_fallback(self, r: AnalysisResult) -> str:
        """Plain-text report generated locally when OpenAI is unavailable."""
        direction = "gained" if r.total_return >= 0 else "lost"
        sentiment = r.sentiment_label or "neutral"
        score     = r.sentiment_score or 0.0

        top = (max(r.anomalies, key=lambda a: abs(a.percent_change))
               if r.anomalies else None)
        top_str = ""
        if top:
            d2        = "surged" if top.percent_change > 0 else "dropped"
            ev_str    = (f", linked to: {top.related_events[0].title}"
                         if top.related_events else "")
            top_str   = (f"The most notable anomaly occurred on {top.date}, "
                         f"when the stock {d2} {top.percent_change:+.2f}%{ev_str}.")

        predicted_str = ""
        if r.predicted_price is not None:
            predicted_str = (f" The Transformer model predicts the next closing "
                             f"price at ${float(r.predicted_price):.2f}.")

        return (
            f"PERFORMANCE:\n"
            f"{r.ticker} {direction} {abs(r.total_return):.2f}% over the period "
            f"from {r.start_date} to {r.end_date}, with {r.anomaly_count()} "
            f"anomalous trading days detected by the two-tier funnel detector.\n\n"
            f"ANOMALIES:\n"
            f"{top_str if top_str else 'No significant anomalies were recorded.'}\n\n"
            f"OUTLOOK:\n"
            f"Current news sentiment is {sentiment} (score: {score:+.2f}).{predicted_str} "
            f"Investors should monitor upcoming catalysts and broader market conditions."
        )