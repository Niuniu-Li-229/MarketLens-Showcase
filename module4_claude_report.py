"""
Module 4 — AI Report Generator
Owner: Person 4

Switched ReportGenerator to use OpenAI API instead of Anthropic.
PromptBuilder classes are unchanged — only the API call layer changed.

        PromptBuilder (abstract)
        ├── StandardReportBuilder    ← unchanged
        └── RiskReportBuilder        ← TODO

        ReportGenerator              ← now uses OpenAI
"""

import os
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv

from models import AnalysisResult

load_dotenv()


# ── Prompt builders (completely unchanged) ────────────────────────────────────

class PromptBuilder(ABC):
    @abstractmethod
    def build(self, result: AnalysisResult) -> str: ...


class StandardReportBuilder(PromptBuilder):
    """
    Builds a structured 3-paragraph analyst report prompt.
    Unchanged from original.
    """

    SYSTEM_PROMPT = (
        "You are a senior financial analyst. Write clear, direct reports. "
        "Avoid generic disclaimers. Be specific about events and their likely causes."
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
        lines = ["Anomalous trading days:"]
        for i, a in enumerate(r.anomalies, 1):
            lines.append(f"\n  {i}. {a.date} — {a.percent_change:+.2f}%")
            lines.append(f"     {a.comment}")
            for e in a.related_events:
                lines.append(f"     [{e.event_type.value}] {e.title} ({e.source})")
        return "\n".join(lines)

    def _intelligence(self, r: AnalysisResult) -> str:
        parts = []
        if r.sentiment_label:
            parts.append(
                f"News sentiment: {r.sentiment_label} (score: {r.sentiment_score:+.2f})"
            )
        if r.predicted_price:
            parts.append(f"LSTM predicted next close: ${r.predicted_price:.2f}")
        return "\n".join(parts) if parts else ""

    def _instruction(self) -> str:
        return (
            "Write a 3-paragraph report:\n"
            "  1. Overall performance summary\n"
            "  2. Key anomaly events and their likely causes\n"
            "  3. Outlook based on sentiment and price prediction\n"
            "Be specific. Reference the events above directly."
        )


class RiskReportBuilder(PromptBuilder):
    """TODO: Variant focused on downside risk and volatility."""
    def build(self, result: AnalysisResult) -> str:
        raise NotImplementedError("Risk report variant not yet implemented.")


# ── Report generator (switched to OpenAI) ────────────────────────────────────

class ReportGenerator:
    """
    Calls OpenAI API to generate a report from the AnalysisResult.
    Accepts any PromptBuilder — swap the builder to change report style.

    Usage:
        generator = ReportGenerator(builder=StandardReportBuilder())
        report    = generator.generate(result)
    """

    def __init__(
        self,
        builder:    PromptBuilder,
        model:      str = "gpt-4o",   # or "gpt-3.5-turbo" for lower cost
        max_tokens: int = 1024,
    ):
        self.builder    = builder
        self.model      = model
        self.max_tokens = max_tokens
        self._client    = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    def generate(self, result: AnalysisResult) -> str:
        prompt = self.builder.build(result)
        response = self._client.chat.completions.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            messages   = [
                {"role": "system", "content": StandardReportBuilder.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content