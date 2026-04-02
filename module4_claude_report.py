"""
Module 4 — Claude AI Report Generator
Owner: Person 4

Design: PromptBuilder is separated from the API call.
To add a new report style (e.g. short summary, risk report, multi-stock),
add a new PromptBuilder subclass — never modify the API call logic.

        PromptBuilder (abstract)
        ├── StandardReportBuilder    ← 3-paragraph analyst report, used now
        └── RiskReportBuilder        ← TODO: risk-focused variant

        ReportGenerator              ← owns the Claude API call, not subclassed
"""

import os
from abc import ABC, abstractmethod
from string import Template
import anthropic

from models import AnalysisResult


# ── Prompt builders ───────────────────────────────────────────────────────────

class PromptBuilder(ABC):
    """
    Abstract interface for prompt construction.
    Separates prompt engineering from API call logic.
    """

    @abstractmethod
    def build(self, result: AnalysisResult) -> str:
        ...


class StandardReportBuilder(PromptBuilder):
    """
    Builds a structured 3-paragraph analyst report prompt.
    Template-based — safe to extend without breaking existing output.
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
        """Optional section — only included when Module 3 data is available."""
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
    """
    TODO: Variant focused on downside risk and volatility.
    Useful for the 'enterprise risk' use case described in the proposal.
    """

    def build(self, result: AnalysisResult) -> str:
        raise NotImplementedError("Risk report variant not yet implemented.")


# ── Report generator ──────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Owns the Claude API call. Accepts any PromptBuilder.
    To change report style: swap the builder, not this class.

    Usage:
        generator = ReportGenerator(builder=StandardReportBuilder())
        report    = generator.generate(result)
    """

    def __init__(
        self,
        builder: PromptBuilder,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ):
        self.builder    = builder
        self.model      = model
        self.max_tokens = max_tokens
        self._client    = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate(self, result: AnalysisResult) -> str:
        prompt = self.builder.build(result)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=StandardReportBuilder.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
