"""
Module 5 — Visualizer
Generates 4 poster-ready charts from pipeline output.

Changes vs previous version:
  - Chart 2 table now shows triggered detector names from module2's
    two-tier funnel (PriceLayer vs individual detector names)
  - Chart 2 annotates top 6 anomalies instead of top 5
  - Chart 3 gracefully handles Mock ForecastResult (empty arrays)
  - Chart 4 mentions two-tier detection in report card

pip install matplotlib yfinance pandas
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from datetime import date, timedelta
from models import PricePoint, AnomalyPoint, AnalysisResult

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE   = "#1d4ed8"
ORANGE = "#f97316"
PURPLE = "#7c3aed"
GRAY   = "#94a3b8"
GREEN  = "#16a34a"
RED    = "#dc2626"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "text.color":        "#1e293b",
    "axes.labelcolor":   "#1e293b",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "axes.edgecolor":    "#e2e8f0",
})


# ── Helper: robust S&P500 fetch ───────────────────────────────────────────────

def _fetch_spy(start_dt: date, end_dt: date, base_price: float):
    """
    Returns (spy_dates, spy_closes_raw, spy_norm).
    All three are empty lists on any failure — never raises.
    """
    import yfinance as yf
    try:
        end_safe  = min(end_dt + timedelta(days=1), date.today())
        df = yf.download("^GSPC",
                          start=str(start_dt), end=str(end_safe),
                          auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            print("[fetch_spy] Empty DataFrame — S&P500 unavailable.")
            return [], [], []
        closes    = df["Close"].values.flatten().astype(float)
        spy_dates = [d.date() for d in df.index]
        spy_norm  = closes / closes[0] * base_price
        print(f"[fetch_spy] OK — {len(spy_dates)} days fetched.")
        return spy_dates, closes, spy_norm
    except Exception as e:
        print(f"[fetch_spy] FAILED: {e}")
        return [], [], []


# ── Chart 1: Price + MA + Volume + S&P500 ────────────────────────────────────

def plot_price_chart(prices: list[PricePoint], ticker: str,
                     save_path: str = None) -> str:
    dates  = [p.date for p in prices]
    closes = [p.close for p in prices]
    vols   = [p.volume / 1_000_000 for p in prices]

    def rolling_mean(arr, w):
        return [np.mean(arr[max(0, i-w+1):i+1]) if i >= w-1 else None
                for i in range(len(arr))]

    ma20 = rolling_mean(closes, 20)
    ma60 = rolling_mean(closes, 60)

    spy_dates, spy_closes_raw, _ = _fetch_spy(dates[0], dates[-1], closes[0])
    has_spy = len(spy_dates) > 0

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1, 1.2]},
        facecolor="white",
    )
    fig.suptitle("MODULE 1 — OUTPUT", fontsize=9, color=GRAY, x=0.01, ha="left")

    # Panel 1: Price + MAs + S&P500 right axis
    ax1.plot(dates, closes, color=BLUE,   lw=1.5, label="Close price", zorder=3)
    ax1.plot(dates, ma20,   color=ORANGE, lw=1.0, label="MA20",  alpha=0.85)
    ax1.plot(dates, ma60,   color=PURPLE, lw=1.0, label="MA60",  alpha=0.85)
    if has_spy:
        ax1_r = ax1.twinx()
        ax1_r.plot(spy_dates, spy_closes_raw, color=GRAY, lw=1.1,
                   linestyle="--", alpha=0.65, label="S&P500 (right)")
        ax1_r.set_ylabel("S&P500", fontsize=8, color=GRAY)
        ax1_r.tick_params(colors=GRAY, labelsize=7)
        ax1_r.spines["top"].set_visible(False)
        ax1_r.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v:.0f}"))
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1_r.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, fontsize=8, framealpha=0, loc="upper left")
    else:
        ax1.legend(fontsize=8, framealpha=0, loc="upper left")

    ax1.set_title(f"{ticker} — price, moving averages & S&P500 comparison",
                  fontsize=13, fontweight="normal", pad=8, loc="left")
    ax1.set_ylabel("Price (USD)", fontsize=9, color=GRAY)
    ax1.tick_params(colors=GRAY, labelsize=8)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.spines[["left","bottom"]].set_color("#e2e8f0")
    ax1.grid(axis="y", color="#f1f5f9", linewidth=0.8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}"))

    # Panel 2: Volume
    ax2.bar(dates, vols, color=GRAY, alpha=0.5, width=1.0)
    ax2.set_ylabel("Vol (M)", fontsize=8, color=GRAY)
    ax2.tick_params(colors=GRAY, labelsize=7)
    ax2.spines[["top","right","left"]].set_visible(False)
    ax2.spines["bottom"].set_color("#e2e8f0")
    ax2.grid(axis="y", color="#f1f5f9", linewidth=0.8)

    # Panel 3: Return %
    if has_spy:
        ticker_pct = [(c - closes[0]) / closes[0] * 100 for c in closes]
        spy_pct    = [(s - spy_closes_raw[0]) / spy_closes_raw[0] * 100
                      for s in spy_closes_raw]
        ax3.plot(dates,     ticker_pct, color=BLUE, lw=1.5, label=f"{ticker} return")
        ax3.plot(spy_dates, spy_pct,    color=GRAY, lw=1.0,
                 linestyle="--", alpha=0.7, label="S&P500 return")
        ax3.axhline(0, color="#e2e8f0", lw=0.8)
        spy_map = dict(zip(spy_dates, spy_pct))
        diff = [t - spy_map.get(dt, 0) for t, dt in zip(ticker_pct, dates)]
        ax3.fill_between(dates, diff, 0, alpha=0.08, color=BLUE)
        ax3.legend(fontsize=8, framealpha=0)
        ax3.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "S&P500 data unavailable",
                 transform=ax3.transAxes, ha="center", fontsize=9, color=GRAY)
    ax3.set_ylabel("Return %", fontsize=8, color=GRAY)
    ax3.tick_params(colors=GRAY, labelsize=7)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.spines[["left","bottom"]].set_color("#e2e8f0")
    ax3.grid(axis="y", color="#f1f5f9", linewidth=0.8)

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis="x", colors=GRAY, labelsize=8)

    total_ret = (closes[-1] - closes[0]) / closes[0] * 100
    ret_color = GREEN if total_ret >= 0 else RED
    spy_ret   = ((spy_closes_raw[-1] - spy_closes_raw[0]) / spy_closes_raw[0] * 100
                 if has_spy else None)

    fig.text(0.01, -0.02, f"${closes[-1]:.2f}", fontsize=14,
             fontweight="bold", color=BLUE)
    fig.text(0.01, -0.05, "Latest close", fontsize=8, color=GRAY)
    fig.text(0.18, -0.02, f"{total_ret:+.1f}%", fontsize=14,
             fontweight="bold", color=ret_color)
    fig.text(0.18, -0.05, f"{ticker} return", fontsize=8, color=GRAY)
    if spy_ret is not None:
        fig.text(0.35, -0.02, f"{spy_ret:+.1f}%", fontsize=14,
                 fontweight="bold", color=GRAY)
        fig.text(0.35, -0.05, "S&P500 return", fontsize=8, color=GRAY)
        outperf = total_ret - spy_ret
        fig.text(0.52, -0.02, f"{outperf:+.1f}%", fontsize=14,
                 fontweight="bold", color=GREEN if outperf >= 0 else RED)
        fig.text(0.52, -0.05, "vs S&P500", fontsize=8, color=GRAY)

    plt.tight_layout()
    path = save_path or f"{ticker}_chart1_price.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Visualizer] Saved {path}")
    return path


# ── Chart 2: Anomaly detection ────────────────────────────────────────────────

def _layer_label(comment: str) -> str:
    """
    Extract a short 'layers triggered' label from the anomaly comment.
    module2's _build_comment writes 'Triggered by: <names>.' in the comment.
    """
    try:
        part = comment.split("Triggered by:")[1].split(".")[0].strip()
        if part == "PriceLayer":
            return "Price >5% (Tier 1)"
        # Shorten long funnel lists
        names = [n.strip() for n in part.split(",")]
        short = [n.split("(")[0].strip() for n in names]
        label = ", ".join(short[:3])
        if len(names) > 3:
            label += f" +{len(names)-3}"
        return label
    except Exception:
        return "—"


def plot_anomaly_chart(prices: list[PricePoint], anomalies: list[AnomalyPoint],
                       ticker: str, save_path: str = None) -> str:
    dates  = [p.date for p in prices]
    closes = [p.close for p in prices]

    top15 = sorted(anomalies, key=lambda a: abs(a.percent_change), reverse=True)[:15]
    top15 = sorted(top15, key=lambda a: a.date)

    gains  = [a for a in top15 if a.percent_change > 0]
    losses = [a for a in top15 if a.percent_change <= 0]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="white")
    fig.suptitle("MODULE 2 — OUTPUT", fontsize=9, color=GRAY, x=0.01, ha="left")

    ax.plot(dates, closes, color=BLUE, lw=1.5, label="Close price", zorder=2)
    if gains:
        ax.scatter([a.date for a in gains],
                   [a.price_point.close for a in gains],
                   color=GREEN, s=80, zorder=4, label="Positive anomaly")
    if losses:
        ax.scatter([a.date for a in losses],
                   [a.price_point.close for a in losses],
                   color=RED, s=80, zorder=4, label="Negative anomaly")

    # Annotate all 15 displayed points
    top6_ann = sorted(top15, key=lambda a: abs(a.percent_change), reverse=True)[:15]
    for a in top6_ann:
        color = GREEN if a.percent_change > 0 else RED
        ax.annotate(f"{a.percent_change:+.1f}%",
                    xy=(a.date, a.price_point.close),
                    xytext=(0, 12), textcoords="offset points",
                    fontsize=7.5, color=color, ha="center", fontweight="bold")

    ax.set_title(
        f"{ticker} — anomaly detection  "
        f"(top 15 of {len(anomalies)} detected, by magnitude)",
        fontsize=12, fontweight="normal", pad=8, loc="left")
    ax.set_ylabel("Price (USD)", fontsize=9, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#e2e8f0")
    ax.grid(axis="y", color="#f1f5f9", linewidth=0.8)
    ax.legend(fontsize=8, framealpha=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}"))
    ax.tick_params(axis="x", colors=GRAY, labelsize=8)

    plt.tight_layout()
    path = save_path or f"{ticker}_chart2_anomalies.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Visualizer] Saved {path}")
    return path


# ── Chart 3: LSTM vs Transformer ─────────────────────────────────────────────

def plot_prediction_chart(lstm_result, tf_result, ticker: str,
                          save_path: str = None) -> str:
    """
    lstm_result slot → Transformer
    tf_result slot   → TFT
    Side-by-side comparison of the two models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle("MODULE 3 — OUTPUT  |  Transformer vs TFT",
                 fontsize=9, color=GRAY, x=0.01, ha="left")

    for ax, res, label in [
        (ax1, lstm_result, "Transformer"),
        (ax2, tf_result,   "TFT"),
    ]:
        n = min(len(res.actual), len(res.predicted), len(res.test_dates))
        if n == 0:
            ax.set_title(f"{ticker} — {label}", fontsize=10,
                         fontweight="normal", pad=6, loc="left")
            ax.text(0.5, 0.5, "No data",
                    transform=ax.transAxes, ha="center", fontsize=9, color=GRAY)
            ax.axis("off")
            continue
        ax.plot(res.test_dates[:n], res.actual[:n],
                color=BLUE, lw=1.8, label="Actual price")
        ax.plot(res.test_dates[:n], res.predicted[:n],
                color=ORANGE, lw=1.5, linestyle="--", label="Prediction")
        ax.set_title(f"{ticker} — {res.model_name}",
                     fontsize=10, fontweight="normal", pad=6, loc="left")
        ax.set_ylabel("Price (USD)", fontsize=9, color=GRAY)
        ax.tick_params(colors=GRAY, labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#e2e8f0")
        ax.grid(axis="y", color="#f1f5f9", linewidth=0.8)
        ax.legend(fontsize=8, framealpha=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}"))
        ax.tick_params(axis="x", colors=GRAY, labelsize=8)
        ax.text(0.02, 0.06,
                f"MAE: ${res.mae:.2f}  |  Dir. acc: {res.dir_accuracy:.1%}",
                transform=ax.transAxes, fontsize=8.5,
                color=GRAY, fontweight="bold")

    plt.tight_layout()
    path = save_path or f"{ticker}_chart3_prediction.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Visualizer] Saved {path}")
    return path

    res = tf_result
    n   = min(len(res.actual), len(res.predicted), len(res.test_dates))

    if n == 0:
        ax.set_title(f"{ticker} — {res.model_name}",
                     fontsize=10, fontweight="normal", pad=6, loc="left")
        ax.text(0.5, 0.5, "Run with real data to see predictions\n"
                "(switch to YFinancePriceFetcher + TransformerForecaster)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=GRAY)
        ax.axis("off")
    else:
        ax.plot(res.test_dates[:n], res.actual[:n],
                color=BLUE, lw=1.8, label="Actual price")
        ax.plot(res.test_dates[:n], res.predicted[:n],
                color=ORANGE, lw=1.5, linestyle="--", label="Prediction")
        ax.set_title(f"{ticker} — {res.model_name}",
                     fontsize=10, fontweight="normal", pad=6, loc="left")
        ax.set_ylabel("Price (USD)", fontsize=9, color=GRAY)
        ax.tick_params(colors=GRAY, labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#e2e8f0")
        ax.grid(axis="y", color="#f1f5f9", linewidth=0.8)
        ax.legend(fontsize=8, framealpha=0)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v:.0f}"))
# ── Chart 4: AI report card ───────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'#{1,6}\s*',     '',    text)
    return text.strip()


def plot_report_card(result: AnalysisResult, report: str, ticker: str,
                     save_path: str = None) -> str:
    import yfinance as yf

    pe_ratio = mkt_cap = w52_pos = upside = rel_perf = None
    vix = beta = target_mean = None
    cap_label = analyst_rec = "n/a"
    rec_color = vix_color = GRAY
    beta_label = vix_label = "n/a"

    try:
        tk   = yf.Ticker(ticker)
        info = tk.info

        pe_ratio    = info.get("trailingPE")
        mkt_cap     = info.get("marketCap")
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low  = info.get("fiftyTwoWeekLow")
        current     = info.get("currentPrice") or info.get("regularMarketPrice")
        beta        = info.get("beta")
        analyst_rec = info.get("recommendationKey", "n/a").upper()
        target_mean = info.get("targetMeanPrice")

        if week52_high and week52_low and current:
            w52_pos = (current - week52_low) / (week52_high - week52_low) * 100
        if target_mean and current:
            upside = (target_mean - current) / current * 100
        if beta:
            beta_label = ("High volatility" if beta > 1.5 else
                          "Above market"    if beta > 1.0 else
                          "Below market"    if beta > 0.5 else "Low volatility")

        rec_colors = {"STRONG_BUY": GREEN, "BUY": GREEN,
                      "HOLD": ORANGE, "SELL": RED, "STRONG_SELL": RED}
        rec_color  = rec_colors.get(analyst_rec, GRAY)

        vix_hist = yf.Ticker("^VIX").history(period="5d")
        vix = round(float(vix_hist["Close"].iloc[-1]), 1) if not vix_hist.empty else None
        if vix:
            vix_label, vix_color = (("Low fear", GREEN)  if vix < 15 else
                                    ("Moderate", ORANGE)  if vix < 25 else
                                    ("High fear", RED))

        hist_stock = tk.history(period="30d")["Close"]
        hist_spy   = yf.Ticker("SPY").history(period="30d")["Close"]
        if len(hist_stock) > 1 and len(hist_spy) > 1:
            ret_stock = (hist_stock.iloc[-1] - hist_stock.iloc[0]) / hist_stock.iloc[0] * 100
            ret_spy   = (hist_spy.iloc[-1]   - hist_spy.iloc[0])   / hist_spy.iloc[0]   * 100
            rel_perf  = ret_stock - ret_spy

        if mkt_cap:
            cap_label = (f"${mkt_cap/1e12:.1f}T" if mkt_cap >= 1e12 else
                         f"${mkt_cap/1e9:.0f}B"  if mkt_cap >= 1e9  else
                         f"${mkt_cap/1e6:.0f}M")
    except Exception as e:
        print(f"[Chart 4] Market indicators fetch failed: {e}")

    # Layout: 3 rows
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.suptitle("MODULE 4 — OUTPUT", fontsize=9, color=GRAY, x=0.01, ha="left")

    gs       = gridspec.GridSpec(3, 1, figure=fig,
                                 height_ratios=[0.8, 0.65, 2.2], hspace=0.45)
    ax_stats = fig.add_subplot(gs[0])
    ax_mkt   = fig.add_subplot(gs[1])
    ax_text  = fig.add_subplot(gs[2])
    for ax in [ax_stats, ax_mkt, ax_text]:
        ax.axis("off")

    ax_stats.set_title(f"{ticker} — AI analysis report (GPT-4o)",
                       fontsize=13, fontweight="normal", pad=8, loc="left")

    # Row 1: Core stats
    sentiment  = result.sentiment_label or "neutral"
    score      = result.sentiment_score or 0.0
    predicted  = float(result.predicted_price) if result.predicted_price else 0.0
    ret        = result.total_return
    ret_color  = GREEN if ret >= 0 else RED
    sent_color = GREEN if score > 0.1 else RED if score < -0.1 else GRAY

    core_stats = [
        ("Ticker",        ticker,                        BLUE),
        ("Period return", f"{ret:+.2f}%",                ret_color),
        ("Sentiment",     f"{score:+.2f} ({sentiment})", sent_color),
        ("Predicted D5",  f"${predicted:.2f}",           PURPLE),
        ("Anomalies",     str(result.anomaly_count()),   ORANGE),
    ]
    for i, (lbl, val, col) in enumerate(core_stats):
        x = 0.01 + i * 0.20
        ax_stats.text(x, 0.88, lbl, transform=ax_stats.transAxes,
                      fontsize=8, color=GRAY)
        ax_stats.text(x, 0.45, val, transform=ax_stats.transAxes,
                      fontsize=12, fontweight="bold", color=col)
    ax_stats.plot([0, 1], [0.12, 0.12], color="#e2e8f0",
                  lw=0.8, transform=ax_stats.transAxes, clip_on=False)

    # Row 2: Market indicators
    def fmt(v, fmt_str, fallback="n/a"):
        try:    return fmt_str.format(v) if v is not None else fallback
        except: return fallback

    market_stats = [
        ("P/E ratio",
         fmt(pe_ratio, "{:.1f}x"),
         BLUE if pe_ratio and pe_ratio < 30 else ORANGE if pe_ratio else GRAY),
        ("Market cap", cap_label, BLUE),
        ("52w position",
         fmt(w52_pos, "{:.0f}% of range"),
         GREEN if w52_pos and w52_pos > 60 else
         RED   if w52_pos and w52_pos < 30 else ORANGE),
        ("Beta",
         fmt(beta, "{:.2f}") + f"  {beta_label}",
         ORANGE if beta and beta > 1.2 else GREEN),
        ("Analyst target",
         fmt(target_mean, "${:.2f}") + (f"  {upside:+.1f}% upside" if upside else ""),
         GREEN if upside and upside > 5 else
         RED   if upside and upside < -5 else GRAY),
        ("Analyst rating", analyst_rec, rec_color),
        ("VIX (fear)",
         fmt(vix, "{:.1f}") + f"  {vix_label}", vix_color),
        ("vs S&P500 (30d)",
         fmt(rel_perf, "{:+.1f}%") + "  rel. perf",
         GREEN if rel_perf and rel_perf > 0 else RED),
    ]
    for i, (lbl, val, col) in enumerate(market_stats):
        x = 0.01 + (i % 4) * 0.25
        y = 0.88 if i < 4 else 0.38
        ax_mkt.text(x, y,        lbl, transform=ax_mkt.transAxes,
                    fontsize=8,  color=GRAY)
        ax_mkt.text(x, y - 0.32, val, transform=ax_mkt.transAxes,
                    fontsize=10, fontweight="bold", color=col)

    # Row 3: Report text
    ax_text.set_title("AI-generated analyst report",
                      fontsize=10, fontweight="normal",
                      pad=6, loc="left", color=GRAY)

    bullet_text = ""
    try:
        from openai import OpenAI as _OAI
        import os as _os
        _client = _OAI(api_key=_os.environ.get("OPENAI_API_KEY"))
        _resp   = _client.chat.completions.create(
            model      = "gpt-4o-mini",
            max_tokens = 400,
            messages   = [{
                "role": "system",
                "content": (
                    "Convert the analyst report into exactly 3 sections: "
                    "PERFORMANCE, ANOMALIES, OUTLOOK. "
                    "Each section: label on its own line, then 2-3 bullet lines "
                    "starting with '• '. Each bullet under 90 characters. "
                    "No markdown, no **, no ##."
                )
            }, {"role": "user", "content": report}]
        )
        bullet_text = _resp.choices[0].message.content
    except Exception:
        clean     = _clean_markdown(report)
        sentences = [s.strip() for s in clean.replace("\n", " ").split(".")
                     if len(s.strip()) > 20]
        bullet_text = "\n".join(f"• {s}." for s in sentences[:12])

    bullet_text = _clean_markdown(bullet_text)

    y_pos = 0.97
    for line in bullet_text.split("\n"):
        line = line.strip()
        if not line:
            y_pos -= 0.018
            continue
        is_header = any(line.upper().startswith(h)
                        for h in ["PERFORMANCE", "ANOMALIES", "OUTLOOK"])
        if is_header:
            ax_text.text(0.01, y_pos, line.upper(), transform=ax_text.transAxes,
                         fontsize=8.5, fontweight="bold", color=GRAY,
                         verticalalignment="top")
            y_pos -= 0.065
        else:
            ax_text.text(0.01, y_pos, line, transform=ax_text.transAxes,
                         fontsize=8.5, color="#334155", verticalalignment="top")
            y_pos -= 0.052
        if y_pos < 0.02:
            break

    plt.tight_layout()
    path = save_path or f"{ticker}_chart4_report.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Visualizer] Saved {path}")
    return path


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_charts(
    ticker:      str,
    prices:      list[PricePoint],
    anomalies:   list[AnomalyPoint],
    result:      AnalysisResult,
    report:      str,
    lstm_result  = None,
    tf_result    = None,
):
    print(f"\n[Visualizer] Generating 4 charts for {ticker}...")
    plot_price_chart(prices, ticker)
    plot_anomaly_chart(prices, anomalies, ticker)
    if lstm_result and tf_result:
        plot_prediction_chart(lstm_result, tf_result, ticker)
    plot_report_card(result, report, ticker)
    print(f"[Visualizer] Done — PNG files saved in current folder.")