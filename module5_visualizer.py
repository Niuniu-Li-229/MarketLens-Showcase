"""
Module 5 — Visualizer
Generates 4 poster-ready charts from pipeline output.
Saves as PNG files in the current directory.

pip install matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from datetime import date
from models import PricePoint, AnomalyPoint, AnalysisResult


# ── Colour palette ────────────────────────────────────────────────────────────

BLUE   = "#1d4ed8"
ORANGE = "#f97316"
PURPLE = "#7c3aed"
GRAY   = "#94a3b8"
GREEN  = "#16a34a"
RED    = "#dc2626"
LIGHT  = "#f8fafc"

# Force white background regardless of system dark/light mode
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

CAT_COLORS = {
    "Positive catalyst":           GREEN,
    "Negative catalyst":           RED,
    "Contradictory — oversold?":   "#d97706",
    "Contradictory — buying opp?": PURPLE,
    "Unknown driver":              GRAY,
}


# ── Chart 1: Price + MA + Volume ─────────────────────────────────────────────

def plot_price_chart(
    prices: list[PricePoint],
    ticker: str,
    save_path: str = None,
):
    import yfinance as yf

    dates  = [p.date for p in prices]
    closes = [p.close for p in prices]
    vols   = [p.volume / 1_000_000 for p in prices]

    def rolling_mean(arr, w):
        return [
            np.mean(arr[max(0, i-w+1): i+1]) if i >= w-1 else None
            for i in range(len(arr))
        ]

    ma20 = rolling_mean(closes, 20)
    ma60 = rolling_mean(closes, 60)

    # ── Fetch S&P500 and normalise to same starting price ────────────────────
    spy_dates, spy_norm = [], []
    try:
        from datetime import timedelta
        spy_df = yf.download(
            "^GSPC",
            start   = str(dates[0]),
            end     = str(dates[-1] + timedelta(days=1)),
            auto_adjust       = True,
            progress          = False,
            multi_level_index = False,
        )
        if not spy_df.empty:
            spy_closes = spy_df["Close"].values
            spy_dates  = [d.date() for d in spy_df.index]
            # Normalise: scale S&P500 so it starts at same price as ticker
            spy_norm = spy_closes / spy_closes[0] * closes[0]
    except Exception as e:
        print(f"[Chart 1] S&P500 fetch failed: {e}")

    # ── Layout: 3 panels (price, volume, S&P500 comparison) ──────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1, 1.2]},
        facecolor="white"
    )
    fig.suptitle("MODULE 1 — OUTPUT", fontsize=9, color=GRAY,
                 x=0.01, ha="left")

    # Panel 1: Price + MAs
    ax1.plot(dates, closes, color=BLUE,   lw=1.5, label="Close price", zorder=3)
    ax1.plot(dates, ma20,   color=ORANGE, lw=1.0, label="MA20", alpha=0.85)
    ax1.plot(dates, ma60,   color=PURPLE, lw=1.0, label="MA60", alpha=0.85)
    if spy_norm is not None and len(spy_norm):
        ax1.plot(spy_dates, spy_norm, color=GRAY, lw=1.0,
                 linestyle="--", alpha=0.6, label="S&P500 (normalised)")
    ax1.set_title(f"{ticker} — price, moving averages & S&P500 comparison",
                  fontsize=13, fontweight="normal", pad=8, loc="left")
    ax1.set_ylabel("Price (USD)", fontsize=9, color=GRAY)
    ax1.tick_params(colors=GRAY, labelsize=8)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.spines[["left","bottom"]].set_color("#e2e8f0")
    ax1.grid(axis="y", color="#f1f5f9", linewidth=0.8)
    ax1.legend(fontsize=8, framealpha=0, loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}"))

    # Panel 2: Volume
    ax2.bar(dates, vols, color=GRAY, alpha=0.5, width=1.0)
    ax2.set_ylabel("Vol (M)", fontsize=8, color=GRAY)
    ax2.tick_params(colors=GRAY, labelsize=7)
    ax2.spines[["top","right","left"]].set_visible(False)
    ax2.spines["bottom"].set_color("#e2e8f0")
    ax2.grid(axis="y", color="#f1f5f9", linewidth=0.8)

    # Panel 3: Relative performance vs S&P500
    if len(spy_norm) and len(spy_dates):
        # Compute % return from start for both
        ticker_pct = [(c - closes[0]) / closes[0] * 100 for c in closes]
        spy_pct    = [(s - spy_closes[0]) / spy_closes[0] * 100
                      for s in spy_closes]
        ax3.plot(dates,     ticker_pct, color=BLUE, lw=1.5,
                 label=f"{ticker} return")
        ax3.plot(spy_dates, spy_pct,    color=GRAY, lw=1.0,
                 linestyle="--", alpha=0.7, label="S&P500 return")
        ax3.axhline(0, color="#e2e8f0", lw=0.8)
        ax3.fill_between(
            dates,
            [t - s for t, s in zip(ticker_pct,
                [next((s for d, s in zip(spy_dates, spy_pct)
                       if d == dt), 0) for dt in dates])],
            0,
            alpha=0.08, color=BLUE,
        )
        ax3.set_ylabel("Return %", fontsize=8, color=GRAY)
        ax3.tick_params(colors=GRAY, labelsize=7)
        ax3.spines[["top","right"]].set_visible(False)
        ax3.spines[["left","bottom"]].set_color("#e2e8f0")
        ax3.grid(axis="y", color="#f1f5f9", linewidth=0.8)
        ax3.legend(fontsize=8, framealpha=0)
        ax3.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "S&P500 data unavailable",
                 transform=ax3.transAxes, ha="center",
                 fontsize=9, color=GRAY)

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis="x", colors=GRAY, labelsize=8)

    # Stat boxes
    total_ret = (closes[-1] - closes[0]) / closes[0] * 100
    ret_color = GREEN if total_ret >= 0 else RED
    spy_ret   = ((spy_closes[-1] - spy_closes[0]) / spy_closes[0] * 100
                 if len(spy_norm) else None)

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
        outperform = total_ret - spy_ret
        fig.text(0.52, -0.02,
                 f"{outperform:+.1f}%", fontsize=14,
                 fontweight="bold",
                 color=GREEN if outperform >= 0 else RED)
        fig.text(0.52, -0.05, "vs S&P500", fontsize=8, color=GRAY)

    plt.tight_layout()
    path = save_path or f"{ticker}_chart1_price.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Visualizer] Saved {path}")
    return path


# ── Chart 2: Anomaly detection ────────────────────────────────────────────────

def plot_anomaly_chart(
    prices:    list[PricePoint],
    anomalies: list[AnomalyPoint],
    ticker:    str,
    save_path: str = None,
):
    dates  = [p.date for p in prices]
    closes = [p.close for p in prices]

    # ── Keep only the most extreme anomalies for a clean chart ───────────────
    # Sort by absolute % change, keep top 15
    top_anomalies = sorted(anomalies,
                           key=lambda a: abs(a.percent_change),
                           reverse=True)[:15]
    top_anomalies = sorted(top_anomalies, key=lambda a: a.date)

    gains  = [a for a in top_anomalies if a.percent_change > 0]
    losses = [a for a in top_anomalies if a.percent_change <= 0]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
    fig.suptitle("MODULE 2 — OUTPUT", fontsize=9, color=GRAY,
                 x=0.01, ha="left")

    ax.plot(dates, closes, color=BLUE, lw=1.5, label="Close price", zorder=2)

    # Green dots = positive anomalies, red dots = negative
    if gains:
        ax.scatter([a.date for a in gains],
                   [a.price_point.close for a in gains],
                   color=GREEN, s=80, zorder=4, label="Positive anomaly")
    if losses:
        ax.scatter([a.date for a in losses],
                   [a.price_point.close for a in losses],
                   color=RED, s=80, zorder=4, label="Negative anomaly")

    # Annotate the 5 most extreme with % labels
    top5 = sorted(top_anomalies,
                  key=lambda a: abs(a.percent_change),
                  reverse=True)[:5]
    for a in top5:
        color = GREEN if a.percent_change > 0 else RED
        ax.annotate(
            f"{a.percent_change:+.1f}%",
            xy=(a.date, a.price_point.close),
            xytext=(0, 12), textcoords="offset points",
            fontsize=7.5, color=color, ha="center", fontweight="bold",
        )

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

    # ── Event table — show top 6 by magnitude ────────────────────────────────
    top6 = sorted(top_anomalies,
                  key=lambda a: abs(a.percent_change),
                  reverse=True)[:6]
    top6 = sorted(top6, key=lambda a: a.date)

    col_labels = ["Date", "Change", "Layers triggered", "Linked event"]
    col_x      = [0.01, 0.10, 0.22, 0.42]
    header_y   = -0.08

    for x, lbl in zip(col_x, col_labels):
        fig.text(x, header_y, lbl, fontsize=8,
                 fontweight="bold", color=GRAY,
                 transform=fig.transFigure)

    row_y = header_y - 0.07
    for a in top6:
        chg     = a.percent_change
        color   = GREEN if chg > 0 else RED
        layers  = ", ".join(
            [type(d).__name__.replace("Detector","") for d in []]
        ) or f"{len(a.related_events)} events"
        headline = (a.related_events[0].title[:55] + "…"
                    if a.related_events and
                       a.related_events[0].title != "No linked news"
                    else "—")
        row_vals = [
            str(a.date),
            f"{chg:+.1f}%",
            f"{len(a.related_events)} news linked",
            headline,
        ]
        for x, val in zip(col_x, row_vals):
            fig.text(x, row_y, val, fontsize=7.5,
                     color=color if val == row_vals[1] else "#334155",
                     transform=fig.transFigure)
        row_y -= 0.06

    plt.tight_layout()
    path = save_path or f"{ticker}_chart2_anomalies.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Visualizer] Saved {path}")
    return path


# ── Chart 3: LSTM prediction vs actual ───────────────────────────────────────

def plot_prediction_chart(
    lstm_result,
    tf_result,
    ticker: str,
    save_path: str = None,
):
    """
    Receives ForecastResult objects directly from Module 3.
    No model training here — just plotting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle("MODULE 3 — OUTPUT", fontsize=9, color=GRAY,
                 x=0.01, ha="left")

    for ax, res in [(ax1, lstm_result), (ax2, tf_result)]:
        n = min(len(res.actual), len(res.predicted), len(res.test_dates))
        ax.plot(res.test_dates[:n], res.actual[:n],
                color=BLUE, lw=1.8, label="Actual price")
        ax.plot(res.test_dates[:n], res.predicted[:n],
                color=ORANGE, lw=1.5, linestyle="--", label="Prediction")
        ax.set_title(f"{ticker} — {res.model_name}",
                     fontsize=10, fontweight="normal", pad=6, loc="left")
        ax.set_ylabel("Price (USD)", fontsize=9, color=GRAY)
        ax.tick_params(colors=GRAY, labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#e2e8f0")
        ax.grid(axis="y", color="#f1f5f9", linewidth=0.8)
        ax.legend(fontsize=8, framealpha=0)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v:.0f}"))
        color = GREEN if res.dir_accuracy >= 0.55 else RED
        ax.text(0.02, 0.06,
                f"Dir. accuracy: {res.dir_accuracy:.1%}   MAE: ${res.mae:.2f}",
                transform=ax.transAxes, fontsize=9,
                color=color, fontweight="bold")

    plt.tight_layout()
    path = save_path or f"{ticker}_chart3_prediction.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Visualizer] Saved {path}")
    return path


# ── Chart 4: AI report card ───────────────────────────────────────────────────

def plot_report_card(
    result:  AnalysisResult,
    report:  str,
    ticker:  str,
    save_path: str = None,
):
    import yfinance as yf
    import textwrap

    # ── Fetch extra indicators from yfinance ──────────────────────────────────
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info

        # Basic
        pe_ratio    = info.get("trailingPE")
        mkt_cap     = info.get("marketCap")
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low  = info.get("fiftyTwoWeekLow")
        current     = info.get("currentPrice") or info.get("regularMarketPrice")
        beta        = info.get("beta")
        analyst_rec = info.get("recommendationKey", "n/a").upper()
        target_mean = info.get("targetMeanPrice")
        buy_pct     = info.get("recommendationMean")  # 1=strong buy, 5=sell

        # Dividend yield
        div_yield = info.get("dividendYield")

        # 52w position (0% = at low, 100% = at high)
        if week52_high and week52_low and current:
            w52_pos = (current - week52_low) / (week52_high - week52_low) * 100
        else:
            w52_pos = None

        # Analyst upside
        if target_mean and current:
            upside = (target_mean - current) / current * 100
        else:
            upside = None

        # Beta label
        if beta:
            if beta > 1.5:   beta_label = "High volatility"
            elif beta > 1.0: beta_label = "Above market"
            elif beta > 0.5: beta_label = "Below market"
            else:            beta_label = "Low volatility"
        else:
            beta_label = "n/a"

        # Recommendation score → label
        rec_colors = {
            "STRONG_BUY": GREEN, "BUY": GREEN,
            "HOLD": ORANGE,
            "SELL": RED, "STRONG_SELL": RED,
        }
        rec_color = rec_colors.get(analyst_rec, GRAY)

        # VIX (market fear index)
        vix_hist = yf.Ticker("^VIX").history(period="5d")
        vix      = round(float(vix_hist["Close"].iloc[-1]), 1) if not vix_hist.empty else None
        if vix:
            if vix < 15:   vix_label, vix_color = "Low fear", GREEN
            elif vix < 25: vix_label, vix_color = "Moderate", ORANGE
            else:          vix_label, vix_color = "High fear", RED
        else:
            vix_label, vix_color = "n/a", GRAY

        # Beta vs S&P500 (relative performance last 30 days)
        hist_stock = tk.history(period="30d")["Close"]
        hist_spy   = yf.Ticker("SPY").history(period="30d")["Close"]
        if len(hist_stock) > 1 and len(hist_spy) > 1:
            ret_stock = (hist_stock.iloc[-1] - hist_stock.iloc[0]) / hist_stock.iloc[0] * 100
            ret_spy   = (hist_spy.iloc[-1]   - hist_spy.iloc[0])   / hist_spy.iloc[0]   * 100
            rel_perf  = ret_stock - ret_spy
        else:
            rel_perf = None

        # Market cap label
        if mkt_cap:
            if mkt_cap >= 1e12:   cap_label = f"${mkt_cap/1e12:.1f}T"
            elif mkt_cap >= 1e9:  cap_label = f"${mkt_cap/1e9:.0f}B"
            else:                 cap_label = f"${mkt_cap/1e6:.0f}M"
        else:
            cap_label = "n/a"

    except Exception as e:
        print(f"  [Chart 4] Extra indicators fetch failed: {e}")
        pe_ratio = beta = w52_pos = upside = rel_perf = None
        vix = vix_label = analyst_rec = cap_label = None
        rec_color = vix_color = GRAY
        beta_label = div_yield = target_mean = "n/a"

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    fig.suptitle("MODULE 4 — OUTPUT", fontsize=9, color=GRAY,
                 x=0.01, ha="left")

    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[1, 1.4], hspace=0.35)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])
    ax_top.axis("off")
    ax_bot.axis("off")

    ax_top.set_title(f"{ticker} — AI analysis report (GPT-4o)",
                     fontsize=13, fontweight="normal", pad=8, loc="left")

    # ── Row 1: Core pipeline results ──────────────────────────────────────────
    sentiment   = result.sentiment_label or "neutral"
    score       = result.sentiment_score or 0.0
    predicted   = result.predicted_price or 0.0
    ret         = result.total_return
    ret_color   = GREEN if ret >= 0 else RED
    sent_color  = GREEN if score > 0.1 else RED if score < -0.1 else GRAY
    pred_chg    = (predicted - (result.anomalies[-1].price_point.close
                                if result.anomalies else predicted))

    core_stats = [
        ("Ticker",      ticker,                    BLUE),
        ("Period return", f"{ret:+.2f}%",          ret_color),
        ("Sentiment",   f"{score:+.2f} ({sentiment})", sent_color),
        ("Predicted D1",f"${predicted:.2f}",       PURPLE),
        ("Anomalies",   str(result.anomaly_count()), ORANGE),
    ]
    for i, (lbl, val, col) in enumerate(core_stats):
        x = 0.01 + i * 0.20
        ax_top.text(x, 0.90, lbl, transform=ax_top.transAxes,
                    fontsize=8, color=GRAY)
        ax_top.text(x, 0.75, val, transform=ax_top.transAxes,
                    fontsize=12, fontweight="bold", color=col)

    ax_top.plot([0, 1], [0.65, 0.65], color="#e2e8f0",
                lw=0.8, transform=ax_top.transAxes, clip_on=False)

    # ── Row 2: Market indicators (8 boxes) ────────────────────────────────────
    def fmt(v, fmt_str, fallback="n/a"):
        try:    return fmt_str.format(v) if v is not None else fallback
        except: return fallback

    market_stats = [
        ("P/E ratio",
         fmt(pe_ratio, "{:.1f}x"),
         BLUE if pe_ratio and pe_ratio < 30 else ORANGE if pe_ratio else GRAY),

        ("Market cap",
         cap_label or "n/a",
         BLUE),

        ("52w position",
         fmt(w52_pos, "{:.0f}% of range"),
         GREEN if w52_pos and w52_pos > 60 else RED if w52_pos and w52_pos < 30 else ORANGE),

        ("Beta",
         fmt(beta, "{:.2f}") + f"\n{beta_label}",
         ORANGE if beta and beta > 1.2 else GREEN),

        ("Analyst target",
         fmt(target_mean, "${:.2f}") + (f"\n{upside:+.1f}% upside" if upside else ""),
         GREEN if upside and upside > 5 else RED if upside and upside < -5 else GRAY),

        ("Analyst rating",
         analyst_rec or "n/a",
         rec_color),

        ("VIX (fear)",
         fmt(vix, "{:.1f}") + f"\n{vix_label}",
         vix_color),

        ("vs S&P500 (30d)",
         fmt(rel_perf, "{:+.1f}%") + "\nrelative perf",
         GREEN if rel_perf and rel_perf > 0 else RED),
    ]

    n_cols = 4
    for i, (lbl, val, col) in enumerate(market_stats):
        row = i // n_cols
        col_i = i % n_cols
        x = 0.01 + col_i * 0.25
        y = 0.55 - row * 0.28
        ax_top.text(x, y,        lbl, transform=ax_top.transAxes,
                    fontsize=8,  color=GRAY)
        ax_top.text(x, y - 0.13, val, transform=ax_top.transAxes,
                    fontsize=10, fontweight="bold", color=col,
                    linespacing=1.4)

    # ── AI report text — bullet points ───────────────────────────────────────
    ax_bot.set_title("AI-generated analyst report", fontsize=10,
                     fontweight="normal", pad=6, loc="left", color=GRAY)

    # Ask GPT to reformat the report as bullet points
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
                    "Convert the analyst report into concise bullet points. "
                    "Use exactly 3 sections: "
                    "PERFORMANCE, ANOMALIES, OUTLOOK. "
                    "Each section has 2-3 bullet points starting with '•'. "
                    "Keep each bullet under 90 characters. "
                    "No markdown, no bold, no headers with ##."
                )
            }, {
                "role": "user",
                "content": report,
            }]
        )
        bullet_text = _resp.choices[0].message.content
    except Exception:
        # Fallback: split into sentences and add bullets manually
        clean       = report.replace("##","").replace("#","").replace("**","")
        sentences   = [s.strip() for s in clean.replace("\n"," ").split(".") if len(s.strip()) > 20]
        bullet_text = "\n".join(f"• {s}." for s in sentences[:12])

    # Draw bullet points line by line
    y_pos = 0.95
    for line in bullet_text.split("\n"):
        line = line.strip()
        if not line:
            y_pos -= 0.015
            continue
        # Section headers (PERFORMANCE / ANOMALIES / OUTLOOK)
        is_header = any(
            line.upper().startswith(h)
            for h in ["PERFORMANCE", "ANOMALIES", "OUTLOOK"]
        )
        if is_header:
            ax_bot.text(0.01, y_pos, line.upper(), transform=ax_bot.transAxes,
                        fontsize=8, fontweight="bold", color=GRAY,
                        verticalalignment="top")
            y_pos -= 0.055
        else:
            ax_bot.text(0.01, y_pos, line, transform=ax_bot.transAxes,
                        fontsize=8, color="#334155", verticalalignment="top")
            y_pos -= 0.048
        if y_pos < 0.02:
            break

    plt.tight_layout()
    path = save_path or f"{ticker}_chart4_report.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Visualizer] Saved {path}")
    return path


# ── Master function — call after run_pipeline ─────────────────────────────────

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
    plot_prediction_chart(lstm_result, tf_result, ticker)
    plot_report_card(result, report, ticker)
    print(f"[Visualizer] Done — 4 PNG files saved in current folder.")