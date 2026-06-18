"""Streamlit app: US–Japan rates vs USD/JPY (2-year history)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from rate_history_engine import (
    BOJ_EVENTS,
    FED_FUNDS_PATH,
    PANEL_END,
    USDJPY_SAMPLE_PATH,
    fed_policy_events_table,
    load_fed_funds_latest,
    load_usdjpy_latest,
    merge_rate_fx_panel,
    policy_event_study,
    realized_fx_vol,
    spread_fx_regression,
    spread_fx_indicator_study,
    summary_metrics,
    fitted_fx_from_spread,
)

BRIGHT_BLUE = "#1E90FF"

st.set_page_config(
    page_title="Japan Rates & FX",
    page_icon="¥",
    layout="wide",
)


def _panel_layout(fig: go.Figure, title: str, *, show_legend: bool = True) -> go.Figure:
    bottom_margin = 90 if show_legend else 56
    fig.update_layout(
        height=400,
        hovermode="x unified",
        margin=dict(t=52, b=bottom_margin, l=56, r=16),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=14),
        ),
        showlegend=show_legend,
    )
    if show_legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                tracegroupgap=16,
                itemsizing="constant",
            ),
        )
    return fig


@st.cache_data(ttl=3600)
def _load_fed_cached() -> "pd.DataFrame":
    return load_fed_funds_latest(FED_FUNDS_PATH)


@st.cache_data(ttl=3600)
def _load_fred_fx_cached() -> "pd.DataFrame":
    return load_usdjpy_latest(USDJPY_SAMPLE_PATH)


st.title("Japan Rates & FX")

st.info(
    """
**16 Jun 2026 — Bank of Japan**  
+25 bps to **1.0%** (effective **17 Jun**) · **31-year high** · passed **7–1**

**17 Jun 2026 — Federal Reserve (Warsh)**  
First FOMC under **Kevin Warsh** · held fed funds at **3.50%–3.75%** · **unanimous** vote

**FX**  
USD/JPY still near **¥160** — US–Japan rate gap remains wide
    """
)


st.subheader("Rate differential vs USD/JPY")
st.caption(
    "USD/JPY: FRED `DEXJPUS` (live refresh + bundled). Fed funds: FRED `DFF`. "
    f"Panel through **{PANEL_END}**."
)

try:
    fed = _load_fed_cached()
    fx = _load_fred_fx_cached()
    panel = merge_rate_fx_panel(fed, fx)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

if len(panel) < 30:
    st.warning("Insufficient overlapping data.")
    st.stop()

metrics = summary_metrics(panel)
reg = spread_fx_regression(panel)
fed_events = fed_policy_events_table(panel)
boj_study = policy_event_study(panel, BOJ_EVENTS)
vol_df = realized_fx_vol(panel, window=21)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("USD/JPY", f"{metrics['fx_end']:.2f}", f"{metrics['fx_change_pct']:+.1f}%")
m2.metric("Spread", f"{metrics['spread_end_bps']:.0f} bps")
m3.metric("BoJ rate", "1.00%", "+25 bps")
m4.metric("Fed funds", f"{metrics['fed_end_pct']:.2f}%")
m5.metric("FX vol (ann.)", f"{metrics['fx_vol_ann_pct']:.1f}%")
m6.metric("Misalignment", f"{reg['residual_pct']:+.1f}%")
st.caption(
    f"*BoJ at **1.0%** effective 17 Jun 2026 (announced 16 Jun). "
    f"Misalignment vs log-linear fit (R²={reg['r2']:.2f}): "
    f"**{reg['residual_pct']:+.1f}%** — positive = yen weaker than spread implies. "
    f"{metrics['start_date'].date()} → {metrics['end_date'].date()}, {metrics['n_days']} days."
)

fig_main = make_subplots(specs=[[{"secondary_y": True}]])
fig_main.add_trace(
    go.Scatter(
        x=panel["date"],
        y=panel["spread_bps"],
        name="US−JP spread (bps)",
        line=dict(color=BRIGHT_BLUE, width=2),
    ),
    secondary_y=False,
)
fig_main.add_trace(
    go.Scatter(
        x=panel["date"],
        y=panel["usd_jpy"],
        name="USD/JPY",
        line=dict(color="#bc002d", width=2),
    ),
    secondary_y=True,
)
for _, ev in boj_study.iterrows():
    _evt_date = pd.Timestamp(ev["Date"])
    fig_main.add_shape(
        type="line",
        x0=_evt_date,
        x1=_evt_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#888", width=1, dash="dot"),
        opacity=0.35,
        layer="below",
    )
fig_main.update_xaxes(title_text="Date")
fig_main.update_yaxes(title_text="Spread (bps)", secondary_y=False)
fig_main.update_yaxes(title_text="USD/JPY", secondary_y=True)
fig_main.update_layout(
    height=420,
    margin=dict(t=52, b=72, l=56, r=56),
    hovermode="x unified",
    title=dict(
        text="US−JP spread vs USD/JPY",
        x=0.5,
        xanchor="center",
        y=0.98,
        yanchor="top",
        font=dict(size=14),
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.14,
        xanchor="center",
        x=0.5,
        tracegroupgap=16,
    ),
)
st.plotly_chart(fig_main, use_container_width=True)

fig_rates = go.Figure()
fig_rates.add_trace(
    go.Scatter(
        x=panel["date"],
        y=panel["fed_funds_pct"],
        name="Fed funds",
        line=dict(color=BRIGHT_BLUE),
    )
)
fig_rates.add_trace(
    go.Scatter(
        x=panel["date"],
        y=panel["boj_rate_pct"],
        name="BoJ policy",
        line=dict(color="#bc002d"),
    )
)
fig_rates.update_yaxes(title_text="Policy rate (%)")
st.plotly_chart(
    _panel_layout(fig_rates, "Policy rates", show_legend=True),
    use_container_width=True,
)

fig_vol = go.Figure()
fig_vol.add_trace(
    go.Scatter(
        x=vol_df["date"],
        y=vol_df["fx_vol_ann_pct"],
        name="21d FX vol",
        line=dict(color="#d62728"),
    )
)
fig_vol.update_yaxes(title_text="Ann. vol (%)")
st.plotly_chart(
    _panel_layout(fig_vol, "Realized USD/JPY volatility", show_legend=False),
    use_container_width=True,
)

st.subheader("Study: is the rate spread a good FX indicator?")
st.markdown(
    """
    **Question:** If US rates are much higher than Japan’s, should the **dollar strengthen**
    vs the yen (higher **USD/JPY**)? And when the **spread narrows**, should the yen firm?

    Below we test that carry-trade logic on **two years of bundled data** — levels (with a
    **log regression** of USD/JPY on the spread), daily changes, and lead/lag — and summarise
    how far the spread alone explains the yen.
    """
)

indicator = spread_fx_indicator_study(panel)
ind_reg = indicator["regression"]

study_l, study_r = st.columns(2)

with study_l:
    alpha = ind_reg["intercept"]
    beta = ind_reg["beta"]
    beta_latex = f"- {abs(beta):.6f}" if beta < 0 else f"+ {beta:.6f}"
    fitted = fitted_fx_from_spread(panel, ind_reg)
    levels_tbl = panel[["date", "spread_bps", "usd_jpy", "log_usd_jpy"]].copy()
    levels_tbl["fitted_usd_jpy"] = fitted.values
    levels_tbl["residual_pct"] = 100.0 * (levels_tbl["usd_jpy"] / levels_tbl["fitted_usd_jpy"] - 1.0)
    levels_tbl = levels_tbl.sort_values("date", ascending=False).reset_index(drop=True)
    levels_tbl["date"] = pd.to_datetime(levels_tbl["date"]).dt.strftime("%Y-%m-%d")
    levels_tbl = levels_tbl.rename(
        columns={
            "date": "Date",
            "spread_bps": "Spread (bps)",
            "usd_jpy": "USD/JPY",
            "log_usd_jpy": "log(USD/JPY)",
            "fitted_usd_jpy": "Fitted USD/JPY",
            "residual_pct": "Residual (%)",
        }
    )

    with st.expander("Plot data — levels (spread vs USD/JPY)"):
        st.dataframe(
            levels_tbl.style.format(
                {
                    "Spread (bps)": "{:.0f}",
                    "USD/JPY": "{:.2f}",
                    "log(USD/JPY)": "{:.4f}",
                    "Fitted USD/JPY": "{:.2f}",
                    "Residual (%)": "{:+.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=320,
        )

    fig_study = go.Figure()
    fig_study.add_trace(
        go.Scatter(
            x=panel["spread_bps"],
            y=panel["usd_jpy"],
            mode="markers",
            marker=dict(size=5, color="#bc002d", opacity=0.45),
            name="Observed",
        )
    )
    order = panel["spread_bps"].argsort()
    fig_study.add_trace(
        go.Scatter(
            x=panel["spread_bps"].iloc[order],
            y=fitted.iloc[order],
            mode="lines",
            line=dict(color=BRIGHT_BLUE, width=2.5),
            name="Log-linear fit",
        )
    )
    fig_study.update_xaxes(title_text="US−JP spread (bps)")
    fig_study.update_yaxes(title_text="USD/JPY")
    st.plotly_chart(
        _panel_layout(
            fig_study,
            "Levels: spread vs USD/JPY (log regression fit)",
            show_legend=True,
        ),
        use_container_width=True,
    )

    with st.expander("How this plot is calculated"):
        st.markdown(
            f"""
            **Inputs (one row per trading day)**  
            - **Spread (bps)** = Fed funds − BoJ policy rate, × 100  
            - **USD/JPY** = FRED `DEXJPUS` (bundled sample)

            **Red dots** — each point is one day: x = spread, y = observed USD/JPY.  
            Vertical stacks appear when policy rates are unchanged for many days but FX still moves.

            **Blue line** — OLS **log regression** on the full sample:  
            log(USD/JPY) = {alpha:.4f} {beta_latex} × spread (bps)  
            R² = {ind_reg['r2']:.2f}. The line is **exp(fitted log level)** mapped back to ¥.

            **Intuition** — carry theory says a wider spread should mean a weaker yen (higher USD/JPY),
            so dots might slope up-right. Here the fit is weak (low R²) and β is negative in this window:
            spread alone does not reliably pin USD/JPY levels.
            """
        )

with study_r:
    lead_lag = indicator["lead_lag"]
    lead_lag_tbl = lead_lag.rename(
        columns={"lag_days": "Lag (days)", "correlation": "Correlation"}
    ).sort_values("Lag (days)")

    with st.expander("Plot data — lead/lag correlations"):
        st.dataframe(
            lead_lag_tbl.style.format({"Correlation": "{:+.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    fig_lag = go.Figure()
    fig_lag.add_trace(
        go.Bar(
            x=lead_lag["lag_days"],
            y=lead_lag["correlation"],
            marker_color=[
                "#2ca02c" if v >= 0 else "#d62728" for v in lead_lag["correlation"]
            ],
            showlegend=False,
        )
    )
    fig_lag.add_hline(y=0, line_color="gray", line_dash="dash", annotation=dict(visible=False))
    fig_lag.update_xaxes(title_text="Lag (days): + = spread change leads FX")
    fig_lag.update_yaxes(title_text="Correlation")
    st.plotly_chart(
        _panel_layout(
            fig_lag,
            "Lead/lag: does spread change predict FX moves?",
            show_legend=False,
        ),
        use_container_width=True,
    )

    with st.expander("How this plot is calculated"):
        st.markdown(
            """
            **Inputs (daily changes)**  
            - **Δspread (bps)** = day-over-day change in US−JP spread  
            - **FX return (%)** = day-over-day % change in USD/JPY

            **Each bar** — Pearson correlation between Δspread and FX return at a different **lag** (−10 to +10 days):

            | Lag | Comparison |
            |-----|------------|
            | **0** | Δspread today vs FX return **same day** |
            | **+k** | Δspread today vs FX return **k days later** (spread **leads** FX) |
            | **−k** | Δspread **k days ago** vs FX return today (FX **leads** or spread lags) |

            **Intuition** — if spread changes predict yen moves, bars should be meaningfully positive at
            **positive lags**. In this sample correlations stay small (roughly −0.10 to +0.07) with no
            stable lead day. Lag 0 is the largest bar and is **negative** (same-day move opposite naive
            carry). Most days Δspread = 0 because rates move in steps, which makes daily timing noisy.
            """
        )

st.subheader("Policy event study — USD/JPY reaction")
ec1, ec2 = st.columns(2)
with ec1:
    st.markdown("**BoJ hikes (2024–2026)**")
    st.dataframe(
        boj_study.style.format(
            {c: "{:+.2f}%" for c in boj_study.columns if "FX" in c}
            | {"Spread (bps)": "{:.0f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
with ec2:
    st.markdown("**Fed policy events**")
    if fed_events.empty:
        st.caption("No Fed events in window.")
    else:
        display_fed = fed_events.rename(
            columns={
                "date": "Date",
                "fed_funds_pct": "Fed funds (%)",
                "change_bps": "Change (bps)",
                "label": "Event",
            }
        )
        display_fed["Date"] = pd.to_datetime(display_fed["Date"]).dt.strftime("%m/%d/%Y")
        st.dataframe(
            display_fed.style.format(
                {"Fed funds (%)": "{:.2f}", "Change (bps)": "{:+.0f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
