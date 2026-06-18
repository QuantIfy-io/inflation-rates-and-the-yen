"""Historical US–Japan rate spread vs USD/JPY analysis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
FED_FUNDS_PATH = DATA_DIR / "fed_funds_rate.csv"
USDJPY_SAMPLE_PATH = DATA_DIR / "usdjpy_fred_sample.csv"

# BoJ uncollateralized overnight call rate target (effective dates, %)
BOJ_POLICY_CHANGES: list[tuple[str, float]] = [
    ("2024-03-19", 0.10),
    ("2024-07-31", 0.25),
    ("2025-01-24", 0.50),
    ("2025-12-19", 0.75),
    ("2026-06-17", 1.00),
]

BOJ_EVENTS: list[tuple[str, str]] = [
    ("2024-07-31", "BoJ hike → 0.25%"),
    ("2025-01-24", "BoJ hike → 0.50%"),
    ("2025-12-19", "BoJ hike → 0.75%"),
    ("2026-06-16", "BoJ +25 bps → 1.00%"),
]

# Notable FOMC announcements with no rate change (not visible in DFF step detection)
FED_HOLD_EVENTS: list[tuple[str, str]] = [
    ("2026-06-17", "FOMC hold — no change (Warsh; 3.50%–3.75%)"),
]

TRADING_DAYS = 252


def _parse_date_column(df: pd.DataFrame) -> pd.Series:
    for col in df.columns:
        if col.lower() in {"date", "observation_date", "time", "datetime"}:
            return pd.to_datetime(df[col])
    return pd.to_datetime(df.iloc[:, 0])


def _parse_value_column(df: pd.DataFrame, hints: tuple[str, ...]) -> pd.Series:
    lower_map = {c.lower(): c for c in df.columns}
    for hint in hints:
        if hint in lower_map:
            return pd.to_numeric(df[lower_map[hint]], errors="coerce")
    numeric_cols = [
        c
        for c in df.columns
        if c.lower() not in {"date", "observation_date", "time", "datetime"}
    ]
    if len(numeric_cols) == 1:
        return pd.to_numeric(df[numeric_cols[0]], errors="coerce")
    raise ValueError(
        "Could not identify value column. Use columns like date, usd_jpy (or close)."
    )


def load_fed_funds(path: Path | str = FED_FUNDS_PATH) -> pd.DataFrame:
    """Daily effective fed funds rate (%). Source: FRED DFF."""
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "date": _parse_date_column(df),
            "fed_funds_pct": _parse_value_column(df, ("dff", "fed_funds", "rate", "value")),
        }
    )
    return out.dropna().sort_values("date").reset_index(drop=True)


def load_usdjpy(source: Path | str | pd.DataFrame) -> pd.DataFrame:
    """USD/JPY from CSV path or uploaded DataFrame."""
    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    out = pd.DataFrame(
        {
            "date": _parse_date_column(df),
            "usd_jpy": _parse_value_column(
                df, ("usd_jpy", "usdjpy", "dexjpus", "close", "price", "value")
            ),
        }
    )
    return out.dropna().sort_values("date").reset_index(drop=True)


def boj_policy_series(dates: pd.DatetimeIndex) -> pd.Series:
    """Step-function BoJ policy rate (%) aligned to dates."""
    if dates.empty:
        return pd.Series(dtype=float)
    changes = pd.Series(
        {pd.Timestamp(d): r for d, r in BOJ_POLICY_CHANGES},
        dtype=float,
    ).sort_index()
    # Rate in force at sample start
    prior = changes[changes.index <= dates.min()]
    level = float(prior.iloc[-1]) if len(prior) else float(changes.iloc[0])
    idx = changes.index.union([dates.min()]).sort_values()
    stepped = changes.reindex(idx).ffill()
    stepped.loc[dates.min()] = level
    stepped = stepped.sort_index().ffill()
    return stepped.reindex(dates, method="ffill").ffill()


def build_boj_daily(fed_df: pd.DataFrame) -> pd.DataFrame:
    """Expand BoJ policy steps to daily dates matching fed funds calendar."""
    dates = pd.DatetimeIndex(fed_df["date"])
    return pd.DataFrame({"date": dates, "boj_rate_pct": boj_policy_series(dates).values})


def merge_rate_fx_panel(
    fed_df: pd.DataFrame,
    fx_df: pd.DataFrame,
    boj_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge fed funds, BoJ policy, and USD/JPY on date."""
    if boj_df is None:
        boj_df = build_boj_daily(fed_df)

    fx_daily = fx_df.set_index("date").sort_index()
    fx_daily = fx_daily.reindex(pd.DatetimeIndex(fed_df["date"])).ffill().bfill()

    panel = fed_df.merge(boj_df, on="date", how="left")
    panel["usd_jpy"] = fx_daily["usd_jpy"].values
    panel = panel.dropna(subset=["fed_funds_pct", "boj_rate_pct", "usd_jpy"])
    panel["spread_bps"] = (panel["fed_funds_pct"] - panel["boj_rate_pct"]) * 100
    panel["log_usd_jpy"] = np.log(panel["usd_jpy"])
    panel["fx_return"] = panel["usd_jpy"].pct_change()
    panel["spread_change_bps"] = panel["spread_bps"].diff()
    panel["daily_carry_pct"] = panel["spread_bps"] / 100 / TRADING_DAYS
    return panel.reset_index(drop=True)


def detect_fed_rate_changes(panel: pd.DataFrame, min_bps: float = 5.0) -> pd.DataFrame:
    """FOMC-effective fed funds moves (step changes in daily series)."""
    chg = panel["fed_funds_pct"].diff()
    events = panel.loc[chg.abs() * 100 >= min_bps, ["date", "fed_funds_pct"]].copy()
    events["change_bps"] = (chg.loc[events.index] * 100).values
    events["label"] = events.apply(
        lambda r: f"Fed → {r['fed_funds_pct']:.2f}% ({r['change_bps']:+.0f} bps)",
        axis=1,
    )
    return events.reset_index(drop=True)


def fed_policy_events_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Fed rate step changes plus annotated hold announcements."""
    events = detect_fed_rate_changes(panel)
    hold_rows: list[dict[str, object]] = []
    for date_str, label in FED_HOLD_EVENTS:
        dt = pd.Timestamp(date_str)
        on_or_before = panel.loc[panel["date"] <= dt, "fed_funds_pct"]
        rate = float(on_or_before.iloc[-1]) if len(on_or_before) else np.nan
        hold_rows.append(
            {
                "date": dt,
                "fed_funds_pct": rate,
                "change_bps": 0.0,
                "label": label,
            }
        )
    if hold_rows:
        events = pd.concat([events, pd.DataFrame(hold_rows)], ignore_index=True)
    return events.sort_values("date").reset_index(drop=True)


def policy_event_study(
    panel: pd.DataFrame,
    event_dates: list[tuple[str, str]],
    horizons: tuple[int, ...] = (1, 5, 21),
) -> pd.DataFrame:
    """USD/JPY % change after BoJ (or other) policy dates."""
    rows: list[dict[str, object]] = []
    indexed = panel.set_index("date")
    for date_str, label in event_dates:
        dt = pd.Timestamp(date_str)
        if dt not in indexed.index:
            # nearest next business day in panel
            future = indexed.index[indexed.index >= dt]
            if len(future) == 0:
                continue
            dt = future[0]
        spot = indexed.loc[dt, "usd_jpy"]
        row: dict[str, object] = {"Event": label, "Date": dt.date()}
        for h in horizons:
            target_idx = indexed.index.get_indexer([dt], method="nearest")[0] + h
            if target_idx < len(indexed):
                end = indexed.iloc[target_idx]["usd_jpy"]
                row[f"FX {h}d %"] = 100.0 * (end / spot - 1.0)
            else:
                row[f"FX {h}d %"] = np.nan
        row["Spread (bps)"] = indexed.loc[dt, "spread_bps"]
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_correlation(
    panel: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Rolling corr between spread changes and FX returns."""
    out = panel[["date"]].copy()
    out["rolling_corr"] = (
        panel["spread_change_bps"]
        .rolling(window)
        .corr(panel["fx_return"])
    )
    return out.dropna()


def cumulative_carry_vs_fx(panel: pd.DataFrame) -> pd.DataFrame:
    """Cumulative implied carry (from spread) vs cumulative FX return."""
    out = panel[["date"]].copy()
    out["cumulative_carry_pct"] = panel["daily_carry_pct"].cumsum() * 100
    out["cumulative_fx_pct"] = (1 + panel["fx_return"].fillna(0)).cumprod() * 100 - 100
    out["cumulative_net_pct"] = out["cumulative_carry_pct"] + out["cumulative_fx_pct"]
    return out


def spread_fx_regression(panel: pd.DataFrame) -> dict[str, float]:
    """OLS: log(USD/JPY) ~ spread. Returns beta, intercept, r2, latest residual."""
    x = panel["spread_bps"].values
    y = panel["log_usd_jpy"].values
    if len(x) < 10:
        return {"beta": 0.0, "intercept": 0.0, "r2": 0.0, "residual_pct": 0.0}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    latest_fitted = slope * panel["spread_bps"].iloc[-1] + intercept
    residual_pct = 100.0 * (np.exp(panel["log_usd_jpy"].iloc[-1] - latest_fitted) - 1.0)
    return {
        "beta": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "residual_pct": float(residual_pct),
    }


def realized_fx_vol(panel: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualised rolling USD/JPY vol (%)."""
    out = panel[["date", "spread_bps"]].copy()
    out["fx_vol_ann_pct"] = (
        panel["fx_return"].rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
    )
    return out.dropna()


def summary_metrics(panel: pd.DataFrame) -> dict[str, float]:
    """Headline statistics for the merged panel."""
    fx_ret = panel["fx_return"].dropna()
    return {
        "start_date": panel["date"].iloc[0],
        "end_date": panel["date"].iloc[-1],
        "n_days": len(panel),
        "fx_start": panel["usd_jpy"].iloc[0],
        "fx_end": panel["usd_jpy"].iloc[-1],
        "fx_change_pct": 100.0 * (panel["usd_jpy"].iloc[-1] / panel["usd_jpy"].iloc[0] - 1),
        "spread_start_bps": panel["spread_bps"].iloc[0],
        "spread_end_bps": panel["spread_bps"].iloc[-1],
        "boj_start_pct": panel["boj_rate_pct"].iloc[0],
        "boj_end_pct": panel["boj_rate_pct"].iloc[-1],
        "fed_start_pct": panel["fed_funds_pct"].iloc[0],
        "fed_end_pct": panel["fed_funds_pct"].iloc[-1],
        "fx_vol_ann_pct": float(fx_ret.std() * np.sqrt(TRADING_DAYS) * 100),
        "corr_spread_fx": float(panel["spread_bps"].corr(panel["usd_jpy"])),
    }


def fitted_fx_from_spread(panel: pd.DataFrame, reg: dict[str, float]) -> pd.Series:
    """Rate-implied log USD/JPY from regression."""
    return np.exp(reg["intercept"] + reg["beta"] * panel["spread_bps"])


def spread_fx_indicator_study(
    panel: pd.DataFrame,
    rolling_window: int = 63,
    max_lag: int = 10,
) -> dict[str, object]:
    """
    Test whether the US−JP rate spread indicates USD/JPY strength.

    Carry-theory sign convention:
    - **Wider spread** (higher US vs JP rates) → **weaker yen** → **higher USD/JPY**
    - **Narrowing spread** → yen should **strengthen** → **lower USD/JPY**
    """
    reg = spread_fx_regression(panel)
    roll = rolling_correlation(panel, window=rolling_window)

    clean = panel.dropna(subset=["spread_change_bps", "fx_return"])
    corr_level = float(panel["spread_bps"].corr(panel["usd_jpy"]))
    corr_change = float(clean["spread_change_bps"].corr(clean["fx_return"] * 100))

    lag_rows: list[dict[str, float | int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            series_a = clean["spread_change_bps"]
            series_b = clean["fx_return"] * 100
        elif lag > 0:
            series_a = clean["spread_change_bps"]
            series_b = (clean["fx_return"] * 100).shift(-lag)
        else:
            series_a = clean["spread_change_bps"].shift(lag)
            series_b = clean["fx_return"] * 100
        lag_rows.append({"lag_days": lag, "correlation": float(series_a.corr(series_b))})
    lead_lag_df = pd.DataFrame(lag_rows)

    spread_chg = panel["spread_bps"].iloc[-1] - panel["spread_bps"].iloc[0]
    fx_chg_pct = 100.0 * (panel["usd_jpy"].iloc[-1] / panel["usd_jpy"].iloc[0] - 1)

    # Theory: spread down → yen up → USD/JPY down (negative co-movement over long window)
    same_direction = (spread_chg < 0 and fx_chg_pct < 0) or (spread_chg > 0 and fx_chg_pct > 0)
    roll_mean = float(roll["rolling_corr"].mean()) if not roll.empty else np.nan
    roll_positive_pct = (
        100.0 * (roll["rolling_corr"] > 0).mean() if not roll.empty else np.nan
    )

    verdict = _spread_indicator_verdict(
        corr_level=corr_level,
        corr_change=corr_change,
        r2=reg["r2"],
        roll_mean=roll_mean,
        same_direction_over_window=bool(same_direction),
        spread_chg_bps=spread_chg,
        fx_chg_pct=fx_chg_pct,
    )

    return {
        "regression": reg,
        "corr_level": corr_level,
        "corr_change": corr_change,
        "spread_change_bps_window": float(spread_chg),
        "fx_change_pct_window": float(fx_chg_pct),
        "rolling_corr_mean": roll_mean,
        "rolling_corr_positive_pct": roll_positive_pct,
        "lead_lag": lead_lag_df,
        "rolling_corr": roll,
        "verdict_title": verdict["title"],
        "verdict_body": verdict["body"],
    }


def _spread_indicator_verdict(
    *,
    corr_level: float,
    corr_change: float,
    r2: float,
    roll_mean: float,
    same_direction_over_window: bool,
    spread_chg_bps: float,
    fx_chg_pct: float,
) -> dict[str, str]:
    """Plain-language conclusion for the indicator study."""
    if corr_level >= 0.5 and r2 >= 0.25:
        level_grade = "moderately useful at explaining **levels**"
    elif corr_level >= 0.25:
        level_grade = "only a **partial** guide to **levels**"
    else:
        level_grade = "a **weak** guide to **levels**"

    if abs(corr_change) >= 0.15:
        change_grade = "daily spread moves align somewhat with same-day FX"
    else:
        change_grade = "day-to-day spread moves do **not** reliably track FX"

    direction_note = (
        "Over this window, a **narrower** spread coincided with a **weaker USD/JPY** "
        "(yen firmer), as carry theory suggests."
        if spread_chg_bps < 0 and fx_chg_pct < 0
        else "Over this window, spread and USD/JPY did **not** move in the textbook "
        "carry direction — other forces (risk, intervention, terms of trade) dominated."
        if not same_direction_over_window
        else "Over this window, spread and USD/JPY moved in the same direction — "
        "yen **weakened** even as the rate gap **narrowed**, contradicting a simple carry read."
    )

    title = f"Spread is {level_grade}; {change_grade}."
    body = (
        f"{direction_note} "
        f"Spread fell **{spread_chg_bps:+.0f} bps** while USD/JPY moved "
        f"**{fx_chg_pct:+.1f}%**. "
        f"Level correlation **{corr_level:.2f}**, change correlation **{corr_change:.2f}**, "
        f"regression R² **{r2:.2f}**. "
        f"Use the spread for **context**, not a standalone FX forecast."
    )
    return {"title": title, "body": body}

