"""
forecasting_utils.py
--------------------
Utility functions for the SPB Winter 2022 retrospective forecasting experiment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

DATE_COL_CANDIDATES = ["date", "Date", "fecha", "Fecha", "Дата", "TIME", "time"]
CASES_COL_CANDIDATES = [
    "confirmed", "confirmed_cases", "cases", "Confirmed", "daily_cases",
    "new_cases", "cases_confirmed", "CONFIRMED",
]


def load_spb_data(csv_path: str) -> tuple:
    """Load and clean the SPB winter 2022 CSV.

    Returns
    -------
    dates : pd.Series of pd.Timestamp
    y_real : np.ndarray of float  (confirmed cases)
    """
    df = pd.read_csv(csv_path)

    # Detect date column
    date_col = None
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(
            f"No date column found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Detect cases column
    cases_col = None
    for c in CASES_COL_CANDIDATES:
        if c in df.columns:
            cases_col = c
            break
    if cases_col is None:
        raise ValueError(
            f"No cases column found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.dropna(subset=[date_col, cases_col])
    df[cases_col] = pd.to_numeric(df[cases_col], errors="coerce")
    df = df.dropna(subset=[cases_col])

    dates = df[date_col].dt.normalize()  # strip time component
    y_real = df[cases_col].to_numpy(dtype=float)

    print(f"[DATA] Loaded {len(y_real)} data points | "
          f"Date range: {dates.iloc[0].date()} → {dates.iloc[-1].date()} | "
          f"Peak cases: {y_real.max():.0f}")

    return dates.reset_index(drop=True), y_real


# ─────────────────────────────────────────────
# 2. PEAK DETECTION
# ─────────────────────────────────────────────

def detect_peak(dates: pd.Series, y_real: np.ndarray,
                mode: str = "raw",
                manual_peak_date: str = "2022-02-08") -> tuple:
    """Return (peak_date, peak_idx).

    Parameters
    ----------
    mode : 'raw'    – use argmax of y_real
           'manual' – use manual_peak_date string
    """
    if mode == "raw":
        peak_idx = int(np.argmax(y_real))
        peak_date = dates.iloc[peak_idx]
    elif mode == "manual":
        target = pd.Timestamp(manual_peak_date)
        diffs = (dates - target).abs()
        peak_idx = int(diffs.argmin())
        peak_date = dates.iloc[peak_idx]
    else:
        raise ValueError(f"Unknown PEAK_MODE: {mode}")

    print(f"[PEAK] mode={mode} | peak_date={peak_date.date()} | "
          f"peak_idx={peak_idx} | peak_cases={y_real[peak_idx]:.0f}")
    return peak_date, peak_idx


# ─────────────────────────────────────────────
# 3. SCENARIO SPLITS
# ─────────────────────────────────────────────

def build_scenario_splits(dates: pd.Series, y_real: np.ndarray,
                          peak_idx: int, horizon: int = 14) -> list:
    """Build the three forecast scenarios.

    Returns list of dicts:
        name, t_cut_idx, t_cut_date,
        y_known, y_hidden,
        dates_known, dates_hidden
    """
    offsets = {"tpeak_minus_14": -14, "tpeak_minus_7": -7, "tpeak_plus_7": +7}
    n = len(y_real)
    scenarios = []

    for name, offset in offsets.items():
        t_cut_idx = peak_idx + offset

        # Validation
        if t_cut_idx < 1:
            raise ValueError(f"[{name}] t_cut_idx={t_cut_idx} is before the series start.")
        if t_cut_idx >= n:
            raise ValueError(f"[{name}] t_cut_idx={t_cut_idx} is beyond the series end (n={n}).")
        remaining = n - t_cut_idx - 1
        if remaining < horizon:
            raise ValueError(
                f"[{name}] Only {remaining} days after t_cut; need at least {horizon}."
            )

        t_cut_date = dates.iloc[t_cut_idx]
        # known: indices 0 … t_cut_idx (inclusive)
        y_known = y_real[: t_cut_idx + 1]
        dates_known = dates.iloc[: t_cut_idx + 1]

        # hidden: indices t_cut_idx+1 … t_cut_idx+horizon (inclusive)
        end_idx = t_cut_idx + horizon
        y_hidden = y_real[t_cut_idx + 1: end_idx + 1]
        dates_hidden = dates.iloc[t_cut_idx + 1: end_idx + 1]

        scenarios.append({
            "name": name,
            "t_cut_idx": t_cut_idx,
            "t_cut_date": t_cut_date,
            "y_known": y_known,
            "y_hidden": y_hidden,
            "dates_known": dates_known.reset_index(drop=True),
            "dates_hidden": dates_hidden.reset_index(drop=True),
        })
        print(f"[SCENARIO] {name}: cut={t_cut_date.date()} | "
              f"known={len(y_known)} pts | hidden={len(y_hidden)} pts")

    return scenarios


def save_observed_hidden_splits(scenarios: list, out_path: str):
    rows = []
    for sc in scenarios:
        for d, v in zip(sc["dates_known"], sc["y_known"]):
            rows.append({"scenario": sc["name"], "date": d.date(), "value": v, "segment": "known"})
        for d, v in zip(sc["dates_hidden"], sc["y_hidden"]):
            rows.append({"scenario": sc["name"], "date": d.date(), "value": v, "segment": "hidden"})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[SAVE] Splits saved → {out_path}")


# ─────────────────────────────────────────────
# 4. SCALE FACTOR
# ─────────────────────────────────────────────

def estimate_scale_factor(y_known: np.ndarray, f_known: np.ndarray) -> float:
    """Least-squares scale factor: a = <y, f> / <f, f>."""
    denom = float(np.dot(f_known, f_known))
    if denom < 1e-12:
        return 0.0
    return max(float(np.dot(y_known, f_known) / denom), 0.0)


# ─────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────

def compute_forecast_metrics(y_hidden: np.ndarray,
                             forecast_mean: np.ndarray,
                             forecast_p10: np.ndarray,
                             forecast_p90: np.ndarray,
                             peak_date_real,
                             dates_hidden: pd.Series) -> dict:
    """Compute all evaluation metrics against the hidden segment."""
    h = len(y_hidden)
    fm = forecast_mean[:h]
    p10 = forecast_p10[:h]
    p90 = forecast_p90[:h]

    rmse = float(np.sqrt(mean_squared_error(y_hidden, fm)))
    mae = float(mean_absolute_error(y_hidden, fm))
    r2 = float(r2_score(y_hidden, fm))

    # Peak error within horizon
    pred_peak_idx = int(np.argmax(fm))
    pred_peak_date = dates_hidden.iloc[pred_peak_idx] if pred_peak_idx < len(dates_hidden) else None
    if pred_peak_date is not None and peak_date_real is not None:
        peak_time_error = (pred_peak_date - peak_date_real).days
    else:
        peak_time_error = float("nan")

    real_peak = float(np.max(y_hidden))
    pred_peak_val = float(np.max(fm))
    if real_peak > 0:
        peak_height_rel_err = (pred_peak_val - real_peak) / real_peak
    else:
        peak_height_rel_err = float("nan")

    # Coverage
    in_band = np.sum((y_hidden >= p10[:h]) & (y_hidden <= p90[:h]))
    coverage = float(in_band) / h if h > 0 else float("nan")

    return {
        "rmse_forecast": rmse,
        "mae_forecast": mae,
        "r2_forecast": r2,
        "peak_time_error_days": peak_time_error,
        "peak_height_relative_error": peak_height_rel_err,
        "coverage_10_90": coverage,
    }
