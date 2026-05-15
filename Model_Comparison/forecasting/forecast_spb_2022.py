"""
forecast_spb_2022.py
--------------------
Retrospective forecasting experiment — SPB Winter 2022.

Run from the project root (Model_Comparison/) with:
    python forecasting/forecast_spb_2022.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path setup so imports work when called from any working directory ──
_HERE = Path(__file__).resolve().parent          # forecasting/
_ROOT = _HERE.parent                             # Model_Comparison/
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / "output" / "ai_sbm" / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME",  str(_ROOT / "output" / "ai_sbm" / "xdg_cache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ── Local imports ──
from LSTM_SBM import (
    run_custom_scenario,
    EpidemicSurrogateNet,
    DATASET_FILE,
    MODEL_FILE,
    TMAX,
)
from test_simulation import MODEL_CONFIG_TEMPLATE, SIMULATION_PARAMS
from forecasting_utils import (
    load_spb_data,
    detect_peak,
    build_scenario_splits,
    save_observed_hidden_splits,
    estimate_scale_factor,
    compute_forecast_metrics,
)

# ═══════════════════════════════════════════════════════
#  CONFIGURATION  (edit these to change experiment)
# ═══════════════════════════════════════════════════════
MODE       = "surrogate"   # "surrogate"  or  "sbm"
PEAK_MODE  = "raw"         # "raw"        or  "manual"
MANUAL_PEAK_DATE = "2022-02-08"
HORIZON    = 14
TOP_K      = 20
N_RANDOM   = 100 if MODE == "surrogate" else 30
NUM_SIMS_SBM = 20          # Monte-Carlo sims per theta in SBM mode

# Baseline parameters (SPB 2022 calibrated)
BASELINE = {
    "beta_network":   0.2436,
    "beta_household": 2.544,
    "delta":          1.0552,
    "fermi_mu":       19.2075,
}
BOUNDS = {
    "beta_network":   (0.05, 0.80),
    "beta_household": (0.50, 4.00),
    "delta":          (0.30, 1.50),
    "fermi_mu":       (4.00, 40.00),
}

# Paths
CSV_FILE   = str(_ROOT / "stopkoronavirus_clean_wave_winter.csv")
OUTPUT_DIR = str(_HERE / "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════
#  SURROGATE LOADER
# ═══════════════════════════════════════════════════════
_surrogate_cache = {}

def load_surrogate():
    if "model" in _surrogate_cache:
        return (_surrogate_cache["model"],
                _surrogate_cache["x_scaler"],
                _surrogate_cache["y_scaler"])

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "Surrogate model not found. Run LSTM_SBM.py first to train the surrogate.\n"
            f"Expected: {MODEL_FILE}"
        )
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_FILE}\n"
            "Run LSTM_SBM.py first to generate it."
        )

    data = np.load(DATASET_FILE)
    X, Y = data["X"], data["Y"]

    x_scaler = StandardScaler().fit(X)
    y_scaler = MinMaxScaler().fit(Y)

    model = EpidemicSurrogateNet(input_dim=X.shape[1], output_dim=Y.shape[1])
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu",
                                     weights_only=True))
    model.eval()

    _surrogate_cache.update({"model": model, "x_scaler": x_scaler,
                              "y_scaler": y_scaler})
    print("[SURROGATE] Model and scalers loaded.")
    return model, x_scaler, y_scaler


# ═══════════════════════════════════════════════════════
#  SIMULATE CURVE
# ═══════════════════════════════════════════════════════
def simulate_curve(theta, mode=None):
    """Return normalised infected fraction curve of length TMAX."""
    if mode is None:
        mode = MODE
    beta_net, beta_hh, delta, fermi_mu = theta

    if mode == "sbm":
        curve, _ = run_custom_scenario(
            beta_net, beta_hh, delta, fermi_mu,
            num_sims=NUM_SIMS_SBM, return_population=True
        )
        return np.asarray(curve, dtype=float)

    elif mode == "surrogate":
        model, x_scaler, y_scaler = load_surrogate()
        params = np.array([[beta_net, beta_hh, delta, fermi_mu]])
        x_sc  = x_scaler.transform(params)
        with torch.no_grad():
            pred_sc = model(torch.FloatTensor(x_sc)).numpy()
        curve = y_scaler.inverse_transform(pred_sc)[0]
        return np.maximum(curve, 0.0)
    else:
        raise ValueError(f"Unknown MODE: {mode}")


# ═══════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════
def loss_fn(theta, y_known):
    try:
        f = simulate_curve(theta)
        n = len(y_known)
        f_k = f[:n]
        a = estimate_scale_factor(y_known, f_k)
        if a < 1e-6:
            return 1e9
        y_hat = a * f_k
        return float(np.sqrt(np.mean((y_known - y_hat) ** 2)))
    except Exception:
        return 1e9


def _sample_theta():
    """Sample one random parameter vector within bounds."""
    return np.array([
        np.random.uniform(*BOUNDS["beta_network"]),
        np.random.uniform(*BOUNDS["beta_household"]),
        np.random.uniform(*BOUNDS["delta"]),
        np.random.uniform(*BOUNDS["fermi_mu"]),
    ])


def _perturb_baseline(frac=0.30):
    """Perturb baseline parameters by ±frac."""
    b = BASELINE
    factors = [
        np.random.uniform(1 - frac, 1 + frac),
        np.random.uniform(1 - frac, 1 + frac),
        np.random.uniform(1 - 0.20, 1 + 0.20),
        np.random.uniform(1 - frac, 1 + frac),
    ]
    vals = [
        b["beta_network"]   * factors[0],
        b["beta_household"] * factors[1],
        b["delta"]          * factors[2],
        b["fermi_mu"]       * factors[3],
    ]
    # clip to bounds
    keys = ["beta_network", "beta_household", "delta", "fermi_mu"]
    return np.array([
        float(np.clip(v, *BOUNDS[k]))
        for v, k in zip(vals, keys)
    ])


def calibrate(y_known, n_random=None, top_k=TOP_K):
    if n_random is None:
        n_random = N_RANDOM

    candidates = []

    # Always include the baseline
    theta0 = np.array([BASELINE["beta_network"], BASELINE["beta_household"],
                       BASELINE["delta"], BASELINE["fermi_mu"]])
    candidates.append(theta0)

    # Perturbations around baseline
    n_perturb = n_random // 3
    for _ in range(n_perturb):
        candidates.append(_perturb_baseline())

    # Pure random samples
    n_rand = n_random - n_perturb - 1
    for _ in range(n_rand):
        candidates.append(_sample_theta())

    print(f"  [CAL] Evaluating {len(candidates)} candidates …")
    results = []
    for i, theta in enumerate(candidates):
        loss = loss_fn(theta, y_known)
        results.append((loss, theta))
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(candidates)}  best_rmse={min(r[0] for r in results):.2f}")

    results.sort(key=lambda x: x[0])
    top = results[:top_k]
    print(f"  [CAL] Top-{top_k} best RMSE: {top[0][0]:.2f} … {top[-1][0]:.2f}")
    return top   # list of (rmse, theta)


# ═══════════════════════════════════════════════════════
#  FORECAST GENERATION
# ═══════════════════════════════════════════════════════
def generate_forecast(top_candidates, y_known, horizon=HORIZON):
    """Return (forecast_matrix, scale_factors) where each row is one trajectory."""
    trajectories = []
    scale_factors = []
    n_known = len(y_known)

    for rmse, theta in top_candidates:
        # Optional tiny perturbation for surrogate mode
        if MODE == "surrogate":
            noise = np.array([
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.03, 0.03),
                np.random.uniform(-0.05, 0.05),
            ])
            theta_j = np.clip(theta * (1 + noise),
                              [BOUNDS[k][0] for k in ["beta_network","beta_household","delta","fermi_mu"]],
                              [BOUNDS[k][1] for k in ["beta_network","beta_household","delta","fermi_mu"]])
        else:
            theta_j = theta

        f = simulate_curve(theta_j)
        a = estimate_scale_factor(y_known, f[:n_known])
        if a < 1e-6:
            continue
        y_hat_full = a * f
        # Extract horizon
        horizon_curve = y_hat_full[n_known: n_known + horizon]
        if len(horizon_curve) < horizon:
            pad = horizon - len(horizon_curve)
            horizon_curve = np.pad(horizon_curve, (0, pad), "edge")
        trajectories.append(horizon_curve)
        scale_factors.append(a)

    if not trajectories:
        raise RuntimeError("No valid forecast trajectories generated.")

    matrix = np.array(trajectories)
    return matrix, scale_factors


# ═══════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════
def plot_scenario(sc, best_fit_known, forecast_mean, forecast_p10, forecast_p90,
                  metrics, best_theta, best_scale, peak_date, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Known data
    ax.scatter(sc["dates_known"], sc["y_known"],
               color="black", s=18, zorder=5, label="Real data (observed)")

    # Hidden data
    ax.scatter(sc["dates_hidden"], sc["y_hidden"],
               color="gray", s=18, zorder=5, marker="^", label="Real data (hidden)")

    # Best fit on known segment
    dates_known_list = list(sc["dates_known"])
    if best_fit_known is not None:
        ax.plot(dates_known_list, best_fit_known,
                color="firebrick", lw=1.5, ls="--", alpha=0.7, label="Best fit (known)")

    # Forecast
    dates_hidden_list = list(sc["dates_hidden"])
    h = len(dates_hidden_list)
    ax.fill_between(dates_hidden_list,
                    forecast_p10[:h], forecast_p90[:h],
                    color="green", alpha=0.20, label="Forecast 10–90%")
    ax.plot(dates_hidden_list, forecast_mean[:h],
            color="green", lw=2.2, label="Forecast mean")

    # Cut-date line
    ax.axvline(sc["t_cut_date"], color="navy", lw=1.5, ls="--", alpha=0.8,
               label=f"Cut date ({sc['t_cut_date'].date()})")

    # Peak date marker
    ax.axvline(peak_date, color="darkorange", lw=1.2, ls=":", alpha=0.6,
               label=f"Peak ({peak_date.date()})")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Confirmed cases", fontsize=12)
    scenario_label = sc["name"].replace("_", " ")
    ax.set_title(f"Retrospective Forecasting: SPB 2022 — {scenario_label}", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Info box
    info = (
        f"mode: {MODE}\n"
        f"peak: {peak_date.date()}\n"
        f"cut:  {sc['t_cut_date'].date()}\n"
        f"RMSE: {metrics['rmse_forecast']:.1f}\n"
        f"R²:   {metrics['r2_forecast']:.3f}\n"
        f"cov:  {metrics['coverage_10_90']:.2%}\n"
        f"β_net={best_theta[0]:.4f}\n"
        f"β_hh={best_theta[1]:.4f}\n"
        f"δ={best_theta[2]:.4f}\n"
        f"μ={best_theta[3]:.4f}"
    )
    ax.text(0.02, 0.97, info, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved → {out_path}")


def plot_all_scenarios(all_data, peak_date, out_path):
    """Three-panel combined figure."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    for ax, d in zip(axes, all_data):
        sc = d["scenario"]
        fm = d["forecast_mean"]
        p10 = d["forecast_p10"]
        p90 = d["forecast_p90"]
        metrics = d["metrics"]
        h = len(sc["dates_hidden"])

        ax.scatter(sc["dates_known"], sc["y_known"],
                   color="black", s=14, zorder=5, label="Observed")
        ax.scatter(sc["dates_hidden"], sc["y_hidden"],
                   color="gray", s=14, zorder=5, marker="^", label="Hidden")

        dates_hidden = list(sc["dates_hidden"])
        ax.fill_between(dates_hidden, p10[:h], p90[:h],
                        color="green", alpha=0.20)
        ax.plot(dates_hidden, fm[:h], color="green", lw=2,
                label="Forecast mean")

        ax.axvline(sc["t_cut_date"], color="navy", lw=1.4, ls="--", alpha=0.8,
                   label="Cut date")
        ax.axvline(peak_date, color="darkorange", lw=1.2, ls=":", alpha=0.6)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        label = sc["name"].replace("_", " ")
        ax.set_title(f"{label} | RMSE={metrics['rmse_forecast']:.1f} "
                     f"R²={metrics['r2_forecast']:.3f} "
                     f"cov={metrics['coverage_10_90']:.2%}", fontsize=12)
        ax.set_ylabel("Confirmed cases", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Date", fontsize=11)
    fig.suptitle("Retrospective Forecasting: SPB Winter 2022\n"
                 f"Mode: {MODE} | Peak: {peak_date.date()}",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Combined figure saved → {out_path}")


# ═══════════════════════════════════════════════════════
#  SUMMARY MARKDOWN
# ═══════════════════════════════════════════════════════
def write_summary(metrics_rows, peak_date, raw_peak_date, out_path):
    lines = [
        "# Retrospective Forecasting Experiment — SPB Winter 2022\n",
        "## Objective",
        "This experiment evaluates whether the SBM-SIR model and/or its surrogate LSTM "
        "approximation can forecast the short-term evolution of the SPB winter 2022 epidemic "
        "wave using only partial observed data.\n",
        "## Method",
        "The observed incidence curve is split into known and hidden segments using three "
        "forecast origins:",
        "- 14 days before the epidemic peak.",
        "- 7 days before the epidemic peak.",
        "- 7 days after the epidemic peak.\n",
        "For each scenario, the model is calibrated only on the known segment. "
        "The following 14 days are then forecasted and compared against the hidden real data.\n",
        "## Peak Definition",
        f"- **Raw peak date (argmax):** {raw_peak_date.date()}",
        f"- **Selected peak mode:** `{PEAK_MODE}`",
        f"- **Peak date used:** {peak_date.date()}\n",
        "The observed SPB winter 2022 wave shows a high-incidence peak window around "
        "February 7–11, 2022. For reproducibility, the main forecasting experiment uses "
        "the raw maximum of the observed series as the official peak date.\n",
        "## Metrics\n",
    ]

    # Table header
    lines.append("| Scenario | Cut Date | RMSE | MAE | R² | Peak Δt (d) | Peak Height Err | Coverage |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in metrics_rows:
        lines.append(
            f"| {r['scenario']} | {r['t_cut_date']} | {r['rmse_forecast']:.1f} | "
            f"{r['mae_forecast']:.1f} | {r['r2_forecast']:.3f} | "
            f"{r['peak_time_error_days']} | {r['peak_height_relative_error']:.3f} | "
            f"{r['coverage_10_90']:.2%} |"
        )

    lines += [
        "\n## Interpretation\n",
        "**t_peak − 14:** Early forecast. Higher uncertainty is expected because the model "
        "sees only the initial growth phase.\n",
        "**t_peak − 7:** Near-peak forecast. The model has more information about the wave "
        "growth, so better peak timing and magnitude are expected.\n",
        "**t_peak + 7:** Post-peak forecast. Less useful for early warning, but useful to "
        "validate whether the model captures the epidemic decline.\n",
        "## Methodological Note\n",
        "This is a retrospective forecasting experiment using real SPB 2022 incidence data. "
        "The model is calibrated only on the observed segment before each forecast origin, "
        "while the hidden future segment is used exclusively for evaluation.\n",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [SUMMARY] Saved → {out_path}")


# ═══════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  SPB Winter 2022 — Retrospective Forecasting")
    print(f"  MODE={MODE}  PEAK_MODE={PEAK_MODE}  HORIZON={HORIZON}")
    print("=" * 60)

    # 1. Load data
    dates, y_real = load_spb_data(CSV_FILE)

    # 2. Peak detection
    raw_peak_date, raw_peak_idx = detect_peak(dates, y_real, mode="raw")
    peak_date, peak_idx = detect_peak(dates, y_real, mode=PEAK_MODE,
                                      manual_peak_date=MANUAL_PEAK_DATE)

    # 3. Scenario splits
    scenarios = build_scenario_splits(dates, y_real, peak_idx, horizon=HORIZON)
    save_observed_hidden_splits(
        scenarios, os.path.join(OUTPUT_DIR, "observed_hidden_splits.csv")
    )

    # Preload surrogate (if needed) once
    if MODE == "surrogate":
        load_surrogate()

    all_metrics_rows = []
    all_params_rows  = []
    all_plot_data    = []

    for sc in scenarios:
        print(f"\n{'─'*50}")
        print(f"[SCENARIO] {sc['name']}")
        print(f"{'─'*50}")

        y_known = sc["y_known"]
        y_hidden = sc["y_hidden"]

        # 4. Calibration
        t0_cal = time.time()
        top_candidates = calibrate(y_known, n_random=N_RANDOM, top_k=TOP_K)
        cal_runtime = time.time() - t0_cal
        print(f"  [CAL] Runtime: {cal_runtime:.1f}s")

        best_rmse, best_theta = top_candidates[0]
        f_best = simulate_curve(best_theta)
        best_scale = estimate_scale_factor(y_known, f_best[:len(y_known)])
        best_fit_known = (best_scale * f_best)[:len(y_known)]

        # 5. Forecast
        t0_fc = time.time()
        fc_matrix, scale_factors = generate_forecast(
            top_candidates, y_known, horizon=HORIZON
        )
        fc_runtime = time.time() - t0_fc

        forecast_mean = np.mean(fc_matrix, axis=0)
        forecast_p10  = np.percentile(fc_matrix, 10, axis=0)
        forecast_p90  = np.percentile(fc_matrix, 90, axis=0)

        # 6. Metrics
        metrics = compute_forecast_metrics(
            y_hidden, forecast_mean, forecast_p10, forecast_p90,
            peak_date, sc["dates_hidden"]
        )
        metrics["calibration_runtime_seconds"] = cal_runtime
        metrics["forecast_runtime_seconds"]    = fc_runtime

        print(f"  RMSE={metrics['rmse_forecast']:.1f}  "
              f"R²={metrics['r2_forecast']:.3f}  "
              f"cov={metrics['coverage_10_90']:.2%}")

        # 7. Plot individual scenario
        out_png = os.path.join(OUTPUT_DIR, f"forecast_{sc['name']}.png")
        plot_scenario(sc, best_fit_known, forecast_mean, forecast_p10, forecast_p90,
                      metrics, best_theta, best_scale, peak_date, out_png)

        # Accumulate
        all_plot_data.append({
            "scenario": sc,
            "forecast_mean": forecast_mean,
            "forecast_p10": forecast_p10,
            "forecast_p90": forecast_p90,
            "metrics": metrics,
        })

        # Metrics row
        metrics_row = {
            "scenario":         sc["name"],
            "mode":             MODE,
            "peak_mode":        PEAK_MODE,
            "peak_date":        peak_date.date(),
            "t_cut_date":       sc["t_cut_date"].date(),
            "forecast_start_date": sc["dates_hidden"].iloc[0].date(),
            "forecast_end_date":   sc["dates_hidden"].iloc[-1].date(),
            **metrics,
        }
        all_metrics_rows.append(metrics_row)

        # Params rows (top-k)
        for rmse_i, theta_i in top_candidates:
            a_i = estimate_scale_factor(y_known, simulate_curve(theta_i)[:len(y_known)])
            all_params_rows.append({
                "scenario":          sc["name"],
                "peak_mode":         PEAK_MODE,
                "peak_date":         peak_date.date(),
                "t_cut_date":        sc["t_cut_date"].date(),
                "beta_network":      theta_i[0],
                "beta_household":    theta_i[1],
                "delta":             theta_i[2],
                "fermi_mu":          theta_i[3],
                "scale_factor":      a_i,
                "calibration_rmse":  rmse_i,
                "mode":              MODE,
                "runtime_seconds":   cal_runtime,
            })

    # 8. Combined figure
    plot_all_scenarios(
        all_plot_data, peak_date,
        os.path.join(OUTPUT_DIR, "forecast_all_scenarios.png")
    )

    # 9. Save CSVs
    metrics_path = os.path.join(OUTPUT_DIR, "forecast_metrics.csv")
    params_path  = os.path.join(OUTPUT_DIR, "forecast_parameters.csv")
    pd.DataFrame(all_metrics_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(all_params_rows).to_csv(params_path, index=False)
    print(f"\n[SAVE] Metrics  → {metrics_path}")
    print(f"[SAVE] Params   → {params_path}")

    # 10. Summary markdown
    write_summary(
        all_metrics_rows, peak_date, raw_peak_date,
        os.path.join(OUTPUT_DIR, "forecast_summary.md")
    )

    print("\n" + "=" * 60)
    print("  Forecasting experiment completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
