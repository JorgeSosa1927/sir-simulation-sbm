"""
forecast_spb_2022_lstm_target.py
---------------------------------
Retrospective forecasting using the LSTM-fitted curve as pseudo-observed target.

Run from Model_Comparison/ with:
    python3 forecasting/forecast_spb_2022_lstm_target.py
"""

import os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from LSTM_SBM import (
    run_custom_scenario, EpidemicSurrogateNet,
    DATASET_FILE, MODEL_FILE, TMAX,
)
from forecasting_utils import (
    load_spb_data, detect_peak, build_scenario_splits,
    estimate_scale_factor, compute_forecast_metrics,
)

# ── CONFIG ──────────────────────────────────────────────
MODE     = "surrogate"
HORIZON  = 14
TOP_K    = 20
N_RANDOM = 100

BASELINE = dict(beta_network=0.2436, beta_household=2.544,
                delta=1.0552, fermi_mu=19.2075)
BOUNDS   = dict(beta_network=(0.05,0.80), beta_household=(0.50,4.00),
                delta=(0.30,1.50), fermi_mu=(4.00,40.00))

CSV_FILE   = str(_ROOT / "stopkoronavirus_clean_wave_winter.csv")
OUTPUT_DIR = str(_HERE / "output_lstm_target")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── SURROGATE LOADER ────────────────────────────────────
_cache = {}

def load_surrogate():
    if "model" in _cache:
        return _cache["model"], _cache["xs"], _cache["ys"]
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Run LSTM_SBM.py first. Missing: {MODEL_FILE}")
    data = np.load(DATASET_FILE)
    xs = StandardScaler().fit(data["X"])
    ys = MinMaxScaler().fit(data["Y"])
    model = EpidemicSurrogateNet(input_dim=data["X"].shape[1], output_dim=data["Y"].shape[1])
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
    model.eval()
    _cache.update({"model": model, "xs": xs, "ys": ys})
    print("[SURROGATE] Loaded.")
    return model, xs, ys

# ── SIMULATE ────────────────────────────────────────────
def simulate_curve(theta):
    beta_net, beta_hh, delta, fermi_mu = theta
    model, xs, ys = load_surrogate()
    x = xs.transform(np.array([[beta_net, beta_hh, delta, fermi_mu]]))
    with torch.no_grad():
        pred = model(torch.FloatTensor(x)).numpy()
    return np.maximum(ys.inverse_transform(pred)[0], 0.0)

# ── LSTM TARGET CURVE ───────────────────────────────────
def get_lstm_target_curve(dates, y_real):
    """Generate the LSTM-smoothed target using baseline params scaled to real data."""
    theta0 = [BASELINE["beta_network"], BASELINE["beta_household"],
               BASELINE["delta"], BASELINE["fermi_mu"]]
    f = simulate_curve(theta0)        # normalised fraction, length TMAX
    n = len(y_real)

    # Interpolate f to match n real data points
    t_f   = np.linspace(0, 1, len(f))
    t_tgt = np.linspace(0, 1, n)
    f_interp = np.interp(t_tgt, t_f, f)

    # Scale using full real curve
    a = estimate_scale_factor(y_real, f_interp)
    y_lstm = a * f_interp

    print(f"[LSTM-TARGET] scale_factor={a:.2f}  "
          f"peak={y_lstm.max():.0f} @ idx={np.argmax(y_lstm)}")
    return y_lstm, a

# ── CALIBRATION ─────────────────────────────────────────
def loss_fn(theta, y_known):
    try:
        f = simulate_curve(theta)
        n = len(y_known)
        # interpolate f to match length
        f_interp = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(f)), f)
        a = estimate_scale_factor(y_known, f_interp)
        if a < 1e-6: return 1e9
        return float(np.sqrt(np.mean((y_known - a*f_interp)**2)))
    except Exception:
        return 1e9

def calibrate(y_known):
    b = BASELINE
    theta0 = np.array([b["beta_network"], b["beta_household"], b["delta"], b["fermi_mu"]])
    candidates = [theta0]
    for _ in range(N_RANDOM // 3):
        frac = 0.30
        v = theta0 * np.array([np.random.uniform(1-frac,1+frac),
                                np.random.uniform(1-frac,1+frac),
                                np.random.uniform(1-0.20,1+0.20),
                                np.random.uniform(1-frac,1+frac)])
        ks = ["beta_network","beta_household","delta","fermi_mu"]
        candidates.append(np.array([np.clip(v[i],*BOUNDS[k]) for i,k in enumerate(ks)]))
    for _ in range(N_RANDOM - len(candidates)):
        candidates.append(np.array([np.random.uniform(*BOUNDS[k])
                                    for k in ["beta_network","beta_household","delta","fermi_mu"]]))
    results = sorted([(loss_fn(th, y_known), th) for th in candidates], key=lambda x: x[0])
    top = results[:TOP_K]
    print(f"  [CAL] best_rmse={top[0][0]:.2f} … {top[-1][0]:.2f}")
    return top

# ── FORECAST ────────────────────────────────────────────
def generate_forecast(top_candidates, y_known):
    n = len(y_known)
    trajectories = []
    for _, theta in top_candidates:
        f = simulate_curve(theta)
        f_interp = np.interp(np.linspace(0,1,n+HORIZON), np.linspace(0,1,len(f)), f)
        a = estimate_scale_factor(y_known, f_interp[:n])
        if a < 1e-6: continue
        trajectories.append((a * f_interp)[n:n+HORIZON])
    if not trajectories:
        raise RuntimeError("No valid trajectories.")
    mat = np.array(trajectories)
    return mat, np.mean(mat,0), np.percentile(mat,10,0), np.percentile(mat,90,0)

# ── METRICS ─────────────────────────────────────────────
def metrics_vs(y_true, fm, p10, p90):
    h = len(y_true)
    rmse = float(np.sqrt(mean_squared_error(y_true, fm[:h])))
    mae  = float(mean_absolute_error(y_true, fm[:h]))
    r2   = float(r2_score(y_true, fm[:h]))
    cov  = float(np.sum((y_true >= p10[:h]) & (y_true <= p90[:h]))) / h
    return rmse, mae, r2, cov

# ── PLOT SINGLE SCENARIO ─────────────────────────────────
def plot_scenario(sc, dates, y_real, y_lstm, fm, p10, p90,
                  metrics, best_theta, lstm_peak_date, real_peak_date, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    t_cut_idx = sc["t_cut_idx"]
    h = len(sc["dates_hidden"])

    # Real data (reference)
    ax.scatter(dates, y_real, color="black", s=12, zorder=3,
               alpha=0.5, label="Real SPB data (reference)")

    # Full LSTM target
    ax.plot(dates, y_lstm, color="firebrick", lw=1.5, ls="--",
            alpha=0.6, label="LSTM target (full)")

    # LSTM known segment
    ax.plot(sc["dates_known"], sc["y_known"], color="firebrick", lw=2.2,
            label="LSTM observed segment")

    # LSTM hidden segment
    ax.plot(sc["dates_hidden"], sc["y_hidden"], color="salmon", lw=1.8,
            ls="-.", label="LSTM hidden segment")

    # Forecast
    dh = list(sc["dates_hidden"])
    ax.fill_between(dh, p10[:h], p90[:h], color="green", alpha=0.20)
    ax.plot(dh, fm[:h], color="green", lw=2.2, label="Forecast mean (SBM surrogate)")

    # Verticals
    ax.axvline(sc["t_cut_date"], color="navy", lw=1.5, ls="--",
               label=f"Cut ({sc['t_cut_date'].date()})")
    ax.axvline(lstm_peak_date, color="firebrick", lw=1.0, ls=":",
               alpha=0.7, label=f"LSTM peak ({lstm_peak_date.date()})")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Date", fontsize=11); ax.set_ylabel("Cases", fontsize=11)
    ax.set_title(f"Retrospective Forecasting Using LSTM Target: {sc['name'].replace('lstm_','').replace('_',' ')}",
                 fontsize=13)
    ax.grid(True, alpha=0.3)

    rmse_l, mae_l, r2_l, cov_l = metrics["lstm"]
    rmse_r, _, r2_r, _ = metrics["real"]
    info = (f"mode: {MODE}\nLSTM peak: {lstm_peak_date.date()}\n"
            f"cut: {sc['t_cut_date'].date()}\n"
            f"RMSE(LSTM): {rmse_l:.1f}\nR²(LSTM): {r2_l:.3f}\n"
            f"cov(LSTM): {cov_l:.2%}\n"
            f"RMSE(real): {rmse_r:.1f}\nR²(real): {r2_r:.3f}\n"
            f"β_net={best_theta[0]:.4f} β_hh={best_theta[1]:.4f}\n"
            f"δ={best_theta[2]:.4f} μ={best_theta[3]:.4f}")
    ax.text(0.02,0.97,info, transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round",facecolor="white",alpha=0.85,edgecolor="gray"))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {out_path}")

# ── COMBINED FIGURE ──────────────────────────────────────
def plot_all(all_data, lstm_peak_date, out_path):
    fig, axes = plt.subplots(3,1, figsize=(12,16))
    for ax, d in zip(axes, all_data):
        sc=d["sc"]; fm=d["fm"]; p10=d["p10"]; p90=d["p90"]
        m=d["metrics"]; h=len(sc["dates_hidden"]); dh=list(sc["dates_hidden"])
        ax.plot(d["dates"], d["y_lstm"], color="firebrick", lw=1.3, ls="--", alpha=0.5)
        ax.plot(sc["dates_known"], sc["y_known"], color="firebrick", lw=2)
        ax.plot(sc["dates_hidden"], sc["y_hidden"], color="salmon", lw=1.6, ls="-.")
        ax.fill_between(dh, p10[:h], p90[:h], color="green", alpha=0.20)
        ax.plot(dh, fm[:h], color="green", lw=2, label="Forecast")
        ax.scatter(d["dates"], d["y_real"], color="black", s=10, alpha=0.4, zorder=3)
        ax.axvline(sc["t_cut_date"], color="navy", lw=1.3, ls="--")
        ax.axvline(lstm_peak_date, color="firebrick", lw=1.0, ls=":", alpha=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        rmse_l,_,r2_l,cov_l = m["lstm"]; rmse_r,_,r2_r,_ = m["real"]
        ax.set_title(f"{sc['name']} | RMSE(LSTM)={rmse_l:.1f} R²={r2_l:.3f} "
                     f"cov={cov_l:.2%} | RMSE(real)={rmse_r:.1f}", fontsize=10)
        ax.set_ylabel("Cases",fontsize=9); ax.grid(True,alpha=0.3); ax.legend(fontsize=8)
    axes[-1].set_xlabel("Date",fontsize=10)
    fig.suptitle(f"LSTM-Target Retrospective Forecasting — SPB 2022\n"
                 f"LSTM peak: {lstm_peak_date.date()}", fontsize=13, y=1.005)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] Combined → {out_path}")

# ── SUMMARY ─────────────────────────────────────────────
def write_summary(metric_rows, lstm_peak_date, real_peak_date, out_path):
    lines = [
        "# Retrospective Forecasting Using LSTM Target — SPB Winter 2022\n",
        "## Objective",
        "This experiment evaluates whether the SBM-SIR / Surrogate + SBM framework can "
        "forecast a smoothed epidemic trajectory generated by the LSTM surrogate, instead "
        "of calibrating directly on noisy real observations.\n",
        "## Difference from real-data forecasting",
        "The previous experiment used the observed SPB 2022 confirmed cases as the "
        "forecasting target. In this experiment, the LSTM-fitted epidemic curve is used "
        "as a pseudo-observed target trajectory.\n",
        "## Peak definition",
        f"- **Real observed peak date:** {real_peak_date.date()}",
        f"- **LSTM target peak date:** {lstm_peak_date.date()}\n",
        "## Method",
        "For each scenario, the LSTM target curve is split into known and hidden segments "
        "around the LSTM peak. The model is calibrated only on the known LSTM segment and "
        "then used to forecast the following 14 days.\n",
        "## Metrics\n",
        "| Scenario | Cut Date | RMSE(LSTM) | R²(LSTM) | Cov(LSTM) | RMSE(real) | R²(real) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in metric_rows:
        lines.append(f"| {r['scenario']} | {r['t_cut_date']} | "
                     f"{r['rmse_forecast_lstm']:.1f} | {r['r2_forecast_lstm']:.3f} | "
                     f"{r['coverage_lstm_10_90']:.2%} | "
                     f"{r['rmse_forecast_real_reference']:.1f} | "
                     f"{r['r2_forecast_real_reference']:.3f} |")
    lines += [
        "\n## Interpretation\n",
        "**t_peak − 14:** Tests whether the model can anticipate the LSTM-smoothed epidemic growth before the peak.\n",
        "**t_peak − 7:** Evaluates near-peak forecasting when the model has already observed most of the rising phase.\n",
        "**t_peak + 7:** Evaluates whether the model can reproduce the decline phase after the LSTM peak.\n",
        "## Methodological Note\n",
        "This experiment uses the LSTM-fitted SPB 2022 curve as a pseudo-observed target. "
        "The real data are shown only as a reference, while calibration and primary "
        "evaluation are performed against the LSTM target trajectory.\n",
    ]
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [SUMMARY] {out_path}")

# ── MAIN ─────────────────────────────────────────────────
def main():
    print("="*60)
    print("  LSTM-Target Retrospective Forecasting — SPB 2022")
    print(f"  MODE={MODE}  HORIZON={HORIZON}")
    print("="*60)

    # 1. Load real data
    dates, y_real = load_spb_data(CSV_FILE)
    real_peak_date, real_peak_idx = detect_peak(dates, y_real, mode="raw")

    # 2. Generate LSTM target
    load_surrogate()
    y_lstm, scale_a = get_lstm_target_curve(dates, y_real)
    lstm_peak_idx  = int(np.argmax(y_lstm))
    lstm_peak_date = dates.iloc[lstm_peak_idx]
    print(f"[LSTM-TARGET] peak_date={lstm_peak_date.date()}  peak_cases={y_lstm[lstm_peak_idx]:.0f}")

    # Save target curve
    pd.DataFrame({"date": [d.date() for d in dates],
                  "real_cases": y_real,
                  "lstm_target_cases": y_lstm}
                 ).to_csv(os.path.join(OUTPUT_DIR,"lstm_target_curve.csv"), index=False)

    # 3. Build scenarios on LSTM curve
    scenarios = build_scenario_splits(dates, y_lstm, lstm_peak_idx, horizon=HORIZON)
    # Rename scenarios
    for sc in scenarios:
        sc["name"] = "lstm_" + sc["name"]

    # Save splits
    rows = []
    for sc in scenarios:
        for d,v in zip(sc["dates_known"],sc["y_known"]):
            rows.append({"scenario":sc["name"],"date":d.date(),"value":v,"segment":"known"})
        for d,v in zip(sc["dates_hidden"],sc["y_hidden"]):
            rows.append({"scenario":sc["name"],"date":d.date(),"value":v,"segment":"hidden"})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR,"observed_hidden_lstm_splits.csv"),index=False)

    all_metrics_rows, all_params_rows, all_plot_data = [], [], []

    for sc in scenarios:
        print(f"\n{'─'*50}\n[SCENARIO] {sc['name']}\n{'─'*50}")
        y_known  = sc["y_known"]
        y_hidden = sc["y_hidden"]

        # 4. Calibrate vs LSTM
        t0 = time.time()
        top = calibrate(y_known)
        cal_rt = time.time() - t0
        best_rmse, best_theta = top[0]

        # 5. Forecast
        t1 = time.time()
        mat, fm, p10, p90 = generate_forecast(top, y_known)
        fc_rt = time.time() - t1

        # 6. Primary metrics vs LSTM hidden
        rmse_l, mae_l, r2_l, cov_l = metrics_vs(y_hidden, fm, p10, p90)
        print(f"  RMSE(LSTM)={rmse_l:.1f}  R²={r2_l:.3f}  cov={cov_l:.2%}")

        # 7. Secondary metrics vs real hidden
        t_cut_idx = sc["t_cut_idx"]
        y_real_hidden = y_real[t_cut_idx+1: t_cut_idx+1+HORIZON]
        h2 = min(len(y_real_hidden), len(fm))
        rmse_r = float(np.sqrt(mean_squared_error(y_real_hidden[:h2], fm[:h2])))
        mae_r  = float(mean_absolute_error(y_real_hidden[:h2], fm[:h2]))
        r2_r   = float(r2_score(y_real_hidden[:h2], fm[:h2]))
        cov_r  = float(np.sum((y_real_hidden[:h2]>=p10[:h2])&(y_real_hidden[:h2]<=p90[:h2])))/h2

        # 8. Plot
        out_png = os.path.join(OUTPUT_DIR, f"forecast_{sc['name']}.png")
        plot_scenario(sc, dates, y_real, y_lstm, fm, p10, p90,
                      {"lstm":(rmse_l,mae_l,r2_l,cov_l),"real":(rmse_r,mae_r,r2_r,cov_r)},
                      best_theta, lstm_peak_date, real_peak_date, out_png)

        all_plot_data.append({"sc":sc,"fm":fm,"p10":p10,"p90":p90,
                               "metrics":{"lstm":(rmse_l,mae_l,r2_l,cov_l),
                                          "real":(rmse_r,mae_r,r2_r,cov_r)},
                               "dates":dates,"y_real":y_real,"y_lstm":y_lstm})

        # Metrics row
        all_metrics_rows.append({
            "scenario": sc["name"], "target_source":"LSTM_surrogate",
            "mode": MODE,
            "lstm_peak_date": lstm_peak_date.date(), "real_peak_date": real_peak_date.date(),
            "t_cut_date": sc["t_cut_date"].date(),
            "forecast_start_date": sc["dates_hidden"].iloc[0].date(),
            "forecast_end_date":   sc["dates_hidden"].iloc[-1].date(),
            "rmse_forecast_lstm": rmse_l, "mae_forecast_lstm": mae_l,
            "r2_forecast_lstm": r2_l, "coverage_lstm_10_90": cov_l,
            "peak_time_error_days_lstm": int(np.argmax(fm[:len(y_hidden)])) - int(np.argmax(y_hidden)),
            "peak_height_relative_error_lstm": (float(np.max(fm[:len(y_hidden)]))-float(np.max(y_hidden)))/max(float(np.max(y_hidden)),1),
            "rmse_forecast_real_reference": rmse_r, "mae_forecast_real_reference": mae_r,
            "r2_forecast_real_reference": r2_r, "coverage_real_reference_10_90": cov_r,
            "calibration_runtime_seconds": cal_rt, "forecast_runtime_seconds": fc_rt,
        })

        for rmse_i, theta_i in top:
            f_i = simulate_curve(theta_i)
            f_i_interp = np.interp(np.linspace(0,1,len(y_known)),np.linspace(0,1,len(f_i)),f_i)
            a_i = estimate_scale_factor(y_known, f_i_interp)
            all_params_rows.append({
                "scenario": sc["name"], "target_source":"LSTM_surrogate",
                "lstm_peak_date": lstm_peak_date.date(), "real_peak_date": real_peak_date.date(),
                "t_cut_date": sc["t_cut_date"].date(),
                "beta_network": theta_i[0], "beta_household": theta_i[1],
                "delta": theta_i[2], "fermi_mu": theta_i[3],
                "scale_factor": a_i, "calibration_rmse_lstm": rmse_i,
                "mode": MODE, "runtime_seconds": cal_rt,
            })

    # Combined plot
    plot_all(all_plot_data, lstm_peak_date,
             os.path.join(OUTPUT_DIR,"forecast_lstm_all_scenarios.png"))

    # Save CSVs
    pd.DataFrame(all_metrics_rows).to_csv(os.path.join(OUTPUT_DIR,"forecast_lstm_metrics.csv"),index=False)
    pd.DataFrame(all_params_rows).to_csv(os.path.join(OUTPUT_DIR,"forecast_lstm_parameters.csv"),index=False)
    print(f"\n[SAVE] Metrics  → {OUTPUT_DIR}/forecast_lstm_metrics.csv")
    print(f"[SAVE] Params   → {OUTPUT_DIR}/forecast_lstm_parameters.csv")

    write_summary(all_metrics_rows, lstm_peak_date, real_peak_date,
                  os.path.join(OUTPUT_DIR,"forecast_lstm_summary.md"))

    print("\n"+"="*60)
    print("  LSTM-target forecasting completed successfully.")
    print("="*60)

if __name__ == "__main__":
    main()
