"""
paper_style_forecast.py
-----------------------
Paper-style retrospective forecasting using a synthetic SBM-SIR target.

Run from Model_Comparison/ with:
    python3 paper_style_forecasting/paper_style_forecast.py
"""

import os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / "output" / "ai_sbm" / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME",  str(_ROOT / "output" / "ai_sbm" / "xdg_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_style_utils import (
    MODE, HORIZON, TOP_K, BASELINE, KEYS, BOUNDS,
    load_surrogate, simulate_curve, build_splits,
    calibrate, generate_forecast, calc_metrics,
)
from generate_sbm_target import generate_synthetic_target, TARGET_PARAMS
from LSTM_SBM import TMAX

OUTPUT_DIR = str(_HERE / "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── PLOTTING ─────────────────────────────────────────────
def plot_scenario(sc, target, t_peak, top_params, mat, fm, p10, p90,
                  metrics, out_path):
    t_cut = sc["t_cut"]
    hidden = sc["hidden"]
    known  = sc["known"]
    t_arr  = np.arange(TMAX)
    h      = HORIZON

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # ── Top panel: trajectories ──────────────────────────
    # Full target
    ax1.plot(t_arr, target, color="black", lw=1.5, alpha=0.5,
             label="Synthetic SBM-SIR target")
    # Known segment
    ax1.plot(np.arange(t_cut + 1), known, color="black", lw=2.5,
             label="Known segment")
    # Hidden segment
    t_hidden = np.arange(t_cut + 1, t_cut + 1 + len(hidden))
    ax1.plot(t_hidden, hidden, color="gray", lw=2, ls="-.",
             label="Hidden segment")

    # Forecast trajectories
    for traj in mat:
        ax1.plot(t_hidden[:h], traj[:h], color="green", alpha=0.12, lw=0.8)

    # Forecast band & mean
    ax1.fill_between(t_hidden[:h], p10[:h], p90[:h],
                     color="green", alpha=0.22, label="10th–90th forecast interval")
    ax1.plot(t_hidden[:h], fm[:h], color="green", lw=2.5,
             label="Forecast mean")

    # Verticals
    ax1.axvline(t_cut, color="navy", lw=1.8, ls="--",
                label=f"Forecast origin (t={t_cut})")
    ax1.axvline(t_peak, color="red", lw=1.2, ls=":", alpha=0.7,
                label=f"Peak (t={t_peak})")

    ax1.set_ylabel("Infected fraction", fontsize=11)
    label = sc["name"].replace("_", " ")
    ax1.set_title(
        f"Paper-Style Retrospective Forecast — {label}\n"
        f"RMSE={metrics['rmse_forecast']:.5f}  "
        f"R²={metrics['r2_forecast']:.3f}  "
        f"cov={metrics['coverage_10_90']:.2%}",
        fontsize=12,
    )
    ax1.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Bottom panel: posterior-like histograms ──────────
    params_arr = np.array([th for _, th in top_params])   # (TOP_K, 4)
    labels_bot = ["β_network", "β_household", "δ", "μ (fermi)"]
    n_p = 4
    for i in range(n_p):
        ax_i = ax2.inset_axes([i / n_p, 0.05, 0.9 / n_p, 0.88])
        ax_i.hist(params_arr[:, i], bins=10, color="steelblue",
                  edgecolor="white", alpha=0.85)
        ax_i.axvline(BASELINE[KEYS[i]], color="red", lw=1.2, ls="--")
        ax_i.set_title(labels_bot[i], fontsize=8)
        ax_i.tick_params(labelsize=6)
        ax_i.set_yticks([])

    ax2.axis("off")
    ax2.set_title("Posterior-like parameter distributions (top-k accepted)",
                  fontsize=9, loc="left", pad=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out_path}")


def plot_all(all_data, t_peak, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    for ax, d in zip(axes, all_data):
        sc   = d["sc"]; fm=d["fm"]; p10=d["p10"]; p90=d["p90"]
        mat  = d["mat"]; m=d["metrics"]; target=d["target"]
        t_cut = sc["t_cut"]; hidden=sc["hidden"]; known=sc["known"]
        h = HORIZON
        t_arr  = np.arange(TMAX)
        t_hid  = np.arange(t_cut + 1, t_cut + 1 + len(hidden))

        ax.plot(t_arr, target, color="black", lw=1.2, alpha=0.4)
        ax.plot(np.arange(t_cut + 1), known, color="black", lw=2)
        ax.plot(t_hid, hidden, color="gray", lw=1.8, ls="-.")
        for traj in mat:
            ax.plot(t_hid[:h], traj[:h], color="green", alpha=0.10, lw=0.7)
        ax.fill_between(t_hid[:h], p10[:h], p90[:h], color="green", alpha=0.20)
        ax.plot(t_hid[:h], fm[:h], color="green", lw=2.2, label="Forecast")
        ax.axvline(t_cut, color="navy", lw=1.5, ls="--")
        ax.axvline(t_peak, color="red", lw=1.0, ls=":", alpha=0.6)
        ax.set_title(
            f"{sc['name'].replace('_',' ')} | "
            f"RMSE={m['rmse_forecast']:.5f}  R²={m['r2_forecast']:.3f}  "
            f"cov={m['coverage_10_90']:.2%}",
            fontsize=10,
        )
        ax.set_ylabel("Infected fraction", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time step", fontsize=10)
    fig.suptitle(f"Paper-Style Forecasting — Synthetic SBM-SIR Target\n"
                 f"MODE={MODE}  t_peak={t_peak}", fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Combined → {out_path}")


def write_summary(metrics_rows, t_peak, peak_val, out_path):
    p = TARGET_PARAMS
    lines = [
        "# Paper-Style Forecasting Experiment\n",
        "## Objective",
        "This experiment replicates the forecasting logic of the hybrid-surrogate reference "
        "study using the proposed SBM-SIR framework.\n",
        "## Method",
        "A synthetic epidemic trajectory is first generated with the SBM-SIR model and treated "
        "as the target curve. The trajectory is then split into known and hidden segments around "
        "its epidemic peak. Three forecast origins are evaluated: 14 days before the peak, "
        "7 days before the peak, and 7 days after the peak. For each scenario, the LSTM surrogate "
        "is calibrated only on the known segment and then used to generate 14-day forecast "
        "trajectories.\n",
        "## Target model parameters",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| beta_network | {p['beta_network']} |",
        f"| beta_household | {p['beta_household']} |",
        f"| delta | {p['delta']} |",
        f"| fermi_mu | {p['fermi_mu']} |",
        f"| t_peak | {t_peak} |",
        f"| peak_fraction | {peak_val:.6f} |\n",
        "## Metrics\n",
        "| Scenario | t_cut | RMSE | MAE | R² | Coverage | Δt_peak | Δh_peak |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in metrics_rows:
        lines.append(
            f"| {r['scenario']} | {r['t_cut']} | {r['rmse_forecast']:.5f} | "
            f"{r['mae_forecast']:.5f} | {r['r2_forecast']:.3f} | "
            f"{r['coverage_10_90']:.2%} | {r['peak_time_error_days']} | "
            f"{r['peak_height_relative_error']:.3f} |"
        )
    lines += [
        "\n## Interpretation\n",
        "**t_peak − 14:** Early forecasting scenario. Higher uncertainty is expected because "
        "the model observes only the rising phase before the peak.\n",
        "**t_peak − 7:** Near-peak forecasting scenario. Better accuracy is expected because "
        "more of the epidemic growth has already been observed.\n",
        "**t_peak + 7:** Post-peak forecasting scenario. This is less useful for early warning "
        "but useful to verify whether the surrogate captures the declining phase.\n",
        "## Methodological Note\n",
        "This experiment follows the retrospective forecasting logic of the reference "
        "hybrid-surrogate study: the target epidemic curve is generated by a baseline "
        "network-based model, only the initial segment is revealed for calibration, and the "
        "following 14 days are forecasted and compared against the hidden target segment.\n",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [SUMMARY] {out_path}")


# ── MAIN ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Paper-Style Forecasting — Synthetic SBM-SIR Target")
    print(f"  MODE={MODE}  HORIZON={HORIZON}  TOP_K={TOP_K}")
    print("=" * 60)

    # 1. Generate synthetic target
    target_frac, population, t_peak = generate_synthetic_target()
    peak_val = float(target_frac[t_peak])
    print(f"[MAIN] t_peak={t_peak}  peak_frac={peak_val:.6f}")

    # 2. Preload surrogate
    if MODE == "surrogate":
        load_surrogate()

    # 3. Build splits
    scenarios = build_splits(target_frac, t_peak, horizon=HORIZON)

    all_metrics_rows, all_traj_rows, all_plot_data = [], [], []
    all_posterior_rows = {sc["name"]: [] for sc in scenarios}

    for sc in scenarios:
        name   = sc["name"]
        known  = sc["known"]
        hidden = sc["hidden"]
        t_cut  = sc["t_cut"]
        print(f"\n{'─'*50}\n[SCENARIO] {name}\n{'─'*50}")

        # 4. Calibrate
        t0 = time.time()
        top = calibrate(known)
        cal_rt = time.time() - t0
        print(f"  [CAL] Runtime: {cal_rt:.1f}s")

        # Save posterior
        for rank, (rmse_i, th_i) in enumerate(top):
            all_posterior_rows[name].append({
                "scenario": name, "rank": rank,
                "beta_network": th_i[0], "beta_household": th_i[1],
                "delta": th_i[2], "fermi_mu": th_i[3],
                "rmse_known": rmse_i, "mode": MODE,
            })

        # 5. Forecast
        t1 = time.time()
        mat, fm, p10, p90 = generate_forecast(top, known, horizon=HORIZON)
        fc_rt = time.time() - t1

        # 6. Metrics
        metrics = calc_metrics(hidden, fm, p10, p90, t_peak, t_cut)
        print(f"  RMSE={metrics['rmse_forecast']:.5f}  "
              f"R²={metrics['r2_forecast']:.3f}  "
              f"cov={metrics['coverage_10_90']:.2%}")

        # 7. Plot individual
        out_png = os.path.join(OUTPUT_DIR, f"forecast_{name}.png")
        plot_scenario(sc, target_frac, t_peak, top, mat, fm, p10, p90, metrics, out_png)

        # Trajectory rows
        t_hid = np.arange(t_cut + 1, t_cut + 1 + HORIZON)
        for tid, traj in enumerate(mat):
            for ti, val in zip(t_hid, traj):
                all_traj_rows.append({
                    "scenario": name, "trajectory_id": tid,
                    "t": int(ti), "forecast_fraction": float(val),
                    "is_forecast_horizon": True,
                })

        # Metrics row
        all_metrics_rows.append({
            "scenario": name, "mode": MODE,
            "t_peak": t_peak, "t_cut": t_cut,
            "forecast_start": t_cut + 1, "forecast_end": t_cut + HORIZON,
            **metrics,
            "calibration_runtime_seconds": cal_rt,
            "forecast_runtime_seconds": fc_rt,
        })

        all_plot_data.append({"sc": sc, "fm": fm, "p10": p10, "p90": p90,
                               "mat": mat, "metrics": metrics,
                               "target": target_frac})

    # 8. Combined figure
    plot_all(all_plot_data, t_peak,
             os.path.join(OUTPUT_DIR, "forecast_all_scenarios.png"))

    # 9. Save CSVs
    pd.DataFrame(all_metrics_rows).to_csv(
        os.path.join(OUTPUT_DIR, "forecast_metrics.csv"), index=False)
    pd.DataFrame(all_traj_rows).to_csv(
        os.path.join(OUTPUT_DIR, "forecast_trajectories.csv"), index=False)
    for name, rows in all_posterior_rows.items():
        pd.DataFrame(rows).to_csv(
            os.path.join(OUTPUT_DIR, f"posterior_parameters_{name}.csv"), index=False)
    print(f"\n[SAVE] CSVs written to {OUTPUT_DIR}")

    # 10. Summary
    write_summary(all_metrics_rows, t_peak, peak_val,
                  os.path.join(OUTPUT_DIR, "paper_style_forecast_summary.md"))

    print("\n" + "=" * 60)
    print("  Paper-style forecasting completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
