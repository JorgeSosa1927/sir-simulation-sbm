"""
predictability_horizon.py
-------------------------
Evaluates the predictability limit (H_max) by analyzing the growth of
the uncertainty band (P90 - P10) relative to the central median trajectory.
Three cut points are used: before the peak, at the peak, and after the peak.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import random
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from paper_style_forecasting.generate_sbm_target import generate_synthetic_target
from paper_style_forecasting.paper_style_utils import (
    load_surrogate, calibrate, generate_forecast, simulate_curve,
    calc_metrics, MODE, TOP_K, TMAX
)

OUTPUT_DIR = str(_HERE / "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CONFIGURATION ──────────────────────────────────────────
# Define predictability loss threshold.
# If the band (P90-P10) is larger than RBW_THRESHOLD times the global
# predicted peak, the forecast is treated as unpredictable.
RBW_THRESHOLD = 0.75
# Minimum threshold for median to avoid division by zero artifacts at the tail
MIN_MEDIAN_VAL = 1e-4

def calculate_predictability_horizon(fm_median, p10, p90, reference_peak):
    """
    Returns the number of days (H_max) until the prediction becomes unreliable.
    Unreliable is defined by the global relative bandwidth:
    (P90 - P10) / reference_peak > RBW_THRESHOLD

    The local relative bandwidth is also returned for diagnostics:
    (P90 - P10) / Median
    """
    global_rbw = []
    local_rbw = []
    h_max = None
    if reference_peak <= 0:
        raise ValueError("reference_peak must be positive.")

    for h, (med, lower, upper) in enumerate(zip(fm_median, p10, p90)):
        spread = upper - lower
        global_val = spread / reference_peak

        if med < MIN_MEDIAN_VAL:
            # If median is extremely small, we are at the tail.
            # We look at the absolute spread. If spread is large while median is tiny, it's useless.
            if spread > MIN_MEDIAN_VAL * 10:
                local_val = float('inf')
            else:
                local_val = 0.0
        else:
            local_val = spread / med

        global_rbw.append(global_val)
        local_rbw.append(local_val)
        
        # Determine H_max if not already found
        if h_max is None and global_val > RBW_THRESHOLD:
            h_max = h
            
    if h_max is None:
        h_max = len(fm_median) # Fully predictable until the end
        
    return global_rbw, local_rbw, h_max


def predicted_global_peak(top_params):
    """
    Estimate the global scale of the predicted epidemic curve from t=0..TMAX.
    Uses the median trajectory across the accepted parameter set, not only the
    future forecast segment.
    """
    full_curves = []
    for _, th in top_params:
        curve = simulate_curve(th)
        if len(curve) < TMAX:
            curve = np.pad(curve, (0, TMAX - len(curve)), "edge")
        full_curves.append(curve[:TMAX])

    if not full_curves:
        raise RuntimeError("No valid full predicted trajectories.")

    median_full_curve = np.median(np.array(full_curves), axis=0)
    return float(np.max(median_full_curve))

def plot_predictability(all_data, t_peak, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    
    for ax, d in zip(axes, all_data):
        sc = d["sc"]
        t_cut = sc["t_cut"]
        known = sc["known"]
        hidden = sc["hidden"]
        target = d["target"]
        fm_median = d["fm_median"]
        p10 = d["p10"]
        p90 = d["p90"]
        h_max = d["h_max"]
        
        h = len(hidden)
        t_arr = np.arange(TMAX)
        t_hid = np.arange(t_cut + 1, t_cut + 1 + h)
        
        ax.plot(t_arr, target, color="black", lw=1.2, alpha=0.4, label="True Target")
        ax.plot(np.arange(t_cut + 1), known, color="black", lw=2.5, label="Known History")
        
        ax.plot(t_hid, fm_median, color="green", lw=2.5, label="Forecast Median")
        ax.fill_between(t_hid, p10, p90, color="green", alpha=0.25, label="10th-90th Interval")
        
        ax.axvline(t_cut, color="navy", lw=2.0, ls="--", label=f"Origin (t={t_cut})")
        ax.axvline(t_peak, color="red", lw=1.5, ls=":", alpha=0.6, label=f"Peak (t={t_peak})")
        
        # Mark Predictability Limit
        limit_t = t_cut + 1 + h_max
        if limit_t < TMAX:
            ax.axvline(limit_t, color="darkorange", lw=2.5, ls="-.", 
                       label=f"Predictability Limit (H_max={h_max} days)")
            ax.axvspan(limit_t, TMAX-1, color="red", alpha=0.05)
        
        ax.set_title(
            f"Scenario: {sc['name']} (Origin t={t_cut}) | "
            f"Predictable Horizon = {h_max} days",
            fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Infected fraction", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
        
    axes[-1].set_xlabel("Time step", fontsize=11)
    fig.suptitle(
        f"Predictability Horizon Analysis "
        f"(Threshold = {RBW_THRESHOLD} x predicted global peak)\n"
        f"MODE={MODE}  |  Target Peak = {t_peak}", 
        fontsize=15, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved predictability figure -> {out_path}")


def plot_predictability_parameters(all_data, out_path):
    from paper_style_forecasting.paper_style_utils import BASELINE, KEYS
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    colors = ["steelblue", "darkorange", "green"]
    x_ticks_labels = ["t_peak - 14", "t_peak", "t_peak + 14"]
    
    for i, key in enumerate(KEYS):
        data_to_plot = []
        for sc_data in all_data:
            top_params = sc_data["top_params"]
            params_arr = np.array([th for _, th in top_params])
            data_to_plot.append(params_arr[:, i])
            
        parts = axes[i].violinplot(data_to_plot, showmeans=False, showmedians=True)
        
        # Color each violin
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
            
        # Style the lines
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1.5)
            if partname == 'cmedians':
                vp.set_color('white') # Make median stand out
            
        # Add true baseline
        axes[i].axhline(BASELINE[key], color="red", lw=2, ls="--", label="True Target Value")
        
        axes[i].set_title(key, fontsize=12, fontweight="bold")
        axes[i].set_xticks([1, 2, 3])
        axes[i].set_xticklabels(x_ticks_labels)
        
        if i == 0:
            axes[i].legend(fontsize=9, loc="best")
            
    fig.suptitle("Evolution of Parameter Uncertainty Across Forecasting Horizons", fontsize=15, y=1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved parameter space figure -> {out_path}")


def main():
    # Force determinism
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 60)
    print("  PREDICTABILITY HORIZON ANALYSIS")
    print("=" * 60)

    # 1. Target Curve
    target_frac, population, t_peak = generate_synthetic_target()
    print(f"[MAIN] t_peak={t_peak}  peak_frac={target_frac[t_peak]:.6f}")

    # 2. Preload Surrogate
    if MODE == "surrogate":
        load_surrogate()

    # 3. Define the three exact cut points requested
    # - Before peak: t_peak - 14
    # - At peak: t_peak
    # - After peak: t_peak + 14
    scenarios = [
        {"name": "pre_peak",  "t_cut": t_peak - 14},
        {"name": "at_peak",   "t_cut": t_peak},
        {"name": "post_peak", "t_cut": t_peak + 14},
    ]

    all_plot_data = []
    stats_rows = []

    for sc in scenarios:
        name = sc["name"]
        t_cut = sc["t_cut"]
        
        # Rest of the curve is hidden
        known = target_frac[:t_cut + 1]
        hidden = target_frac[t_cut + 1:]
        horizon = len(hidden)
        
        sc["known"] = known
        sc["hidden"] = hidden
        
        print(f"\n──────────────────────────────────────────────────")
        print(f"[SCENARIO] {name} | t_cut={t_cut} | horizon={horizon}")
        print(f"──────────────────────────────────────────────────")
        
        # Calibration
        top = calibrate(known, mode=MODE)
        
        # Forecast until TMAX
        mat, fm_mean, fm_median, p10, p90 = generate_forecast(top, known, horizon=horizon)
        
        # Predictability Analysis
        reference_peak_predicted = predicted_global_peak(top)
        global_rbw_curve, local_rbw_curve, h_max = calculate_predictability_horizon(
            fm_median, p10, p90, reference_peak_predicted
        )
        print(f"  -> Predictable Horizon (H_max): {h_max} days")
        print(f"  -> Predicted reference peak: {reference_peak_predicted:.6f}")
        
        all_plot_data.append({
            "sc": sc,
            "target": target_frac,
            "fm_median": fm_median,
            "fm_mean": fm_mean,
            "p10": p10,
            "p90": p90,
            "mat": mat,
            "h_max": h_max,
            "global_rbw": global_rbw_curve,
            "local_rbw": local_rbw_curve,
            "reference_peak_predicted": reference_peak_predicted,
            "top_params": top
        })
        
        # Collect stats
        for h_idx, (med, p_low, p_high, global_rbw_val, local_rbw_val) in enumerate(
            zip(fm_median, p10, p90, global_rbw_curve, local_rbw_curve)
        ):
            stats_rows.append({
                "scenario": name,
                "t_origin": t_cut,
                "t_target": t_cut + 1 + h_idx,
                "days_ahead": h_idx + 1,
                "target_val": hidden[h_idx],
                "forecast_median": med,
                "p10": p_low,
                "p90": p_high,
                "reference_peak_predicted": reference_peak_predicted,
                "global_relative_bandwidth": global_rbw_val,
                "local_relative_bandwidth": local_rbw_val,
                "is_predictable": h_idx < h_max
            })

    # Save outputs
    out_png = os.path.join(OUTPUT_DIR, "predictability_horizon.png")
    plot_predictability(all_plot_data, t_peak, out_png)
    
    out_params_png = os.path.join(OUTPUT_DIR, "predictability_parameters.png")
    plot_predictability_parameters(all_plot_data, out_params_png)
    
    out_csv = os.path.join(OUTPUT_DIR, "predictability_stats.csv")
    pd.DataFrame(stats_rows).to_csv(out_csv, index=False)
    print(f"  [CSV] Saved predictability stats -> {out_csv}")
    print("=" * 60)
    print("  Predictability analysis completed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
