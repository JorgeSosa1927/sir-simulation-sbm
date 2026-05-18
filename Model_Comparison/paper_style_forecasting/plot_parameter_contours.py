"""
plot_parameter_contours.py
--------------------------
Generates a confidence contour plot (corner plot) of the 4D parameter space
across the three forecasting scenarios (pre_peak, at_peak, post_peak).
Uses Seaborn to plot 2D KDEs to visualize parameter trade-offs.
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from paper_style_forecasting.generate_sbm_target import generate_synthetic_target
from paper_style_forecasting.paper_style_utils import (
    load_surrogate, loss_fn, _sample, _perturb, BASELINE, KEYS, MODE
)

OUTPUT_DIR = str(_HERE / "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# For a smooth contour plot, we need more samples than standard forecasting
N_SAMPLES = 2000
TOP_K = 150

def calibrate_for_contours(known):
    """Custom calibration that generates enough points for smooth KDEs."""
    theta0 = np.array([BASELINE[k] for k in KEYS])
    candidates = [theta0]
    
    # 30% perturbed from baseline, 70% random
    for _ in range(int(N_SAMPLES * 0.3)):
        candidates.append(_perturb())
    while len(candidates) < N_SAMPLES:
        candidates.append(_sample())

    print(f"  [CAL] Evaluating {len(candidates)} candidates for contours…")
    results = []
    for i, th in enumerate(candidates):
        results.append((loss_fn(th, known), th))
        if (i + 1) % 500 == 0:
            best = min(r[0] for r in results)
            print(f"    {i+1}/{len(candidates)} best_rmse={best:.5f}")

    results.sort(key=lambda x: x[0])
    top = results[:TOP_K]
    print(f"  [CAL] Top-{TOP_K} RMSE range: {top[0][0]:.5f} … {top[-1][0]:.5f}")
    return top

def main():
    # Force determinism
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 60)
    print("  PARAMETER CONFIDENCE CONTOURS ANALYSIS")
    print("=" * 60)

    target_frac, population, t_peak = generate_synthetic_target()
    print(f"[MAIN] Target peak at day {t_peak}")

    if MODE == "surrogate":
        load_surrogate()

    scenarios = [
        {"name": "pre_peak (t-14)", "t_cut": t_peak - 14},
        {"name": "at_peak (t)",     "t_cut": t_peak},
        {"name": "post_peak (t+14)","t_cut": t_peak + 14},
    ]

    all_params = []
    
    for sc in scenarios:
        name = sc["name"]
        t_cut = sc["t_cut"]
        known = target_frac[:t_cut + 1]
        
        print(f"\n──────────────────────────────────────────────────")
        print(f"[SCENARIO] {name} | t_cut={t_cut}")
        
        top = calibrate_for_contours(known)
        
        # Extract parameters and store in a DataFrame
        for rmse, th in top:
            row = {"Scenario": name}
            for i, key in enumerate(KEYS):
                row[key] = th[i]
            all_params.append(row)

    df_all = pd.DataFrame(all_params)
    
    print("\n[PLOT] Generating Separate Corner Plots with Seaborn...")
    
    # Set seaborn style
    sns.set_theme(style="ticks")
    palette = {"pre_peak (t-14)": "steelblue", 
               "at_peak (t)": "darkorange", 
               "post_peak (t+14)": "green"}
               
    for scenario_name in df_all["Scenario"].unique():
        df_sub = df_all[df_all["Scenario"] == scenario_name]
        color = palette[scenario_name]
        
        # Create the PairGrid
        g = sns.PairGrid(df_sub, corner=True, height=2.5)
        
        # Map the lower triangle to KDE contour plots
        g.map_lower(sns.kdeplot, color=color, fill=True, alpha=0.5, levels=4, warn_singular=False)
        
        # Map the diagonal to KDE density plots
        g.map_diag(sns.kdeplot, color=color, fill=True, linewidth=2, alpha=0.6, warn_singular=False)
        
        # Add the True Target Value markers
        for i in range(4):
            for j in range(4):
                if i >= j:
                    ax = g.axes[i, j]
                    key_x = KEYS[j]
                    key_y = KEYS[i]
                    
                    # Plot true value lines
                    if i == j:
                        ax.axvline(BASELINE[key_x], color="red", ls="--", lw=2, zorder=10)
                    else:
                        ax.axvline(BASELINE[key_x], color="red", ls="--", lw=1.5, alpha=0.7, zorder=10)
                        ax.axhline(BASELINE[key_y], color="red", ls="--", lw=1.5, alpha=0.7, zorder=10)
                        ax.plot(BASELINE[key_x], BASELINE[key_y], marker="*", color="red", markersize=12, zorder=11)

        # Add title
        g.figure.suptitle(f"Parameter Confidence Contours | {scenario_name}", 
                          fontsize=18, y=1.02, fontweight="bold")
        
        safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("-", "minus")
        out_png = os.path.join(OUTPUT_DIR, f"parameter_confidence_contours_{safe_name}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [PLOT] Saved confidence contour plot -> {out_png}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
