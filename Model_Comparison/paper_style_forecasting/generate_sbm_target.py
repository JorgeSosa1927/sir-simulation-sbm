"""
generate_sbm_target.py
----------------------
Generate a stable synthetic SBM-SIR target trajectory.

Run standalone:
    python3 paper_style_forecasting/generate_sbm_target.py
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / "output" / "ai_sbm" / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME",  str(_ROOT / "output" / "ai_sbm" / "xdg_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from LSTM_SBM import run_custom_scenario, TMAX

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed parameters — SPB-calibrated used as synthetic scenario
TARGET_PARAMS = dict(
    beta_network=0.2436,
    beta_household=2.544,
    delta=1.0552,
    fermi_mu=19.2075,
)
NUM_SIMS_TARGET = 100


def generate_synthetic_target(num_sims=NUM_SIMS_TARGET):
    print(f"[TARGET] Running SBM-SIR with {num_sims} replicas …")
    p = TARGET_PARAMS
    curve_frac, population = run_custom_scenario(
        p["beta_network"], p["beta_household"], p["delta"], p["fermi_mu"],
        num_sims=num_sims, return_population=True,
    )
    curve_frac = np.asarray(curve_frac, dtype=float)
    curve_cases = curve_frac * population

    t_peak = int(np.argmax(curve_frac))
    print(f"[TARGET] TMAX={TMAX}  t_peak={t_peak}  "
          f"peak_frac={curve_frac[t_peak]:.6f}  "
          f"peak_cases≈{curve_cases[t_peak]:.0f}  "
          f"population≈{population}")

    # Validate horizon availability
    for off in [-14, -7, +7]:
        t_cut = t_peak + off
        assert t_cut >= 1, f"t_cut={t_cut} before series start"
        assert t_cut < TMAX, f"t_cut={t_cut} beyond TMAX={TMAX}"
        assert TMAX - t_cut - 1 >= 14, f"Not enough points after t_cut={t_cut}"

    # Save CSV
    csv_path = OUTPUT_DIR / "sbm_synthetic_target.csv"
    pd.DataFrame({
        "t": np.arange(TMAX),
        "I_target_fraction": curve_frac,
        "I_target_cases_optional": curve_cases,
    }).to_csv(csv_path, index=False)
    print(f"[TARGET] Saved → {csv_path}")

    # Save figure
    fig_path = OUTPUT_DIR / "sbm_synthetic_target.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(TMAX)
    ax.plot(t, curve_frac, color="black", lw=2, label="Synthetic SBM-SIR target")
    ax.axvline(t_peak, color="red", ls="--", lw=1.5, alpha=0.8,
               label=f"Peak (t={t_peak}, I={curve_frac[t_peak]:.4f})")
    for off, col, lbl in [(-14,"steelblue","t_peak−14"),
                           (-7,"darkorange","t_peak−7"),
                           (+7,"green","t_peak+7")]:
        ax.axvline(t_peak + off, color=col, ls=":", lw=1.2, alpha=0.7, label=lbl)
    ax.set_title("Synthetic SBM-SIR Target Trajectory", fontsize=13)
    ax.set_xlabel("Time step", fontsize=11)
    ax.set_ylabel("Mean infected fraction", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[TARGET] Figure saved → {fig_path}")

    return curve_frac, population, t_peak


if __name__ == "__main__":
    generate_synthetic_target()
