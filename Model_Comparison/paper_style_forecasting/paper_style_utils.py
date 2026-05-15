"""
paper_style_utils.py
--------------------
Shared utilities for paper-style retrospective forecasting.
"""

import os, sys
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / "output" / "ai_sbm" / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME",  str(_ROOT / "output" / "ai_sbm" / "xdg_cache"))

import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from LSTM_SBM import run_custom_scenario, EpidemicSurrogateNet, DATASET_FILE, MODEL_FILE, TMAX

# ── CONFIG ───────────────────────────────────────────────
MODE     = "surrogate"   # "surrogate" or "sbm"
HORIZON  = 14
TOP_K    = 30
N_RANDOM = {"surrogate": 300, "sbm": 50}

BASELINE = dict(beta_network=0.2436, beta_household=2.544,
                delta=1.0552, fermi_mu=19.2075)
BOUNDS   = dict(beta_network=(0.05, 0.80), beta_household=(0.50, 4.00),
                delta=(0.30, 1.50), fermi_mu=(4.00, 40.00))
KEYS     = ["beta_network", "beta_household", "delta", "fermi_mu"]

# ── SURROGATE CACHE ──────────────────────────────────────
_cache = {}

def load_surrogate():
    if "model" in _cache:
        return _cache["model"], _cache["xs"], _cache["ys"]
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Surrogate model not found. Please run LSTM_SBM.py first.\nExpected: {MODEL_FILE}")
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")
    data = np.load(DATASET_FILE)
    xs = StandardScaler().fit(data["X"])
    ys = MinMaxScaler().fit(data["Y"])
    model = EpidemicSurrogateNet(input_dim=data["X"].shape[1], output_dim=data["Y"].shape[1])
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
    model.eval()
    _cache.update({"model": model, "xs": xs, "ys": ys})
    print("[SURROGATE] Loaded.")
    return model, xs, ys

def simulate_curve(theta, mode=None, num_sims=20):
    """Return normalised infected-fraction curve of length TMAX."""
    if mode is None:
        mode = MODE
    beta_net, beta_hh, delta, fermi_mu = theta
    if mode == "sbm":
        curve, _ = run_custom_scenario(beta_net, beta_hh, delta, fermi_mu,
                                       num_sims=num_sims, return_population=True)
        return np.asarray(curve, dtype=float)
    model, xs, ys = load_surrogate()
    x = xs.transform(np.array([[beta_net, beta_hh, delta, fermi_mu]]))
    with torch.no_grad():
        pred = model(torch.FloatTensor(x)).numpy()
    return np.maximum(ys.inverse_transform(pred)[0], 0.0)

# ── SPLITS ───────────────────────────────────────────────
def build_splits(curve, peak_idx, horizon=HORIZON):
    n = len(curve)
    offsets = {"tpeak_minus_14": -14, "tpeak_minus_7": -7, "tpeak_plus_7": +7}
    scenarios = []
    for name, off in offsets.items():
        t_cut = peak_idx + off
        if t_cut < 1:
            raise ValueError(f"[{name}] t_cut={t_cut} before start.")
        if t_cut >= n:
            raise ValueError(f"[{name}] t_cut={t_cut} beyond end (n={n}).")
        if n - t_cut - 1 < horizon:
            raise ValueError(f"[{name}] Not enough points after t_cut.")
        scenarios.append({
            "name": name,
            "t_cut": t_cut,
            "known": curve[:t_cut + 1],
            "hidden": curve[t_cut + 1: t_cut + 1 + horizon],
        })
        print(f"[SPLIT] {name}: t_cut={t_cut} | known={t_cut+1} | hidden={horizon}")
    return scenarios

# ── CALIBRATION ──────────────────────────────────────────
def _sample():
    return np.array([np.random.uniform(*BOUNDS[k]) for k in KEYS])

def _perturb(frac=0.40):
    b = [BASELINE[k] for k in KEYS]
    fracs = [frac, frac, 0.25, frac]
    v = [b[i] * np.random.uniform(1 - fracs[i], 1 + fracs[i]) for i in range(4)]
    return np.array([np.clip(v[i], *BOUNDS[KEYS[i]]) for i in range(4)])

def loss_fn(theta, known):
    """RMSE between normalised known target and normalised sim curve."""
    try:
        f = simulate_curve(theta)
        n = len(known)
        f_k = f[:n]
        if np.max(f_k) < 1e-10:
            return 1e9
        # Both are normalised fractions — no scale factor needed
        return float(np.sqrt(np.mean((known - f_k) ** 2)))
    except Exception:
        return 1e9

def calibrate(known, mode=None):
    if mode is None:
        mode = MODE
    n_rand = N_RANDOM.get(mode, 300)
    theta0 = np.array([BASELINE[k] for k in KEYS])
    candidates = [theta0]
    for _ in range(n_rand // 3):
        candidates.append(_perturb())
    while len(candidates) < n_rand:
        candidates.append(_sample())

    print(f"  [CAL] Evaluating {len(candidates)} candidates …")
    results = []
    for i, th in enumerate(candidates):
        results.append((loss_fn(th, known), th))
        if (i + 1) % 50 == 0:
            best = min(r[0] for r in results)
            print(f"    {i+1}/{len(candidates)}  best_rmse={best:.5f}")

    results.sort(key=lambda x: x[0])
    top = results[:TOP_K]
    print(f"  [CAL] Top-{TOP_K}: {top[0][0]:.5f} … {top[-1][0]:.5f}")
    return top

# ── FORECAST ─────────────────────────────────────────────
def generate_forecast(top, known, horizon=HORIZON):
    n = len(known)
    traj = []
    for _, th in top:
        f = simulate_curve(th)
        if len(f) < n + horizon:
            f = np.pad(f, (0, n + horizon - len(f)), "edge")
        seg = f[n: n + horizon]
        if np.any(np.isnan(seg)):
            continue
        traj.append(seg)
    if not traj:
        raise RuntimeError("No valid forecast trajectories.")
    mat = np.array(traj)
    return (mat,
            np.mean(mat, 0),
            np.median(mat, 0),
            np.percentile(mat, 10, 0),
            np.percentile(mat, 90, 0))

# ── METRICS ──────────────────────────────────────────────
def calc_metrics(hidden, fm_mean, fm_median, p10, p90, t_peak_global, t_cut):
    """Compute metrics for both median (primary) and mean (secondary)."""
    h = len(hidden)
    med_ = fm_median[:h]; mean_ = fm_mean[:h]
    p10_ = p10[:h];       p90_  = p90[:h]
    tpv  = float(np.max(hidden))

    # ── Median (primary) ──
    rmse_med = float(np.sqrt(mean_squared_error(hidden, med_)))
    mae_med  = float(mean_absolute_error(hidden, med_))
    r2_med   = float(r2_score(hidden, med_))
    cov      = float(np.sum((hidden >= p10_) & (hidden <= p90_))) / h
    p_pk_med = int(np.argmax(med_))
    t_pk_med = int(np.argmax(hidden))
    pt_err_med = p_pk_med - t_pk_med
    ph_err_med = (float(np.max(med_)) - tpv) / tpv if tpv > 0 else float("nan")

    # ── Mean (secondary) ──
    rmse_mn = float(np.sqrt(mean_squared_error(hidden, mean_)))
    mae_mn  = float(mean_absolute_error(hidden, mean_))
    r2_mn   = float(r2_score(hidden, mean_))
    p_pk_mn = int(np.argmax(mean_))
    pt_err_mn = p_pk_mn - t_pk_med
    ph_err_mn = (float(np.max(mean_)) - tpv) / tpv if tpv > 0 else float("nan")

    return dict(
        rmse_median=rmse_med, mae_median=mae_med, r2_median=r2_med,
        rmse_mean=rmse_mn,   mae_mean=mae_mn,   r2_mean=r2_mn,
        coverage_10_90=cov,
        peak_time_error_days_median=pt_err_med,
        peak_height_relative_error_median=ph_err_med,
        peak_time_error_days_mean=pt_err_mn,
        peak_height_relative_error_mean=ph_err_mn,
    )
