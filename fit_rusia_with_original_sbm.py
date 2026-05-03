import argparse
import os
import re
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "ai_sbm"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "xdg_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from simple_sbm_generator import (
    GeneradorSBM,
    ModeloConfig,
    construir_red_manzanas_con_proyeccion_hubs,
)
from test_simulation import MODEL_CONFIG_TEMPLATE, SIMULATION_PARAMS


CSV_FILE = ROOT / "Data_Rusia_2022.csv"
DATASET_FILE = OUTPUT_DIR / "dataset_normalized.npz"
SURROGATE_RESULTS_FILE = OUTPUT_DIR / "ajuste_rusia_surrogate_shift.txt"
PLOT_FILE = OUTPUT_DIR / "ajuste_rusia_sbm_original_20sims.png"
RESULTS_FILE = OUTPUT_DIR / "ajuste_rusia_sbm_original_20sims.txt"


def load_russia_data():
    df = pd.read_csv(CSV_FILE)
    df["Casos nuevos"] = df["Casos nuevos"].astype(str).str.replace(",", "", regex=False).astype(float)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def load_dataset_bounds():
    data = np.load(DATASET_FILE)
    x = data["X"]
    return x.min(axis=0), x.max(axis=0)


def load_surrogate_seed():
    text = SURROGATE_RESULTS_FILE.read_text(encoding="utf-8")
    keys = ["beta_net", "beta_hh", "delta", "fermi_mu", "shift_days", "scale_cases"]
    values = {}
    for key in keys:
        match = re.search(rf"^{key}:\s*([-+0-9.eE]+)", text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"No pude leer {key} desde {SURROGATE_RESULTS_FILE}")
        values[key] = float(match.group(1))
    return values


def intersect_bounds(center, rel_width, global_min, global_max):
    local_min = center * (1.0 - rel_width)
    local_max = center * (1.0 + rel_width)
    lo = max(global_min, local_min)
    hi = min(global_max, local_max)
    if lo >= hi:
        return float(global_min), float(global_max)
    return float(lo), float(hi)


def build_search_bounds(seed, x_min, x_max):
    return [
        intersect_bounds(seed["beta_net"], 0.20, x_min[0], x_max[0]),
        intersect_bounds(seed["beta_hh"], 0.20, x_min[1], x_max[1]),
        intersect_bounds(seed["delta"], 0.15, x_min[2], x_max[2]),
        intersect_bounds(seed["fermi_mu"], 0.25, x_min[3], x_max[3]),
        (seed["shift_days"] - 7.0, seed["shift_days"] + 7.0),
    ]


def shifted_curve(curve, shift, n_points):
    source_t = np.arange(len(curve), dtype=float)
    target_t = np.arange(n_points, dtype=float) - shift
    return np.interp(target_t, source_t, curve, left=curve[0], right=curve[-1])


def optimal_scale(pred, target):
    denom = float(np.dot(pred, pred))
    if denom <= 1e-12:
        return 0.0
    return max(float(np.dot(target, pred) / denom), 0.0)


def normalized(values):
    peak = float(np.max(values))
    if peak <= 0.0:
        return values.copy()
    return values / peak


def stable_param_seed(params):
    rounded = np.round(np.asarray(params, dtype=float), 5)
    return int(abs(hash(tuple(rounded))) % 100000)


def run_sbm_average(beta_net, beta_hh, delta, fermi_mu, num_sims=20, seed_offset=0):
    cfg_data = deepcopy(MODEL_CONFIG_TEMPLATE)
    cfg_data["fermi_mu"] = float(fermi_mu)
    cfg_data["fermi_beta"] = 0.2

    cfg = ModeloConfig(**cfg_data)
    generador = GeneradorSBM(cfg)
    g0 = generador.generar_original()
    graph = construir_red_manzanas_con_proyeccion_hubs(g0, cfg)

    base_packet = deepcopy(SIMULATION_PARAMS)
    base_packet["G"] = graph
    base_packet["beta_network"] = float(beta_net)
    base_packet["beta_household"] = float(beta_hh)
    base_packet["delta"] = float(delta)

    base_seed = int(SIMULATION_PARAMS["seed"]) + int(seed_offset)
    curves = []
    for i in range(num_sims):
        packet = deepcopy(base_packet)
        packet["seed"] = base_seed + 1000 + i
        out = generador.simulate(packet)
        n_tot = int(out.meta["N_tot"])
        curve = out.I / n_tot
        if len(curve) < int(SIMULATION_PARAMS["tmax"]):
            curve = np.pad(curve, (0, int(SIMULATION_PARAMS["tmax"]) - len(curve)), "edge")
        else:
            curve = curve[: int(SIMULATION_PARAMS["tmax"])]
        curves.append(curve)

    return np.mean(curves, axis=0), np.asarray(curves)


def fit_original_sbm(df, seed, bounds, num_sims, maxiter, maxfev):
    real_cases = df["Casos nuevos"].to_numpy(dtype=float)
    n_days = min(len(real_cases), int(SIMULATION_PARAMS["tmax"]))
    real_cases = real_cases[:n_days]
    real_norm = normalized(real_cases)
    cache = {}
    eval_count = {"n": 0}

    def objective(theta):
        beta_net, beta_hh, delta, fermi_mu, shift = theta
        key = tuple(np.round(theta, 4))
        if key in cache:
            return cache[key]["loss"]

        eval_count["n"] += 1
        try:
            seed_offset = stable_param_seed(theta)
            mean_curve, _ = run_sbm_average(
                beta_net,
                beta_hh,
                delta,
                fermi_mu,
                num_sims=num_sims,
                seed_offset=seed_offset,
            )
            shifted = shifted_curve(mean_curve, shift, n_days)
            loss = mean_squared_error(real_norm, normalized(shifted))
        except Exception as exc:
            print(f"Eval {eval_count['n']:03d} fallo: {exc}")
            loss = 1e6
            shifted = None
            mean_curve = None

        cache[key] = {"loss": float(loss), "mean_curve": mean_curve, "shifted": shifted}
        print(
            f"Eval {eval_count['n']:03d} | loss={loss:.6f} | "
            f"beta_net={beta_net:.4f}, beta_hh={beta_hh:.4f}, "
            f"delta={delta:.4f}, fermi_mu={fermi_mu:.4f}, shift={shift:.2f}"
        )
        return float(loss)

    x0 = np.array(
        [
            seed["beta_net"],
            seed["beta_hh"],
            seed["delta"],
            seed["fermi_mu"],
            seed["shift_days"],
        ],
        dtype=float,
    )
    x0 = np.array([np.clip(value, lo, hi) for value, (lo, hi) in zip(x0, bounds)])

    result = minimize(
        objective,
        x0=x0,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "maxfev": maxfev,
            "xtol": 1e-3,
            "ftol": 1e-4,
            "disp": True,
        },
    )

    beta_net, beta_hh, delta, fermi_mu, shift = result.x
    final_seed_offset = stable_param_seed(result.x)
    mean_curve, all_curves = run_sbm_average(
        beta_net,
        beta_hh,
        delta,
        fermi_mu,
        num_sims=num_sims,
        seed_offset=final_seed_offset,
    )
    shifted = shifted_curve(mean_curve, shift, n_days)
    scale = optimal_scale(shifted, real_cases)
    pred_cases = scale * shifted
    lower = scale * shifted_curve(np.percentile(all_curves, 10, axis=0), shift, n_days)
    upper = scale * shifted_curve(np.percentile(all_curves, 90, axis=0), shift, n_days)

    params = {
        "beta_net": beta_net,
        "beta_hh": beta_hh,
        "delta": delta,
        "fermi_mu": fermi_mu,
        "shift_days": shift,
        "scale_cases": scale,
        "num_sims": num_sims,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "objective_evaluations": eval_count["n"],
    }
    metrics = {
        "mse": mean_squared_error(real_cases, pred_cases),
        "mae": mean_absolute_error(real_cases, pred_cases),
        "r2": r2_score(real_cases, pred_cases),
        "shape_mse": mean_squared_error(real_norm, normalized(shifted)),
        "optimizer_loss": float(result.fun),
    }
    return params, metrics, pred_cases, lower, upper


def save_results(df, params, metrics, pred_cases, lower, upper, seed, bounds):
    dates = df["Fecha"].iloc[:len(pred_cases)]
    real_cases = df["Casos nuevos"].to_numpy(dtype=float)[:len(pred_cases)]

    plt.figure(figsize=(13, 7))
    plt.plot(dates, real_cases, color="firebrick", linewidth=2.2, marker="o", markersize=3.5, label="Rusia datos reales")
    plt.plot(dates, pred_cases, color="darkgreen", linewidth=2.5, label="SBM original ajustado")
    plt.title("Ajuste de datos de Rusia 2022 con SBM original", fontsize=15)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Casos nuevos", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.xticks(rotation=45)

    text = "\n".join(
        [
            f"beta_net = {params['beta_net']:.4f}",
            f"beta_hh = {params['beta_hh']:.4f}",
            f"delta = {params['delta']:.4f}",
            f"fermi_mu = {params['fermi_mu']:.4f}",
            f"shift = {params['shift_days']:.2f} dias",
            f"escala = {params['scale_cases']:.2e}",
            f"simulaciones = {int(params['num_sims'])}",
            f"R2 = {metrics['r2']:.4f}",
            f"Norm. MSE = {metrics['shape_mse']:.6f}",
        ]
    )
    plt.gca().text(
        0.02,
        0.96,
        text,
        transform=plt.gca().transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("Ajuste de datos de Rusia 2022 con SBM original\n")
        f.write("Nota: ajuste fenomenologico; el SBM produce fraccion infectada y se escala a casos nuevos.\n\n")
        f.write("Semilla surrogate:\n")
        for key, value in seed.items():
            f.write(f"{key}: {value:.10g}\n")
        f.write("\nBounds usados:\n")
        for name, (lo, hi) in zip(["beta_net", "beta_hh", "delta", "fermi_mu", "shift_days"], bounds):
            f.write(f"{name}: [{lo:.10g}, {hi:.10g}]\n")
        f.write("\nParametros optimos:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\nMetricas:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.10g}\n")
        f.write(f"\nGrafica: {PLOT_FILE}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Ajusta el SBM original a datos de Rusia usando semilla del surrogate.")
    parser.add_argument("--num-sims", type=int, default=20, help="Numero de replicas SBM por evaluacion.")
    parser.add_argument("--maxiter", type=int, default=25, help="Iteraciones maximas de Powell.")
    parser.add_argument("--maxfev", type=int, default=80, help="Evaluaciones maximas de la funcion objetivo.")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_russia_data()
    seed = load_surrogate_seed()
    x_min, x_max = load_dataset_bounds()
    bounds = build_search_bounds(seed, x_min, x_max)

    print("Semilla surrogate:")
    for key, value in seed.items():
        print(f"  {key}: {value:.6g}")
    print("Bounds de busqueda:")
    for name, (lo, hi) in zip(["beta_net", "beta_hh", "delta", "fermi_mu", "shift_days"], bounds):
        print(f"  {name}: [{lo:.6g}, {hi:.6g}]")
    print(f"Simulaciones por evaluacion: {args.num_sims}")

    params, metrics, pred_cases, lower, upper = fit_original_sbm(
        df,
        seed,
        bounds,
        num_sims=args.num_sims,
        maxiter=args.maxiter,
        maxfev=args.maxfev,
    )
    save_results(df, params, metrics, pred_cases, lower, upper, seed, bounds)

    print("Ajuste SBM original completado.")
    print(f"Grafica: {PLOT_FILE}")
    print(f"Resultados: {RESULTS_FILE}")
    print("Parametros optimos:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("Metricas:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6g}")


if __name__ == "__main__":
    main()
