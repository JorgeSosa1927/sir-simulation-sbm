import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "ai_sbm"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "xdg_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution


CSV_FILE = ROOT / "Data_Rusia_2022.csv"
PLOT_FILE = OUTPUT_DIR / "ajuste_rusia_sir_normal.png"
RESULTS_FILE = OUTPUT_DIR / "ajuste_rusia_sir_normal.txt"
TMAX = 100


def load_russia_data():
    df = pd.read_csv(CSV_FILE)
    df["Casos nuevos"] = df["Casos nuevos"].astype(str).str.replace(",", "", regex=False).astype(float)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def shifted_curve(curve, shift, n_points):
    source_t = np.arange(len(curve), dtype=float)
    target_t = np.arange(n_points, dtype=float) - shift
    return np.interp(target_t, source_t, curve, left=curve[0], right=curve[-1])


def normalized(values):
    peak = float(np.max(values))
    if peak <= 0.0:
        return values.copy()
    return values / peak


def regression_metrics(real, pred):
    residual = real - pred
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((real - np.mean(real)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    return {"mse": mse, "mae": mae, "r2": float(r2)}


def simulate_sir_daily_cases(beta, gamma, n_population, i0, tmax=TMAX):
    if i0 <= 0.0 or n_population <= i0 or beta <= 0.0 or gamma <= 0.0:
        return None

    s0 = n_population - i0
    y0 = [s0, i0, 0.0, 0.0]
    t_eval = np.arange(tmax + 1, dtype=float)

    def rhs(_t, y):
        s, i, _r, _c = y
        incidence = beta * s * i / n_population
        recovery = gamma * i
        return [-incidence, incidence - recovery, recovery, incidence]

    sol = solve_ivp(
        rhs,
        (0.0, float(tmax)),
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-3,
    )
    if not sol.success or sol.y.shape[1] != tmax + 1:
        return None

    cumulative_cases = np.maximum.accumulate(sol.y[3])
    daily_cases = np.diff(cumulative_cases)
    return np.maximum(daily_cases, 0.0)


def decode_theta(theta):
    beta, gamma, log10_n, log10_i0, shift = theta
    n_population = 10.0**log10_n
    i0 = 10.0**log10_i0
    return beta, gamma, n_population, i0, shift


def fit_sir_to_russia(df, maxiter, popsize, seed):
    real_cases = df["Casos nuevos"].to_numpy(dtype=float)
    n_days = min(len(real_cases), TMAX)
    real_cases = real_cases[:n_days]
    real_peak = float(np.max(real_cases))
    real_norm = normalized(real_cases)

    bounds = [
        (0.01, 2.0),  # beta
        (0.01, 1.0),  # gamma
        (5.0, 8.8),  # log10(N): 100k to about 631M
        (0.0, 7.0),  # log10(I0): 1 to 10M
        (-20.0, 20.0),  # shift_days
    ]

    cache = {}

    def objective(theta):
        key = tuple(np.round(theta, 8))
        if key in cache:
            return cache[key]

        beta, gamma, n_population, i0, shift = decode_theta(theta)
        if i0 >= 0.5 * n_population:
            return 1e6

        pred_cases = simulate_sir_daily_cases(beta, gamma, n_population, i0)
        if pred_cases is None:
            return 1e6

        pred_shifted = shifted_curve(pred_cases, shift, n_days)
        magnitude_loss = np.mean(((pred_shifted - real_cases) / real_peak) ** 2)
        shape_loss = np.mean((normalized(pred_shifted) - real_norm) ** 2)
        loss = float(magnitude_loss + 0.25 * shape_loss)
        cache[key] = loss
        return loss

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-8,
        seed=seed,
        polish=True,
        workers=1,
        updating="immediate",
    )

    beta, gamma, n_population, i0, shift = decode_theta(result.x)
    pred_cases_full = simulate_sir_daily_cases(beta, gamma, n_population, i0)
    pred_cases = shifted_curve(pred_cases_full, shift, n_days)

    params = {
        "beta": beta,
        "gamma": gamma,
        "r0": beta / gamma,
        "N": n_population,
        "I0": i0,
        "S0": n_population - i0,
        "R0_initial": 0.0,
        "shift_days": shift,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_loss": float(result.fun),
    }
    metrics = regression_metrics(real_cases, pred_cases)
    metrics["shape_mse"] = float(np.mean((real_norm - normalized(pred_cases)) ** 2))
    return params, metrics, pred_cases, bounds


def save_results(df, params, metrics, pred_cases, bounds):
    dates = df["Fecha"].iloc[: len(pred_cases)]
    real_cases = df["Casos nuevos"].to_numpy(dtype=float)[: len(pred_cases)]

    plt.figure(figsize=(13, 7))
    plt.plot(dates, real_cases, color="firebrick", linewidth=2.2, marker="o", markersize=3.5, label="Rusia datos reales")
    plt.plot(dates, pred_cases, color="black", linewidth=2.5, label="SIR normal ajustado")
    plt.title("Ajuste de datos de Rusia 2022 con SIR normal", fontsize=15)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Casos nuevos", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.xticks(rotation=45)

    text = "\n".join(
        [
            f"beta = {params['beta']:.4f}",
            f"gamma = {params['gamma']:.4f}",
            f"R0 = {params['r0']:.4f}",
            f"N = {params['N']:.2e}",
            f"I0 = {params['I0']:.2e}",
            f"shift = {params['shift_days']:.2f} dias",
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
        f.write("Ajuste de datos de Rusia 2022 con SIR normal\n")
        f.write("Modelo: SIR determinista homogeneo sin red.\n")
        f.write("Nota: no se usa scale_cases; N es un parametro ajustado y la curva comparada es incidencia diaria.\n\n")
        f.write("Ecuaciones:\n")
        f.write("dS/dt = -beta*S*I/N\n")
        f.write("dI/dt = beta*S*I/N - gamma*I\n")
        f.write("dR/dt = gamma*I\n")
        f.write("casos_nuevos(t) = integral diaria de beta*S*I/N\n\n")
        f.write("Bounds usados:\n")
        for name, (lo, hi) in zip(["beta", "gamma", "log10_N", "log10_I0", "shift_days"], bounds):
            f.write(f"{name}: [{lo:.10g}, {hi:.10g}]\n")
        f.write("\nParametros optimos:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\nMetricas:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.10g}\n")
        f.write(f"\nGrafica: {PLOT_FILE}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Ajusta un SIR normal a datos de Rusia sin factor scale_cases.")
    parser.add_argument("--maxiter", type=int, default=160, help="Iteraciones maximas de differential_evolution.")
    parser.add_argument("--popsize", type=int, default=15, help="Tamano relativo de poblacion del optimizador.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla del optimizador.")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_russia_data()
    params, metrics, pred_cases, bounds = fit_sir_to_russia(df, args.maxiter, args.popsize, args.seed)
    save_results(df, params, metrics, pred_cases, bounds)

    print("Ajuste SIR normal completado.")
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
