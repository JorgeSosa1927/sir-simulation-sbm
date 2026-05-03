import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "output" / "ai_sbm" / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / "output" / "ai_sbm" / "xdg_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from AI_SBM import DATASET_FILE, MODEL_FILE, OUTPUT_DIR, TMAX, EpidemicSurrogateNet


CSV_FILE = "Data_Rusia_2022.csv"
PLOT_FILE = os.path.join(OUTPUT_DIR, "ajuste_rusia_surrogate_shift.png")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "ajuste_rusia_surrogate_shift.txt")


def load_russia_data():
    df = pd.read_csv(CSV_FILE)
    df["Casos nuevos"] = df["Casos nuevos"].astype(str).str.replace(",", "", regex=False).astype(float)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def load_model_and_scalers():
    dataset = np.load(DATASET_FILE)
    X = dataset["X"]
    Y = dataset["Y"]

    x_scaler = StandardScaler().fit(X)
    y_scaler = MinMaxScaler().fit(Y)

    model = EpidemicSurrogateNet(input_dim=X.shape[1], output_dim=Y.shape[1])
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()

    return model, x_scaler, y_scaler, X


def predict_curve(model, x_scaler, y_scaler, params):
    x_scaled = x_scaler.transform(np.asarray(params, dtype=float).reshape(1, -1))
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(x_scaled)).numpy()
    pred = y_scaler.inverse_transform(pred_scaled)[0]
    return np.maximum(pred, 0.0)


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
    if peak <= 0:
        return values.copy()
    return values / peak


def fit_surrogate_to_russia(df, model, x_scaler, y_scaler, X_train):
    real_cases = df["Casos nuevos"].to_numpy(dtype=float)
    n_days = min(len(real_cases), TMAX)
    real_cases = real_cases[:n_days]
    real_norm = normalized(real_cases)

    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    bounds = [
        (x_min[0], x_max[0]),
        (x_min[1], x_max[1]),
        (x_min[2], x_max[2]),
        (x_min[3], x_max[3]),
        (-20.0, 20.0),
    ]

    cache = {}

    def objective(theta):
        beta_net, beta_hh, delta, fermi_mu, shift = theta
        key = tuple(np.round(theta, 8))
        if key in cache:
            return cache[key]

        pred_fraction = predict_curve(model, x_scaler, y_scaler, [beta_net, beta_hh, delta, fermi_mu])
        pred_shifted = shifted_curve(pred_fraction, shift, n_days)
        pred_norm = normalized(pred_shifted)
        loss = mean_squared_error(real_norm, pred_norm)

        cache[key] = loss
        return loss

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=80,
        popsize=12,
        tol=1e-7,
        seed=42,
        polish=True,
        workers=1,
        updating="immediate",
    )

    beta_net, beta_hh, delta, fermi_mu, shift = result.x
    pred_fraction = predict_curve(model, x_scaler, y_scaler, [beta_net, beta_hh, delta, fermi_mu])
    pred_shifted = shifted_curve(pred_fraction, shift, n_days)
    scale = optimal_scale(pred_shifted, real_cases)
    pred_cases = scale * pred_shifted

    metrics = {
        "mse": mean_squared_error(real_cases, pred_cases),
        "mae": mean_absolute_error(real_cases, pred_cases),
        "r2": r2_score(real_cases, pred_cases),
        "shape_mse": mean_squared_error(real_norm, normalized(pred_shifted)),
        "optimizer_loss": result.fun,
    }
    params = {
        "beta_net": beta_net,
        "beta_hh": beta_hh,
        "delta": delta,
        "fermi_mu": fermi_mu,
        "shift_days": shift,
        "scale_cases": scale,
    }
    return params, metrics, pred_cases


def save_results(df, params, metrics, pred_cases):
    dates = df["Fecha"].iloc[:len(pred_cases)]
    real_cases = df["Casos nuevos"].to_numpy(dtype=float)[:len(pred_cases)]

    plt.figure(figsize=(13, 7))
    plt.plot(dates, real_cases, color="firebrick", linewidth=2.2, marker="o", markersize=3.5, label="Rusia datos reales")
    plt.plot(dates, pred_cases, color="navy", linewidth=2.4, label="Surrogate ajustado")
    plt.title("Ajuste de datos de Rusia 2022 con red surrogate entrenada", fontsize=15)
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
        f.write("Ajuste de datos de Rusia 2022 con surrogate entrenado\n")
        f.write("Modelo: EpidemicSurrogateNet autoregresivo\n")
        f.write("Nota: ajuste fenomenologico; la red predice fraccion infectada y se escala a casos nuevos.\n\n")
        f.write("Parametros optimos:\n")
        for key, value in params.items():
            f.write(f"{key}: {value:.10g}\n")
        f.write("\nMetricas:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.10g}\n")
        f.write(f"\nGrafica: {PLOT_FILE}\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_russia_data()
    model, x_scaler, y_scaler, X_train = load_model_and_scalers()
    params, metrics, pred_cases = fit_surrogate_to_russia(df, model, x_scaler, y_scaler, X_train)
    save_results(df, params, metrics, pred_cases)

    print("Ajuste completado.")
    print(f"Grafica: {os.path.abspath(PLOT_FILE)}")
    print(f"Resultados: {os.path.abspath(RESULTS_FILE)}")
    print("Parametros optimos:")
    for key, value in params.items():
        print(f"  {key}: {value:.6g}")
    print("Metricas:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6g}")


if __name__ == "__main__":
    main()
