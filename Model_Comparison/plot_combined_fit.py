import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from copy import deepcopy
import random

# Import for LSTM
from LSTM_SBM import EpidemicSurrogateNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import for SBM
from simple_sbm_generator import (
    GeneradorSBM,
    ModeloConfig,
    construir_red_manzanas_con_proyeccion_hubs,
)
from test_simulation import MODEL_CONFIG_TEMPLATE, SIMULATION_PARAMS

# Aesthetic setup
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

OUTPUT_DIR = "output/ai_sbm"
TMAX = 100

def load_surrogate_and_scalers():
    DATASET_FILE = os.path.join(OUTPUT_DIR, "dataset_normalized.npz")
    MODEL_FILE = os.path.join(OUTPUT_DIR, "surrogate_model_normalized.pth")
    data = np.load(DATASET_FILE)
    X, Y = data['X'], data['Y']
    X_scaler = StandardScaler().fit(X)
    Y_scaler = MinMaxScaler().fit(Y)
    
    input_dim = 4
    model = EpidemicSurrogateNet(input_dim=input_dim, output_dim=TMAX)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    return model, X_scaler, Y_scaler

def run_sbm_average(beta_net, beta_hh, delta, fermi_mu, num_sims=15):
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

    base_seed = int(SIMULATION_PARAMS["seed"])
    curves = []
    for i in range(num_sims):
        packet = deepcopy(base_packet)
        packet["seed"] = base_seed + 1000 + i
        try:
            out = generador.simulate(packet)
            n_tot = int(out.meta["N_tot"])
            curve = out.I / n_tot
            if len(curve) < TMAX:
                curve = np.pad(curve, (0, TMAX - len(curve)), "edge")
            else:
                curve = curve[:TMAX]
            curves.append(curve)
        except Exception:
            continue
            
    if len(curves) == 0:
        return np.zeros(TMAX), np.zeros((1, TMAX))
        
    return np.mean(curves, axis=0), np.asarray(curves)

def shifted_curve_array(curve, shift, n_points):
    source_t = np.arange(len(curve), dtype=float)
    target_t = np.arange(n_points, dtype=float) - shift
    return np.interp(target_t, source_t, curve, left=0.0, right=curve[-1])

def get_lstm_curve(beta_net, beta_hh, delta, fermi_mu, scale, shift, real_data_len, model, X_scaler, Y_scaler):
    x_input = X_scaler.transform([[beta_net, beta_hh, delta, fermi_mu]])
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(x_input)).numpy()
    I_pred = Y_scaler.inverse_transform(pred_scaled)[0]
    I_pred = np.maximum(I_pred, 0)
    
    return scale * shifted_curve_array(I_pred, shift, real_data_len)

def main():
    print("Loading empirical data...")
    df = pd.read_csv("stopkoronavirus_clean_wave_winter.csv")
    real_data = df["CONFIRMED"].values
    if "TIME" in df.columns:
        dates = pd.to_datetime(df["TIME"]).dt.date.iloc[:len(real_data)]
    else:
        dates = np.arange(len(real_data))
        
    n_days = len(real_data)
    
    print("Generating LSTM Surrogate curve...")
    model, X_scaler, Y_scaler = load_surrogate_and_scalers()
    lstm_params = [0.2143, 2.4335, 1.1722, 22.4418, 1724663.41, 4.0]
    lstm_curve = get_lstm_curve(*lstm_params, n_days, model, X_scaler, Y_scaler)
    lstm_r2 = r2_score(real_data, lstm_curve)
    
    print("Generating Mechanistic SBM curve (averaging 15 runs)...")
    sbm_beta_net = 0.2235724878845205
    sbm_beta_hh = 2.5483942846491474
    sbm_delta = 1.0552091989317305
    sbm_fermi_mu = 19.207485965140435
    sbm_shift = 4.0
    sbm_scale = 1447002.8910357412
    
    mean_sbm, all_sbm_curves = run_sbm_average(sbm_beta_net, sbm_beta_hh, sbm_delta, sbm_fermi_mu, num_sims=15)
    sbm_curve = sbm_scale * shifted_curve_array(mean_sbm, sbm_shift, n_days)
    sbm_r2 = r2_score(real_data, sbm_curve)
    
    lower = sbm_scale * shifted_curve_array(np.percentile(all_sbm_curves, 10, axis=0), sbm_shift, n_days)
    upper = sbm_scale * shifted_curve_array(np.percentile(all_sbm_curves, 90, axis=0), sbm_shift, n_days)

    print("Plotting combined results...")
    plt.figure(figsize=(14, 8))
    
    # Real Data
    plt.plot(dates, real_data, color="black", linewidth=0, marker="o", markersize=4, alpha=0.7, label="SpB Data Covid 2022")
    
    # LSTM Fit
    plt.plot(dates, lstm_curve, color="#E74C3C", linewidth=2.5, linestyle="--", label=f"LSTM Surrogate Fit ($R^2$={lstm_r2:.3f})")
    
    # Mechanistic SBM Fit
    plt.plot(dates, sbm_curve, color="#27AE60", linewidth=3.0, label=f"Surrogate + SBM Fit ($R^2$={sbm_r2:.3f})")
    plt.fill_between(dates, lower, upper, color="#27AE60", alpha=0.2, label="Surrogate + SBM 10th-90th Percentile")
    
    plt.title("Model Validation: LSTM Surrogate and SBM Model", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=14, labelpad=10)
    plt.ylabel("Confirmed Cases", fontsize=14, labelpad=10)
    
    plt.grid(True, linestyle="--", alpha=0.6, color="#a0a0a0")
    plt.xticks(rotation=45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    text = (
        "Fit Execution Times:\n"
        "  LSTM Surrogate: ~31 s\n"
        "  Surrogate + SBM: ~187 s\n\n"
        "Surrogate + SBM Optimal Parameters:\n"
        f"  $\\beta_{{net}}$ = {sbm_beta_net:.4f}\n"
        f"  $\\beta_{{hh}}$ = {sbm_beta_hh:.4f}\n"
        f"  $\\delta$ = {sbm_delta:.4f}\n"
        f"  $\\mu$ = {sbm_fermi_mu:.4f}"
    )
    plt.gca().text(
        0.02, 0.96, text,
        transform=plt.gca().transAxes,
        va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )
    
    plt.legend(loc="upper right", fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    
    out_img = os.path.join(OUTPUT_DIR, "combined_wave_winter_fit.png")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to: {out_img}")

if __name__ == "__main__":
    main()
