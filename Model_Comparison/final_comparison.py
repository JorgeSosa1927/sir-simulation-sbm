import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from copy import deepcopy
import random
import pandas as pd

# Import existing simulator components
from simple_sbm_generator import (
    GeneradorSBM,
    ModeloConfig,
    construir_red_manzanas_con_proyeccion_hubs,
)
from test_simulation import MODEL_CONFIG_TEMPLATE, SIMULATION_PARAMS
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import architectures
from LSTM_SBM import EpidemicSurrogateNet as LSTMNet
from Autoencouder_Smoth_SBM import EpidemicSurrogateNet as SmoothAENet
from BasicAE_SBM import EpidemicSurrogateNet as BasicAENet

OUTPUT_DIR = "output/ai_sbm"
TMAX = SIMULATION_PARAMS["tmax"]

def run_scenario(beta_net, beta_hh, delta, fermi_mu, num_sims=100):
    cfg_data = deepcopy(MODEL_CONFIG_TEMPLATE)
    cfg_data["fermi_mu"] = fermi_mu
    cfg_data["fermi_beta"] = 0.2
    cfg = ModeloConfig(**cfg_data)
    generador = GeneradorSBM(cfg)
    G0 = generador.generar_original()
    H_multi = construir_red_manzanas_con_proyeccion_hubs(G0, cfg)
    
    packet = deepcopy(SIMULATION_PARAMS)
    packet["beta_network"] = beta_net
    packet["beta_household"] = beta_hh
    packet["delta"] = delta
    packet["G"] = H_multi
    
    all_I = []
    base_seed = packet["seed"]
    for i in range(num_sims):
        packet["seed"] = base_seed + i + random.randint(1000, 9000)
        out = generador.simulate(packet)
        N_tot = int(out.meta["N_tot"])
        I_curve = out.I / N_tot
        if len(I_curve) < TMAX:
            I_curve = np.pad(I_curve, (0, TMAX - len(I_curve)), 'edge')
        else:
            I_curve = I_curve[:TMAX]
        all_I.append(I_curve)
    return np.mean(all_I, axis=0)

def load_scalers():
    data = np.load(os.path.join(OUTPUT_DIR, "dataset_normalized.npz"))
    X, Y = data['X'], data['Y']
    X_scaler = StandardScaler().fit(X)
    Y_scaler = MinMaxScaler().fit(Y)
    return X_scaler, Y_scaler

def main():
    print("=== Generando Comparativa Final Corta vs Larga Distancia ===")
    X_scaler, Y_scaler = load_scalers()
    
    # Load Models
    input_dim = 4
    lstm_model = LSTMNet(input_dim=input_dim, output_dim=TMAX)
    lstm_model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "surrogate_model_normalized.pth")))
    lstm_model.eval()
    
    smooth_ae = SmoothAENet(input_dim=input_dim, output_dim=TMAX)
    smooth_ae.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "surrogate_model_smooth_normalized.pth")))
    smooth_ae.eval()
    
    basic_ae = BasicAENet(input_dim=input_dim, output_dim=TMAX)
    basic_ae.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "surrogate_model_basic_normalized.pth")))
    basic_ae.eval()
    
    # Scenarios
    beta_net, beta_hh, delta = 0.4469, 2.289, 0.9592
    mu_short, mu_long = 5.0, 15.0
    
    print("Obteniendo Ground Truth...")
    real_short = run_scenario(beta_net, beta_hh, delta, mu_short)
    real_long = run_scenario(beta_net, beta_hh, delta, mu_long)
    
    # Predictions
    params_s = X_scaler.transform([[beta_net, beta_hh, delta, mu_short]])
    params_l = X_scaler.transform([[beta_net, beta_hh, delta, mu_long]])
    
    with torch.no_grad():
        p_lstm_s = np.clip(Y_scaler.inverse_transform(lstm_model(torch.FloatTensor(params_s)).numpy())[0], a_min=0, a_max=None)
        p_lstm_l = np.clip(Y_scaler.inverse_transform(lstm_model(torch.FloatTensor(params_l)).numpy())[0], a_min=0, a_max=None)
        
        p_smooth_s = np.clip(Y_scaler.inverse_transform(smooth_ae(torch.FloatTensor(params_s)).numpy())[0], a_min=0, a_max=None)
        p_smooth_l = np.clip(Y_scaler.inverse_transform(smooth_ae(torch.FloatTensor(params_l)).numpy())[0], a_min=0, a_max=None)
        
        p_basic_s = np.clip(Y_scaler.inverse_transform(basic_ae(torch.FloatTensor(params_s)).numpy())[0], a_min=0, a_max=None)
        p_basic_l = np.clip(Y_scaler.inverse_transform(basic_ae(torch.FloatTensor(params_l)).numpy())[0], a_min=0, a_max=None)

    t = np.arange(TMAX)
    
    # === GUARDAR DATOS EN CSV PARA VERIFICACIÓN ===
    df_short = pd.DataFrame({
        'Dia': t,
        'Ground_Truth': real_short,
        'LSTM_Pred': p_lstm_s,
        'SmoothAE_Pred': p_smooth_s,
        'BasicAE_Pred': p_basic_s
    })
    df_short.to_csv(os.path.join(OUTPUT_DIR, "datos_corta_distancia.csv"), index=False)
    
    df_long = pd.DataFrame({
        'Dia': t,
        'Ground_Truth': real_long,
        'LSTM_Pred': p_lstm_l,
        'SmoothAE_Pred': p_smooth_l,
        'BasicAE_Pred': p_basic_l
    })
    df_long.to_csv(os.path.join(OUTPUT_DIR, "datos_larga_distancia.csv"), index=False)
    print(f"Datos exportados a CSV en {OUTPUT_DIR} para verificación de negativos.")
    # ===============================================
    
    # Plot 1: Short Distance
    plt.figure(figsize=(10, 6))
    plt.plot(t, real_short, 'k--', label="Ground Truth (Mu=5.0)", linewidth=2)
    plt.plot(t, p_lstm_s, label=f"LSTM (R2: {r2_score(real_short, p_lstm_s):.4f})", color="red")
    plt.plot(t, p_smooth_s, label=f"Smooth AE (R2: {r2_score(real_short, p_smooth_s):.4f})", color="green")
    plt.plot(t, p_basic_s, label=f"Basic AE (R2: {r2_score(real_short, p_basic_s):.4f})", color="blue")
    plt.title("Short Distance Comparison (Mu=5.0)")
    plt.xlabel("Days")
    plt.ylabel("Infected Fraction")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6, color="#a0a0a0")
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(OUTPUT_DIR, "comparativa_corta_distancia.png"), dpi=300)
    plt.close()
    
    # Plot 2: Long Distance
    plt.figure(figsize=(10, 6))
    plt.plot(t, real_long, 'k--', label="Ground Truth (Mu=15.0)", linewidth=2)
    plt.plot(t, p_lstm_l, label=f"LSTM (R2: {r2_score(real_long, p_lstm_l):.4f})", color="red")
    plt.plot(t, p_smooth_l, label=f"Smooth AE (R2: {r2_score(real_long, p_smooth_l):.4f})", color="green")
    plt.plot(t, p_basic_l, label=f"Basic AE (R2: {r2_score(real_long, p_basic_l):.4f})", color="blue")
    plt.title("Long Distance Comparison (Mu=15.0)")
    plt.xlabel("Days")
    plt.ylabel("Infected Fraction")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6, color="#a0a0a0")
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(OUTPUT_DIR, "comparativa_larga_distancia.png"), dpi=300)
    plt.close()
    
    print("Gráficas guardadas.")
    
    # Save specific metrics
    with open(os.path.join(OUTPUT_DIR, "final_comparison_metrics.txt"), "w") as f:
        f.write("Métricas por Escenario:\n")
        f.write(f"SHORT (Mu=5.0):\n")
        f.write(f"  LSTM R2: {r2_score(real_short, p_lstm_s):.4f}\n")
        f.write(f"  SmoothAE R2: {r2_score(real_short, p_smooth_s):.4f}\n")
        f.write(f"  BasicAE R2: {r2_score(real_short, p_basic_s):.4f}\n")
        f.write(f"LONG (Mu=15.0):\n")
        f.write(f"  LSTM R2: {r2_score(real_long, p_lstm_l):.4f}\n")
        f.write(f"  SmoothAE R2: {r2_score(real_long, p_smooth_l):.4f}\n")
        f.write(f"  BasicAE R2: {r2_score(real_long, p_basic_l):.4f}\n")

if __name__ == "__main__":
    main()
