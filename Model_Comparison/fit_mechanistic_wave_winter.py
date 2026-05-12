import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from copy import deepcopy
import random

# Importar componentes del simulador
from simple_sbm_generator import (
    GeneradorSBM,
    ModeloConfig,
    construir_red_manzanas_con_proyeccion_hubs,
)
from test_simulation import MODEL_CONFIG_TEMPLATE, SIMULATION_PARAMS

# Configuración de estilo
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

TMAX = 100
OUTPUT_DIR = "output/ai_sbm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_mechanistic_scenario(beta_net, beta_hh, delta, fermi_mu, num_sims=10):
    cfg_data = deepcopy(MODEL_CONFIG_TEMPLATE)
    cfg_data["fermi_mu"] = fermi_mu
    cfg_data["fermi_beta"] = 0.2  # Mantenido fijo según simulaciones anteriores
    
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
        try:
            out = generador.simulate(packet)
            N_tot = int(out.meta["N_tot"])
            I_curve = out.I / N_tot  # Fracción de infectados
            if len(I_curve) < TMAX:
                I_curve = np.pad(I_curve, (0, TMAX - len(I_curve)), 'edge')
            else:
                I_curve = I_curve[:TMAX]
            all_I.append(I_curve)
        except Exception as e:
            continue
            
    if len(all_I) == 0:
        return np.zeros(TMAX)
        
    return np.mean(all_I, axis=0)

def main():
    start_time = time.time()
    
    print("1. Cargando datos reales...")
    df = pd.read_csv("stopkoronavirus_clean_wave_winter.csv")
    real_data = df["CONFIRMED"].values
    
    # Parámetros óptimos encontrados previamente
    beta_net = 0.2143
    beta_hh = 2.4335
    delta = 1.1722
    fermi_mu = 22.4418
    scale = 1724663.41
    shift = 4
    
    print("2. Ejecutando Simulador Mecanístico SBM-SIR...")
    print(f"Usando parámetros: beta_net={beta_net}, beta_hh={beta_hh}, delta={delta}, mu={fermi_mu}")
    print("Promediando 10 simulaciones para reducir ruido estocástico...")
    
    I_pred = run_mechanistic_scenario(beta_net, beta_hh, delta, fermi_mu, num_sims=10)
    
    # Escalar y desplazar según los parámetros encontrados
    I_pred_scaled = I_pred * scale
    
    shifted_pred = np.zeros(len(real_data))
    if shift >= 0:
        length = min(len(I_pred_scaled), len(real_data) - shift)
        if length > 0:
            shifted_pred[shift:shift+length] = I_pred_scaled[:length]
    else:
        length = min(len(I_pred_scaled) + shift, len(real_data))
        if length > 0:
            shifted_pred[:length] = I_pred_scaled[-shift:-shift+length]
            
    # Calcular Métricas
    r2 = r2_score(real_data, shifted_pred)
    mse = mean_squared_error(real_data, shifted_pred)
    mse_relativo = mse / np.var(real_data)
    
    exec_time = time.time() - start_time
    
    print("\n" + "="*40)
    print(" RESULTADOS SIMULADOR MECANÍSTICO")
    print("="*40)
    print(f"R²               : {r2:.4f}")
    print(f"MSE Relativo     : {mse_relativo:.4f}")
    print(f"Tiempo Ejecución : {exec_time:.2f} segundos")
    print("="*40)
    
    # Graficar
    plt.figure(figsize=(12, 7))
    plt.plot(real_data, 'ko', markersize=4, label='Datos Reales (Wave Winter)', alpha=0.6)
    plt.plot(shifted_pred, color="#3498DB", linewidth=3, label=f'SBM Mecanístico (R²={r2:.3f})')
    
    plt.title("Ajuste de Ola de Invierno: Simulador Mecanístico SBM-SIR", fontsize=18, fontweight='bold', pad=20, color="#2c3e50")
    plt.xlabel("Días", fontsize=14, labelpad=10)
    plt.ylabel("Casos Confirmados", fontsize=14, labelpad=10)
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, "fit_mechanistic_wave_winter.png")
    plt.savefig(out_img, dpi=300)
    print(f"\nGráfica de validación guardada en: {out_img}")

if __name__ == "__main__":
    main()
