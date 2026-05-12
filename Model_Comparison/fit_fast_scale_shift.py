import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
from copy import deepcopy
import random

from simple_sbm_generator import (
    GeneradorSBM,
    ModeloConfig,
    construir_red_manzanas_con_proyeccion_hubs,
)
from test_simulation import MODEL_CONFIG_TEMPLATE, SIMULATION_PARAMS

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

def run_mechanistic_scenario(beta_net, beta_hh, delta, fermi_mu, num_sims=10):
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
        try:
            out = generador.simulate(packet)
            N_tot = int(out.meta["N_tot"])
            I_curve = out.I / N_tot
            if len(I_curve) < TMAX:
                I_curve = np.pad(I_curve, (0, TMAX - len(I_curve)), 'edge')
            else:
                I_curve = I_curve[:TMAX]
            all_I.append(I_curve)
        except Exception:
            continue
            
    if len(all_I) == 0:
        return np.zeros(TMAX)
        
    return np.mean(all_I, axis=0)

def shifted_curve(curve, shift, n_points):
    source_t = np.arange(len(curve), dtype=float)
    target_t = np.arange(n_points, dtype=float) - shift
    return np.interp(target_t, source_t, curve, left=0.0, right=curve[-1])

def main():
    start_time = time.time()
    
    df = pd.read_csv("stopkoronavirus_clean_wave_winter.csv")
    real_data = df["CONFIRMED"].values
    
    # Usar los parámetros como fijos
    beta_net = 0.2143
    beta_hh = 2.4335
    delta = 1.1722
    fermi_mu = 22.4418
    
    print("1. Generando la curva base SBM (promedio de 15 corridas)...")
    base_curve = run_mechanistic_scenario(beta_net, beta_hh, delta, fermi_mu, num_sims=15)
    
    print("2. Optimizando Scale y Shift...")
    n_days = len(real_data)
    
    def objective(params):
        scale, shift = params
        pred = scale * shifted_curve(base_curve, shift, n_days)
        return mean_squared_error(real_data, pred)
        
    # Semilla para el optimizador
    x0 = np.array([1724663.41, 4.0])
    bounds = [(100000, 5000000), (-30, 30)]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    best_scale, best_shift = res.x
    pred_cases = best_scale * shifted_curve(base_curve, best_shift, n_days)
    
    r2 = r2_score(real_data, pred_cases)
    mse_rel = mean_squared_error(real_data, pred_cases) / np.var(real_data)
    exec_time = time.time() - start_time
    
    print("\n" + "="*40)
    print(" RESULTADOS AJUSTE RÁPIDO (SCALE & SHIFT)")
    print("="*40)
    print(f"Beta Net Fijo : {beta_net}")
    print(f"Beta HH Fijo  : {beta_hh}")
    print(f"Delta Fijo    : {delta}")
    print(f"Fermi Mu Fijo : {fermi_mu}")
    print("-" * 40)
    print(f"Scale Óptimo  : {best_scale:.2f}")
    print(f"Shift Óptimo  : {best_shift:.2f} días")
    print(f"R²            : {r2:.4f}")
    print(f"MSE Relativo  : {mse_rel:.4f}")
    print(f"T. Ejecución  : {exec_time:.2f} s")
    print("="*40)
    
    plt.figure(figsize=(12, 7))
    plt.plot(real_data, 'ko', markersize=4, label='Datos Reales (Wave Winter)', alpha=0.6)
    plt.plot(pred_cases, color="#8E44AD", linewidth=3, label=f'SBM Ajustado (R²={r2:.3f})')
    
    plt.title("Ajuste de Escala y Desfase sobre Curva SBM Fija", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Días", fontsize=14)
    plt.ylabel("Casos Confirmados", fontsize=14)
    plt.legend(fontsize=12, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, "fit_fast_scale_shift.png")
    plt.savefig(out_img, dpi=300)
    print(f"\nGráfica guardada en: {out_img}")

if __name__ == "__main__":
    main()
