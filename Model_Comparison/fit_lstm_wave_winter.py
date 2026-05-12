import numpy as np
import pandas as pd
import torch
import time
import os
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

# Importar arquitectura del surrogate
from LSTM_SBM import EpidemicSurrogateNet

TMAX = 100
OUTPUT_DIR = "output/ai_sbm"
DATASET_FILE = os.path.join(OUTPUT_DIR, "dataset_normalized.npz")
MODEL_FILE = os.path.join(OUTPUT_DIR, "surrogate_model_normalized.pth")

def load_surrogate_and_scalers():
    data = np.load(DATASET_FILE)
    X, Y = data['X'], data['Y']
    X_scaler = StandardScaler().fit(X)
    Y_scaler = MinMaxScaler().fit(Y)
    
    input_dim = 4
    model = EpidemicSurrogateNet(input_dim=input_dim, output_dim=TMAX)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    
    return model, X_scaler, Y_scaler

def objective(params, model, X_scaler, Y_scaler, real_data):
    beta_net, beta_hh, delta, fermi_mu, scale, shift = params
    shift = int(round(shift))
    
    # Obtener predicción del surrogate
    x_input = X_scaler.transform([[beta_net, beta_hh, delta, fermi_mu]])
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(x_input)).numpy()
        
    I_pred = Y_scaler.inverse_transform(pred_scaled)[0]
    I_pred = np.maximum(I_pred, 0)
    
    # Escalar y desplazar
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
            
    return mean_squared_error(real_data, shifted_pred)

def main():
    start_time = time.time()
    
    print("1. Cargando datos reales y modelo LSTM...")
    model, X_scaler, Y_scaler = load_surrogate_and_scalers()
    
    df = pd.read_csv("stopkoronavirus_clean_wave_winter.csv")
    real_data = df["CONFIRMED"].values
    
    print("2. Iniciando optimización con Differential Evolution...")
    # Límites de búsqueda para los parámetros
    bounds = [
        (0.1, 0.8),    # beta_net
        (1.0, 3.5),    # beta_hh
        (0.6, 1.2),    # delta
        (4.0, 40.0),   # fermi_mu
        (1e4, 5e6),    # factor de escala
        (-30, 30)      # shift temporal en días
    ]
    
    result = differential_evolution(
        objective,
        bounds,
        args=(model, X_scaler, Y_scaler, real_data),
        maxiter=60,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True
    )
    
    best_params = result.x
    beta_net, beta_hh, delta, fermi_mu, scale, shift = best_params
    shift = int(round(shift))
    
    # Generar la mejor predicción
    x_input = X_scaler.transform([[beta_net, beta_hh, delta, fermi_mu]])
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(x_input)).numpy()
    I_pred = Y_scaler.inverse_transform(pred_scaled)[0]
    I_pred = np.maximum(I_pred, 0)
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
    mse_relativo = mse / np.var(real_data)  # MSE normalizado por la varianza de los datos
    
    exec_time = time.time() - start_time
    
    print("\n" + "="*40)
    print(" RESULTADOS DE LA OPTIMIZACIÓN (LSTM)")
    print("="*40)
    print(f"Beta Net: {beta_net:.4f}")
    print(f"Beta HH : {beta_hh:.4f}")
    print(f"Delta   : {delta:.4f}")
    print(f"Fermi Mu: {fermi_mu:.4f}")
    print(f"Scale   : {scale:.2f}")
    print(f"Shift   : {shift} días")
    print("-" * 40)
    print(f"R²               : {r2:.4f}")
    print(f"MSE Relativo     : {mse_relativo:.4f}")
    print(f"Tiempo Ejecución : {exec_time:.2f} segundos")
    print("="*40)
    
    # Graficar
    plt.figure(figsize=(12, 7))
    plt.plot(real_data, 'ko', markersize=4, label='Datos Reales (Wave Winter)', alpha=0.6)
    plt.plot(shifted_pred, color="#FF4B4B", linewidth=3, label=f'Ajuste LSTM Surrogate (R²={r2:.3f})')
    
    plt.title("Calibración del Modelo LSTM con Datos de la Ola de Invierno", fontsize=18, fontweight='bold', pad=20, color="#2c3e50")
    plt.xlabel("Días", fontsize=14, labelpad=10)
    plt.ylabel("Casos Confirmados", fontsize=14, labelpad=10)
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, "fit_lstm_wave_winter.png")
    plt.savefig(out_img, dpi=300)
    print(f"\nGráfica de ajuste guardada en: {out_img}")

if __name__ == "__main__":
    main()
