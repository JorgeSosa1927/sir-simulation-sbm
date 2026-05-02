import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from copy import deepcopy

# Import existing simulator components
from simple_sbm_generator import (
    GeneradorSBM,
    ModeloConfig,
    construir_red_manzanas_con_proyeccion_hubs,
)
from test_simulation import MODEL_CONFIG_TEMPLATE, SIMULATION_PARAMS

# Configuración Global
OUTPUT_DIR = "output/ai_sbm"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TMAX = SIMULATION_PARAMS["tmax"]
DATASET_FILE = os.path.join(OUTPUT_DIR, "dataset_normalized.npz")
MODEL_FILE = os.path.join(OUTPUT_DIR, "surrogate_model_normalized.pth")
METRICS_FILE = os.path.join(OUTPUT_DIR, "eval_metrics_normalized.txt")


# 1. Dataset Generation
def run_custom_scenario(beta_net, beta_hh, delta, fermi_mu, num_sims=10):
    """Ejecuta el simulador y retorna la curva promedio de fraccion infectada.

    Cada replica se normaliza primero por su poblacion total y despues se
    promedia. Esto permite comparar generaciones con distinto numero de
    personas sin que el modelo aprenda la escala poblacional absoluta.
    """
    cfg_data = deepcopy(MODEL_CONFIG_TEMPLATE)
    cfg_data["fermi_mu"] = fermi_mu
    cfg_data["fermi_beta"] = 0.2  # Mantenemos este espacial fijo
    
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
            # Asegurar longitud igual a TMAX
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

def generate_dataset(num_samples=150, num_sims_per_sample=8):
    """Genera datos sintéticos variando los parámetros clave."""
    print("Generando dataset sintético normalizado usando el simulador original SBM-SIR...")
    dataset_file = DATASET_FILE
    
    if os.path.exists(dataset_file):
        print("Cargando dataset existente...")
        data = np.load(dataset_file)
        return data['X'], data['Y']
    
    X = []
    Y = []
    
    for i in range(num_samples):
        # Muestreo aleatorio de parámetros
        beta_net = np.random.uniform(0.1, 0.8)
        beta_hh = np.random.uniform(1.0, 3.5)
        delta = np.random.uniform(0.6, 1.2)
        fermi_mu = np.random.uniform(4.0, 40.0) # Corta vs Larga distancia
        
        # Consideramos incluir muestras específicas en los extremos
        if i < num_samples // 4:
            fermi_mu = np.random.uniform(4.0, 6.0) # Asegurar datos de corta distancia
        elif i < num_samples // 2:
            fermi_mu = np.random.uniform(14.0, 40.0) # Asegurar datos de larga distancia
            
        params = [beta_net, beta_hh, delta, fermi_mu]
        I_mean = run_custom_scenario(*params, num_sims=num_sims_per_sample)
        
        X.append(params)
        Y.append(I_mean)
        
        if (i+1) % 10 == 0:
            print(f"Progreso dataset: {i+1}/{num_samples}...")
            
    X = np.array(X)
    Y = np.array(Y)
    
    np.savez(dataset_file, X=X, Y=Y)
    print(f"Dataset normalizado generado y guardado en {dataset_file}")
    return X, Y

# 2. Arquitectura temporal autoregresiva (Surrogate Model)
class EpidemicSurrogateNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=TMAX, num_layers=2):
        super(EpidemicSurrogateNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Los parámetros epidemiológicos definen el estado inicial de la LSTM.
        self.param_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim * num_layers * 2)
        )
        
        # La curva se genera con dependencia temporal usando I(t-1) como entrada.
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        batch_size = x.size(0)
        device = x.device
        
        h0_c0 = self.param_encoder(x)
        h0_c0 = h0_c0.view(batch_size, self.num_layers, 2, self.hidden_dim)
        h_t = h0_c0[:, :, 0, :].permute(1, 0, 2).contiguous()
        c_t = h0_c0[:, :, 1, :].permute(1, 0, 2).contiguous()
        
        outputs = []
        input_t = torch.zeros(batch_size, 1, device=device)
        
        for t in range(self.output_dim):
            out, (h_t, c_t) = self.lstm(input_t.unsqueeze(1), (h_t, c_t))
            pred = torch.sigmoid(self.decoder(out.squeeze(1)))
            outputs.append(pred)
            
            use_teacher_forcing = (
                target is not None
                and teacher_forcing_ratio > 0.0
                and torch.rand(1, device=device).item() < teacher_forcing_ratio
            )
            input_t = target[:, t:t + 1] if use_teacher_forcing else pred
        
        return torch.cat(outputs, dim=1)

# 3. Entrenamiento
def train_model(X, Y, epochs=500, batch_size=32):
    print("Normalizando datos y preparando particiones...")
    
    X_scaler = StandardScaler()
    Y_scaler = MinMaxScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    Y_scaled = Y_scaler.fit_transform(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.15, random_state=42)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = EpidemicSurrogateNet(input_dim=X.shape[1], output_dim=Y.shape[1])
    
    # Optimizador AdamW y Scheduler para mejor convergencia
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    # Función de pérdida personalizada:
    # MSE + pico + primera derivada temporal + segunda derivada temporal.
    def criterion(pred, target):
        mse = nn.MSELoss()(pred, target)
        
        # Penalizar diferencia en la altura del pico de infectados
        peak_pred, _ = torch.max(pred, dim=1)
        peak_target, _ = torch.max(target, dim=1)
        peak_loss = nn.MSELoss()(peak_pred, peak_target)
        
        # Primera derivada discreta: pendiente diaria de la curva.
        d1_pred = pred[:, 1:] - pred[:, :-1]
        d1_target = target[:, 1:] - target[:, :-1]
        first_derivative_loss = nn.MSELoss()(d1_pred, d1_target)
        
        # Segunda derivada discreta: cambios en la pendiente/curvatura.
        d2_pred = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
        d2_target = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
        second_derivative_loss = nn.MSELoss()(d2_pred, d2_target)
        
        return (
            mse
            + 0.6 * peak_loss
            + 0.3 * first_derivative_loss
            + 0.2 * second_derivative_loss
        )
    
    print("Iniciando entrenamiento robusto del modelo sustituto...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, target=batch_Y, teacher_forcing_ratio=0.5)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(train_loader)
        scheduler.step(avg_loss)
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    print("Entrenamiento finalizado.")
    
    # Guardar modelo
    model_path = MODEL_FILE
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")
    
    return model, X_test, Y_test, X_scaler, Y_scaler

# 4. Evaluación y Métricas
def evaluate_model(model, X_test, Y_test, Y_scaler):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy()
        
    preds_orig = Y_scaler.inverse_transform(preds)
    Y_test_orig = Y_scaler.inverse_transform(Y_test)
    
    mse = mean_squared_error(Y_test_orig, preds_orig)
    mae = mean_absolute_error(Y_test_orig, preds_orig)
    r2 = r2_score(Y_test_orig.flatten(), preds_orig.flatten())
    
    print("\n--- Resultados de Evaluación en Test ---")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"R²:  {r2:.4f}")
    
    metrics_file = METRICS_FILE
    with open(metrics_file, "w") as f:
        f.write("Metricas de Evaluacion del Surrogate Model (curvas normalizadas por poblacion):\n")
        f.write(f"MSE: {mse:.8f}\n")
        f.write(f"MAE: {mae:.8f}\n")
        f.write(f"R2: {r2:.4f}\n")
        
    return mse, mae, r2

# 5. Comparación Conjunta Completa (Original vs Surrogate en Corta y Larga Distancia)
def plot_ultimate_validation(model, X_scaler, Y_scaler):
    print("Corriendo modelo original para comparación directa de métricas conjuntas...")
    
    # Parámetros base (Ajustados con factor 1.09)
    beta_net = 0.4469
    beta_hh = 2.289
    delta = 0.9592
    
    mu_short = 5.0
    mu_long = 15.0
    
    # Simulación Original
    print("Simulando Corta Distancia (Mu=5.0)...")
    I_real_short = run_custom_scenario(beta_net, beta_hh, delta, mu_short, num_sims=20)
    
    print("Simulando Larga Distancia (Mu=15.0)...")
    I_real_long = run_custom_scenario(beta_net, beta_hh, delta, mu_long, num_sims=20)
    
    # Predicciones Surrogate
    params_short = np.array([[beta_net, beta_hh, delta, mu_short]])
    params_long = np.array([[beta_net, beta_hh, delta, mu_long]])
    
    x_short_scaled = X_scaler.transform(params_short)
    x_long_scaled = X_scaler.transform(params_long)
    
    model.eval()
    with torch.no_grad():
        pred_short_scaled = model(torch.FloatTensor(x_short_scaled)).numpy()
        pred_long_scaled = model(torch.FloatTensor(x_long_scaled)).numpy()
        
    I_pred_short = Y_scaler.inverse_transform(pred_short_scaled)[0]
    I_pred_long = Y_scaler.inverse_transform(pred_long_scaled)[0]
    
    # Evitar negativos
    I_pred_short = np.maximum(I_pred_short, 0)
    I_pred_long = np.maximum(I_pred_long, 0)
    
    # Calcular Métricas
    r2_short = r2_score(I_real_short, I_pred_short)
    mse_short = mean_squared_error(I_real_short, I_pred_short)
    
    r2_long = r2_score(I_real_long, I_pred_long)
    mse_long = mean_squared_error(I_real_long, I_pred_long)
    
    t = np.arange(TMAX)
    
    plt.figure(figsize=(12, 7))
    
    # Trazar Larga Distancia (Mu=15) - Predominancia Libre
    plt.plot(t, I_real_long, label=f"Original (Larga Dist., Mu=15.0)", color="blue", linestyle="--", linewidth=2, alpha=0.6)
    plt.plot(t, I_pred_long, label=f"Surrogate (Larga Dist., Mu=15.0)", color="blue", linestyle="-", linewidth=2.5)
    
    # Trazar Corta Distancia (Mu=5) - Predominancia Local
    plt.plot(t, I_real_short, label=f"Original (Corta Dist., Mu=5.0)", color="green", linestyle="--", linewidth=2, alpha=0.6)
    plt.plot(t, I_pred_short, label=f"Surrogate (Corta Dist., Mu=5.0)", color="green", linestyle="-", linewidth=2.5)
    
    plt.title("Validación de Modelos: Mecanicista SBM-SIR vs Autoencoder Surrogate", fontsize=14)
    plt.xlabel("Tiempo (Pasos)", fontsize=12)
    plt.ylabel("Fracción infectada promedio", fontsize=12)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Mostrar métricas agrupadas en una caja de texto
    textstr = '\n'.join((
        r'$\bf{Métricas\ Larga\ Distancia}$',
        f'$R^2$: {r2_long:.3f}',
        f'MSE: {mse_long:.6f}',
        '',
        r'$\bf{Métricas\ Corta\ Distancia}$',
        f'$R^2$: {r2_short:.3f}',
        f'MSE: {mse_short:.6f}'
    ))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    out_img = os.path.join(OUTPUT_DIR, "validacion_surrogate_comparativa_normalizada.png")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfica unificada final guardada en {out_img}")

def main():
    print("=== Iniciando Pipeline de AI SBM Mejorado ===")
    
    # Conservamos el dataset normalizado si ya existe; solo borramos el modelo viejo
    model_file = MODEL_FILE
    if os.path.exists(model_file): os.remove(model_file)

    # 1. Dataset normalizado: si existe, se carga directamente
    X, Y = generate_dataset(num_samples=300, num_sims_per_sample=20)
    
    # 2. Entrenar con nueva arquitectura y pérdida de pico
    model, X_test, Y_test, X_scaler, Y_scaler = train_model(X, Y, epochs=600)
    
    # 3. Evaluar
    evaluate_model(model, X_test, Y_test, Y_scaler)
    
    # 4. Generar plots comparativos
    plot_ultimate_validation(model, X_scaler, Y_scaler)
    
    print("=== Pipeline completado con éxito ===")

if __name__ == "__main__":
    main()
