import numpy as np
import matplotlib.pyplot as plt
import os

# Configuración
DATASET_PATH = "output/ai_sbm/dataset_normalized.npz"
OUTPUT_IMG = "output/ai_sbm/visualizacion_dataset.png"

def plot_dataset_samples(num_to_plot=50):
    if not os.path.exists(DATASET_PATH):
        print(f"Error: No se encontró el dataset en {DATASET_PATH}")
        return

    data = np.load(DATASET_PATH)
    X = data['X'] # [num_samples, 4] -> [beta_net, beta_hh, delta, mu]
    Y = data['Y'] # [num_samples, 100] -> Curvas de infectados

    plt.figure(figsize=(12, 7))
    
    # Graficar una muestra aleatoria de curvas
    indices = np.random.choice(len(Y), min(num_to_plot, len(Y)), replace=False)
    
    for idx in indices:
        mu_val = X[idx, 3]
        # Color basado en Mu (Corta vs Larga distancia)
        color = 'green' if mu_val < 10 else 'blue'
        alpha = 0.2
        plt.plot(Y[idx], color=color, alpha=alpha, linewidth=0.8)

    # Añadir líneas de referencia de color para la leyenda
    plt.plot([], [], color='green', label='Corta Distancia (Mu bajo)', alpha=0.6)
    plt.plot([], [], color='blue', label='Larga Distancia (Mu alto)', alpha=0.6)

    plt.title(f"Visualización del Dataset de Entrenamiento ({num_to_plot} muestras de {len(Y)})", fontsize=16)
    plt.xlabel("Pasos de Tiempo", fontsize=14)
    plt.ylabel("Fracción de Infectados (Normalizada)", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Estética premium
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualización del dataset guardada en: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_dataset_samples()
