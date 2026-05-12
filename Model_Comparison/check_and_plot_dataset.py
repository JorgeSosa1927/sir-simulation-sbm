import numpy as np
import matplotlib.pyplot as plt
import os

DATASET_PATH = "output/ai_sbm/dataset_normalized.npz"
OUTPUT_IMG = "output/ai_sbm/visualizacion_dataset.png"

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} no encontrado.")
        return

    data = np.load(DATASET_PATH)
    X = data['X']
    Y = data['Y']
    
    # 1. Verificar si hay familias de soluciones en 0 (curvas completamente en 0)
    zero_curves = []
    for i in range(len(Y)):
        if np.all(Y[i] == 0):
            zero_curves.append(i)
            
    print(f"Total de simulaciones en el dataset: {len(Y)}")
    if len(zero_curves) > 0:
        print(f"¡Atención! Hay {len(zero_curves)} curvas que son exactamente 0 en todos los días.")
        print(f"Índices de curvas en cero: {zero_curves}")
    else:
        print("No hay ninguna curva que sea exactamente 0 en todos los días. Todas tienen al menos algún infectado en algún momento.")
        
    # También verificar cuántas curvas terminan en 0 al día 100
    zeros_at_end = np.sum(Y[:, -1] == 0)
    print(f"Curvas que terminan en exactamente 0 infectados al final de los 100 días: {zeros_at_end}")

    # 2. Graficar TODAS las soluciones en escala logarítmica
    plt.figure(figsize=(12, 8))
    
    for idx in range(len(Y)):
        mu_val = X[idx, 3]
        color = 'green' if mu_val < 10 else 'blue'
        
        # Para graficar en log, reemplazamos los 0 absolutos con un número muy pequeño (1e-6)
        # para que matplotlib no dé error matemático al calcular log(0).
        y_safe = np.maximum(Y[idx], 1e-6)
        plt.plot(y_safe, color=color, alpha=0.15, linewidth=0.8)

    # Leyenda
    plt.plot([], [], color='green', label='Corta Distancia (Mu < 10)', alpha=0.8)
    plt.plot([], [], color='blue', label='Larga Distancia (Mu >= 10)', alpha=0.8)

    plt.title("Visualización del Dataset (300 Escenarios) - Escala Logarítmica", fontsize=16)
    plt.xlabel("Días", fontsize=14)
    plt.ylabel("Fracción de Infectados (escala log, min=1e-6)", fontsize=14)
    
    # Establecer escala logarítmica
    plt.yscale('log')
    plt.ylim(bottom=1e-6, top=1.0)
    
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.2)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfica guardada en: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
