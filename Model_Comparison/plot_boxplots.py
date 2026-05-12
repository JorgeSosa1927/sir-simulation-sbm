import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Configuración de estética premium
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
SHORT_CSV = os.path.join(OUTPUT_DIR, "datos_corta_distancia.csv")
LONG_CSV = os.path.join(OUTPUT_DIR, "datos_larga_distancia.csv")

def get_errors(csv_path):
    df = pd.read_csv(csv_path)
    # Calcular Error Absoluto para cada modelo
    lstm_err = (df['LSTM_Pred'] - df['Ground_Truth']).abs().values
    smooth_err = (df['SmoothAE_Pred'] - df['Ground_Truth']).abs().values
    basic_err = (df['BasicAE_Pred'] - df['Ground_Truth']).abs().values
    return [lstm_err, smooth_err, basic_err]

def main():
    if not os.path.exists(SHORT_CSV) or not os.path.exists(LONG_CSV):
        print("Error: No se encontraron los archivos CSV en la carpeta de salida.")
        return

    errors_short = get_errors(SHORT_CSV)
    errors_long = get_errors(LONG_CSV)
    
    # Colores
    colors = ["#FF4B4B", "#2ECC71", "#3498DB"]
    labels = ["LSTM", "Smooth AE", "Basic AE"]

    fig, ax = plt.subplots(figsize=(15, 6))

    # Posiciones para las cajas
    # Queremos agrupar por escenario: [LSTM_S, Smooth_S, Basic_S] y [LSTM_L, Smooth_L, Basic_L]
    positions_short = [1, 2, 3]
    positions_long = [5, 6, 7]
    
    all_errors = errors_short + errors_long
    all_positions = positions_short + positions_long
    
    # Crear boxplot
    bp = ax.boxplot(all_errors, positions=all_positions, patch_artist=True, widths=0.6)

    # Colorear las cajas
    for i, patch in enumerate(bp['boxes']):
        color_idx = i % 3
        patch.set_facecolor(colors[color_idx])
        patch.set_alpha(0.7)
        patch.set_edgecolor(colors[color_idx])
        patch.set_linewidth(2)

    # Estilo de medianas y bigotes
    for whisker in bp['whiskers']:
        whisker.set(color='#757575', linewidth=1.5, linestyle=":")
    for cap in bp['caps']:
        cap.set(color='#757575', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='white', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e74c3c', alpha=0.5, markersize=3)

    # Etiquetas de ejes
    ax.set_xticks([2, 6])
    ax.set_xticklabels(["Short Distance\n(Mu=5.0)", "Long Distance\n(Mu=15.0)"], fontsize=13, fontweight='medium')
    
    ax.set_title("Absolute Error Distribution by Model", fontsize=20, fontweight='bold', pad=25, color="#2c3e50")
    ax.set_ylabel("Absolute Error (Infected Fraction)", fontsize=14, labelpad=15)
    
    # Grid y estética
    ax.grid(True, linestyle="--", alpha=0.6, color="#a0a0a0")
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Leyenda personalizada
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=labels[i]) for i in range(3)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, shadow=True, title="Models")

    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, "boxplot_errores_modelos.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Boxplot guardado en: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
