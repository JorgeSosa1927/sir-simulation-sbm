import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "output" / "ai_sbm" / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / "output" / "ai_sbm" / "xdg_cache"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "ai_sbm"
MODEL_FILE = OUTPUT_DIR / "surrogate_model_normalized.pth"
OUTPUT_FILE = OUTPUT_DIR / "arquitectura_red_entrenada_colormap.png"


def load_state_dict():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado: {MODEL_FILE}")
    return torch.load(MODEL_FILE, map_location="cpu")


def to_numpy(state_dict, key):
    return state_dict[key].detach().cpu().numpy()


def linear_scores(weight, bias):
    """Valor firmado por neurona: bias + promedio de pesos entrantes."""
    return bias + weight.mean(axis=1)


def lstm_scores(state_dict, layer):
    w_ih = to_numpy(state_dict, f"lstm.weight_ih_l{layer}")
    w_hh = to_numpy(state_dict, f"lstm.weight_hh_l{layer}")
    b_ih = to_numpy(state_dict, f"lstm.bias_ih_l{layer}")
    b_hh = to_numpy(state_dict, f"lstm.bias_hh_l{layer}")
    return b_ih + b_hh + w_ih.mean(axis=1) + w_hh.mean(axis=1)


def aggregate(values, size):
    values = np.asarray(values).ravel()
    if len(values) == size:
        return values
    chunks = np.array_split(values, size)
    return np.array([chunk.mean() for chunk in chunks])


def build_layer_values(state_dict):
    input_scores = to_numpy(state_dict, "param_encoder.0.weight").mean(axis=0)
    enc_64 = linear_scores(
        to_numpy(state_dict, "param_encoder.0.weight"),
        to_numpy(state_dict, "param_encoder.0.bias"),
    )
    enc_128 = linear_scores(
        to_numpy(state_dict, "param_encoder.2.weight"),
        to_numpy(state_dict, "param_encoder.2.bias"),
    )
    state_512 = linear_scores(
        to_numpy(state_dict, "param_encoder.4.weight"),
        to_numpy(state_dict, "param_encoder.4.bias"),
    )
    lstm_0 = lstm_scores(state_dict, 0)
    lstm_1 = lstm_scores(state_dict, 1)
    decoder = linear_scores(
        to_numpy(state_dict, "decoder.weight"),
        to_numpy(state_dict, "decoder.bias"),
    )

    return [
        input_scores,
        aggregate(enc_64, 12),
        aggregate(enc_128, 18),
        aggregate(state_512, 20),
        aggregate(lstm_0, 20),
        aggregate(lstm_1, 20),
        decoder,
    ]


def draw_neural_net(ax, left, right, bottom, top, layer_values, cmap, norm):
    layer_sizes = [len(values) for values in layer_values]
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    for n, (size_a, size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (size_a - 1) / 2.0 + (top + bottom) / 2.0
        layer_top_b = v_spacing * (size_b - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(size_a):
            for o in range(size_b):
                line = plt.Line2D(
                    [n * h_spacing + left, (n + 1) * h_spacing + left],
                    [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                    c="#6b7280",
                    alpha=0.12,
                    lw=0.45,
                    zorder=1,
                )
                ax.add_artist(line)

    for n, values in enumerate(layer_values):
        layer_size = len(values)
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        for m, value in enumerate(values):
            circle = plt.Circle(
                (n * h_spacing + left, layer_top - m * v_spacing),
                v_spacing / 3.1,
                color=cmap(norm(value)),
                ec="#111827",
                lw=1.1,
                zorder=4,
            )
            ax.add_artist(circle)


def plot_trained_schematic():
    state_dict = load_state_dict()
    layer_values = build_layer_values(state_dict)
    all_values = np.concatenate([np.ravel(v) for v in layer_values])
    vmax = np.percentile(np.abs(all_values), 98)
    vmax = float(vmax) if vmax > 0 else 1.0

    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(17, 9.5), facecolor="white")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_neural_net(ax, 0.08, 0.90, 0.18, 0.82, layer_values, cmap, norm)

    h_spacing = (0.90 - 0.08) / float(len(layer_values) - 1)
    layer_names = [
        "Input\nParams\nreal: 4",
        "Encoder H1\nDense\nreal: 64",
        "Encoder H2\nDense\nreal: 128",
        "h0 / c0\nInit LSTM\nreal: 512",
        "LSTM capa 0\n4 gates x 128\nreal: 512",
        "LSTM capa 1\n4 gates x 128\nreal: 512",
        "Decoder\nI(t)\nreal: 1",
    ]

    for n, name in enumerate(layer_names):
        ax.text(
            n * h_spacing + 0.08,
            0.90,
            name,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#1f2933",
        )

    ax.text(
        0.5,
        1.03,
        "Red neuronal entrenada: nodos coloreados por pesos aprendidos",
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color="#111827",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.985,
        "Representacion esquematica de la LSTM autoregresiva condicionada por parametros",
        ha="center",
        va="center",
        fontsize=13,
        color="#4b5563",
        transform=ax.transAxes,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("bias + promedio de pesos entrantes", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.text(0.10, 0.075, "Azul: valor negativo", color=cmap(norm(-vmax)), fontsize=12, fontweight="bold")
    ax.text(0.32, 0.075, "Blanco: cercano a cero", color="#4b5563", fontsize=12, fontweight="bold")
    ax.text(0.56, 0.075, "Rojo: valor positivo", color=cmap(norm(vmax)), fontsize=12, fontweight="bold")
    ax.text(
        0.10,
        0.035,
        "Nota: los nodos grandes de LSTM y h0/c0 estan agregados para que la figura sea legible.",
        color="#4b5563",
        fontsize=10,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafica generada exitosamente en:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    plot_trained_schematic()
