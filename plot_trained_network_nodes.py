import os
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "output" / "ai_sbm" / "surrogate_model_normalized.pth"
OUTPUT_DIR = ROOT / "output" / "ai_sbm"
OUT_FILE = OUTPUT_DIR / "nodos_red_entrenada_colormap.svg"


PARAM_NAMES = ["beta_net", "beta_hh", "delta", "fermi_mu"]
GATE_NAMES = ["input gate", "forget gate", "cell gate", "output gate"]


def load_state_dict(path):
    if not path.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado: {path}")
    return torch.load(path, map_location="cpu")


def tensor(state_dict, key):
    return state_dict[key].detach().cpu().numpy()


def linear_node_scores(weight, bias):
    """Signed score per output node using learned bias and mean incoming weight."""
    return bias + weight.mean(axis=1)


def lstm_gate_scores(state_dict, layer):
    w_ih = tensor(state_dict, f"lstm.weight_ih_l{layer}")
    w_hh = tensor(state_dict, f"lstm.weight_hh_l{layer}")
    b_ih = tensor(state_dict, f"lstm.bias_ih_l{layer}")
    b_hh = tensor(state_dict, f"lstm.bias_hh_l{layer}")
    return b_ih + b_hh + w_ih.mean(axis=1) + w_hh.mean(axis=1)


def robust_scale(values):
    joined = np.concatenate([np.ravel(v) for v in values if np.size(v) > 0])
    scale = np.percentile(np.abs(joined), 98)
    return float(scale) if scale > 0 else 1.0


def color_for(value, scale):
    v = float(np.clip(value / scale, -1.0, 1.0))
    if v < 0:
        t = -v
        r = int(246 * (1 - t) + 37 * t)
        g = int(247 * (1 - t) + 99 * t)
        b = int(249 * (1 - t) + 235 * t)
    else:
        t = v
        r = int(246 * (1 - t) + 220 * t)
        g = int(247 * (1 - t) + 38 * t)
        b = int(249 * (1 - t) + 38 * t)
    return f"rgb({r},{g},{b})"


def text(x, y, value, size=13, weight="400", anchor="middle", color="#1f2933"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{escape(str(value))}</text>'
    )


def rect(x, y, w, h, fill, stroke="#22303c", sw=1.2, rx=8):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def node_grid(x, y, title, values, scale, cols=16, cell=12, gap=3, label_every=None):
    values = np.asarray(values).ravel()
    rows = int(np.ceil(len(values) / cols))
    width = cols * cell + (cols - 1) * gap
    height = rows * cell + (rows - 1) * gap
    svg = [
        text(x + width / 2, y - 18, title, size=15, weight="700"),
        rect(x - 12, y - 8, width + 24, height + 18, "#ffffff", stroke="#d1d7de", sw=1.0, rx=10),
    ]
    for idx, value in enumerate(values):
        row = idx // cols
        col = idx % cols
        cx = x + col * (cell + gap)
        cy = y + row * (cell + gap)
        svg.append(rect(cx, cy, cell, cell, color_for(value, scale), stroke="#ffffff", sw=0.45, rx=cell / 2))
        if label_every and idx % label_every == 0:
            svg.append(text(cx + cell / 2, cy - 3, idx, size=7, color="#52606d"))
    svg.append(text(x + width / 2, y + height + 25, f"{len(values)} nodos", size=11, color="#52606d"))
    return "\n".join(svg), width, height


def arrow(x1, y1, x2, y2, label=None):
    label_svg = ""
    if label:
        lx = (x1 + x2) / 2
        ly = (y1 + y2) / 2 - 10
        label_svg = rect(lx - 72, ly - 17, 144, 24, "#ffffff", stroke="none", sw=0, rx=7)
        label_svg += text(lx, ly, label, size=11)
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="#22303c" '
        f'stroke-width="2.0" marker-end="url(#arrowhead)"/>{label_svg}'
    )


def legend(x, y, scale):
    svg = [text(x + 150, y - 16, "Mapa de color del valor aprendido", size=14, weight="700")]
    steps = 60
    for i in range(steps):
        v = -scale + 2 * scale * i / (steps - 1)
        svg.append(rect(x + i * 5, y, 5, 22, color_for(v, scale), stroke="none", sw=0, rx=0))
    svg.append(rect(x, y, steps * 5, 22, "none", stroke="#22303c", sw=1, rx=0))
    svg.append(text(x, y + 42, f"-{scale:.3g}", size=11, anchor="start"))
    svg.append(text(x + steps * 2.5, y + 42, "0", size=11))
    svg.append(text(x + steps * 5, y + 42, f"+{scale:.3g}", size=11, anchor="end"))
    svg.append(text(x + 150, y + 68, "azul = negativo, blanco = cercano a cero, rojo = positivo", size=11))
    return "\n".join(svg)


def input_nodes(x, y):
    svg = [text(x + 60, y - 18, "Entrada", size=15, weight="700")]
    for i, name in enumerate(PARAM_NAMES):
        cy = y + i * 42
        svg.append(rect(x, cy, 120, 28, "#d9ecff", stroke="#22303c", sw=1.1, rx=14))
        svg.append(text(x + 60, cy + 19, name, size=11))
    return "\n".join(svg)


def build_svg(state_dict):
    enc64 = linear_node_scores(tensor(state_dict, "param_encoder.0.weight"), tensor(state_dict, "param_encoder.0.bias"))
    enc128 = linear_node_scores(tensor(state_dict, "param_encoder.2.weight"), tensor(state_dict, "param_encoder.2.bias"))
    state512 = linear_node_scores(tensor(state_dict, "param_encoder.4.weight"), tensor(state_dict, "param_encoder.4.bias"))
    lstm0 = lstm_gate_scores(state_dict, 0)
    lstm1 = lstm_gate_scores(state_dict, 1)
    decoder = linear_node_scores(tensor(state_dict, "decoder.weight"), tensor(state_dict, "decoder.bias"))
    scale = robust_scale([enc64, enc128, state512, lstm0, lstm1, decoder])

    width = 1800
    height = 1180
    svg = [
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrowhead" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto">
    <polygon points="0 0, 12 4, 0 8" fill="#22303c"/>
  </marker>
</defs>
<rect width="{width}" height="{height}" fill="#fbfcfd"/>
{text(width / 2, 48, "Nodos de la red neuronal entrenada", size=26, weight="700")}
{text(width / 2, 78, "Cada nodo se colorea con bias + promedio de pesos entrantes; en LSTM se muestra por compuerta.", size=14)}
""",
        legend(80, 105, scale),
        input_nodes(80, 285),
    ]

    grid1, w1, h1 = node_grid(250, 250, "Encoder Linear 4 -> 64", enc64, scale, cols=8, cell=15, gap=5)
    grid2, w2, h2 = node_grid(470, 230, "Encoder Linear 64 -> 128", enc128, scale, cols=16, cell=12, gap=4)
    grid3, w3, h3 = node_grid(790, 200, "Estados h0/c0 del encoder (512)", state512, scale, cols=32, cell=9, gap=3)
    svg.extend([grid1, grid2, grid3])

    svg.append(arrow(205, 370, 250, 370, "4 -> 64"))
    svg.append(arrow(405, 370, 470, 370, "64 -> 128"))
    svg.append(arrow(730, 370, 790, 370, "128 -> 512"))

    x_lstm0 = 80
    y_lstm = 640
    for gate_idx, gate_name in enumerate(GATE_NAMES):
        values = lstm0[gate_idx * 128:(gate_idx + 1) * 128]
        grid, _, _ = node_grid(x_lstm0 + gate_idx * 265, y_lstm, f"LSTM capa 0 - {gate_name}", values, scale, cols=16, cell=10, gap=3)
        svg.append(grid)

    x_lstm1 = 80
    y_lstm2 = 865
    for gate_idx, gate_name in enumerate(GATE_NAMES):
        values = lstm1[gate_idx * 128:(gate_idx + 1) * 128]
        grid, _, _ = node_grid(x_lstm1 + gate_idx * 265, y_lstm2, f"LSTM capa 1 - {gate_name}", values, scale, cols=16, cell=10, gap=3)
        svg.append(grid)

    svg.append(arrow(970, 535, 970, 625, "h0/c0 inicializa LSTM"))
    svg.append(arrow(610, 810, 610, 850, "salida capa 0"))

    grid_dec, _, _ = node_grid(1325, 520, "Decoder Linear 128 -> 1", decoder, scale, cols=1, cell=28, gap=4)
    svg.append(grid_dec)
    svg.append(rect(1470, 515, 210, 92, "#ffe0d1", stroke="#22303c", sw=1.5, rx=12))
    svg.append(text(1575, 550, "Sigmoid + I(t)", size=16, weight="700"))
    svg.append(text(1575, 578, "salida en [0, 1]", size=13))
    svg.append(arrow(1280, 565, 1325, 565, "128 -> 1"))
    svg.append(arrow(1365, 565, 1470, 565))

    svg.append(rect(1225, 685, 500, 245, "#ffffff", stroke="#d1d7de", sw=1.2, rx=14))
    svg.append(text(1475, 725, "Resumen de parametros visualizados", size=17, weight="700"))
    svg.append(text(1255, 765, "Encoder: 64 + 128 + 512 nodos", size=13, anchor="start"))
    svg.append(text(1255, 792, "LSTM capa 0: 4 compuertas x 128 nodos", size=13, anchor="start"))
    svg.append(text(1255, 819, "LSTM capa 1: 4 compuertas x 128 nodos", size=13, anchor="start"))
    svg.append(text(1255, 846, "Decoder: 1 nodo antes de sigmoid", size=13, anchor="start"))
    svg.append(text(1255, 883, "Nota: el color no es activacion con datos.", size=13, anchor="start"))
    svg.append(text(1255, 910, "Es una firma de pesos entrenados por nodo.", size=13, anchor="start"))

    svg.append("</svg>\n")
    return "\n".join(svg)


def main():
    state_dict = load_state_dict(MODEL_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUT_FILE.write_text(build_svg(state_dict), encoding="utf-8")
    print(f"Grafica guardada en {OUT_FILE}")


if __name__ == "__main__":
    main()
