import os
import re
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "ai_sbm"
OUT_FILE = OUTPUT_DIR / "estructura_red_lstm_surrogate.svg"


def read_tmax(default=100):
    text = (ROOT / "test_simulation.py").read_text(encoding="utf-8")
    match = re.search(r'"tmax"\s*:\s*(\d+)', text)
    return int(match.group(1)) if match else default


def text_lines(x, y, lines, size=15, weight="400", color="#1f2933", anchor="middle", line_height=20):
    svg = []
    for i, line in enumerate(lines):
        svg.append(
            f'<text x="{x}" y="{y + i * line_height}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">{escape(line)}</text>'
        )
    return "\n".join(svg)


def box(x, y, w, h, title, lines, fill, stroke="#22303c"):
    title_y = y + 32
    body_y = y + 68
    return f"""
<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
{text_lines(x + w / 2, title_y, [title], size=17, weight="700")}
{text_lines(x + w / 2, body_y, lines, size=13, line_height=18)}
"""


def arrow(x1, y1, x2, y2, label=None, curve=0):
    if curve:
        mx = (x1 + x2) / 2
        path = f"M {x1} {y1} Q {mx} {y1 + curve} {x2} {y2}"
    else:
        path = f"M {x1} {y1} L {x2} {y2}"
    label_svg = ""
    if label:
        lx = (x1 + x2) / 2
        ly = (y1 + y2) / 2 - 12
        label_svg = f"""
<rect x="{lx - 92}" y="{ly - 18}" width="184" height="26" rx="7" fill="#ffffff" opacity="0.92"/>
{text_lines(lx, ly, [label], size=12)}
"""
    return f"""
<path d="{path}" fill="none" stroke="#22303c" stroke-width="2.2" marker-end="url(#arrowhead)"/>
{label_svg}
"""


def build_svg():
    tmax = read_tmax()
    width = 1500
    height = 820

    content = [
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrowhead" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto">
    <polygon points="0 0, 12 4, 0 8" fill="#22303c"/>
  </marker>
</defs>
<rect width="{width}" height="{height}" fill="#fbfcfd"/>
{text_lines(width / 2, 55, ["Estructura de la red: LSTM autoregresiva condicionada por parametros"], size=24, weight="700")}
""",
        box(
            70,
            145,
            210,
            130,
            "Entrada X",
            ["4 parametros", "beta_net, beta_hh", "delta, fermi_mu"],
            "#d9ecff",
        ),
        box(
            360,
            125,
            255,
            170,
            "Param encoder",
            ["Linear 4 -> 64 + ReLU", "Linear 64 -> 128 + ReLU", "Linear 128 -> 512"],
            "#dcf5df",
        ),
        box(
            695,
            125,
            245,
            170,
            "Estado inicial",
            ["512 valores", "reshape a h0 y c0", "2 capas, hidden=128"],
            "#fff1c8",
        ),
        box(
            1015,
            125,
            235,
            170,
            "LSTM",
            ["input_size=1", "hidden_size=128", "num_layers=2"],
            "#e9ddff",
        ),
        box(
            1310,
            145,
            135,
            130,
            "Decoder",
            ["Linear 128 -> 1", "Sigmoid"],
            "#ffe0d1",
        ),
        box(
            70,
            505,
            210,
            120,
            "I(0)",
            ["cero inicial", "batch x 1"],
            "#f0f2f4",
        ),
        box(
            360,
            485,
            255,
            160,
            "Paso temporal t",
            ["entra I(t-1)", "actualiza h_t, c_t", "produce estado oculto"],
            "#e9ddff",
        ),
        box(
            695,
            485,
            245,
            160,
            "Prediccion",
            ["I(t) en [0, 1]", "se reutiliza como", "entrada siguiente"],
            "#ffe0d1",
        ),
        box(
            1015,
            485,
            235,
            160,
            "Secuencia",
            [f"repetir t = 1..{tmax}", "concatenar", "predicciones"],
            "#e7eaee",
        ),
        box(
            1310,
            505,
            135,
            120,
            "Salida",
            ["curva I(t)", f"batch x {tmax}"],
            "#e7eaee",
        ),
        arrow(280, 210, 360, 210),
        arrow(615, 210, 695, 210, "h0 + c0"),
        arrow(940, 210, 1015, 210),
        arrow(1250, 210, 1310, 210),
        arrow(175, 275, 175, 505, "arranque"),
        arrow(280, 565, 360, 565),
        arrow(615, 565, 695, 565),
        arrow(940, 565, 1015, 565),
        arrow(1250, 565, 1310, 565),
        arrow(815, 485, 485, 485, "I(t) -> I(t+1)", curve=-90),
        arrow(1135, 295, 490, 485, "h_t, c_t persisten", curve=95),
        text_lines(
            width / 2,
            755,
            [
                "Tipo de modelo: surrogate secuencial autoregresivo condicionado por parametros.",
                "No es VAE: no usa mu/logvar, reparametrizacion ni termino KL.",
            ],
            size=15,
        ),
        "</svg>\n",
    ]
    return "\n".join(content)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUT_FILE.write_text(build_svg(), encoding="utf-8")
    print(f"Grafica guardada en {OUT_FILE}")


if __name__ == "__main__":
    main()
