import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
AI_OUTPUT_DIR = ROOT / "output" / "ai_sbm"
ENGLISH_DIR = AI_OUTPUT_DIR / "english"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(AI_OUTPUT_DIR / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(AI_OUTPUT_DIR / "xdg_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from AI_SBM import (
    DATASET_FILE,
    MODEL_FILE,
    TMAX,
    EpidemicSurrogateNet,
    run_custom_scenario,
)
from fit_rusia_with_original_sbm import run_sbm_average, shifted_curve as shift_sbm, stable_param_seed
from fit_rusia_with_sir_normal import shifted_curve as shift_sir
from fit_rusia_with_sir_normal import simulate_sir_daily_cases
from fit_rusia_with_surrogate import predict_curve, shifted_curve as shift_surrogate


def load_russia_data():
    df = pd.read_csv(ROOT / "Data_Rusia_2022.csv")
    df["Casos nuevos"] = df["Casos nuevos"].astype(str).str.replace(",", "", regex=False).astype(float)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def read_results(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        raw = raw.strip()
        if re.fullmatch(r"[-+0-9.eE]+", raw):
            values[key.strip()] = float(raw)
    return values


def normalized(values):
    peak = float(np.max(values))
    return values.copy() if peak <= 0 else values / peak


def optimal_scale(pred, target):
    denom = float(np.dot(pred, pred))
    if denom <= 1e-12:
        return 0.0
    return max(float(np.dot(target, pred) / denom), 0.0)


def load_surrogate_model_and_scalers():
    data = np.load(DATASET_FILE)
    x = data["X"]
    y = data["Y"]
    x_scaler = StandardScaler().fit(x)
    y_scaler = MinMaxScaler().fit(y)
    model = EpidemicSurrogateNet(input_dim=x.shape[1], output_dim=y.shape[1])
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()
    return model, x_scaler, y_scaler


def save_russia_data_plot(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Fecha"], df["Casos nuevos"], color="firebrick", linewidth=2, marker="o", markersize=4)
    ax.set_title("Reported New Cases in Russia, Early 2022", fontsize=15, pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("New cases", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(ENGLISH_DIR / "plot_russia_2022.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_surrogate_fit_plot(df, model, x_scaler, y_scaler):
    params = read_results(AI_OUTPUT_DIR / "ajuste_rusia_surrogate_shift.txt")
    dates = df["Fecha"].iloc[:TMAX]
    real_cases = df["Casos nuevos"].to_numpy(dtype=float)[:TMAX]
    pred_fraction = predict_curve(
        model,
        x_scaler,
        y_scaler,
        [params["beta_net"], params["beta_hh"], params["delta"], params["fermi_mu"]],
    )
    shifted = shift_surrogate(pred_fraction, params["shift_days"], len(real_cases))
    pred_cases = params.get("scale_cases", optimal_scale(shifted, real_cases)) * shifted
    shape_mse = mean_squared_error(normalized(real_cases), normalized(shifted))
    r2 = r2_score(real_cases, pred_cases)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(dates, real_cases, color="firebrick", linewidth=2.2, marker="o", markersize=3.5, label="Russia observed data")
    ax.plot(dates, pred_cases, color="navy", linewidth=2.4, label="Fitted surrogate")
    ax.set_title("Fit to Russia 2022 Data with the Trained Surrogate Network", fontsize=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("New cases", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.45)
    plt.xticks(rotation=45)
    text = "\n".join(
        [
            f"beta_net = {params['beta_net']:.4f}",
            f"beta_hh = {params['beta_hh']:.4f}",
            f"delta = {params['delta']:.4f}",
            f"fermi_mu = {params['fermi_mu']:.4f}",
            f"shift = {params['shift_days']:.2f} days",
            f"scale = {params['scale_cases']:.2e}",
            f"R2 = {r2:.4f}",
            f"Norm. MSE = {shape_mse:.6f}",
        ]
    )
    ax.text(0.02, 0.96, text, transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ENGLISH_DIR / "ajuste_rusia_surrogate_shift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_original_sbm_fit_plot(df):
    params = read_results(AI_OUTPUT_DIR / "ajuste_rusia_sbm_original_20sims.txt")
    dates = df["Fecha"].iloc[:TMAX]
    real_cases = df["Casos nuevos"].to_numpy(dtype=float)[:TMAX]
    mean_curve, _ = run_sbm_average(
        params["beta_net"],
        params["beta_hh"],
        params["delta"],
        params["fermi_mu"],
        num_sims=int(params.get("num_sims", 20)),
        seed_offset=stable_param_seed(
            [
                params["beta_net"],
                params["beta_hh"],
                params["delta"],
                params["fermi_mu"],
                params["shift_days"],
            ]
        ),
    )
    shifted = shift_sbm(mean_curve, params["shift_days"], len(real_cases))
    pred_cases = params.get("scale_cases", optimal_scale(shifted, real_cases)) * shifted
    shape_mse = mean_squared_error(normalized(real_cases), normalized(shifted))
    r2 = r2_score(real_cases, pred_cases)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(dates, real_cases, color="firebrick", linewidth=2.2, marker="o", markersize=3.5, label="Russia observed data")
    ax.plot(dates, pred_cases, color="darkgreen", linewidth=2.5, label="Fitted original SBM")
    ax.set_title("Fit to Russia 2022 Data with the Original SBM", fontsize=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("New cases", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.45)
    plt.xticks(rotation=45)
    text = "\n".join(
        [
            f"beta_net = {params['beta_net']:.4f}",
            f"beta_hh = {params['beta_hh']:.4f}",
            f"delta = {params['delta']:.4f}",
            f"fermi_mu = {params['fermi_mu']:.4f}",
            f"shift = {params['shift_days']:.2f} days",
            f"scale = {params['scale_cases']:.2e}",
            f"simulations = {int(params.get('num_sims', 20))}",
            f"R2 = {r2:.4f}",
            f"Norm. MSE = {shape_mse:.6f}",
        ]
    )
    ax.text(0.02, 0.96, text, transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ENGLISH_DIR / "ajuste_rusia_sbm_original_20sims.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_sir_fit_plot(df):
    params = read_results(AI_OUTPUT_DIR / "ajuste_rusia_sir_normal.txt")
    dates = df["Fecha"].iloc[:TMAX]
    real_cases = df["Casos nuevos"].to_numpy(dtype=float)[:TMAX]
    pred_full = simulate_sir_daily_cases(params["beta"], params["gamma"], params["N"], params["I0"])
    pred_cases = shift_sir(pred_full, params["shift_days"], len(real_cases))
    shape_mse = mean_squared_error(normalized(real_cases), normalized(pred_cases))
    r2 = r2_score(real_cases, pred_cases)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(dates, real_cases, color="firebrick", linewidth=2.2, marker="o", markersize=3.5, label="Russia observed data")
    ax.plot(dates, pred_cases, color="black", linewidth=2.5, label="Fitted standard SIR")
    ax.set_title("Fit to Russia 2022 Data with a Standard SIR Model", fontsize=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("New cases", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.45)
    plt.xticks(rotation=45)
    text = "\n".join(
        [
            f"beta = {params['beta']:.4f}",
            f"gamma = {params['gamma']:.4f}",
            f"R0 = {params['r0']:.4f}",
            f"N = {params['N']:.2e}",
            f"I0 = {params['I0']:.2e}",
            f"shift = {params['shift_days']:.2f} days",
            f"R2 = {r2:.4f}",
            f"Norm. MSE = {shape_mse:.6f}",
        ]
    )
    ax.text(0.02, 0.96, text, transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ENGLISH_DIR / "ajuste_rusia_sir_normal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_validation_plot(model, x_scaler, y_scaler, filename):
    beta_net = 0.4469
    beta_hh = 2.289
    delta = 0.9592
    mu_short = 5.0
    mu_long = 15.0

    i_real_short, pop_short = run_custom_scenario(beta_net, beta_hh, delta, mu_short, num_sims=20, return_population=True)
    i_real_long, pop_long = run_custom_scenario(beta_net, beta_hh, delta, mu_long, num_sims=20, return_population=True)

    params_short = np.array([[beta_net, beta_hh, delta, mu_short]])
    params_long = np.array([[beta_net, beta_hh, delta, mu_long]])
    with torch.no_grad():
        pred_short_scaled = model(torch.FloatTensor(x_scaler.transform(params_short))).numpy()
        pred_long_scaled = model(torch.FloatTensor(x_scaler.transform(params_long))).numpy()
    i_pred_short = np.maximum(y_scaler.inverse_transform(pred_short_scaled)[0], 0)
    i_pred_long = np.maximum(y_scaler.inverse_transform(pred_long_scaled)[0], 0)

    r2_short = r2_score(i_real_short, i_pred_short)
    mse_short = mean_squared_error(i_real_short, i_pred_short)
    r2_long = r2_score(i_real_long, i_pred_long)
    mse_long = mean_squared_error(i_real_long, i_pred_long)

    t = np.arange(TMAX)
    reference_population = int(round(np.mean([pop_short, pop_long])))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(t, i_real_long, label="Original (long distance, mu=15.0)", color="blue", linestyle="--", linewidth=2, alpha=0.6)
    ax.plot(t, i_pred_long, label="Surrogate (long distance, mu=15.0)", color="blue", linestyle="-", linewidth=2.5)
    ax.plot(t, i_real_short, label="Original (short distance, mu=5.0)", color="green", linestyle="--", linewidth=2, alpha=0.6)
    ax.plot(t, i_pred_short, label="Surrogate (short distance, mu=5.0)", color="green", linestyle="-", linewidth=2.5)
    ax.set_title("Model Validation: Mechanistic SBM-SIR vs Surrogate", fontsize=14)
    ax.set_xlabel("Time steps", fontsize=12)
    ax.set_ylabel("Mean infected fraction", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(right=0.83)
    ax.tick_params(axis="y", right=True, labelright=False)
    y_min, y_max = ax.get_ylim()
    for tick in ax.get_yticks():
        if y_min <= tick <= y_max:
            ax.text(1.01, tick, f"{tick * reference_population:,.0f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=10, color="dimgray")
    ax.text(1.10, 0.5, f"Approximate infected\n(reference N={reference_population:,})", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=12, color="dimgray")
    text = "\n".join(
        [
            r"$\bf{Long\ Distance\ Metrics}$",
            f"$R^2$: {r2_long:.3f}",
            f"MSE: {mse_long:.6f}",
            "",
            r"$\bf{Short\ Distance\ Metrics}$",
            f"$R^2$: {r2_short:.3f}",
            f"MSE: {mse_short:.6f}",
        ]
    )
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))
    fig.savefig(ENGLISH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def text_svg(x, y, value, size=13, weight="400", anchor="middle", color="#1f2933"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(str(value))}</text>'


def text_lines(x, y, lines, size=15, weight="400", color="#1f2933", anchor="middle", line_height=20):
    return "\n".join(text_svg(x, y + i * line_height, line, size=size, weight=weight, color=color, anchor=anchor) for i, line in enumerate(lines))


def rect_svg(x, y, w, h, fill, stroke="#22303c", sw=1.2, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def save_network_structure_svg():
    def box(x, y, w, h, title, lines, fill):
        return f'{rect_svg(x, y, w, h, fill, sw=2, rx=10)}\n{text_lines(x + w / 2, y + 32, [title], size=17, weight="700")}\n{text_lines(x + w / 2, y + 68, lines, size=13, line_height=18)}'

    def arrow(x1, y1, x2, y2, label=None, curve=0):
        path = f"M {x1} {y1} L {x2} {y2}" if not curve else f"M {x1} {y1} Q {(x1 + x2) / 2} {y1 + curve} {x2} {y2}"
        label_svg = ""
        if label:
            lx = (x1 + x2) / 2
            ly = (y1 + y2) / 2 - 12
            label_svg = f'{rect_svg(lx - 92, ly - 18, 184, 26, "#ffffff", stroke="none", sw=0, rx=7)}\n{text_lines(lx, ly, [label], size=12)}'
        return f'<path d="{path}" fill="none" stroke="#22303c" stroke-width="2.2" marker-end="url(#arrowhead)"/>\n{label_svg}'

    width = 1500
    height = 820
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrowhead" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto"><polygon points="0 0, 12 4, 0 8" fill="#22303c"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="#fbfcfd"/>',
        text_lines(width / 2, 55, ["Network Structure: Autoregressive LSTM Conditioned on Parameters"], size=24, weight="700"),
        box(70, 145, 210, 130, "Input X", ["4 parameters", "beta_net, beta_hh", "delta, fermi_mu"], "#d9ecff"),
        box(360, 125, 255, 170, "Parameter encoder", ["Linear 4 -> 64 + ReLU", "Linear 64 -> 128 + ReLU", "Linear 128 -> 512"], "#dcf5df"),
        box(695, 125, 245, 170, "Initial state", ["512 values", "reshape to h0 and c0", "2 layers, hidden=128"], "#fff1c8"),
        box(1015, 125, 235, 170, "LSTM", ["input_size=1", "hidden_size=128", "num_layers=2"], "#e9ddff"),
        box(1310, 145, 135, 130, "Decoder", ["Linear 128 -> 1", "Sigmoid"], "#ffe0d1"),
        box(70, 505, 210, 120, "I(0)", ["initial zero", "batch x 1"], "#f0f2f4"),
        box(360, 485, 255, 160, "Time step t", ["input I(t-1)", "update h_t, c_t", "produce hidden state"], "#e9ddff"),
        box(695, 485, 245, 160, "Prediction", ["I(t) in [0, 1]", "reused as the", "next input"], "#ffe0d1"),
        box(1015, 485, 235, 160, "Sequence", [f"repeat t = 1..{TMAX}", "concatenate", "predictions"], "#e7eaee"),
        box(1310, 505, 135, 120, "Output", ["curve I(t)", f"batch x {TMAX}"], "#e7eaee"),
        arrow(280, 210, 360, 210),
        arrow(615, 210, 695, 210, "h0 + c0"),
        arrow(940, 210, 1015, 210),
        arrow(1250, 210, 1310, 210),
        arrow(175, 275, 175, 505, "start"),
        arrow(280, 565, 360, 565),
        arrow(615, 565, 695, 565),
        arrow(940, 565, 1015, 565),
        arrow(1250, 565, 1310, 565),
        arrow(815, 485, 485, 485, "I(t) -> I(t+1)", curve=-90),
        arrow(1135, 295, 490, 485, "h_t, c_t persist", curve=95),
        text_lines(width / 2, 755, ["Model type: sequential autoregressive surrogate conditioned on parameters.", "Not a VAE: no mu/logvar, reparameterization, or KL term."], size=15),
        "</svg>\n",
    ]
    (ENGLISH_DIR / "estructura_red_lstm_surrogate.svg").write_text("\n".join(content), encoding="utf-8")


def load_state_dict():
    return torch.load(MODEL_FILE, map_location="cpu")


def to_numpy(state_dict, key):
    return state_dict[key].detach().cpu().numpy()


def linear_scores(weight, bias):
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
    return np.array([chunk.mean() for chunk in np.array_split(values, size)])


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


def save_trained_schematic_png(state_dict):
    input_scores = to_numpy(state_dict, "param_encoder.0.weight").mean(axis=0)
    layer_values = [
        input_scores,
        aggregate(linear_scores(to_numpy(state_dict, "param_encoder.0.weight"), to_numpy(state_dict, "param_encoder.0.bias")), 12),
        aggregate(linear_scores(to_numpy(state_dict, "param_encoder.2.weight"), to_numpy(state_dict, "param_encoder.2.bias")), 18),
        aggregate(linear_scores(to_numpy(state_dict, "param_encoder.4.weight"), to_numpy(state_dict, "param_encoder.4.bias")), 20),
        aggregate(lstm_scores(state_dict, 0), 20),
        aggregate(lstm_scores(state_dict, 1), 20),
        linear_scores(to_numpy(state_dict, "decoder.weight"), to_numpy(state_dict, "decoder.bias")),
    ]
    all_values = np.concatenate([np.ravel(v) for v in layer_values])
    vmax = float(np.percentile(np.abs(all_values), 98)) or 1.0
    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(17, 9.5), facecolor="white")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    layer_sizes = [len(values) for values in layer_values]
    v_spacing = (0.82 - 0.18) / float(max(layer_sizes))
    h_spacing = (0.90 - 0.08) / float(len(layer_values) - 1)
    for n, (size_a, size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        top_a = v_spacing * (size_a - 1) / 2.0 + 0.5
        top_b = v_spacing * (size_b - 1) / 2.0 + 0.5
        for m in range(size_a):
            for o in range(size_b):
                ax.add_artist(plt.Line2D([n * h_spacing + 0.08, (n + 1) * h_spacing + 0.08], [top_a - m * v_spacing, top_b - o * v_spacing], c="#6b7280", alpha=0.12, lw=0.45, zorder=1))
    for n, values in enumerate(layer_values):
        top = v_spacing * (len(values) - 1) / 2.0 + 0.5
        for m, value in enumerate(values):
            ax.add_artist(plt.Circle((n * h_spacing + 0.08, top - m * v_spacing), v_spacing / 3.1, color=cmap(norm(value)), ec="#111827", lw=1.1, zorder=4))
    layer_names = ["Input\nParams\nactual: 4", "Encoder H1\nDense\nactual: 64", "Encoder H2\nDense\nactual: 128", "h0 / c0\nLSTM init\nactual: 512", "LSTM layer 0\n4 gates x 128\nactual: 512", "LSTM layer 1\n4 gates x 128\nactual: 512", "Decoder\nI(t)\nactual: 1"]
    for n, name in enumerate(layer_names):
        ax.text(n * h_spacing + 0.08, 0.90, name, ha="center", va="center", fontsize=11, fontweight="bold", color="#1f2933")
    ax.text(0.5, 1.03, "Trained Neural Network: Nodes Colored by Learned Weights", ha="center", va="center", fontsize=21, fontweight="bold", color="#111827", transform=ax.transAxes)
    ax.text(0.5, 0.985, "Schematic representation of the autoregressive LSTM conditioned on parameters", ha="center", va="center", fontsize=13, color="#4b5563", transform=ax.transAxes)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("bias + mean incoming weight", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    ax.text(0.10, 0.075, "Blue: negative value", color=cmap(norm(-vmax)), fontsize=12, fontweight="bold")
    ax.text(0.32, 0.075, "White: near zero", color="#4b5563", fontsize=12, fontweight="bold")
    ax.text(0.56, 0.075, "Red: positive value", color=cmap(norm(vmax)), fontsize=12, fontweight="bold")
    ax.text(0.10, 0.035, "Note: large LSTM and h0/c0 layers are aggregated to keep the figure readable.", color="#4b5563", fontsize=10)
    fig.tight_layout()
    fig.savefig(ENGLISH_DIR / "arquitectura_red_entrenada_colormap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def node_grid(x, y, title, values, scale, cols=16, cell=12, gap=3):
    values = np.asarray(values).ravel()
    rows = int(np.ceil(len(values) / cols))
    width = cols * cell + (cols - 1) * gap
    height = rows * cell + (rows - 1) * gap
    svg = [text_svg(x + width / 2, y - 18, title, size=15, weight="700"), rect_svg(x - 12, y - 8, width + 24, height + 18, "#ffffff", stroke="#d1d7de", sw=1.0, rx=10)]
    for idx, value in enumerate(values):
        row = idx // cols
        col = idx % cols
        svg.append(rect_svg(x + col * (cell + gap), y + row * (cell + gap), cell, cell, color_for(value, scale), stroke="#ffffff", sw=0.45, rx=cell / 2))
    svg.append(text_svg(x + width / 2, y + height + 25, f"{len(values)} nodes", size=11, color="#52606d"))
    return "\n".join(svg)


def save_trained_nodes_svg(state_dict):
    enc64 = linear_scores(to_numpy(state_dict, "param_encoder.0.weight"), to_numpy(state_dict, "param_encoder.0.bias"))
    enc128 = linear_scores(to_numpy(state_dict, "param_encoder.2.weight"), to_numpy(state_dict, "param_encoder.2.bias"))
    state512 = linear_scores(to_numpy(state_dict, "param_encoder.4.weight"), to_numpy(state_dict, "param_encoder.4.bias"))
    lstm0 = lstm_scores(state_dict, 0)
    lstm1 = lstm_scores(state_dict, 1)
    decoder = linear_scores(to_numpy(state_dict, "decoder.weight"), to_numpy(state_dict, "decoder.bias"))
    scale = float(np.percentile(np.abs(np.concatenate([enc64, enc128, state512, lstm0, lstm1, decoder])), 98)) or 1.0
    width = 1800
    height = 1180
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrowhead" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto"><polygon points="0 0, 12 4, 0 8" fill="#22303c"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="#fbfcfd"/>',
        text_svg(width / 2, 48, "Trained Neural Network Nodes", size=26, weight="700"),
        text_svg(width / 2, 78, "Each node is colored by bias + mean incoming weight; LSTM nodes are grouped by gate.", size=14),
        text_svg(230, 89, "Color Map of Learned Value", size=14, weight="700"),
    ]
    for i in range(60):
        value = -scale + 2 * scale * i / 59
        svg.append(rect_svg(80 + i * 5, 105, 5, 22, color_for(value, scale), stroke="none", sw=0, rx=0))
    svg.extend([
        rect_svg(80, 105, 300, 22, "none", sw=1, rx=0),
        text_svg(80, 147, f"-{scale:.3g}", size=11, anchor="start"),
        text_svg(230, 147, "0", size=11),
        text_svg(380, 147, f"+{scale:.3g}", size=11, anchor="end"),
        text_svg(230, 173, "blue = negative, white = near zero, red = positive", size=11),
        text_svg(140, 267, "Input", size=15, weight="700"),
    ])
    for i, name in enumerate(["beta_net", "beta_hh", "delta", "fermi_mu"]):
        svg.append(rect_svg(80, 285 + i * 42, 120, 28, "#d9ecff", sw=1.1, rx=14))
        svg.append(text_svg(140, 304 + i * 42, name, size=11))
    svg.extend([
        node_grid(250, 250, "Encoder Linear 4 -> 64", enc64, scale, cols=8, cell=15, gap=5),
        node_grid(470, 230, "Encoder Linear 64 -> 128", enc128, scale, cols=16, cell=12, gap=4),
        node_grid(790, 200, "Encoder h0/c0 states (512)", state512, scale, cols=32, cell=9, gap=3),
    ])
    for gate_idx, gate_name in enumerate(["input gate", "forget gate", "cell gate", "output gate"]):
        svg.append(node_grid(80 + gate_idx * 265, 640, f"LSTM layer 0 - {gate_name}", lstm0[gate_idx * 128:(gate_idx + 1) * 128], scale, cols=16, cell=10, gap=3))
        svg.append(node_grid(80 + gate_idx * 265, 865, f"LSTM layer 1 - {gate_name}", lstm1[gate_idx * 128:(gate_idx + 1) * 128], scale, cols=16, cell=10, gap=3))
    svg.extend([
        node_grid(1325, 520, "Decoder Linear 128 -> 1", decoder, scale, cols=1, cell=28, gap=4),
        rect_svg(1470, 515, 210, 92, "#ffe0d1", sw=1.5, rx=12),
        text_svg(1575, 550, "Sigmoid + I(t)", size=16, weight="700"),
        text_svg(1575, 578, "output in [0, 1]", size=13),
        rect_svg(1225, 685, 500, 245, "#ffffff", stroke="#d1d7de", sw=1.2, rx=14),
        text_svg(1475, 725, "Summary of Visualized Parameters", size=17, weight="700"),
        text_svg(1255, 765, "Encoder: 64 + 128 + 512 nodes", size=13, anchor="start"),
        text_svg(1255, 792, "LSTM layer 0: 4 gates x 128 nodes", size=13, anchor="start"),
        text_svg(1255, 819, "LSTM layer 1: 4 gates x 128 nodes", size=13, anchor="start"),
        text_svg(1255, 846, "Decoder: 1 node before sigmoid", size=13, anchor="start"),
        text_svg(1255, 883, "Note: color is not data activation.", size=13, anchor="start"),
        text_svg(1255, 910, "It is a node-level signature of trained weights.", size=13, anchor="start"),
        "</svg>\n",
    ])
    (ENGLISH_DIR / "nodos_red_entrenada_colormap.svg").write_text("\n".join(svg), encoding="utf-8")


def main():
    ENGLISH_DIR.mkdir(parents=True, exist_ok=True)
    df = load_russia_data()
    model, x_scaler, y_scaler = load_surrogate_model_and_scalers()
    state_dict = load_state_dict()

    save_russia_data_plot(df)
    save_surrogate_fit_plot(df, model, x_scaler, y_scaler)
    save_original_sbm_fit_plot(df)
    save_sir_fit_plot(df)
    save_validation_plot(model, x_scaler, y_scaler, "validacion_surrogate_comparativa_normalizada.png")
    save_validation_plot(model, x_scaler, y_scaler, "validacion_surrogate_comparativa.png")
    save_network_structure_svg()
    save_trained_schematic_png(state_dict)
    save_trained_nodes_svg(state_dict)
    print(f"English figures written to: {ENGLISH_DIR}")


if __name__ == "__main__":
    main()
