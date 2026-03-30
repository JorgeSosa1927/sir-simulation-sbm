import os
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from simple_sbm_generator import (
    GeneradorSBM,
    ModeloConfig,
    construir_red_manzanas_con_proyeccion_hubs,
)

# -----------------------------
#  Centralized Numerical Configuration
# -----------------------------
ETIQUETAS_BLOQUE = {
    0: "social blocks",
    1: "hub",
    2: "non-social blocks",
}

HUB_SUBTYPE_PROBS = {
    "office": 0.4,
    "transport": 0.1,
    "school": 0.4,
    "shop": 0.1,
}

PERSONAS_REGLAS = {
    0: (200, 280, 0.7),
    2: (200, 280, 0.4),
    
}

INTERACTION_WEIGHTS = {
    # Original edges (neighborhood)
    "social-social": 0.10,
    "social-nosocial": 0.06,
    "nosocial-nosocial": 0.03,

    # Projected edges (hubs) -> MUCH lower
    "office": 0.03,
    "transport": 0.02,
    "school": 0.99,
    "shop": 0.01,
}


# No noise (fixed factor). Note: the generator accepts low == high.
WEIGHT_NOISE_RANGE = [1.0, 1.0]
WEIGHT_NOISE_OFFSET = 999



def _build_model_config_data(tamanos_bloques, matriz_mezcla, semilla):
    data = {
        "tamanos_bloques": tamanos_bloques,
        "matriz_mezcla": matriz_mezcla,
        "semilla": semilla,
        "etiquetas_bloque": deepcopy(ETIQUETAS_BLOQUE),
        "hub_subtype_probs": deepcopy(HUB_SUBTYPE_PROBS),
        "personas_reglas": deepcopy(PERSONAS_REGLAS),
        "interaction_weights": deepcopy(INTERACTION_WEIGHTS),
        "weight_noise_range": list(WEIGHT_NOISE_RANGE),
        "weight_noise_offset": WEIGHT_NOISE_OFFSET,
    }
    return data




tamanos_bloques = [435, 75, 490]
n_net = sum(tamanos_bloques)

# Base intensities (approx. average degree constants)
# These values / n_net = connection probability
base_matrix = [
    [80.0, 4.0, 5.0],
    [4.0, 0.0, 1.0],
    [5.0, 1.0, 1.0],
]

MODEL_CONFIG_TEMPLATE = _build_model_config_data(
    tamanos_bloques=tamanos_bloques,
    matriz_mezcla=[[(v / n_net) for v in row] for row in base_matrix],
    semilla=123,
)

factor = 1.09
SIMULATION_PARAMS = {
    "beta_network": 0.41*factor,
    "beta_household": 2.1*factor,
    "delta": 0.88*factor,
    "init_inf_frac": 0.0003,
    "min_initial_infected": 20,
    "seed_mobile_bias": 0.7,
    "tmax": 100,
    "seed": 123,
}

SIMULATION_PLOT_SETTINGS = {
    "figsize": (10, 6),
    "series_styles": {
        "S": {"label": "Susceptible", "color": "blue", "linestyle": "--", "linewidth": 1},
        "I": {"label": "Infected", "color": "red", "linestyle": "-", "linewidth": 2},
        "R": {"label": "Recovered", "color": "green", "linestyle": "-", "linewidth": 1},
    },
    "title": "SIR Dynamics in Simple SBM Network",
    "xlabel": "Time (steps)",
    "ylabel": "Number of Individuals",
    "grid_alpha": 0.3,
    "dpi": 300,
    "output_path": "output/test_simulation_plot.png",
}

VISUALIZATION_COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c']
VISUALIZATION_BLOCKS_FOR_PANEL_TWO = [0, 2]

VISUALIZATION_DRAW_SETTINGS = {
    "subplot_shape": (1, 2),
    "figsize": (20, 10),
    "axis_indices": {"original": 0, "projected": 1},
    "node_size": 100,
    "panel_one_edge_alpha": 0.3,
    "panel_two_original_edge_alpha": 0.2,
    "panel_two_original_edge_color": "gray",
    "projected_edge_color": "red",
    "projected_edge_width": 2.5,
    "projected_edge_alpha": 0.7,
    "legend_edge_width": 2,
    "legend_edge_alpha": 1.0,
    "legend_location": "upper right",
    "legend_line_coords": ([0], [0]),
    "legend_labels": {
        "original": "Neighborhood",
        "projected": "Work (Projected)",
    },
    "title_fontsize": 16,
    "output_path": "output/simple_sbm_comparison.png",
    "dpi": 300,
    "bbox_inches": "tight",
    "sample_edge_count": 5,
    "weight_precision": 4,
}


def create_model_config(template=MODEL_CONFIG_TEMPLATE):
    # Important: deepcopy prevents global modifications if modified externally
    return ModeloConfig(**deepcopy(template))



def _plot_comparative_experiment(t,
                                results_small,
                                results_large,
                                output_path="output/infectados_mu_small_vs_mu_infty.png"):
    
    plt.figure(figsize=(10, 6))
    
    # 1) MU SMALL (Restrictive) - Green
    I_small = results_small["I"]
    mean_I_small = np.mean(I_small, axis=0)
    for i in range(len(I_small)):
        plt.plot(t, I_small[i], color="green", alpha=0.1, linewidth=0.7)
    plt.plot(t, mean_I_small, color="green", label="Small Mu (Restrictive)", linewidth=3)

    # 2) MU LARGE (Free) - Blue
    I_large = results_large["I"]
    mean_I_large = np.mean(I_large, axis=0)
    for i in range(len(I_large)):
        plt.plot(t, I_large[i], color="blue", alpha=0.1, linewidth=0.7)
    plt.plot(t, mean_I_large, color="blue", label="Large Mu (Free)", linewidth=3)

    plt.title("Infection Dynamics: Distance Restrictiveness Comparison", fontsize=14)
    plt.xlabel("Time (steps)", fontsize=12)
    plt.ylabel("Infected Individuals", fontsize=12)
    plt.xlim(0, 76)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Clean aesthetics
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    print(f"Saving comparative plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_experiment_scenario(scenario_name, mu_value, beta_value, num_sims=50):
    print(f"\n--- Running Scenario: {scenario_name} (mu={mu_value}, beta={beta_value}) ---")
    
    # 1. Configure model with Fermi parameters
    # Note: Using template but overwriting Fermi values
    cfg_data = deepcopy(MODEL_CONFIG_TEMPLATE)
    cfg_data["fermi_mu"] = mu_value
    cfg_data["fermi_beta"] = beta_value
    
    cfg = ModeloConfig(**cfg_data)
    generador = GeneradorSBM(cfg)

    # 2. Generate original network (this fixes positions and blocks)
    G0 = generador.generar_original()
    
    # 3. Project (HERE Fermi modulation is applied to original edges)
    # Note: Positions are already in G0, so distance is deterministic given the seed
    H_multi = construir_red_manzanas_con_proyeccion_hubs(G0, cfg)

    # --- Network Statistics (calculated only once per scenario) ---
    # Count node types
    node_types = {}
    total_pop = 0
    
    # H_multi has the topology entering simulation
    # Note: G0 has original nodes
    
    for n, d in G0.nodes(data=True):
        ntype = d.get(cfg.node_type_attr, "unknown")
        node_types[ntype] = node_types.get(ntype, 0) + 1
        
        # Population (sum mobile + static)
        pm = d.get(cfg.personas_moviles_attr, 0)
        pe = d.get(cfg.personas_estaticas_attr, 0)
        total_pop += (pm + pe)

    # Count Edges by origin
    n_edges_orig = 0
    n_edges_proj = 0
    if H_multi is not None:
        for u, v, k, d in H_multi.edges(keys=True, data=True):
            orig = d.get(cfg.edge_origin_attr)
            if orig == "original":
                n_edges_orig += 1
            elif orig == "projected":
                n_edges_proj += 1

    # Calculate max distance (geometry)
    max_dist = 0.0
    nodes_with_pos = [d['pos'] for n, d in G0.nodes(data=True) if 'pos' in d]
    if len(nodes_with_pos) > 1:
        # Calculate all pairwise distances (brute force ok for 100 nodes)
        import itertools
        for p1, p2 in itertools.combinations(nodes_with_pos, 2):
            d = np.linalg.norm(np.array(p1) - np.array(p2))
            if d > max_dist:
                max_dist = d

    graph_stats = {
        "n_nodes": G0.number_of_nodes(),
        "node_types": node_types,
        "edges_original": n_edges_orig,
        "edges_projected": n_edges_proj,
        "max_distance": max_dist
    }
    
    pop_stats = {
        "total_pop": total_pop
    }

    base_packet = dict(SIMULATION_PARAMS)
    base_packet["G"] = H_multi
    
    all_I = []
    all_R = []
    
    base_seed = SIMULATION_PARAMS["seed"]
    
    for i in range(num_sims):
        packet = base_packet.copy()
        packet["seed"] = base_seed + i + 1000 # Offset to vary contagion seed
        
        # Simulate
        try:
            out = generador.simulate(packet)
            all_I.append(out.I)
            all_R.append(out.R)
        except Exception as e:
            print(f"Error in simulation {i}: {e}")
            continue

    return {
        "I": np.array(all_I),
        "R": np.array(all_R),
        "params": {"mu": mu_value, "beta": beta_value}
    }, graph_stats, pop_stats


def test_simulation_plot():
    print("Starting Comparative Distance Experiment (Fermi-Dirac)...")
    os.makedirs("output", exist_ok=True)

    # Experiment parameters
    FERMI_BETA = 0.2 # Abruptness
    
    # Case A: Small Mu (distance matters a lot)
    # St. Petersburg short distance threshold (~5km / 38km city)
    MU_SMALL = 5 
    
    # Case B: Large Mu (distance irrelevant, f(d) ~ 1)
    MU_LARGE = 15

    NUM_SIMS = 100  # Adjust as needed for performance




    # Run scenarios
    results_small, graph_stats, pop_stats = run_experiment_scenario("Small Mu", MU_SMALL, FERMI_BETA, NUM_SIMS)
    results_large, _, _ = run_experiment_scenario("Large Mu", MU_LARGE, FERMI_BETA, NUM_SIMS)

    print("\n" + "="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)

    # 1. Network and Population (Common)
    print(f"Total Nodes: {graph_stats['n_nodes']}")
    print(f"  - By type: {graph_stats['node_types']}")
    print(f"Total Population: {pop_stats['total_pop']}")
    print(f"Max Distance between nodes: {graph_stats['max_distance']:.4f} (abstract units)")
    
    print("-" * 30)
    print("Edges (Topology):")
    print(f"  - Original (Neighborhood): {graph_stats['edges_original']}")
    print(f"  - Projected (Hubs): {graph_stats['edges_projected']}")
    print("-" * 30)

    # 2. Infection Results
    def print_simulation_stats(label, results):
        I_matrix = results["I"] # (num_sims, tmax)
        R_matrix = results["R"] # (num_sims, tmax)
        
        peaks = np.max(I_matrix, axis=1) # Peak of each run
        
        # Infected at the last iteration
        last_I = I_matrix[:, -1]
        # Recovered at the last iteration
        last_R = R_matrix[:, -1]
        
        # Accumulated infected (I + R at the end)
        acc_infected = last_I + last_R
        
        print(f"Scenario {label}:")
        print(f"  - Peak Infected (Mean): {np.mean(peaks):.2f} (+/- {np.std(peaks):.2f})")
        print(f"  - Last Iteration Infected (Mean): {np.mean(last_I):.2f} (+/- {np.std(last_I):.2f})")
        print(f"  - Total Accumulated Infected (Mean): {np.mean(acc_infected):.2f} (+/- {np.std(acc_infected):.2f})")
        print(f"  - Range [Min, Max] peaks: [{np.min(peaks):.2f}, {np.max(peaks):.2f}]")

    print_simulation_stats("A (Small Mu - Restrictive)", results_small)
    print_simulation_stats("B (Large Mu - Free)", results_large)
    print("="*50 + "\n")

    # Verify we have data
    if len(results_small["I"]) == 0 or len(results_large["I"]) == 0:
        print("Error: No valid results to plot.")
        return

    # Reference time
    tmax = SIMULATION_PARAMS["tmax"]
    t_ref = np.arange(tmax)

    # Plot
    _plot_comparative_experiment(t_ref, results_small, results_large)
    
    print("\nExperiment completed successfully.")

if __name__ == "__main__":
    test_simulation_plot()
