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
    "transport": 0.3,
    "school": 0.1,
    "shop": 0.2,
}

PERSONAS_REGLAS = {
    0: (1, 150, 0.7),
    2: (1, 150, 0.4),
}

INTERACTION_WEIGHTS = {
    # Original edges (neighborhood)
    "social-social": 0.10,
    "social-nosocial": 0.06,
    "nosocial-nosocial": 0.03,

    # Projected edges (hubs) -> MUCH lower
    "office": 0.03,
    "transport": 0.02,
    "school": 0.05,
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




MODEL_CONFIG_TEMPLATE = _build_model_config_data(
    # (2) Fewer hubs: from 15 -> 7 (can try 6–8)
    tamanos_bloques=[40, 7, 45],

    # (1) Adjusted mixing matrix:
    #   - Keep social neighborhood relatively high
    #   - Keep non-social neighborhood very low
    #   - Reduce strong connections with hubs (to avoid "clique explosion" when projecting)
    matriz_mezcla=[
        #   S       H       N
        [0.10,   0.02,  0.010],  # S -> (S,H,N)
        [0.02,  0.000,  0.004],  # H -> (S,H,N)
        [0.010,  0.004,  0.001],  # N -> (S,H,N)
    ],

    semilla=123,
)


SIMULATION_PARAMS = {
    "beta_network": 0.10,
    "beta_household": 0.05,
    "delta": 0.10,
    "init_inf_frac": 0.01,
    "min_initial_infected": 5,
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
    
    # 1) MU SMALL (Restrictive)
    mean_I_small = np.mean(results_small["I"], axis=0)
    std_I_small = np.std(results_small["I"], axis=0)
    
    plt.plot(t, mean_I_small, color="blue", label=f"Small Mu (Restrictive)", linewidth=2.5)
    plt.fill_between(t, mean_I_small - std_I_small, mean_I_small + std_I_small, color="blue", alpha=0.2)

    # 2) MU LARGE (Free)
    mean_I_large = np.mean(results_large["I"], axis=0)
    std_I_large = np.std(results_large["I"], axis=0)
    
    plt.plot(t, mean_I_large, color="green", label=f"Large Mu (Free)", linewidth=2.5, linestyle="--")
    plt.fill_between(t, mean_I_large - std_I_large, mean_I_large + std_I_large, color="green", alpha=0.2)

    plt.title("Comparison: Effect of Distance on Propagation (Uncertainty Bands)")
    plt.xlabel("Time (steps)")
    plt.ylabel("Infected (Mean +/- Std Dev)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"Saving comparative plot to: {output_path}")
    plt.savefig(output_path, dpi=300)
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

    # 4. Monte Carlo Simulation
    base_packet = dict(SIMULATION_PARAMS)
    base_packet["G"] = H_multi
    
    all_I = []
    
    base_seed = SIMULATION_PARAMS["seed"]
    
    for i in range(num_sims):
        packet = base_packet.copy()
        packet["seed"] = base_seed + i + 1000 # Offset to vary contagion seed
        
        # Simulate
        try:
            out = generador.simulate(packet)
            all_I.append(out.I)
        except Exception as e:
            print(f"Error in simulation {i}: {e}")
            continue

    return {
        "I": np.array(all_I),
        "params": {"mu": mu_value, "beta": beta_value}
    }, graph_stats, pop_stats


def test_simulation_plot():
    print("Starting Comparative Distance Experiment (Fermi-Dirac)...")
    os.makedirs("output", exist_ok=True)

    # Experiment parameters
    FERMI_BETA = 15.0  # Abruptness
    
    # Case A: Small Mu (distance matters a lot)
    # Assuming spring_layout in [-1, 1], typical distances ~0.1 - 1.5
    MU_SMALL = 0.025 
    
    # Case B: Large Mu (distance irrelevant, f(d) ~ 1)
    MU_LARGE = 100.0

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
    def print_peak_stats(label, results):
        I_matrix = results["I"] # (num_sims, tmax)
        peaks = np.max(I_matrix, axis=1) # Peak of each run
        
        mean_peak = np.mean(peaks)
        std_peak = np.std(peaks)
        min_peak = np.min(peaks)
        max_peak = np.max(peaks)
        
        print(f"Scenario {label}:")
        print(f"  - Peak Infected (Mean): {mean_peak:.2f}")
        print(f"  - Uncertainty (Std Dev): +/- {std_peak:.2f}")
        print(f"  - Range [Min, Max] peaks: [{min_peak:.2f}, {max_peak:.2f}]")

    print_peak_stats("A (Small Mu - Restrictive)", results_small)
    print_peak_stats("B (Large Mu - Free)", results_large)
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
