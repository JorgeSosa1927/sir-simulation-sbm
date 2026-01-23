import os
import sys
import networkx as nx
import matplotlib.pyplot as plt

from simple_sbm_generator import (
    GeneradorSBM,
    construir_red_manzanas_con_proyeccion_hubs
)
from test_simulation import (
    MODEL_CONFIG_TEMPLATE,
    VISUALIZATION_DRAW_SETTINGS,
    VISUALIZATION_COLOR_PALETTE,
    VISUALIZATION_BLOCKS_FOR_PANEL_TWO,
    create_model_config,
)

def ejecutar_visualizacion():
    print("Initializing Simple SBM Visualization with new configuration...")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    cfg = create_model_config(MODEL_CONFIG_TEMPLATE)
    
    # 2. Generate Original SBM Graph
    generador = GeneradorSBM(cfg)
    G = generador.generar_original()
    
    print(f"Original Graph Generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 3. Configure figure for side-by-side comparison
    subplot_shape = VISUALIZATION_DRAW_SETTINGS["subplot_shape"]
    fig, axes = plt.subplots(
        subplot_shape[0],
        subplot_shape[1],
        figsize=VISUALIZATION_DRAW_SETTINGS["figsize"]
    )
    
    # --- PLOT 1: Original Network (Left) ---
    axis_indices = VISUALIZATION_DRAW_SETTINGS["axis_indices"]
    ax1 = axes[axis_indices["original"]]
    
    # Retrieve positions saved in the graph (robust version)
    pos = {n: data.get('pos') for n, data in G.nodes(data=True) if data.get('pos') is not None}
    
    colores = VISUALIZATION_COLOR_PALETTE
    etiquetas_bloque = cfg.etiquetas_bloque
    
    # Draw by block on ax1
    for block_id in sorted(etiquetas_bloque.keys()):
        nodos = [n for n, d in G.nodes(data=True) if d.get('block') == block_id]
        label = etiquetas_bloque[block_id]
        if nodos:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodos,
                node_color=colores[block_id],
                label=label,
                node_size=VISUALIZATION_DRAW_SETTINGS["node_size"],
                ax=ax1,
            )

    ax1.set_title(
        "Original SBM Network (with Hubs)",
        fontsize=VISUALIZATION_DRAW_SETTINGS["title_fontsize"],
    )
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=VISUALIZATION_DRAW_SETTINGS["panel_one_edge_alpha"],
        ax=ax1,
    )
    ax1.legend()
    ax1.axis('off')

    # --- PLOT 2: Projected Network (Right) ---
    print("Generating hub projection...")
    H = construir_red_manzanas_con_proyeccion_hubs(G, cfg)
    print(f"Projected Graph Generated: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    
    # --- SMOKE TEST (Weight Verification) ---
    print("\n--- Weight Verification (Smoke Test) ---")
    missing_weights = [e for u, v, k, e in H.edges(keys=True, data=True) if cfg.weight_attr not in e]
    if missing_weights:
        print(f"ERROR: Found {len(missing_weights)} edges without weight.")
    else:
        print("SUCCESS: All edges have weight attribute assigned.")
    
    # Statistics by origin
    origins = [d.get(cfg.edge_origin_attr) for u, v, k, d in H.edges(keys=True, data=True)]
    from collections import Counter
    counts = Counter(origins)
    print(f"Count by origin: {dict(counts)}")
    
    # Sample edges
    sample_count = VISUALIZATION_DRAW_SETTINGS["sample_edge_count"]
    print(f"Sample of {sample_count} edges with weights:")
    sample = list(H.edges(keys=True, data=True))[:sample_count]
    peso_precision = VISUALIZATION_DRAW_SETTINGS["weight_precision"]
    for u, v, k, d in sample:
        orig = d.get(cfg.edge_origin_attr)
        w = d.get(cfg.weight_attr)
        print(f"  {u}->{v} ({orig}): {w:.{peso_precision}f}")
    print("------------------------------------------\n")

    ax2 = axes[axis_indices["projected"]]
    
    # Retrieve positions for projected subgraph (Robustness)
    pos_h = {n: data.get('pos') for n, data in H.nodes(data=True) if data.get('pos') is not None}
    
    # Draw Blocks (Block 0 and 2) on ax2
    for block_id in VISUALIZATION_BLOCKS_FOR_PANEL_TWO:
        nodos = [n for n, d in H.nodes(data=True) if d.get('block') == block_id]
        if nodos:
            label = etiquetas_bloque[block_id]
            nx.draw_networkx_nodes(
                H,
                pos_h,
                nodelist=nodos,
                node_color=colores[block_id],
                label=label,
                node_size=VISUALIZATION_DRAW_SETTINGS["node_size"],
                ax=ax2,
            )
            
    # Original edges (soft gray) - with Keys=True
    edges_orig = [(u, v, k) for u, v, k, d in H.edges(keys=True, data=True)
              if d.get(cfg.edge_origin_attr) == 'original']
              
    if edges_orig:
        nx.draw_networkx_edges(
            H,
            pos_h,
            edgelist=edges_orig,
            edge_color=VISUALIZATION_DRAW_SETTINGS["panel_two_original_edge_color"],
            alpha=VISUALIZATION_DRAW_SETTINGS["panel_two_original_edge_alpha"],
            label=VISUALIZATION_DRAW_SETTINGS["legend_labels"]["original"],
            ax=ax2,
        )
    
    # Projected edges (red) - with Keys=True
    # Note: filtering with keys allows distinguishing edges in MultiGraph if separate processing was needed,
    # though for simple visual superposition it's similar.
    edges_proj = [(u, v, k) for u, v, k, d in H.edges(keys=True, data=True)
              if d.get(cfg.edge_origin_attr) == 'projected']
              
    if edges_proj:
        nx.draw_networkx_edges(
            H,
            pos_h,
            edgelist=edges_proj,
            edge_color=VISUALIZATION_DRAW_SETTINGS["projected_edge_color"],
            width=VISUALIZATION_DRAW_SETTINGS["projected_edge_width"],
            alpha=VISUALIZATION_DRAW_SETTINGS["projected_edge_alpha"],
            label=VISUALIZATION_DRAW_SETTINGS["legend_labels"]["projected"],
            ax=ax2,
        )

    ax2.set_title(
        "Projected Block Network (Projected Hubs)",
        fontsize=VISUALIZATION_DRAW_SETTINGS["title_fontsize"],
    )
    # Manual legend creation for edges in 2nd plot to avoid clutter
    from matplotlib.lines import Line2D
    legend_coords = VISUALIZATION_DRAW_SETTINGS["legend_line_coords"]
    legend_elements = [
        Line2D(
            legend_coords[0],
            legend_coords[1],
            color=VISUALIZATION_DRAW_SETTINGS["panel_two_original_edge_color"],
            lw=VISUALIZATION_DRAW_SETTINGS["legend_edge_width"],
            alpha=VISUALIZATION_DRAW_SETTINGS["legend_edge_alpha"],
            label=VISUALIZATION_DRAW_SETTINGS["legend_labels"]["original"],
        ),
        Line2D(
            legend_coords[0],
            legend_coords[1],
            color=VISUALIZATION_DRAW_SETTINGS["projected_edge_color"],
            lw=VISUALIZATION_DRAW_SETTINGS["legend_edge_width"],
            alpha=VISUALIZATION_DRAW_SETTINGS["legend_edge_alpha"],
            label=VISUALIZATION_DRAW_SETTINGS["legend_labels"]["projected"],
        ),
    ]
    # Add nodes to legend too if desired, or trust consistent colors
    ax2.legend(
        handles=legend_elements,
        loc=VISUALIZATION_DRAW_SETTINGS["legend_location"],
    )
    ax2.axis('off')
    
    plt.tight_layout()
    ruta_salida = VISUALIZATION_DRAW_SETTINGS["output_path"]
    plt.savefig(
        ruta_salida,
        dpi=VISUALIZATION_DRAW_SETTINGS["dpi"],
        bbox_inches=VISUALIZATION_DRAW_SETTINGS["bbox_inches"],
    )
    plt.close()
    print(f"Comparative visualization saved to {ruta_salida}")

if __name__ == "__main__":
    ejecutar_visualizacion()
