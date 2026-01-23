import numpy as np
import networkx as nx
from itertools import combinations
from model_output import SIRModelOutput


# -----------------------------
# 1) Configuration
# -----------------------------
class ModeloConfig:
    def __init__(
        self,
        tamanos_bloques,
        matriz_mezcla,
        semilla,
        etiquetas_bloque,
        hub_subtype_probs,
        personas_reglas,
        interaction_weights,
        weight_noise_range,
        weight_noise_offset,
        node_type_attr="tipo_bloque",
        hub_subtype_attr="hub_subtype",
        personas_moviles_attr="personas_moviles",
        personas_estaticas_attr="personas_estaticas",
        base_edge_type="neighborhood",
        projected_edge_type="work_projected",
        edge_type_attr="edge_type",
        edge_origin_attr="edge_origin",
        pair_type_attr="manzana_pair_type",
        weight_attr="weight",
        fermi_beta=10.0,
        fermi_mu=0.5
    ):
        self.tamanos_bloques = tamanos_bloques
        self.matriz_mezcla = matriz_mezcla
        self.semilla = semilla

        self.etiquetas_bloque = etiquetas_bloque
        self.node_type_attr = node_type_attr

        self.hub_subtype_probs = hub_subtype_probs
        self.hub_subtype_attr = hub_subtype_attr

        # personas_reglas: dict block_id -> (total_min, total_max, p_mobile)
        self.personas_reglas = personas_reglas
        self.personas_moviles_attr = personas_moviles_attr
        self.personas_estaticas_attr = personas_estaticas_attr

        self.base_edge_type = base_edge_type
        self.projected_edge_type = projected_edge_type
        self.edge_type_attr = edge_type_attr
        self.edge_origin_attr = edge_origin_attr
        self.pair_type_attr = pair_type_attr

        # Weight configuration
        self.interaction_weights = interaction_weights
        self.weight_noise_range = weight_noise_range
        self.weight_noise_offset = weight_noise_offset
        self.weight_attr = weight_attr
        self.fermi_beta = fermi_beta
        self.fermi_mu = fermi_mu


# -----------------------------
# 2) SBM Generator + Attributes
# -----------------------------
class GeneradorSBM:
    def __init__(self, config):
        self.cfg = config
        self.matriz_mezcla = np.array(config.matriz_mezcla, dtype=float)
        self.rng = np.random.default_rng(config.semilla)

        num_bloques = len(config.tamanos_bloques)
        if self.matriz_mezcla.shape != (num_bloques, num_bloques):
            raise ValueError(
                f"Mixing matrix must be {num_bloques}x{num_bloques}, "
                f"but it is {self.matriz_mezcla.shape}."
            )
        
        # Probability value validation (0 <= p <= 1)
        if np.any(self.matriz_mezcla < 0) or np.any(self.matriz_mezcla > 1):
             raise ValueError("All mixing matrix values must be between 0 and 1.")

        s = sum(config.hub_subtype_probs.values())
        if s <= 0:
            raise ValueError("hub_subtype_probs must sum to > 0.")
            
        # Derive hub block ID from labels
        hub_ids = [k for k, v in config.etiquetas_bloque.items() if v == "hub"]
        if not hub_ids:
             raise ValueError("No 'hub' label found in etiquetas_bloque.")
        self.hub_block_id = hub_ids[0]

    def _asignar_personas_a_bloque(self, G, block_id, total_min, total_max, p_moviles):
        for n, data in G.nodes(data=True):
            if data.get("block") != block_id:
                continue

            total = int(self.rng.integers(total_min, total_max + 1))  # inclusive
            moviles = int(self.rng.binomial(total, p_moviles))
            estaticas = total - moviles

            data[self.cfg.personas_moviles_attr] = moviles
            data[self.cfg.personas_estaticas_attr] = estaticas

    def _asignar_hub_subtype(self, G):
        subtipos = list(self.cfg.hub_subtype_probs.keys())
        probs = np.array(list(self.cfg.hub_subtype_probs.values()), dtype=float)
        probs = probs / probs.sum()

        for n, data in G.nodes(data=True):
            if data.get("block") == self.hub_block_id:
                data[self.cfg.hub_subtype_attr] = self.rng.choice(subtipos, p=probs)

    def generar_original(self):
        G = nx.stochastic_block_model(
            sizes=self.cfg.tamanos_bloques,
            p=self.matriz_mezcla,
            seed=self.cfg.semilla,
            directed=False,
            selfloops=False
        )

        # Readable label per block
        for n, data in G.nodes(data=True):
            b = data.get("block")
            data[self.cfg.node_type_attr] = self.cfg.etiquetas_bloque.get(b, "unknown")

        # Hub attributes
        self._asignar_hub_subtype(G)

        # Person attributes per rules
        for block_id, regla in self.cfg.personas_reglas.items():
            total_min, total_max, p_moviles = regla
            self._asignar_personas_a_bloque(G, block_id, total_min, total_max, p_moviles)

        # Filter: Keep only the Largest Connected Component (LCC)
        if len(G) > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        # Generate and save persistent coordinates for visual consistency
        pos = nx.spring_layout(G, seed=self.cfg.semilla) 
        for n, coords in pos.items():
            G.nodes[n]['pos'] = coords

        return G
        
    def simulate(self, packet):
        """SIR Simulation (toy model) on a *projected network*.

        Key rules (based on project decisions):
        - Input: a *single* ordered object (dict) with the projected MultiGraph and parameters.
        - The MultiGraph is collapsed to a simple Graph combining probabilities via
          W_ij = 1 - Π_k (1 - w_ijk).
        - If the graph is not connected, extract and use the Largest Connected Component (GCC).
        - Population stratified by node: mobile (M) and static (E) are already node attributes.
        - Global initial seed K with minimum and mobile bias.
        - External force (toy): scales also by S_i^M.

        Expected parameters in `packet` (keys):
          G, beta_network, beta_household, delta, init_inf_frac, min_initial_infected,
          seed_mobile_bias, tmax, seed

        Output: SIRModelOutput with aggregated series S/I/R (totals) and extra attributes
        with series by stratum (SM/IM/RM and SE/IE/RE) and GCC metadata.
        """

        if not isinstance(packet, dict):
            raise TypeError("simulate() expects an ordered dict as 'packet'.")

        Gm = packet.get("G")
        if Gm is None:
            raise KeyError("The packet must include key 'G' with the projected MultiGraph.")

        # -----------------------------
        # 1) MultiGraph -> Simple Graph (combine probabilities by pair)
        # -----------------------------
        weight_attr = self.cfg.weight_attr
        Gs = nx.Graph()
        Gs.add_nodes_from(Gm.nodes(data=True))

        # Group multi-edges by undirected pair
        pair_prod = {}  # (u,v) -> Π(1-w)
        if isinstance(Gm, (nx.MultiGraph, nx.MultiDiGraph)):
            edge_iter = Gm.edges(keys=True, data=True)
        else:
            edge_iter = ((u, v, None, d) for u, v, d in Gm.edges(data=True))

        for u, v, _k, d in edge_iter:
            if u == v:
                continue
            if weight_attr not in d:
                raise KeyError(f"Edge ({u},{v}) does not have weight attribute '{weight_attr}'.")
            w = float(d[weight_attr])
            if not (0.0 <= w <= 1.0):
                raise ValueError(
                    f"Weight out of scale [0,1] on edge ({u},{v}): {w}. "
                    "Adjust interaction_weights / noise before simulating."
                )

            # Stable key (independent of orientation)
            uu, vv = sorted((u, v), key=lambda x: str(x))
            key = (uu, vv)
            pair_prod[key] = pair_prod.get(key, 1.0) * (1.0 - w)

        for (u, v), prod in pair_prod.items():
            W = 1.0 - prod
            if not (0.0 <= W <= 1.0 + 1e-12):
                raise ValueError(f"Invalid combined weight for ({u},{v}): {W}")
            W = min(max(W, 0.0), 1.0)
            # save with the same weight name used by the rest of the pipeline
            Gs.add_edge(u, v, **{weight_attr: W})

        if Gs.number_of_nodes() == 0:
            raise ValueError("Input graph has no nodes.")
        if Gs.number_of_edges() == 0:
            raise ValueError("Collapsed simple graph ended up with no edges.")

        # -----------------------------
        # 2) Extract Largest Connected Component (GCC)
        # -----------------------------
        comps = list(nx.connected_components(Gs))
        if not comps:
            raise ValueError("Could not obtain connected components from graph.")
        sizes = [len(c) for c in comps]
        gcc_nodes = comps[int(np.argmax(sizes))]
        G = Gs.subgraph(gcc_nodes).copy()

        meta = {
            "n_components": len(comps),
            "component_sizes": sorted(sizes, reverse=True),
            "gcc_size": len(gcc_nodes),
            "n_total_nodes_before_gcc": Gs.number_of_nodes(),
            "gcc_fraction": float(len(gcc_nodes) / max(1, Gs.number_of_nodes())),
        }

        # -----------------------------
        # 3) Prepare arrays per node (stable order)
        # -----------------------------
        nodes = list(G.nodes())
        n = len(nodes)
        idx = {node: i for i, node in enumerate(nodes)}

        attr_M = self.cfg.personas_moviles_attr
        attr_E = self.cfg.personas_estaticas_attr
        N_M = np.zeros(n, dtype=int)
        N_E = np.zeros(n, dtype=int)

        for node, i in idx.items():
            nd = G.nodes[node]
            if attr_M not in nd or attr_E not in nd:
                raise KeyError(
                    f"Node {node} missing population attributes '{attr_M}'/'{attr_E}'."
                )
            N_M[i] = int(nd[attr_M])
            N_E[i] = int(nd[attr_E])
            if N_M[i] < 0 or N_E[i] < 0:
                raise ValueError(f"Negative populations at node {node}: {N_M[i]}, {N_E[i]}")

        N_tot = int(N_M.sum() + N_E.sum())
        if N_tot <= 0:
            raise ValueError("Total population N_tot must be > 0.")

        # -----------------------------
        # 4) Read packet parameters
        # -----------------------------
        beta_network = float(packet["beta_network"])
        beta_household = float(packet["beta_household"])
        delta = float(packet["delta"])
        init_inf_frac = float(packet["init_inf_frac"])
        min_initial_infected = int(packet.get("min_initial_infected", 1))
        seed_mobile_bias = float(packet.get("seed_mobile_bias", 0.5))
        tmax = int(packet["tmax"])
        seed = int(packet.get("seed", self.cfg.semilla))
        rng = np.random.default_rng(seed)

        if not (0.0 <= seed_mobile_bias <= 1.0):
            raise ValueError("seed_mobile_bias must be in [0,1].")
        if init_inf_frac < 0:
            raise ValueError("init_inf_frac must be >= 0.")
        if min_initial_infected < 0:
            raise ValueError("min_initial_infected must be >= 0.")
        if tmax <= 0:
            raise ValueError("tmax must be > 0.")

        # -----------------------------
        # 5) SIR Initialization per stratum
        # -----------------------------
        S_M = N_M.copy()
        S_E = N_E.copy()
        I_M = np.zeros(n, dtype=int)
        I_E = np.zeros(n, dtype=int)
        R_M = np.zeros(n, dtype=int)
        R_E = np.zeros(n, dtype=int)

        # Global seed K
        K = max(min_initial_infected, int(np.round(init_inf_frac * N_tot)))
        if K > N_tot:
            raise ValueError(f"K={K} exceeds total population N_tot={N_tot}.")

        K_M = int(np.round(seed_mobile_bias * K))
        K_E = int(K - K_M)

        cap_M = int(N_M.sum())
        cap_E = int(N_E.sum())
        if K_M > cap_M:
            K_M = cap_M
            K_E = K - K_M
        if K_E > cap_E:
            K_E = cap_E
            K_M = K - K_E
        if K_M > cap_M or K_E > cap_E:
            raise ValueError("Cannot seed K while respecting M/E capacities.")

        def _seed_individuals(S, I, k):
            for _ in range(int(k)):
                total = int(S.sum())
                if total <= 0:
                    raise ValueError("No susceptibles available to seed initial infection.")
                p = S / total
                i = int(rng.choice(n, p=p))
                S[i] -= 1
                I[i] += 1

        _seed_individuals(S_M, I_M, K_M)
        _seed_individuals(S_E, I_E, K_E)

        # -----------------------------
        # 6) Precompute edge list (indices + weight)
        # -----------------------------
        edges = []
        for u, v, d in G.edges(data=True):
            w = float(d.get(weight_attr, d.get("weight")))
            edges.append((idx[u], idx[v], w))

        # -----------------------------
        # 7) Temporal simulation
        # -----------------------------
        t = np.arange(tmax, dtype=int)
        S_hist = np.zeros(tmax, dtype=float)
        I_hist = np.zeros(tmax, dtype=float)
        R_hist = np.zeros(tmax, dtype=float)

        SM_hist = np.zeros(tmax, dtype=float)
        IM_hist = np.zeros(tmax, dtype=float)
        RM_hist = np.zeros(tmax, dtype=float)
        SE_hist = np.zeros(tmax, dtype=float)
        IE_hist = np.zeros(tmax, dtype=float)
        RE_hist = np.zeros(tmax, dtype=float)

        p_rec = 1.0 - np.exp(-delta)

        for tt in range(tmax):
            # Record
            SM_hist[tt], IM_hist[tt], RM_hist[tt] = S_M.sum(), I_M.sum(), R_M.sum()
            SE_hist[tt], IE_hist[tt], RE_hist[tt] = S_E.sum(), I_E.sum(), R_E.sum()
            S_hist[tt] = SM_hist[tt] + SE_hist[tt]
            I_hist[tt] = IM_hist[tt] + IE_hist[tt]
            R_hist[tt] = RM_hist[tt] + RE_hist[tt]

            # (a) Recovery
            new_RM = rng.binomial(I_M, p_rec)
            new_RE = rng.binomial(I_E, p_rec)
            I_M = I_M - new_RM
            R_M = R_M + new_RM
            I_E = I_E - new_RE
            R_E = R_E + new_RE

            # Save base I_M for external force (avoids immediate feedback)
            I_M_base = I_M.copy()

            # (b) Internal infection (same for M and E)
            I_tot_node = I_M + I_E
            lam_int = beta_household * I_tot_node
            p_int = 1.0 - np.exp(-lam_int)
            new_IM_int = rng.binomial(S_M, p_int)
            new_IE_int = rng.binomial(S_E, p_int)

            S_M = S_M - new_IM_int
            I_M = I_M + new_IM_int
            S_E = S_E - new_IE_int
            I_E = I_E + new_IE_int

            # (c) External infection (toy): beta * S_M * sum_j W_ij I^M_j
            X = np.zeros(n, dtype=float)
            for i, j, w in edges:
                X[i] += w * I_M_base[j]
                X[j] += w * I_M_base[i]

            rate_ext = beta_network * S_M * X
            if np.any(rate_ext < 0) or (not np.all(np.isfinite(rate_ext))):
                raise ValueError("Invalid rate_ext (negative or non-finite).")

            new_IM_ext = rng.poisson(rate_ext)
            new_IM_ext = np.minimum(new_IM_ext, S_M)

            S_M = S_M - new_IM_ext
            I_M = I_M + new_IM_ext

            # (d) Invariants / sanity
            if np.any(S_M < 0) or np.any(S_E < 0) or np.any(I_M < 0) or np.any(I_E < 0):
                raise ValueError("Negative states detected in simulation.")
            if not np.all(S_M + I_M + R_M == N_M):
                raise ValueError("Conservation broken in M stratum.")
            if not np.all(S_E + I_E + R_E == N_E):
                raise ValueError("Conservation broken in E stratum.")

        out = SIRModelOutput(t, S_hist, I_hist, R_hist)
        # Extra attributes (aggregated series per stratum)
        out.SM, out.IM, out.RM = SM_hist, IM_hist, RM_hist
        out.SE, out.IE, out.RE = SE_hist, IE_hist, RE_hist
        out.meta = meta
        out.meta.update({
            "seed": seed,
            "K": K,
            "K_M": int(K_M),
            "K_E": int(K_E),
            "n_nodes_sim": n,
            "n_edges_sim": G.number_of_edges(),
        })
        return out


# -----------------------------
# 3) New network without hubs + hub projection to edges
# -----------------------------
def construir_red_manzanas_con_proyeccion_hubs(G, cfg):
    social_label = cfg.etiquetas_bloque[0]
    nosocial_label = cfg.etiquetas_bloque[2]
    # Dynamically derive Hub Label
    hub_ids = [k for k, v in cfg.etiquetas_bloque.items() if v == "hub"]
    if not hub_ids:
         raise ValueError("No 'hub' label found in etiquetas_bloque.")
    hub_block_id = hub_ids[0]
    hub_label = cfg.etiquetas_bloque[hub_block_id]
    
    # Independent RNG for weight noise (prevents correlation)
    rng_weights = np.random.default_rng(cfg.semilla + cfg.weight_noise_offset)

    def aplicar_ruido(peso_base):
        low, high = cfg.weight_noise_range
        if low > high:
            raise ValueError(f"Invalid noise range: {cfg.weight_noise_range}")
        # Allow low == high as "no noise" (fixed factor)
        if low == high:
            return peso_base * float(low)
        factor = rng_weights.uniform(low, high)
        return peso_base * factor

    def fermi_dirac(dist):
        # f(d) = 1 / (exp(beta * (d - mu)) + 1)
        # alpha controls abruptness, mu is the distance threshold
        val = 1.0 / (np.exp(cfg.fermi_beta * (dist - cfg.fermi_mu)) + 1.0)
        return val

    def pair_type(tipo_u, tipo_v):
        tipos = {tipo_u, tipo_v}
        if tipos == {social_label}:
            return "social-social"
        if tipos == {nosocial_label}:
            return "nosocial-nosocial"
        if tipos == {social_label, nosocial_label}:
            return "social-nosocial"
        return "unknown"

    # 1) keep only 'manzanas' (blocks)
    manzanas = []
    for n, data in G.nodes(data=True):
        if data.get(cfg.node_type_attr) in (social_label, nosocial_label):
            manzanas.append(n)

    # 2) base subgraph (only manzanas) -> keeps original block-block edges
    H_base = G.subgraph(manzanas).copy()

    # 3) Final MultiGraph (to allow multiple projected edges via different hubs)
    H = nx.MultiGraph()
    H.add_nodes_from(H_base.nodes(data=True))

    # 3a) original edges labeled by pair type
    for u, v, edata in H_base.edges(data=True):
        new_edata = dict(edata)
        new_edata.setdefault(cfg.edge_origin_attr, "original")
        new_edata.setdefault(cfg.edge_type_attr, cfg.base_edge_type)

        tipo_u = H_base.nodes[u].get(cfg.node_type_attr)
        tipo_v = H_base.nodes[v].get(cfg.node_type_attr)
        
        p_type = pair_type(tipo_u, tipo_v)
        new_edata[cfg.pair_type_attr] = p_type

        # Assign Weight
        if p_type not in cfg.interaction_weights:
            raise KeyError(f"Weight not defined for pair_type '{p_type}' in interaction_weights")
        peso_base = cfg.interaction_weights[p_type]
        
        # --- MODIFICATION: Fermi-Dirac Spatial Dependence ---
        # 1. Get positions
        pos_u = np.array(H_base.nodes[u].get('pos', [0, 0]))
        pos_v = np.array(H_base.nodes[v].get('pos', [0, 0]))
        # 2. Calculate Euclidean distance
        dist = np.linalg.norm(pos_u - pos_v)
        # 3. Calculate Fermi factor
        f_dist = fermi_dirac(dist)
        
        # Final Weight = BaseWeight * Noise * Fermi
        # Note: Noise is applied multiplicatively as before
        w_noisy = aplicar_ruido(peso_base)
        w_final = w_noisy * f_dist
        
        new_edata[cfg.weight_attr] = w_final
        # Save distance metadata for debug/analysis if desired
        new_edata["distance"] = float(dist)
        new_edata["fermi_factor"] = float(f_dist)

        H.add_edge(u, v, **new_edata)

    # 4) hub projection: if two blocks connected to the same hub, create extra edge
    for h, hdata in G.nodes(data=True):
        if hdata.get(cfg.node_type_attr) != hub_label:
            continue
        
        # Validate subtype existence
        hub_subtype = hdata.get(cfg.hub_subtype_attr)
        if not hub_subtype:
             raise KeyError(f"Hub {h} does not have attribute '{cfg.hub_subtype_attr}'")

        if hub_subtype not in cfg.interaction_weights:
             raise KeyError(f"Weight not defined for hub_subtype '{hub_subtype}' in interaction_weights")
        
        peso_base = cfg.interaction_weights[hub_subtype]

        # explicit and maintainable version:
        vecinos_manzana = []
        for vecino in G.neighbors(h):
            if H.has_node(vecino):  # only manzanas (H does not have hubs)
                vecinos_manzana.append(vecino)

        # Fixed rule: min 2 neighbors to project
        if len(vecinos_manzana) < 2:
            continue

        for u, v in combinations(vecinos_manzana, 2):
            attrs = {
                cfg.edge_origin_attr: "projected",
                cfg.edge_type_attr: cfg.projected_edge_type,
                "hub_id": h,
                cfg.weight_attr: aplicar_ruido(peso_base)
            }

            # pair type also for projected (optional but useful)
            tipo_u = H.nodes[u].get(cfg.node_type_attr)
            tipo_v = H.nodes[v].get(cfg.node_type_attr)
            attrs[cfg.pair_type_attr] = pair_type(tipo_u, tipo_v)

            # inherit selected hub attributes (avoid inflating graph)
            for k in ("hub_subtype", "tipo_bloque", "block"):
                if k in hdata:
                    attrs[f"hub_{k}"] = hdata[k]

            H.add_edge(u, v, **attrs)

    return H
