"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""
import pandas as pd
import networkx as nx
import numpy as np
import cvxpy as cp
import time
import platform
from typing import Tuple

def build_base_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    """Build the base network from the universal node and edge specifications."""
    G = nx.from_pandas_edgelist(
        edges,
        source="busbar_u",
        target="busbar_v",
        edge_attr=True,
        create_using=nx.DiGraph,
    )
    nodes = nodes.set_index("busbar_id")
    nx.set_node_attributes(
        G=G,
        values=nodes.to_dict(orient="index")
    )
    return G


def create_network_scenario(G: nx.Graph, params: dict) -> nx.Graph:
    """Create a scenario by adding load, generation, capacities, and reactances to the network."""
    if params["randomize"]:
        G = randomize_network_scenario(G, params)
    return G

def randomize_network_scenario(G: nx.Graph, params: dict) -> nx.Graph:
    """Add random load, generation, capacities, and reactances to the network."""
    tot_power = params["load_gen_factor"] / params["scale_factor"]
    capacity_multiplier = params["flow_factor"] / params["scale_factor"]
    G = create_loads(
        G,
        min=params["load_power"]["min_norm"],
        max=params["load_power"]["max_norm"],
        system_total_power=tot_power,
        seed=params["random_seed"],
    )
    G = create_gens(
        G,
        min=params["gen_power"]["min_norm"],
        max=params["gen_power"]["max_norm"],
        system_total_power=tot_power,
        seed=params["random_seed"]+1,
    )
    G = create_capacities(
        G,
        min=params["capacity"]["min_norm"],
        max=params["capacity"]["max_norm"],
        capacity_multiplier=capacity_multiplier,
        interconnect_multiplier=params["capacity"]["interconnect_multiplier"],
        seed=params["random_seed"]+2,
    )
    G = create_reactances(
        G,
        min=params["reactance"]["min_norm"],
        max=params["reactance"]["max_norm"],
        interconnect_multiplier=params["reactance"]["interconnect_multiplier"],
        seed=params["random_seed"]+3,
    )
    return G


def create_loads(G: nx.Graph, min: float, max: float, system_total_power: float, seed: int=None) -> nx.Graph:
    """Add loads to the network nodes."""
    rng = np.random.default_rng(seed=seed)
    loads = rng.uniform(low=min, high=max, size=len(G.nodes))
    loads = loads / np.sum(loads) * system_total_power
    loads = {i+1:{"load":loads[i]} for i in np.arange(loads.shape[0])}
    nx.set_node_attributes(G, loads)
    return G


def create_gens(
        G: nx.Graph,
        min: float,
        max: float,
        system_total_power: float,
        seed: int=None,
    ) -> nx.Graph:
    """Add generations to the network nodes."""
    rng = np.random.default_rng(seed=seed)
    gens = rng.uniform(low=min, high=max, size=len(G.nodes))
    gens = gens / np.sum(gens) * system_total_power
    gens = {i+1:{"genr":gens[i]} for i in np.arange(gens.shape[0])}
    nx.set_node_attributes(G, gens)
    return G


def create_capacities(
        G: nx.Graph,
        min: float,
        max: float,
        capacity_multiplier: float,
        interconnect_multiplier: float,
        seed: int=None,
    ) -> nx.Graph:
    """Add capacities to the network edges."""
    rng = np.random.default_rng(seed=seed)
    for u, v, a in G.edges(data=True):
        if not a["is_breaker"]:
            cap = rng.uniform(low=min, high=max) * capacity_multiplier
            if a["is_interconnect"]:
                cap = cap * interconnect_multiplier
            a.update({"capacity": cap})
    return G


def create_reactances(
        G: nx.Graph,
        min: float,
        max: float,
        interconnect_multiplier: float,
        seed: int=None,
    ) -> nx.Graph:
    """Add reactances to the network edges."""
    rng = np.random.default_rng(seed=seed)
    for u, v, a in G.edges(data=True):
        if not a["is_breaker"]:
            react = rng.uniform(low=min, high=max)
            if a["is_interconnect"]:
                react = react * interconnect_multiplier
            a.update({"reactance": react})
    return G


def optimize_scenario(G: nx.Graph, params: dict) -> nx.Graph:
    """Apply optimization to network."""
    tic = time.perf_counter()
    # System-wide parameters, variables, and constraints
    G = set_optim_vars(G, mode=params["mode"])

    c = []
    # Node-specific constraints
    buses_z1 = [v for v in G.nodes if G.nodes[v]["zone"] == 1]
    buses_z2 = [v for v in G.nodes if G.nodes[v]["zone"] == 2]
    t_g_1 = sum([G.nodes[v]["genr"] for v in buses_z1])
    t_g_2 = sum([G.nodes[v]["genr"] for v in buses_z2])
    t_l_1 = sum([G.nodes[v]["load"] for v in buses_z1])
    t_l_2 = sum([G.nodes[v]["load"] for v in buses_z2])
    alpha = t_g_1 / t_l_2
    beta = (t_g_2 - t_l_1) / t_l_2
    lam = cp.Variable()
    for v, a in G.nodes(data=True):
        # Conservation of energy
        if a["zone"] == 1:
            genr = lam * a["genr"]
            load = a["load"]
        else:
            genr = a["genr"]
            load = (alpha * lam + beta) * a["load"]

        net_out_flow = 0
        for _, j in G.out_edges(v):
            net_out_flow += G.edges[v, j]["flow"]
        for i, _ in G.in_edges(v):
            net_out_flow -= G.edges[i, v]["flow"]
        c.append(net_out_flow + load - genr == 0)

        # Connected components
        if a["n_breakers"] >= 1:
            n = 0
            for e in get_in_out_edges(G, v):
                if G.edges[e]["is_breaker"] and G.edges[e]["breaker_closed"]:
                    n += 1
            c.append(n >= 1)

    # Edge-specific constraints
    big_M = max([abs(a["load"] - a["genr"]) * params["big_m_scale"] for v, a in G.nodes(data=True)])
    for u, v, a in G.edges(data=True):
        if a["is_breaker"]:
            rhs_a = big_M * a["breaker_closed"]
            c.append(a["flow"] <= rhs_a)
            c.append(-1 * a["flow"] <= rhs_a)
            rhs_b = big_M * (1 - a["breaker_closed"])
            c.append((G.nodes[u]["angle"] - G.nodes[v]["angle"]) <= rhs_b)
            c.append((G.nodes[v]["angle"] - G.nodes[u]["angle"]) <= rhs_b)
        else:
            c.append(cp.abs(a["flow"]) <= a["capacity"])
            
    obj = cp.Maximize(lam)
    prob = cp.Problem(
        objective=obj,
        constraints=c,
    )
    prob.solve(solver=cp.XPRESS, verbose=True)

    # Make cvxpy variables concrete on the network
    for i, a in G.nodes(data=True):
        for k, v in a.items():
            if isinstance(v, cp.Expression):
                val = v.value
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    if np.issubdtype(val.dtype, np.floating):
                        val = float(val)
                    elif np.issubdtype(val.dtype, np.integer):
                        val = int(val)
                a[k] = val

    for i, j, a in G.edges(data=True):
        for k, v in a.items():
            if isinstance(v, cp.Expression):
                val = v.value
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    if np.issubdtype(val.dtype, np.floating):
                        val = float(val)
                    elif np.issubdtype(val.dtype, np.integer):
                        val = int(val)
                a[k] = val
    toc = time.perf_counter()

    res = {
        "network": G,
        "obj_val": lam.value,
        "time_elapsed": toc-tic,
        "platform": platform.platform()
    }
    return res


def get_in_out_edges(G: nx.DiGraph, v) -> list[Tuple]:
    """Get list of in and out edges in graph G from node v."""
    return list(G.in_edges(v)) + list(G.out_edges(v))


def set_optim_vars(G: nx.DiGraph, mode: str) -> nx.DiGraph:
    """Set the optimization variables on the network.
    
    Args:
        G:
            the network on which to set the variables
        mode:
            if "label", then leave the "breaker_closed" edge attribute as a
            variable, but if "eval", then assume that it already exists.
    
    Returns: network with variables added on
    """
    assert mode in ["label", "eval"]
    # Node-specific parameters, variables, and constraints
    for v, a in G.nodes(data=True):
        if v == 1: # Set angle reference point
            a["angle"] = 0.
        else:
            a["angle"] = cp.Variable(name=f"angle_{v}")
        breaker_edges = [G.edges[e]["is_breaker"] for e in get_in_out_edges(G, v)]
        a["n_breakers"] = sum(breaker_edges)

    # Edge-specific parameters, variables, and constraints
    for u, v, a in G.edges(data=True):
        if a["is_breaker"]:
            if mode == "label":
                a["breaker_closed"] = cp.Variable(boolean=True)
            elif mode == "eval" and "breaker_closed" not in a:
                raise Exception("In eval mode, 'breaker_closed' must be pre-defined for all breakers.")
            a["flow"] = cp.Variable()
        else:
            a["flow"] = (G.nodes[u]["angle"] - G.nodes[v]["angle"]) / a["reactance"]
    
    return G
