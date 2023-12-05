from typing import Tuple

import cvxpy as cp
import networkx as nx
import numpy as np


def define_problem(G: nx.DiGraph, mode: str, params: dict) -> cp.Problem:
    """Define the problem from the network, but don't solve."""
    G = set_optim_vars(G, mode=mode)

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
    return prob


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

def concretize_network_attrs(G: nx.DiGraph) -> nx.DiGraph:
    """Transforms cvxpy expressions into concrete values."""
    for i, a in G.nodes(data=True):
        concretize_dict(a)

    for i, j, a in G.edges(data=True):
        concretize_dict(a)

    return G

def concretize_dict(a: dict) -> None:
    """Concretize cvxpy expressions in a dictionary inplace."""
    for k, v in a.items():
        if isinstance(v, cp.Expression):
            val = v.value
            if isinstance(val, np.ndarray) and val.ndim == 0:
                if np.issubdtype(val.dtype, np.floating):
                    val = float(val)
                elif np.issubdtype(val.dtype, np.integer):
                    val = int(val)
            a[k] = val