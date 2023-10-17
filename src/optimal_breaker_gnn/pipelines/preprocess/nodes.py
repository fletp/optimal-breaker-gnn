"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""
import pandas as pd
import networkx as nx
import numpy as np

def build_base_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    """Build the base network from the universal node and edge specifications."""
    G = nx.from_pandas_edgelist(
        edges,
        source="busbar_u",
        target="busbar_v",
        edge_attr=True
    )
    nodes = nodes.set_index("busbar_id")
    nx.set_node_attributes(
        G=G,
        values=nodes.to_dict(orient="index")
    )
    return G


def create_network_scenario(G: nx.Graph, params: dict) -> nx.Graph:
    """Create a scenario by adding load, generation, capacities, and reactances to the network."""
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
        seed=params["random_seed"],
    )
    G = create_capacities(
        G,
        min=params["capacity"]["min_norm"],
        max=params["capacity"]["max_norm"],
        capacity_multiplier=capacity_multiplier,
        interconnect_multiplier=params["capacity"]["interconnect_multiplier"],
        seed=params["random_seed"],
    )
    G = create_reactances(
        G,
        min=params["reactance"]["min_norm"],
        max=params["reactance"]["max_norm"],
        interconnect_multiplier=params["reactance"]["interconnect_multiplier"],
        seed=params["random_seed"],
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
    gens = {i+1:{"generation":gens[i]} for i in np.arange(gens.shape[0])}
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