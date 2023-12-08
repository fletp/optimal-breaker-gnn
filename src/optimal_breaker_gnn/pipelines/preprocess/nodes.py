"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""
from typing import Callable, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from deepsnap.dataset import GraphDataset
from deepsnap.hetero_graph import HeteroGraph
from torch_geometric.data import HeteroData


def join_partitions(
        partitions: dict[str, Callable], 
        params: dict
    ) -> list[dict]:
    """Join dataset partitions together into a single set.
    
    Args:
        partitions: dict of (partition name, partition loader) pairs
        params: parameters for selecting dataset size for debugging
    
    Returns:
        list of dictionaries giving labeled datasets and descriptive metrics
    """
    compiled = []
    for partition_key, partition_load_func in sorted(partitions.items()):
        partition_data = partition_load_func()
        if isinstance(partition_data, Tuple):
            compiled.extend(partition_data)
        else:
            compiled.append(partition_data)
    if params["use_debug_dataset_size"]:
        compiled = compiled[:params["debug_dataset_size"]]
    return compiled


def label_cycle_counts(G_ls: list[dict], params: dict) -> list[dict]:
    """Label each graph with the node cycle counts."""
    for d in G_ls:
        d["network"] = label_node_cycle_counts(
            G=d["network"],
            n_cycles=params["n_cycles"],
            max_length=params["max_cycle_length"]
        )
    return G_ls


def label_node_cycle_counts(G: nx.Graph, n_cycles: int, max_length: int) -> nx.Graph:
    """Get the number of cycles of each length each node resides in.

    Each indicator vector has counts for lengths i = 1, ..., max_length
    
    Args:
        G: NetworkX graph to search for cycles
        n_cycles: maximum number of cycles to consider. If more are identified,
            then they will be randomly sampled down to this number.
        max_length: maximum length of cycles to consider 

    Returns:
        G with a new node attribute "cycle_count" giving the vector of counts
            of cycles of each length that the node participates in.
    """
    # Count cycles
    H = G.to_undirected()
    cycles = nx.cycle_basis(H)

    # If number of cycles we want to look at is less than the total number, use a random sample
    if len(cycles) > n_cycles:
        sample_idx = np.random.randint(0, len(cycles), n_cycles)
        cycles = [cycles[i] for i in sample_idx]

    # Generate graph labels from cycles
    n_bins = max_length - 3 + 1 # Since the minimum length of a cycle is 3
    counts = {u: [0] * n_bins for u in G.nodes}
    for cyc in cycles:
        len_bin = min(len(cyc), max_length) - 3
        for node in cyc:
            counts[node][len_bin] += 1
    nx.set_node_attributes(G, counts, name="cycle_count")
    return G


def label_graphs_deepsnap(G_ls: list[dict]) -> list[nx.DiGraph]:
    """Apply graph labeling to all graphs."""
    for d in G_ls:
        d["network"] = label_single_graph_deepsnap(d["network"])
    return G_ls


def label_single_graph_deepsnap(G: nx.DiGraph) -> nx.DiGraph:
    """Add node, edge, and graph attributes to be consumed by DeepSnap."""
    # Create clean graph with no attributes but same structure
    H = G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)

    # Set node attributes for DeepSnap
    nx.set_node_attributes(H, "busbar", name="node_type")
    node_feats = {}
    for u, a in G.nodes(data=True):
        scalars = torch.tensor([a["load"], a["genr"], a["n_breakers"]], dtype=torch.float)
        cycles = torch.tensor(a["cycle_count"], dtype=torch.float)
        node_feats[u] = torch.cat([scalars, cycles], dim=0)
    nx.set_node_attributes(H, node_feats, name="node_feature")

    # Set edge attributes for DeepSnap
    edge_types = {}
    edge_labels = {}
    edge_feats = {}
    for u, v, a in G.edges(data=True):
        if a["is_breaker"]: # This is the type of edge we're trying to label and predict
            edge_types[(u, v)] = "breaker"
            edge_labels[(u, v)] = int(a["breaker_closed"])
            edge_feats[(u, v)] = torch.zeros(size=(2,), dtype=torch.float)
        else:
            edge_feats[(u, v)] = torch.tensor([a["capacity"], a["reactance"]], dtype=torch.float)
            if a["is_interconnect"]:
                edge_types[(u, v)] = "interconnect"
            else:
                edge_types[(u, v)] = "branch"
    nx.set_edge_attributes(H, edge_types, name="edge_type")
    nx.set_edge_attributes(H, edge_labels, name="edge_label")
    nx.set_edge_attributes(H, edge_feats, name="edge_feature")
    return H


def build_deepsnap_datasets(G_ls: list[dict]) -> Tuple[GraphDataset, HeteroGraph]:
    """Create a HeteroGraph dataset.
    
    Args:
        G_ls: list of dicts of NetworkX graphs with descriptions

    Returns:
        DeepSnap GraphDataset and an example HeteroGraph to use for node and
        message type checks.
    """
    H_ls = [HeteroGraph(d["network"].to_undirected()) for d in G_ls]

    # Create dataset
    D = GraphDataset(
        graphs=H_ls,
        task="edge"
    )
    example = H_ls[0]
    return D, example
