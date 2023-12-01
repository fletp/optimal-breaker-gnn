"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""
import pandas as pd
import networkx as nx
import numpy as np
from typing import Callable, Tuple
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from deepsnap.hetero_graph import HeteroGraph
from deepsnap.dataset import GraphDataset

def join_partitions(partitions: dict[str, Callable]) -> list[dict]:
    compiled = []
    for partition_key, partition_load_func in sorted(partitions.items()):
        partition_data = partition_load_func()
        if isinstance(partition_data, Tuple):
            compiled.extend(partition_data)
        else:
            compiled.append(partition_data)
    return compiled


def label_graphs(G_ls: list[dict]) -> list[nx.DiGraph]:
    """Apply graph labeling to all graphs."""
    return [label_single_graph(d["network"]) for d in G_ls]


def label_single_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Add node, edge, and graph attributes to be consumed by DeepSnap."""
    # Create clean graph with no attributes but same structure
    H = G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)

    # Set graph attributes compatible with DeepSnap
    nx.set_node_attributes(H, "busbar", name="node_type")
    node_feats = {}
    for u, a in G.nodes(data=True):
        node_feats[u] = torch.tensor([a["load"], a["genr"], a["n_breakers"]])
    nx.set_node_attributes(H, node_feats, name="node_feature")
    edge_types = {}
    edge_labels = {}
    edge_feats = {}
    for u, v, a in G.edges(data=True):
        if a["is_breaker"]:
            edge_types[(u, v)] = "breaker"
            edge_labels[(u, v)] = int(a["breaker_closed"])
        else:
            edge_feats[(u, v)] = torch.tensor([a["capacity"], a["reactance"]])
            if a["is_interconnect"]:
                edge_types[(u, v)] = "interconnect"
            else:
                edge_types[(u, v)] = "branch"
    nx.set_edge_attributes(H, edge_types, name="edge_type")
    nx.set_edge_attributes(H, edge_labels, name="edge_label")
    nx.set_edge_attributes(H, edge_feats, name="edge_feature")

    # Convert graph to undirected
    H = H.to_undirected()
    return H


def build_deepsnap_datasets(G_ls: nx.Graph) -> GraphDataset:
    """Create a HeteroGraph dataset."""
    H_ls = [HeteroGraph(G) for G in G_ls]

    # Create dataset
    D = GraphDataset(
        graphs=H_ls,
        task="edge"
    )
    example = H_ls[0]
    return D, example


def augment_graphs(G_ls: list[dict]) -> list[nx.DiGraph]:
    result = [alter_graph(d["network"]) for d in G_ls]
    return result


def alter_graph(G: nx.DiGraph) -> nx.DiGraph:
    n = G.number_of_nodes()
    i = 1
    edges = list(G.edges)
    for edge in edges:
        u, v = edge
        attr = G.get_edge_data(u, v)
        attr.update({"src_targ":(u, v)})
        G.add_node(n + i)
        node_attr = {n+i: attr}
        nx.set_node_attributes(G, node_attr)
        nx.add_path(G, [u, n+i, v])
        # Add original attributes
        edge_attr = {(u, n+i): attr, (n+i, v): attr}
        nx.set_edge_attributes(G, edge_attr)
        # Add identifying attribute
        if attr["is_breaker"]:
            edge_attr_id = {(u, n+i): {"edge_type": "busbar_breaker"},
                            (n+i, v): {"edge_type": "breaker_busbar"}}
        else: 
            edge_attr_id = {(u, n+i): {"edge_type": "busbar_branch"},
                            (n+i, v): {"edge_type": "branch_busbar"}}
        nx.set_edge_attributes(G, edge_attr_id)
        G.remove_edge(u,v)
        i += 1
    return G

def build_heterograph_datasets(G_ls: list[nx.DiGraph]) -> list[HeteroData]:
    data = [to_heterograph(G) for G in G_ls]
    metadata = data[0].metadata()
    return data, metadata

def to_heterograph(G):
    df_nodes = pd.DataFrame.from_dict(G.nodes, orient='index')
    df_nodes = df_nodes.drop(columns=["src_targ"])
    # Need to alter edge index source and target to refer to correct values
    # Node index for each type should start at 0 
    df_nodes["node_id"] = df_nodes.index.values
    df_nodes["new_index"] = 0
    df_nodes.loc[df_nodes['is_breaker'].isnull(), "new_index"] = np.arange(len(df_nodes[df_nodes['is_breaker'].isnull()]))
    df_nodes.loc[df_nodes['is_breaker'] == True, "new_index"] = np.arange(len(df_nodes[df_nodes['is_breaker'] == True]))
    df_nodes.loc[df_nodes['is_breaker'] == False, "new_index"] = np.arange(len(df_nodes[df_nodes['is_breaker'] == False]))
    df_edges = nx.to_pandas_edgelist(G)
    df_edges = df_edges.merge(df_nodes[["node_id", "new_index"]], how="left", left_on = ["source"], right_on = ["node_id"])
    df_edges = df_edges.merge(df_nodes[["node_id", "new_index"]], how="left", left_on = ["target"], right_on = ["node_id"])
    df_edges = df_edges.rename(columns = {"new_index_x": "new_source", "new_index_y": "new_target"})
    
    data = HeteroData()
    data['busbar'].node_id = torch.from_numpy(df_nodes[df_nodes['is_breaker'].isnull()].index.values)
    data['busbar'].x = torch.from_numpy(df_nodes[df_nodes['is_breaker'].isnull()][["load", "genr"]].values).to(torch.float32)
    data['breaker'].node_id = torch.from_numpy(df_nodes[df_nodes["is_breaker"] == True].index.values)
    data['breaker'].x = torch.from_numpy(np.random.normal(size=(len(df_nodes[df_nodes['is_breaker'] == True]), 5))).to(torch.float32)
    data['breaker'].y = torch.from_numpy(df_nodes[df_nodes["is_breaker"] == True]["breaker_closed"].values.astype(int)).to(torch.float32)
    data['branch'].node_id = torch.from_numpy(df_nodes[df_nodes["is_breaker"] == False].index.values)
    data['branch'].x = torch.from_numpy(df_nodes[df_nodes['is_breaker'] == False][["reactance", "capacity"]].values).to(torch.float32)
    
    data['busbar', 'switched_by', 'breaker'].edge_index = torch.t(torch.from_numpy(df_edges[df_edges["edge_type"] == "busbar_breaker"][["new_source", "new_target"]].values)) 
    data['breaker', 'switches', 'busbar'].edge_index = torch.t(torch.from_numpy(df_edges[df_edges["edge_type"] == "breaker_busbar"][["new_source", "new_target"]].values))
    data['busbar', 'connected_by', 'branch'].edge_index = torch.t(torch.from_numpy(df_edges[df_edges["edge_type"] == "busbar_branch"][["new_source", "new_target"]].values))
    data['branch', 'connects', 'busbar'].edge_index = torch.t(torch.from_numpy(df_edges[df_edges["edge_type"] == "branch_busbar"][["new_source", "new_target"]].values))
    data['busbar', 'connected_by', 'branch'].edge_attr = torch.t(torch.from_numpy(df_edges[df_edges["edge_type"] == "busbar_branch"][["reactance", "capacity"]].values)).to(torch.float32) 
    data['branch', 'connects', 'busbar'].edge_attr = torch.t(torch.from_numpy(df_edges[df_edges["edge_type"] == "branch_busbar"][["reactance", "capacity"]].values)).to(torch.float32)

    data = T.ToUndirected()(data)
    return data
