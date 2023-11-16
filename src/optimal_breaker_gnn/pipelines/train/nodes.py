"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.13
"""
import cvxpy as cp
import networkx as nx
import pandas as pd
import numpy as np
import time
import platform
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch
import copy
from optimal_breaker_gnn.models.gnn import train, evaluate, HGT_Model
import torch.nn as nn
from typing import Tuple, Callable, List
from torch.utils.data import random_split, Subset
from optimal_breaker_gnn.models.optim import define_problem, concretize_network_attrs

def join_partitions(partitions: dict[str, Callable]) -> list[dict]:
    compiled = []
    for partition_key, partition_load_func in sorted(partitions.items()):
        partition_data = partition_load_func()
        compiled.extend(partition_data)
    return compiled


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
    metadata = (['busbar', 'breaker', 'branch'],
                [('busbar', 'breaker', 'breaker'), ('breaker', 'breaker', 'busbar'),
                ('busbar', 'branch', 'branch'),('branch', 'branch', 'busbar')])
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

def build_dataloaders(data: list, params: dict) -> dict[DataLoader]:
    split_names = list(params["splits"].keys())
    split_fracs = [params["splits"][k]["frac"] for k in split_names]

    rng = torch.Generator().manual_seed(params["seed"])
    splits = random_split(data, lengths=split_fracs, generator=rng)
    split_dict = {split_names[i]:splits[i] for i in range(len(split_names))}

    loaders = {}
    for name, sub in split_dict.items():
        cur_data = [data[i] for i in sub.indices]
        loader = DataLoader(
            cur_data,
            batch_size=params["splits"][name]["batch_size"],
            shuffle=params["splits"][name]["shuffle"]
        )
        loaders.update({name: loader})
    return loaders, split_dict


def train_model(loaders: dict, metadata: Tuple[List[str], List[Tuple[str, str, str]]], params_struct: dict, params_device: dict):
    model = HGT_Model(
        params_struct['hidden_dim'],
        params_struct['output_dim'],
        metadata,
        params_struct['num_layers'],
        params_struct['num_heads'],
        params_struct['dropout'],
        params_struct["gain"],
        params_struct["bias"],
        ).to(params_device['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=params_struct['lr'])
    loss_fn =  nn.BCEWithLogitsLoss()

    best_model = None
    best_valid_acc = 0

    logs = []
    for epoch in range(1, 1 + params_struct["epochs"]):
        print('Training...')
        loss = train(model, params_device['device'], loaders["train"], optimizer, loss_fn)

        print('Evaluating...')
        train_acc, train_ones = evaluate(model, params_device["device"], loaders["train"], loss_fn)
        valid_acc, valid_ones = evaluate(model, params_device["device"], loaders["valid"], loss_fn)
        test_acc, test_ones = evaluate(model, params_device["device"], loaders["test"], loss_fn)


        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
        
        logs.append(
            {
                'epoch': epoch,
                'loss': loss,
                'train_acc': train_acc,
                'validation_acc': valid_acc,
                'test_acc': test_acc,
                'train_ones': train_ones,
                'validation_ones': valid_ones,
                'test_ones': test_ones,
            }
        )
        print(f'Epoch: {epoch:02d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {100 * train_acc:.2f}%, '
            f'Valid: {100 * valid_acc:.2f}% '
            f'Test: {100 * test_acc:.2f}%'
            f'Train % Ones: {100 * train_ones:.2f}%, '
            f'Valid % Ones: {100 * valid_ones:.2f}%, '
            f'Test % Ones: {100 * test_ones:.2f}%')
    log_df = pd.DataFrame(logs)
    return best_model, log_df


def apply_preds_to_networks(model, heterodata:HeteroData, splits:list[Subset], augments:list[nx.DiGraph], networks:list[dict], params:dict) -> list[dict]:
    """Apply predictions to networks."""
    idx = splits[params["subset"]].indices
    for i in idx: # Looping through networks
        with torch.no_grad():
            preds = F.sigmoid(torch.squeeze(model(heterodata[i]))).round().cpu().detach().numpy()
        networks[i]["network_pred"] = copy.deepcopy(networks[i]["network"])
        for b in range(len(preds)): # Looping through predicted values / breakers
            closed = preds[b]
            aug_id = int(heterodata[i]["breaker"].node_id[b])
            u, v = augments[i].nodes[aug_id]["src_targ"]
            networks[i]["network_pred"].edges[u, v]["breaker_closed"] = closed
    return [networks[i] for i in idx]


def eval_preds_by_optim(networks:list[dict], params:dict) -> dict:
    """Evaluate predictions by optimization."""
    evals = [eval_single_pred(d["network_pred"], params=params) for d in networks]
    return evals
    

def eval_single_pred(G:nx.DiGraph, params:dict) -> dict:
    """Evaluate a single prediction using optimization."""
    tic = time.perf_counter()
    prob = define_problem(G=G, mode="eval", params=params)
    prob.solve(solver=cp.XPRESS, verbose=True)
    toc = time.perf_counter()
    cnstrs = prob.constraints
    viols = [not c.value() for c in cnstrs]
    n_viols = sum(viols)

    tic_1 = time.perf_counter()
    G = concretize_network_attrs(G)
    toc_1 = time.perf_counter()

    res = {
        "network": G,
        "obj_val": float(prob.objective.value),
        "time_elapsed": toc-tic+toc_1-tic_1,
        "num_viols": n_viols,
        "platform": platform.platform()
    }
    return res

