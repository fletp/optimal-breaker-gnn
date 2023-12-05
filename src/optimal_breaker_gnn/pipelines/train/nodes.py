"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.13
"""
import cvxpy as cp
import networkx as nx
import pandas as pd
import time
import platform
from torch_geometric.data import HeteroData
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import copy
import torch.nn as nn
from typing import Tuple, List
from torch.utils.data import random_split, Subset
from optimal_breaker_gnn.models.optim import define_problem, concretize_network_attrs
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from optimal_breaker_gnn.models.hetero_gnn import train, evaluate, HeteroGNN
from deepsnap.hetero_graph import HeteroGraph


def build_dataloaders(data: list, params: dict) -> dict[DataLoader]:
    split_names = list(params["splits"].keys())
    split_fracs = [params["splits"][k]["frac"] for k in split_names]

    rng = torch.Generator().manual_seed(params["seed"])
    splits = random_split(data, lengths=split_fracs, generator=rng)
    split_dict = {split_names[i]:splits[i].indices for i in range(len(split_names))}

    loaders = {}
    for name, idx in split_dict.items():
        cur_data = data[idx]
        cur_data = cur_data[:params["splits"][name]["n_examples_from_frac"]]
        loader = DataLoader(
            cur_data,
            batch_size=params["splits"][name]["batch_size"],
            shuffle=params["splits"][name]["shuffle"],
            collate_fn=Batch.collate(),
        )
        loaders.update({name: loader})
    return loaders, split_dict, params


def train_model(loaders: dict, example_graph: HeteroGraph, params_struct: dict, params_train: dict):
    model = HeteroGNN(
        example_graph=example_graph,
        params=params_struct,
    ).to(params_train["device"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params_train['lr'],
        weight_decay=params_train["weight_decay"],
    )

    best_model = None
    best_valid_acc = 0
    best_metrics = None

    logs = []
    for epoch in range(1, 1 + params_train["epochs"]):
        print('Training...')
        loss = train(model, optimizer, loaders["train"])
        
        print('Evaluating...')
        train_acc, train_ones = evaluate(model, loaders["train"])
        valid_acc, valid_ones = evaluate(model, loaders["valid"])
        test_acc, test_ones = evaluate(model, loaders["test"])

        log_dict = {
                'epoch': epoch,
                'loss': loss,
                'train_acc': train_acc,
                'validation_acc': valid_acc,
                'test_acc': test_acc,
                'train_ones': train_ones,
                'validation_ones': valid_ones,
                'test_ones': test_ones,
            }
        logs.append(log_dict)

        print(f'Epoch: {epoch:02d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {100 * train_acc:.2f}%, '
            f'Valid: {100 * valid_acc:.2f}%, '
            f'Train % Ones: {100 * train_ones:.2f}%, '
            f'Valid % Ones: {100 * valid_ones:.2f}%'
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
            best_metrics = copy.deepcopy(log_dict)
        
    log_df = pd.DataFrame(logs)
    return best_model, best_metrics, params_struct, log_df


def apply_preds_to_networks(
        model,
        heterodata:GraphDataset,
        networks:list[dict],
        splits:dict[list],
        params:dict,
    ) -> list[dict]:
    """Apply predictions to networks."""
    idx = splits[params["subset"]]
    edge_type = tuple(params["predict_edge_type"])
    for i in idx: # Looping through networks
        g = heterodata[i]
        corresp = g.edge_label_index[edge_type]
        networks[i]["network_pred"] = copy.deepcopy(networks[i]["network"])
        with torch.no_grad():
            preds = model(g.node_feature, g.edge_index, g.edge_feature)
            preds_edge = preds[edge_type].round().cpu().detach().numpy()
            for b in range(len(preds)): # Looping through predicted values / breakers
                closed = preds_edge[b]
                u, v = int(corresp[0, b]) + 1, int(corresp[1, b]) + 1 # Adding one because we set networks to be 1-indexed, whereas DeepSnap must internally 0-index
                if (u, v) in networks[i]["network_pred"].edges: # Because we're moving from an undirected graph to a directed graph
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

