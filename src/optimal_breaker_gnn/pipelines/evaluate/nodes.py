"""
This is a boilerplate pipeline 'evaluate'
generated using Kedro 0.18.13
"""
import copy
import platform
import time

import networkx as nx
import torch
import torch.nn as nn

from deepsnap.dataset import GraphDataset
from optimal_breaker_gnn.models.optim import concretize_network_attrs, define_problem

def apply_preds_to_networks(
        model: nn.Module,
        heterodata:GraphDataset,
        networks:list[dict],
        splits:dict[list],
        params:dict,
    ) -> list[dict]:
    """Apply model predictions to original networks.
    
    Args:
        model: pytorch module which defines the model
        heterodata: DeepSnap GraphDataset giving all the graphs used in this
            project (training, validation, and testing) for use as a lookup
        networks: original NetworkX graphs with labelled optimal results
        splits: train, validation, test dataset split indices
        params: dict of parameters
    
    Returns:
        list of dicts containing NetworkX graphs with model-derived labels applied
    """
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


def eval_preds_by_optim(
        networks:list[dict], 
        params:dict
    ) -> list[dict]:
    """Evaluate model predictions by optimization of labeled networks.
    
    Args:
        networks: list of dicts containing NetworkX graphs with model-derived
            labels applied.
        params: parameters for the optimization problem

    Returns:
        list of dicts containing optimized NetworkX graphs along with result
        metrics.
    """
    evals = [eval_single_pred(d["network_pred"], params=params) for d in networks]
    return evals
    

def eval_single_pred(
        G:nx.DiGraph, 
        params:dict
    ) -> dict:
    """Evaluate a single prediction using optimization of a labeled network."""
    tic = time.perf_counter()
    prob = define_problem(G=G, mode="eval", params=params)
    prob.solve(solver=params["solver"], verbose=True)
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

