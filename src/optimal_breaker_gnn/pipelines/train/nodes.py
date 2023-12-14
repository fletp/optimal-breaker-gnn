"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.13
"""
import copy
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from deepsnap.batch import Batch
from deepsnap.hetero_graph import HeteroGraph
from torch.utils.data import DataLoader, random_split

from optimal_breaker_gnn.models.hetero_gnn import HeteroGNN, evaluate, train


def build_dataloaders(
        data: list[HeteroGraph], 
        params: dict
    ) -> dict[DataLoader]:
    """Apply train-valid-test split to list of graphs."""

    # Apply splits
    split_names = list(params["splits"].keys())
    split_fracs = [params["splits"][k]["frac"] for k in split_names]

    rng = torch.Generator().manual_seed(params["seed"])
    splits = random_split(data, lengths=split_fracs, generator=rng)
    split_dict = {split_names[i]:splits[i].indices for i in range(len(split_names))}

    # Create corresponding dataloaders
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


def train_model(
        loaders: dict[DataLoader], 
        example_graph: HeteroGraph, 
        params_struct: dict, 
        params_train: dict
    ) -> Tuple[nn.Module, dict, dict, pd.DataFrame]:
    """Train the model and evaluate performance across epochs, saving the best model.
    
    Args:
        loaders: dict of dataloaders for training, validation, and testing
        example_graph: HeteroGraph with the same entity types and numbers
            of features for each entity as the graphs to be processed.
        params_struct: dict of model structure parameters
        params_train: dict of training process parameters

    Returns:
        best_model: the best model identified by validation score
        best_metrics: the metrics achieved by the best_model
        params_struct: the structural parameters that define the best model
        log_df: the trajectory of metrics across epochs
    """
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
    best_valid_score = 0
    best_metrics = None

    logs = []
    for epoch in range(1, 1 + params_train["epochs"]):
        print('Training...')
        loss = train(model, optimizer, loaders["train"], params_train["device"])
        
        print('Evaluating...')
        train_score, train_ones = evaluate(model, loaders["train"], params_train["device"])
        valid_score, valid_ones = evaluate(model, loaders["valid"], params_train["device"])
        test_score, test_ones = evaluate(model, loaders["test"], params_train["device"])

        log_dict = {
                'epoch': epoch,
                'loss': loss,
                'train_score': train_score,
                'validation_score': valid_score,
                'test_score': test_score,
                'train_ones': train_ones,
                'validation_ones': valid_ones,
                'test_ones': test_ones,
            }
        logs.append(log_dict)

        print(f'Epoch: {epoch:02d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {100 * train_score:.2f}%, '
            f'Valid: {100 * valid_score:.2f}%, '
            f'Train % Ones: {100 * train_ones:.2f}%, '
            f'Valid % Ones: {100 * valid_ones:.2f}%'
        )

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_model = copy.deepcopy(model)
            best_metrics = copy.deepcopy(log_dict)
        
    log_df = pd.DataFrame(logs)
    return best_model, best_metrics, params_struct, log_df
