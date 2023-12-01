"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_dataloaders,
    train_model,
    apply_preds_to_networks,
    eval_preds_by_optim,
    sparsify_edge_indices,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=build_dataloaders,
                inputs=["heterodata", "params:dataloaders"],
                outputs=["dataloaders", "splits", "param_dataloaders"],
                name="build_dataloaders",
            ),
            node(
                func=sparsify_edge_indices,
                inputs="dataloaders",
                outputs="sparse_dataloaders",
                name="sparsify_edge_indices",
            ),
            node(
                func=train_model,
                inputs=["sparse_dataloaders", "example_heterograph", "params:structure", "params:train"],
                outputs=["trained_model_best", "best_metrics", "param_struct", "training_logs"],
                name="train_model",
            ),
            node(
                func=apply_preds_to_networks,
                inputs=["trained_model_best", "heterodata", "splits", "training_networks_augmented", "training_networks", "params:predict"],
                outputs="prediction_networks",
                name="apply_preds_to_networks",
            ),
            node(
                func=eval_preds_by_optim,
                inputs=["prediction_networks", "params:optimize"],
                outputs="violations",
                name="eval_preds_by_optim",
            ),
        ]
    )
    return pipe
