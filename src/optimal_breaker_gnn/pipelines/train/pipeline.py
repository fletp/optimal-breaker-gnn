"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    augment_graphs,
    build_heterograph_datasets,
    build_dataloaders,
    train_model,
    join_partitions,
    apply_preds_to_networks,
    eval_preds_by_optim,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=join_partitions,
                inputs="optim_res_dict",
                outputs="training_networks",
                name="join_partitions",
            ),
            node(
                func=augment_graphs,
                inputs="training_networks",
                outputs="training_networks_augmented",
                name="augment_graphs",
            ),
            node(
                func=build_heterograph_datasets,
                inputs="training_networks_augmented",
                outputs=["heterodata", "graph_metadata"],
                name="build_heterograph_datasets",
            ),
            node(
                func=build_dataloaders,
                inputs=["heterodata", "params:dataloaders"],
                outputs=["dataloaders", "splits", "param_dataloaders"],
                name="build_dataloaders",
            ),
            node(
                func=train_model,
                inputs=["dataloaders", "graph_metadata", "params:structure", "params:platform"],
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
