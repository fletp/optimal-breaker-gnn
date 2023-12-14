"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_dataloaders,
    train_model,
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
                func=train_model,
                inputs=["dataloaders", "example_heterograph", "params:structure", "params:train"],
                outputs=["trained_model_best", "best_metrics", "param_struct", "training_logs"],
                name="train_model",
            ),
        ]
    )
    return pipe
