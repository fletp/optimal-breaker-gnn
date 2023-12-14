"""
This is a boilerplate pipeline 'evaluate'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    apply_preds_to_networks,
    eval_preds_by_optim,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=apply_preds_to_networks,
                inputs=["trained_model_best", "heterodata", "training_networks", "splits", "params:predict"],
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
