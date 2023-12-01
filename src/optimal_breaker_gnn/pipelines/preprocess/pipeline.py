"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    join_partitions,
    label_graphs,
    build_deepsnap_datasets,
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
                func=label_graphs,
                inputs="training_networks",
                outputs="training_networks_augmented",
                name="label_graphs",
            ),
            node(
                func=build_deepsnap_datasets,
                inputs="training_networks_augmented",
                outputs=["heterodata", "example_heterograph"],
                name="build_deepsnap_datasets",
            ),
        ]
    )
    return pipe