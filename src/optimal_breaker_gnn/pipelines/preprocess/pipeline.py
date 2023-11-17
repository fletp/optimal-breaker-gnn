"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    join_partitions,
    augment_graphs,
    build_heterograph_datasets,
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
        ]
    )
    return pipe