"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_deepsnap_datasets, join_partitions, label_graphs


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=join_partitions,
                inputs=["optim_res_dict", "params:data_size"],
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