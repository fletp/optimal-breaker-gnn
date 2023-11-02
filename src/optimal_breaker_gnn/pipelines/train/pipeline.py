"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import augment_graphs, build_heterograph_datasets, build_dataloaders, train_model, join_partitions


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=join_partitions,
                inputs="training_network_partitions",
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
                outputs="heterodata",
                name="build_heterograph_datasets",
            ),
            node(
                func=build_dataloaders,
                inputs=["heterodata", "params:dataloaders"],
                outputs="dataloaders",
                name="build_dataloaders",
            ),
            node(
                func=train_model,
                inputs=["dataloaders", "params:structure", "params:platform"],
                outputs=["trained_model_best", "training_logs"],
                name="train_model",
            ),
        ]
    )
    return pipe
