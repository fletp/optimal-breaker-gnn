"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import build_base_network, create_network_scenario


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=build_base_network,
                inputs=["nodes", "edges"],
                outputs="network_base",
                name="build_base_network",
            ),
            node(
                func=create_network_scenario,
                inputs=["network_base", "params:network_scenario"],
                outputs="network_scenario",
                name="create_network_scenario",
            ),
        ]
    )
    return pipe