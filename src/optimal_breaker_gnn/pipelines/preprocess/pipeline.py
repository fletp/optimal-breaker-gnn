"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import build_base_network, create_network_scenario, optimize_scenario, combine_optimized_networks


def create_pipeline(**kwargs) -> Pipeline:
    pipe_ls = []
    build_pipe = pipeline(
        [
            node(
                func=build_base_network,
                inputs=["nodes", "edges"],
                outputs="network_base",
                name="build_base_network",
            ),
        ]
    )
    pipe_ls.append(build_pipe)

    optim_pipe = pipeline(
        [
            node(
                func=create_network_scenario,
                inputs=["network_base", "params:network_scenario"],
                outputs="network_scenario",
                name="create_network_scenario",
            ),
            node(
                func=optimize_scenario,
                inputs=["network_scenario", "params:optimize"],
                outputs="optim_res_dict",
                name="optimize_scenario",
            ),
        ]
    )

    n_examples = 100
    for p in range(n_examples):
        cur_pipe = pipeline(
            pipe=optim_pipe,
            inputs={"network_base"},
            parameters={"params:network_scenario", "params:optimize"},
            namespace=f"run_{p}",
        )
        pipe_ls.append(cur_pipe)

    
    combine_pipe = pipeline(
        [
            node(
                func=combine_optimized_networks,
                inputs=[f"run_{p}.optim_res_dict" for p in range(n_examples)],
                outputs="training_network_list",
                name="combine_optimized_networks"
            ),
        ]
    )
    pipe_ls.append(combine_pipe)

    return sum(pipe_ls)