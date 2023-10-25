"""
This is a boilerplate pipeline 'run_parallel'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_slurm_script


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=generate_slurm_script,
                inputs=["params:slurm"],
                outputs="slurm_script",
                name="generate_slurm_script",
            ),
        ]
    )
    return pipe
