"""
This is a boilerplate pipeline 'run_parallel'
generated using Kedro 0.18.13
"""

def generate_slurm_script(params: dict) -> str:
    """Generate SLURM script for running many preprocessing optimizations."""
    
    lines = ["#!/bin/bash"]
    for k, v in params["sbatch"].items():
        lines.append(f"#SBATCH --{k}={v}")
    lines.append("")
    lines.append("cd /home/users/passow/cs224w/optimal-breaker-gnn")
    lines.append("kedro run -p preprocess")

    sh = "\n".join(lines)

    if not params["dry"]:
        # Run the job directly from here
        pass

    return sh