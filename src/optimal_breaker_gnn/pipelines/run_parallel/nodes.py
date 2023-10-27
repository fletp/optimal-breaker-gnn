"""
This is a boilerplate pipeline 'run_parallel'
generated using Kedro 0.18.13
"""

def generate_slurm_script(params: dict) -> str:
    """Generate SLURM script for running many preprocessing optimizations."""
    
    lines = ["#!/bin/bash"]

    def build_opt(k, v):
        return f"#SBATCH --{k}={v}"
    
    lines.append(build_opt("job-name", params["job-name"]))
    for k, v in params["logging"].items():
        v = v.replace("*", params["job-name"])
        lines.append(build_opt(k, v))
    for k, v in params["resources"].items():
        lines.append(build_opt(k, v))
    lines.append("")
    if params["ruse"]:
        lines.append("module load system ruse")
        cmd_prefix = "ruse" + " "
    else:
        cmd_prefix = ""
    lines.append("cd /home/users/passow/cs224w/optimal-breaker-gnn")

    cmds = []
    for k, v in params["kedro"].items():
        cmds.append(f"--{k}={v}")
    cmd_opts = " ".join(cmds)

    lines.append(f"{cmd_prefix}kedro run {cmd_opts}")

    sh = "\n".join(lines)
    return sh