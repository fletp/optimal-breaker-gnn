"""
This is a boilerplate pipeline 'run_parallel'
generated using Kedro 0.18.13
"""
from typing import List
import copy

def generate_slurm_script(params: dict) -> str:
    """Generate SLURM script for running many preprocessing optimizations."""
    
    lines = ["#!/bin/bash"]

    # Set up resources
    slurm = params["slurm"]
    res_opts = {}
    if slurm["command"] == "sbatch":
        prefix = ""
        # Label batches with job-name
        for k, v in slurm["reporting"].items():
            if isinstance(v, str):
                v = v.replace("*", slurm["reporting"]["job-name"])
        
        # Add options to full option list
        res_opts.update(slurm["reporting"])
        res_opts.update(slurm["multi_job"])
        template = ["#SBATCH --", 0, "=", 1, "\n"]
    elif slurm["command"] == "salloc":
        prefix = "salloc"
        template = [" --", 0, "=", 1]
    res_opts.update(slurm["resources"])
    slurm_opts = build_opts(d=res_opts, template=template)
    slurm_opts = "".join([prefix, slurm_opts])
    lines.append(slurm_opts)

    # Set up commands to run on resources
    lines.append("")
    if params["ruse"]:
        lines.append("module load system ruse")
        cmd_prefix = "ruse" + " "
    else:
        cmd_prefix = ""
    
    lines.append("cd /home/users/passow/cs224w/optimal-breaker-gnn")

    kedro_opts = build_opts(d=params["kedro"], template=[" --", 0, ":", 1])
    cmds = f"{cmd_prefix}kedro run {kedro_opts}"

    cmds_ls = [cmds] * params["n_sequential_jobs"]
    lines.extend(cmds_ls)

    # Build file
    sh = "\n".join(lines)
    return sh


def build_opts(d: dict[str: str | int | float], template: List[str | int]) -> str:
    """Build options based on a dictionary of key, value pairs and a string 
    template."""
    opt_lines = []
    for k, v in d.items():
        line = []
        for elem in template:
            # Place in keys and values
            if elem == 0:
                elem = str(k)
            elif elem == 1:
                elem = str(v)
            line.extend(elem)
        opt_lines.extend(line)
    out_str = "".join(opt_lines)
    return out_str