"""
This is a boilerplate pipeline 'run_parallel'
generated using Kedro 0.18.13
"""

def generate_slurm_script(params: dict) -> str:
    """Generate SLURM script for running many preprocessing optimizations."""
    
    lines = ["#!/bin/bash"]

    def build_opt(k, v):
        return f"#SBATCH --{k}={v}"
    
    for k, v in params["sbatch"].items():
        if isinstance(v, str):
            v = v.replace("*", params["sbatch"]["job-name"])
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
    cmds = f"{cmd_prefix}kedro run {cmd_opts}"

    email_command_str = """
secs_to_human(){
echo "$(( ${1} / 3600 )):$(( (${1} / 60) % 60 )):$(( ${1} % 60 ))"
}
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}\n"

### exec task here
( << replace with your task here >> ) \
&& (cat JOB$SLURM_JOB_ID.out |mail -s "$SLURM_JOB_NAME Ended after $(secs_to_human $(($(date +%s) - ${start}))) id=$SLURM_JOB_ID" my@email.com && echo mail sended) \
|| (cat JOB$SLURM_JOB_ID.out |mail -s "$SLURM_JOB_NAME Failed after $(secs_to_human $(($(date +%s) - ${start}))) id=$SLURM_JOB_ID" my@email.com && echo mail sended && exit $?)"""

    email_command_str = email_command_str.replace("<< replace with your task here >>", cmds)

    lines.append(email_command_str)

    sh = "\n".join(lines)
    return sh