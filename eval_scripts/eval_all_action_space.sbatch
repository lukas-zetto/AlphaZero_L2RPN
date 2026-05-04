#!/bin/bash
#SBATCH --job-name=eval-all-action-space
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=35G
#SBATCH --partition=cpu
#SBATCH --time=8:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_all_action_space_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_all_action_space_%j.err

echo "Starting multi-run evaluation of ALL action space experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

export PATH=/usr/local/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval

enroot start --root --rw --mount /home/ka/ka_iai/ka_jp7970:/work dqn_idf_2023 bash -c '
    echo "Inside container - multi-run action space evaluation..."
    
    cd /work/PdF_L2RPN/AlphaZero_L2RPN
    export PYTHONPATH="/workspace/src:/workspace:/work/PdF_L2RPN/AlphaZero_L2RPN/src:${PYTHONPATH}"
    
    if [ ! -d "/workspace" ]; then
        mkdir -p /workspace
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN /workspace/AlphaZero_L2RPN
        ln -sfn /work /workspace
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/src /workspace/src
        mkdir -p /workspace/logs
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/logs/eval /workspace/logs/
    fi
    
    export DATA_DIR="/work/data_grid2op"
    export PYTHONUNBUFFERED=1
    export AGENT_DEBUG=false
    
    echo "Running multi-run action space evaluation script..."
    python3 scripts/eval_all_action_space.py
    
    echo "Multi-run action space evaluation completed at: $(date)"
'

echo "Job completed at: $(date)"