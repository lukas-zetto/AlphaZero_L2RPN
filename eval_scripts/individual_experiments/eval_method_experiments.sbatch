#!/bin/bash
#SBATCH --job-name=eval-method
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_method_experiments_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_method_experiments_%j.err

echo "Starting evaluation of method_experiments_fixed_8"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

export PATH=/usr/local/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval

enroot start --root --rw --mount /home/ka/ka_iai/ka_jp7970:/work dqn_idf_2023 bash -c '
    echo "Inside container - evaluating method_experiments_fixed_8..."
    
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
    
    echo "Running evaluation for method_experiments_fixed_8..."
    python3 scripts/evaluate_all_experiments.py \
        --checkpoints-dir /work/PdF_L2RPN/AlphaZero_L2RPN/checkpoints \
        --eval-dir /work/PdF_L2RPN/AlphaZero_L2RPN/eval \
        --experiment method_experiments_fixed_8
    
    echo "Method experiments fixed_8 evaluation completed at: $(date)"
'

echo "Job completed at: $(date)"