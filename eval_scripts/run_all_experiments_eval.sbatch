#!/bin/bash
#SBATCH --job-name=eval-all-experiments
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_all_experiments_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_all_experiments_%j.err

echo "Starting comprehensive evaluation of all experiments..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================"

export PATH=/usr/local/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval

enroot start --root --rw --mount /home/ka/ka_iai/ka_jp7970:/work dqn_idf_2023 bash -c '
    echo "Inside container - setting up environment..."
    echo "Container started at: $(date)"
    
    # Set up paths - use same approach as training script + support src-prefixed imports
    cd /work/PdF_L2RPN/AlphaZero_L2RPN
    export PYTHONPATH="/workspace/src:/workspace:/work/PdF_L2RPN/AlphaZero_L2RPN/src:${PYTHONPATH}"
    
    # Create symlink for Grid2Op data if needed
    if [ ! -d "/workspace" ]; then
        echo "Setting up workspace symlinks..."
        mkdir -p /workspace
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN /workspace/AlphaZero_L2RPN
        ln -sfn /work /workspace
        # Create the /workspace/src symlink that training expects
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/src /workspace/src
        mkdir -p /workspace/logs
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/logs/eval /workspace/logs/
    fi
    
    # Set environment variables for the evaluation script
    export DATA_DIR="/work/data_grid2op"
    export PYTHONUNBUFFERED=1
    export AGENT_DEBUG=false
    
    echo "========================================"
    echo "Starting comprehensive experiment evaluation..."
    echo "Timestamp: $(date)"
    echo "Working directory: $(pwd)"
    echo "Python path: $PYTHONPATH"
    echo "========================================"
    
    # Run the comprehensive evaluation
    python3 scripts/evaluate_all_experiments.py \
        --checkpoints-dir /work/PdF_L2RPN/AlphaZero_L2RPN/checkpoints \
        --eval-dir /work/PdF_L2RPN/AlphaZero_L2RPN/eval \
        --max-parallel-experiments 2
    
    echo "========================================"
    echo "Evaluation completed at: $(date)"
    echo "========================================"
'

echo "Job completed at: $(date)"