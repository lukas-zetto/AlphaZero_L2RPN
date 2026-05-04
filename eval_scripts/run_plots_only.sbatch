#!/bin/bash
#SBATCH --job-name=plot-experiments
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/plot_experiments_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/plot_experiments_%j.err

echo "Starting plotting of existing evaluation results..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

export PATH=/usr/local/bin:$PATH

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval

enroot start --root --rw --mount /home/ka/ka_iai/ka_jp7970:/work dqn_idf_2023 bash -c '
    echo "Inside container - setting up environment for plotting..."
    
    # Set up paths
    cd /work/PdF_L2RPN/AlphaZero_L2RPN
    export PYTHONPATH="/workspace/src:/workspace:/work/PdF_L2RPN/AlphaZero_L2RPN/src:${PYTHONPATH}"
    
    # Create symlink for Grid2Op data if needed
    if [ ! -d "/workspace" ]; then
        echo "Setting up workspace symlinks..."
        mkdir -p /workspace
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN /workspace/AlphaZero_L2RPN
        ln -sfn /work /workspace
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/src /workspace/src
        mkdir -p /workspace/logs
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/logs/eval /workspace/logs/
    fi
    
    # Set environment variables
    export DATA_DIR="/work/data_grid2op"
    export PYTHONUNBUFFERED=1
    export AGENT_DEBUG=false
    
    echo "========================================"
    echo "Creating plots from existing evaluation results..."
    echo "Timestamp: $(date)"
    echo "========================================"
    
    # Run plotting only (no re-evaluation)
    python3 scripts/evaluate_all_experiments.py \
        --checkpoints-dir /work/PdF_L2RPN/AlphaZero_L2RPN/checkpoints \
        --eval-dir /work/PdF_L2RPN/AlphaZero_L2RPN/eval \
        --plot-only
    
    echo "========================================"
    echo "Plotting completed at: $(date)"
    echo "========================================"
'

echo "Job completed at: $(date)"