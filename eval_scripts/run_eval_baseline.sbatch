#!/bin/bash
#SBATCH --job-name=eval-baseline-seq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --time=15:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_baseline_seq_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_baseline_seq_%j.err

    echo "Starting evaluation of checkpoints_baseline_seq in docker container..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

export PATH=/usr/local/bin:$PATH

enroot start --root --rw --mount /home/ka/ka_iai/ka_jp7970:/work dqn_idf_2023 bash -c '
    echo "Inside container - setting up environment..."
    
    # Set up paths - use same approach as training script + support src-prefixed imports
    cd /work/PdF_L2RPN/AlphaZero_L2RPN
    export PYTHONPATH="/workspace/src:/workspace:/work/PdF_L2RPN/AlphaZero_L2RPN:${PYTHONPATH}"
    
    # Create symlink for Grid2Op data if needed
    if [ ! -d "/workspace" ]; then
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
    
    echo "Running evaluation script on checkpoints_baseline_seq..."
    python3 scripts/evaluate_all_checkpoints.py \
        --checkpoint-dir /work/PdF_L2RPN/AlphaZero_L2RPN/checkpoints_baseline_seq \
        --results-dir /work/PdF_L2RPN/AlphaZero_L2RPN/eval/evaluation_results_baseline_seq
    
    echo "Evaluation completed at $(date)"
'

echo "Job completed."