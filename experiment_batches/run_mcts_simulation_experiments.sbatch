#!/bin/bash
#SBATCH --job-name=exp_mcts_sims
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=75
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/mcts_experiments2_fresh/exp_mcts_sims_%A_%a.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/mcts_experiments2_fresh/exp_mcts_sims_%A_%a.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/mcts_experiments2_fresh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "========================================="
echo "EXPERIMENT SUITE: MCTS Simulation Count Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Container setup
CONTAINER_IMAGE="dqn_idf_2023"

# Base configuration
export num_cycles=3

# Define experiment configurations
exp_names=("MCTS 50 (Fast)" "MCTS 200 (Medium)" "MCTS 400 (Standard)" "MCTS 800 (High)" "MCTS 1600 (Very High)")
mcts_simulations=(50 200 400 800 1600)
checkpoint_dirs=("checkpoints/mcts_experiments2_fresh/checkpoints_mcts50" "checkpoints/mcts_experiments2_fresh/checkpoints_mcts200" "checkpoints/mcts_experiments2_fresh/checkpoints_mcts400" "checkpoints/mcts_experiments2_fresh/checkpoints_mcts800" "checkpoints/mcts_experiments2_fresh/checkpoints_mcts1600")

# Get current experiment parameters based on array index
exp_name="${exp_names[$SLURM_ARRAY_TASK_ID]}"
mcts_sims="${mcts_simulations[$SLURM_ARRAY_TASK_ID]}"
checkpoint_dir="${checkpoint_dirs[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "🧪 Starting Experiment: $exp_name"
echo "   Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "   MCTS Simulations: $mcts_sims"
echo "   Checkpoint Dir: $checkpoint_dir"
echo "   Time: $(date)"
echo "----------------------------------------"

# Create experiment-specific output directories
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/training_data_fresh/mcts_experiments2/$(basename $checkpoint_dir)"

# Set experiment-specific environment variables
export MCTS_SIMULATIONS=$mcts_sims
export CHECKPOINT_DIR="/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
export training_data_dir="training_data_fresh/mcts_experiments/$(basename $checkpoint_dir)"

# Run training
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    ${CONTAINER_IMAGE} \
    bash -c "
        cd /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN && \
        export PYTHONPATH=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        export MCTS_SIMULATIONS=$mcts_sims && \
        export PARALLEL_WORKERS=70 && \
        export CHECKPOINT_DIR=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir && \
        export AGENT_DEBUG=false && \
        python -u scripts/train_alphazero_v2.py \
            2>&1
    "

echo "✅ Completed Experiment: $exp_name at $(date)"
echo "========================================="
echo "EXPERIMENT $SLURM_ARRAY_TASK_ID COMPLETED"
echo "Finished at: $(date)"
echo "========================================"