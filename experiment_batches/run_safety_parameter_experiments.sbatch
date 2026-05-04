#!/bin/bash
#SBATCH --job-name=exp_safety_params
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=75
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/safety_experiments_fresh/exp_safety_params_%A_%a.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/safety_experiments_fresh/exp_safety_params_%A_%a.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/safety_experiments_fresh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "========================================="
echo "EXPERIMENT SUITE: Safety Parameter (t_skipped/t_stopping) Comparison"
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
exp_names=("Very Aggressive (10/5)" "Aggressive (50/10)" "Balanced (100/20)" "Conservative (200/20)" "Ultra Conservative (200/50)")
t_skipped_values=(10 50 100 200 200)
t_stopping_values=(5 10 20 20 50)
checkpoint_dirs=("checkpoints/safety_experiments_fresh/checkpoints_safety_10_5" "checkpoints/safety_experiments_fresh/checkpoints_safety_50_10" "checkpoints/safety_experiments_fresh/checkpoints_safety_100_20" "checkpoints/safety_experiments_fresh/checkpoints_safety_200_20" "checkpoints/safety_experiments_fresh/checkpoints_safety_200_50")

# Get current experiment parameters based on array index
exp_name="${exp_names[$SLURM_ARRAY_TASK_ID]}"
t_skipped="${t_skipped_values[$SLURM_ARRAY_TASK_ID]}"
t_stopping="${t_stopping_values[$SLURM_ARRAY_TASK_ID]}"
checkpoint_dir="${checkpoint_dirs[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "🧪 Starting Experiment: $exp_name"
echo "   Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "   t_skipped (safe states for recovery): $t_skipped"
echo "   t_stopping (recovery nodes before stopping): $t_stopping"
echo "   Checkpoint Dir: $checkpoint_dir"
echo "   Time: $(date)"
echo "----------------------------------------"

# Create experiment-specific output directories
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/training_data_fresh/safety_experiments/$(basename $checkpoint_dir)"

# Set experiment-specific environment variables
export T_SKIPPED=$t_skipped
export T_STOPPING=$t_stopping
export CHECKPOINT_DIR=$checkpoint_dir
export training_data_dir="training_data_fresh/safety_experiments/$(basename $checkpoint_dir)"

# Run training
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    ${CONTAINER_IMAGE} \
    bash -c "
        cd /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN && \
        export PYTHONPATH=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        export T_SKIPPED=$t_skipped && \
        export T_STOPPING=$t_stopping && \
        export CHECKPOINT_DIR=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir && \
        export PARALLEL_WORKERS=70 && \
        export AGENT_DEBUG=false && \
        export training_data_dir=\"training_data_fresh/safety_experiments/$(basename $checkpoint_dir)\" && \
        python -u scripts/train_alphazero_v2.py \
            2>&1
    "

echo "✅ Completed Experiment: $exp_name at $(date)"
echo "========================================="
echo "EXPERIMENT $SLURM_ARRAY_TASK_ID COMPLETED"
echo "Finished at: $(date)"
echo "========================================"