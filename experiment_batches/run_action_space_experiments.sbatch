#!/bin/bash
#SBATCH --job-name=exp_actions
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/action_space_experiments_fixed_8/exp_actions_%A_%a.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/action_space_experiments_fixed_8/exp_actions_%A_%a.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/action_space_experiments_fixed_8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "========================================="
echo "EXPERIMENT SUITE: Action Space Comparison - RUN _8"
echo "🎲 Fresh Seeds: main=1415, chronic=359"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Container setup
CONTAINER_IMAGE="dqn_idf_2023"

# Base configuration
export num_cycles=3  # Numbe of training iterations per experiment

# Define experiment configurations
exp_names=("Symmetry Reduction (SYM)" "No Reduction (N0)" "N1 Reduction")
action_reductions=("SYM" "N0" "N1")
use_reduced_set=("false" "false" "false")
checkpoint_dirs=("checkpoints/action_space_experiments_fixed_8/checkpoints_action_sym" "checkpoints/action_space_experiments_fixed_8/checkpoints_action_n0" "checkpoints/action_space_experiments_fixed_8/checkpoints_action_n1")

# Get current experiment parameters based on array index
exp_name="${exp_names[$SLURM_ARRAY_TASK_ID]}"
action_reduction="${action_reductions[$SLURM_ARRAY_TASK_ID]}"
use_reduced="${use_reduced_set[$SLURM_ARRAY_TASK_ID]}"
checkpoint_dir="${checkpoint_dirs[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "🧪 Starting Experiment: $exp_name"
echo "   Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "   Action Reduction: $action_reduction"
echo "   Use Reduced Set: $use_reduced"
echo "   Checkpoint Dir: $checkpoint_dir"
echo "   Time: $(date)"
echo "----------------------------------------"
# Create experiment-specific output directories
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/training_data_fresh/action_space_experiments/$(basename $checkpoint_dir)"

# Set experiment-specific environment variables
export REDUCTION=$action_reduction
export USE_REDUCED_ACTION_SPACE=$use_reduced
export CHECKPOINT_DIR=$checkpoint_dir
export PARALLEL_WORKERS=60
export training_data_dir="training_data_fresh/action_space_experiments/$(basename $checkpoint_dir)"

# Run training
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    ${CONTAINER_IMAGE} \
    bash -c "
        cd /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN && \
        export PYTHONPATH=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        export REDUCTION=$action_reduction && \
        export USE_REDUCED_ACTION_SPACE=$use_reduced && \
        export CHECKPOINT_DIR="/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir" && \
        export PARALLEL_WORKERS=65 && \
        export AGENT_DEBUG=false && \
        python -u scripts/train_alphazero_v2.py \
            2>&1
    "

echo "✅ Completed Experiment: $exp_name at $(date)"
echo "========================================="
echo "EXPERIMENT $SLURM_ARRAY_TASK_ID COMPLETED" 
echo "Finished at: $(date)"
echo "======================================="