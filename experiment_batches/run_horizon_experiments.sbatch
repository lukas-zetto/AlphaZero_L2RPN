#!/bin/bash
#SBATCH --job-name=h_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=75
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/horizon_experiments3_fresh/exp_horizons_%A_%a.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/horizon_experiments3_fresh/exp_horizons_%A_%a.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/horizon_experiments3_fresh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "========================================="
echo "EXPERIMENT SUITE: Heuristic Value Horizon Comparison"
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
exp_names=("Horizon 10 (Short)" "Horizon 30 (Medium)" "Horizon 100 (Long)" "Horizon 200 (Extended)" "Horizon 500 (Very Long)")
heuristic_horizons=(10 30 100 200 500)
checkpoint_dirs=("checkpoints/horizon_experiments3_fresh/checkpoints_horizon10" "checkpoints/horizon_experiments3_fresh/checkpoints_horizon30" "checkpoints/horizon_experiments3_fresh/checkpoints_horizon100" "checkpoints/horizon_experiments3_fresh/checkpoints_horizon200" "checkpoints/horizon_experiments3_fresh/checkpoints_horizon500")

# Get current experiment parameters based on array index
exp_name="${exp_names[$SLURM_ARRAY_TASK_ID]}"
heuristic_horizon="${heuristic_horizons[$SLURM_ARRAY_TASK_ID]}"
checkpoint_dir="${checkpoint_dirs[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "🧪 Starting Experiment: $exp_name"
echo "   Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "   Heuristic Value Horizon: $heuristic_horizon"
echo "   Checkpoint Dir: $checkpoint_dir"
echo "   Time: $(date)"
echo "----------------------------------------"

# Create experiment-specific output directories
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/training_data_fresh/horizon_experiments3/$(basename $checkpoint_dir)"

# Set experiment-specific environment variables
export HEURISTIC_VALUE_HORIZON=$heuristic_horizon
export CHECKPOINT_DIR=$checkpoint_dir
export training_data_dir="training_data_fresh/horizon_experiments3/$(basename $checkpoint_dir)"

# Run training
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    ${CONTAINER_IMAGE} \
    bash -c "
        cd /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN && \
        export PYTHONPATH=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        export HEURISTIC_VALUE_HORIZON=$heuristic_horizon && \
        export CHECKPOINT_DIR=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir && \
        export PARALLEL_WORKERS=70 && \
        export AGENT_DEBUG=false && \
        export training_data_dir=\"training_data_fresh/horizon_experiments3/$(basename $checkpoint_dir)\" && \
        python -u scripts/train_alphazero_v2.py \
            --method heuristic \
            2>&1
    "

echo "✅ Completed Experiment: $exp_name at $(date)"
echo "========================================="
echo "EXPERIMENT $SLURM_ARRAY_TASK_ID COMPLETED"
echo "Finished at: $(date)"
echo "========================================"