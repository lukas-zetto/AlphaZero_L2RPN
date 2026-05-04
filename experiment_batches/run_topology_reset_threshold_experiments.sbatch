#!/bin/bash
#SBATCH --job-name=exp_topo_reset
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=75
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/topology_reset_threshold_experiments_fresh/exp_topo_reset_%A_%a.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/topology_reset_threshold_experiments_fresh/exp_topo_reset_%A_%a.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/topology_reset_threshold_experiments_fresh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "========================================="
echo "EXPERIMENT SUITE: Topology Reset Threshold Comparison"
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
exp_names=("Topo Reset 0.75" "Topo Reset 0.80" "Topo Reset 0.85" "Topo Reset 0.90" "Topo Reset 0.95")
reset_thresholds=(0.75 0.80 0.85 0.90 0.95)
checkpoint_dirs=("checkpoints/topology_reset_threshold_experiments_fresh/checkpoints_topo_reset_075" "checkpoints/topology_reset_threshold_experiments_fresh/checkpoints_topo_reset_080" "checkpoints/topology_reset_threshold_experiments_fresh/checkpoints_topo_reset_085" "checkpoints/topology_reset_threshold_experiments_fresh/checkpoints_topo_reset_090" "checkpoints/topology_reset_threshold_experiments_fresh/checkpoints_topo_reset_095")

# Get current experiment parameters based on array index
exp_name="${exp_names[$SLURM_ARRAY_TASK_ID]}"
reset_threshold="${reset_thresholds[$SLURM_ARRAY_TASK_ID]}"
checkpoint_dir="${checkpoint_dirs[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "🧪 Starting Experiment: $exp_name"
echo "   Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "   Topology Reset Threshold: $reset_threshold"
echo "   Checkpoint Dir: $checkpoint_dir"
echo "   Time: $(date)"
echo "----------------------------------------"

# Create experiment-specific output directories
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/training_data_fresh/topology_reset_threshold_experiments/$(basename $checkpoint_dir)"

# Set experiment-specific environment variables
# This maps to get_config_value('topology_reset_threshold', ...) in scripts/train_alphazero_v2.py
export TOPOLOGY_RESET_THRESHOLD=$reset_threshold
export CHECKPOINT_DIR=$checkpoint_dir
export training_data_dir="training_data_fresh/topology_reset_threshold_experiments/$(basename $checkpoint_dir)"

# Run training
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    ${CONTAINER_IMAGE} \
    bash -c "
        cd /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN && \
        export PYTHONPATH=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        export TOPOLOGY_RESET_THRESHOLD=$reset_threshold && \
        export CHECKPOINT_DIR=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir && \
        export PARALLEL_WORKERS=70 && \
        export AGENT_DEBUG=false && \
        export training_data_dir=\"training_data_fresh/topology_reset_threshold_experiments/$(basename $checkpoint_dir)\" && \
        python -u scripts/train_alphazero_v2.py \
            2>&1
    "

echo "✅ Completed Experiment: $exp_name at $(date)"
echo "========================================="
echo "EXPERIMENT $SLURM_ARRAY_TASK_ID COMPLETED"
echo "Finished at: $(date)"
echo "========================================"