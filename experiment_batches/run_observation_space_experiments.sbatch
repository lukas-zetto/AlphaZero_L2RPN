#!/bin/bash
#SBATCH --job-name=exp_obs_space
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/observation_experimentsfixed13_8/exp_obs_space_%A_%a.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/observation_experimentsfixed13_8/exp_obs_space_%A_%a.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/observation_experimentsfixed13_8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "========================================="
echo "EXPERIMENT SUITE: Observation Space Comparison - RUN _8"
echo "🎲 Fresh Seeds: main=1415, chronic=359"
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
exp_names=("Minimal Obs (rho only)" "Essential Obs (power+topo)" "Custom Obs (full features)" "Gym Obs (traditional RL)")
obs_space_types=("minimal" "essential" "custom" "gym")
checkpoint_dirs=("checkpoints/observation_experimentsfixed13_8/checkpoints_obs_minimal" "checkpoints/observation_experimentsfixed13_8/checkpoints_obs_essential" "checkpoints/observation_experimentsfixed13_8/checkpoints_obs_custom" "checkpoints/observation_experimentsfixed13_8/checkpoints_obs_gym")

# Get current experiment parameters based on array index
exp_name="${exp_names[$SLURM_ARRAY_TASK_ID]}"
obs_space_type="${obs_space_types[$SLURM_ARRAY_TASK_ID]}"
checkpoint_dir="${checkpoint_dirs[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "🧪 Starting Experiment: $exp_name"
echo "   Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "   Observation Space Type: $obs_space_type"
echo "   Checkpoint Dir: $checkpoint_dir"
echo "   Time: $(date)"
echo "----------------------------------------"

# Create experiment-specific output directories
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/training_data_fresh/observation_experiments/$(basename $checkpoint_dir)"

# Set experiment-specific environment variables
export OBS_SPACE_TYPE=$obs_space_type
export CHECKPOINT_DIR=$checkpoint_dir
export training_data_dir="training_data_fresh/$checkpoint_dir"

# Run training
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    ${CONTAINER_IMAGE} \
    bash -c "
        cd /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN && \
        export PYTHONPATH=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        export CHECKPOINT_DIR=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir && \
        export PARALLEL_WORKERS=65 && \
        export AGENT_DEBUG=false && \
        export training_data_dir=\"training_data_fresh/observation_experiments/$(basename $checkpoint_dir)\" && \
        python -u scripts/train_alphazero_v2.py \
            --obs_space_type $obs_space_type \
            2>&1
    "

echo "✅ Completed Experiment: $exp_name at $(date)"
echo "========================================="
echo "EXPERIMENT $SLURM_ARRAY_TASK_ID COMPLETED"
echo "Finished at: $(date)"
echo "========================================"