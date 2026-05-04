#!/bin/bash
#SBATCH --job-name=test_config
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/test_config_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/test_config_%j.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "========================================="
echo "TEST: Configuration Fix Verification"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Container setup
CONTAINER_IMAGE="dqn_idf_2023"

# Simple test configuration
export num_cycles=1
export max_training_iterations=1
export episodes_per_iteration=2

# Test with minimal observation space
obs_space_type="minimal"
checkpoint_dir="checkpoints/test_config_fix"

echo ""
echo "🧪 Testing Configuration Fix"
echo "   Observation Space Type: $obs_space_type"
echo "   Checkpoint Dir: $checkpoint_dir"
echo "   Parallel Workers: 1"
echo "   Time: $(date)"
echo "----------------------------------------"

# Create test output directories
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/training_data_fresh/test_config"

# Set test environment variables
export OBS_SPACE_TYPE=$obs_space_type
export CHECKPOINT_DIR=$checkpoint_dir
export training_data_dir="training_data_fresh/test_config"
export debug_fast_training=true

# Run minimal training test
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    ${CONTAINER_IMAGE} \
    bash -c "
        cd /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN && \
        export PYTHONPATH=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        export CHECKPOINT_DIR=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir && \
        export PARALLEL_WORKERS=1 && \
        export AGENT_DEBUG=true && \
        export training_data_dir=\"training_data_fresh/test_config\" && \
        export OBS_SPACE_TYPE=$obs_space_type && \
        export num_cycles=1 && \
        export max_training_iterations=1 && \
        export episodes_per_iteration=2 && \
        export debug_fast_training=true && \
        echo '🔍 DEBUG: Environment variables before training:' && \
        echo \"  OBS_SPACE_TYPE=\$OBS_SPACE_TYPE\" && \
        echo \"  PARALLEL_WORKERS=\$PARALLEL_WORKERS\" && \
        echo \"  CHECKPOINT_DIR=\$CHECKPOINT_DIR\" && \
        echo \"  num_cycles=\$num_cycles\" && \
        echo \"  max_training_iterations=\$max_training_iterations\" && \
        echo \"  episodes_per_iteration=\$episodes_per_iteration\" && \
        echo \"  debug_fast_training=\$debug_fast_training\" && \
        python -u scripts/train_alphazero_v2.py \
            2>&1
    "

echo "✅ Completed Configuration Test at $(date)"
echo "========================================="