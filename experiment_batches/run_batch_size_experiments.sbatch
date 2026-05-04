#!/bin/bash
#SBATCH --job-name=exp_episode_batches
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=128G
#SBATCH --time=71:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/batch_size_experiments5_final/exp_episode_batches_%A_%a.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/batch_size_experiments5_final/exp_episode_batches_%A_%a.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/batch_size_experiments5_final

# Print job info
echo "========================================="
echo "EXPERIMENT SUITE: Episode Batch Size Comparison"
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
exp_names=("Small Episodes (50)" "Medium Episodes (150)" "Large Episodes (400)" "Very Large Episodes (903 - Original)" "Optimized Episodes (300)")
episodes_per_iter=(50 150 400 903 300)
train_every_values=(50 150 400 903 300)
replay_min_sizes=(150 450 1200 2709 903)
replay_buffer_sizes=(300 900 2400 5418 3612)
checkpoint_dirs=("checkpoints/batch_size_experiments5_final/checkpoints_episodes50" "checkpoints/batch_size_experiments5_final/checkpoints_episodes150" "checkpoints/batch_size_experiments5_final/checkpoints_episodes400" "checkpoints/batch_size_experiments5_final/checkpoints_episodes903" "checkpoints/batch_size_experiments5_final/checkpoints_episodes300")

# Get current experiment parameters based on array index
exp_name="${exp_names[$SLURM_ARRAY_TASK_ID]}"
episodes_per_iteration="${episodes_per_iter[$SLURM_ARRAY_TASK_ID]}"
train_every="${train_every_values[$SLURM_ARRAY_TASK_ID]}"
replay_min_size="${replay_min_sizes[$SLURM_ARRAY_TASK_ID]}"
replay_buffer_size="${replay_buffer_sizes[$SLURM_ARRAY_TASK_ID]}"
checkpoint_dir="${checkpoint_dirs[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "🧪 Starting Experiment: $exp_name"
echo "   Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "   Episodes per Iteration: $episodes_per_iteration"
echo "   Train Every: $train_every episodes"
echo "   Episodes after Buffer Full: $episodes_per_iteration"
echo "   Replay Buffer Min Size: $replay_min_size (3x)"
echo "   Replay Buffer Size: $replay_buffer_size (6x)"
echo "   Checkpoint Dir: $checkpoint_dir"
echo "   Time: $(date)"
echo "----------------------------------------"

# Create experiment-specific output directories
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/training_data_fresh/batch_size_experiments5_final/$(basename $checkpoint_dir)"

# Set experiment-specific environment variables
export training_data_dir="training_data_fresh/batch_size_experiments5_final/$(basename $checkpoint_dir)"
# Run training
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    ${CONTAINER_IMAGE} \
    bash -c "
        cd /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN && \
        export PYTHONPATH=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        export EPISODES_PER_ITERATION=$episodes_per_iteration && \
        export EPISODES_PER_ITERATION_AFTER_BUFFER_FULL=$episodes_per_iteration && \
        export TRAIN_EVERY=$train_every && \
        export PARALLEL_WORKERS=65 && \
        export CHECKPOINT_DIR="/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir" && \
        export REPLAY_BUFFER_MIN_SIZE=$replay_min_size && \
        export REPLAY_BUFFER_SIZE=$replay_buffer_size && \
        export AGENT_DEBUG=false && \
        export training_data_dir="training_data_fresh/batch_size_experiments5_final/$(basename $checkpoint_dir)" && \
        python -u scripts/train_alphazero_v2.py \
            2>&1
    "

echo "✅ Completed Experiment: $exp_name at $(date)"
echo "========================================="
echo "EXPERIMENT $SLURM_ARRAY_TASK_ID COMPLETED"
echo "Finished at: $(date)"
echo "========================================="