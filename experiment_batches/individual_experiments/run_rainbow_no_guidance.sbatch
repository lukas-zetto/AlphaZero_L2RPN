#!/bin/bash
#SBATCH --job-name=rainbow_no_guidance
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=75
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/rainbow_experiments/rainbow_no_guidance_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/rainbow_experiments/rainbow_no_guidance_%j.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/rainbow_experiments

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "========================================="
echo "RAINBOW/FRANKENSTEIN RUN: No Guidance Method"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================="
echo ""
echo "🌈 Configuration Summary:"
echo "   Method: no_guidance (pure MCTS)"
echo "   Safety Parameters: t_skipped=50, t_stopping=10"
echo "   Topology Reset Threshold: 0.80"
echo "   Observation Space: minimal"
echo "   Action Space: SYM (symmetry reduction)"
echo "   Parallel Workers: 70"
echo "----------------------------------------"

# Container setup
CONTAINER_IMAGE="dqn_idf_2023"

# Base configuration
export num_cycles=3

# Rainbow experiment settings
checkpoint_dir="checkpoints/rainbow_experiments/checkpoints_rainbow_no_guidance"
training_data_subdir="rainbow_experiments/checkpoints_rainbow_no_guidance"

# Create experiment-specific output directories
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir"
mkdir -p "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/training_data_fresh/$training_data_subdir"

# Run training with combined best settings
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    ${CONTAINER_IMAGE} \
    bash -c "
        cd /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN && \
        export PYTHONPATH=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        export T_SKIPPED=50 && \
        export T_STOPPING=10 && \
        export TOPOLOGY_RESET_THRESHOLD=0.80 && \
        export REDUCTION=SYM && \
        export CHECKPOINT_DIR=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/$checkpoint_dir && \
        export PARALLEL_WORKERS=70 && \
        export AGENT_DEBUG=false && \
        export training_data_dir=\"training_data_fresh/$training_data_subdir\" && \
        python -u scripts/train_alphazero_v2.py \
            --method no_guidance \
            --obs_space_type minimal \
            2>&1
    "

echo "✅ Completed Rainbow No Guidance Experiment at $(date)"
echo "========================================="
echo "RAINBOW NO GUIDANCE COMPLETED"
echo "Finished at: $(date)"
echo "========================================"
