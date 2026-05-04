#!/bin/bash
#SBATCH --job-name=eval-single-exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_single_exp_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_single_exp_%j.err

# Usage: sbatch --export=EXPERIMENT_NAME=horizon_experiments3 eval_scripts/run_single_experiment_eval.sbatch
# Or modify the EXPERIMENT_NAME variable below

# Set the experiment to evaluate (can be overridden with --export)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"horizon_experiments3"}

echo "Starting evaluation of experiment: $EXPERIMENT_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================"

export PATH=/usr/local/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval

enroot start --root --rw --mount /home/ka/ka_iai/ka_jp7970:/work dqn_idf_2023 bash -c "
    echo 'Inside container - setting up environment for experiment: $EXPERIMENT_NAME'
    echo 'Container started at: \$(date)'
    
    # Set up paths
    cd /work/PdF_L2RPN/AlphaZero_L2RPN
    export PYTHONPATH='/workspace/src:/workspace:/work/PdF_L2RPN/AlphaZero_L2RPN/src:\${PYTHONPATH}'
    
    # Create symlink for Grid2Op data if needed
    if [ ! -d '/workspace' ]; then
        echo 'Setting up workspace symlinks...'
        mkdir -p /workspace
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN /workspace/AlphaZero_L2RPN
        ln -sfn /work /workspace
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/src /workspace/src
        mkdir -p /workspace/logs
        ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/logs/eval /workspace/logs/
    fi
    
    # Set environment variables
    export DATA_DIR='/work/data_grid2op'
    export PYTHONUNBUFFERED=1
    export AGENT_DEBUG=false
    
    echo '========================================'
    echo 'Starting single experiment evaluation...'
    echo 'Experiment: $EXPERIMENT_NAME'
    echo 'Timestamp: \$(date)'
    echo 'Working directory: \$(pwd)'
    echo '========================================'
    
    # Run the evaluation for specific experiment
    python3 scripts/evaluate_all_experiments.py \
        --checkpoints-dir /work/PdF_L2RPN/AlphaZero_L2RPN/checkpoints \
        --eval-dir /work/PdF_L2RPN/AlphaZero_L2RPN/eval \
        --experiment '$EXPERIMENT_NAME'
    
    echo '========================================'
    echo 'Single experiment evaluation completed at: \$(date)'
    echo '========================================'
"

echo "Job completed at: $(date)"