#!/bin/bash
#SBATCH --job-name=exp_all_suites
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/logs/exp_all_suites_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/logs/exp_all_suites_%j.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "========================================="
echo "MASTER EXPERIMENT SUITE: All Configurations"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
echo "Expected Duration: ~7 days"
echo "========================================="

# Container setup
CONTAINER_IMAGE="/home/ka/ka_iai/ka_jp7970/teacher_container.sqsh"

# Reduced configuration for faster execution across all experiments
export num_cycles=2
export episodes_per_iteration=25
export num_iterations=20

# Function to run a comprehensive experiment
run_comprehensive_experiment() {
    local exp_name="$1"
    local obs_space="$2"
    local batch_size="$3"
    local use_reduced_actions="$4"
    local reduction="$5"
    local reconnection="$6"
    local topology_reset="$7"
    local checkpoint_dir="$8"
    
    echo ""
    echo "🎆 Starting Comprehensive Experiment: $exp_name"
    echo "   Observation Space: $obs_space"
    echo "   Batch Size: $batch_size"
    echo "   Reduced Actions: $use_reduced_actions"
    echo "   Reduction: $reduction"
    echo "   Reconnection: $reconnection"
    echo "   Topology Reset: $topology_reset"
    echo "   Checkpoint Dir: $checkpoint_dir"
    echo "   Time: $(date)"
    echo "========================================"
    
    # Set all experiment parameters
    export obs_space_type=$obs_space
    export batch_size=$batch_size
    export use_reduced_action_space=$use_reduced_actions
    export reduction=$reduction
    export reconnection_module_enabled=$reconnection
    export topology_reset_module_enabled=$topology_reset
    export disconnect_module_enabled="False"  # Keep disconnect off for most experiments
    export auto_reconnect=$reconnection
    export checkpoint_dir=$checkpoint_dir
    export gym_obs_normalize="True"
    export drop_identity="False"
    export reduced_actions="[0, 9, 17, 22, 26, 29, 54, 59]"
    
    # Run training
    enroot start --rw \
        --mount /home/ka/ka_iai/ka_jp7970:/workspace \
        --mount /home/ka/ka_iai/ka_jp7970/grid2op_data:/root/data_grid2op \
        ${CONTAINER_IMAGE} \
        bash -c "
            cd /workspace/PdF_L2RPN/queno-project-download/scripts && \
            export PYTHONPATH=/workspace/PdF_L2RPN/queno-project-download/src:\$PYTHONPATH && \
            python -u train_alphazero_v2.py \
                --method mcts_all_nodes \
                --num_iterations $num_iterations \
                --episodes_per_iteration $episodes_per_iteration \
                2>&1
        "
    
    echo "✅ Completed: $exp_name at $(date)"
    echo ""
}

# Run systematic experiments covering key combinations
echo "
🔬 PHASE 1: BASELINE EXPERIMENTS"
run_comprehensive_experiment "Minimal-Fast" "minimal" 32 "True" "N1" "True" "False" "checkpoints_minimal_fast"
run_comprehensive_experiment "Custom-Full" "custom" 64 "False" "N1" "True" "True" "checkpoints_custom_full"

echo "
🔬 PHASE 2: OBSERVATION SPACE COMPARISON"
run_comprehensive_experiment "Minimal-Obs" "minimal" 64 "False" "N1" "True" "False" "checkpoints_obs_minimal_sys"
run_comprehensive_experiment "Gym-Obs" "gym" 64 "False" "N1" "True" "False" "checkpoints_obs_gym_sys"
run_comprehensive_experiment "Custom-Obs" "custom" 64 "False" "N1" "True" "False" "checkpoints_obs_custom_sys"

echo "
🔬 PHASE 3: ACTION SPACE COMPARISON"
run_comprehensive_experiment "Reduced-Actions" "minimal" 64 "True" "N1" "True" "False" "checkpoints_act_reduced_sys"
run_comprehensive_experiment "N0-Reduction" "minimal" 64 "False" "N0" "True" "False" "checkpoints_act_n0_sys"
run_comprehensive_experiment "SYM-Reduction" "minimal" 64 "False" "SYM" "True" "False" "checkpoints_act_sym_sys"

echo "
🔬 PHASE 4: BATCH SIZE COMPARISON"
run_comprehensive_experiment "Batch-32" "minimal" 32 "False" "N1" "True" "False" "checkpoints_batch32_sys"
run_comprehensive_experiment "Batch-128" "minimal" 128 "False" "N1" "True" "False" "checkpoints_batch128_sys"

echo "
🔬 PHASE 5: TOPOLOGY RECOVERY COMPARISON"
run_comprehensive_experiment "No-Recovery" "minimal" 64 "False" "N1" "False" "False" "checkpoints_no_recovery"
run_comprehensive_experiment "Full-Recovery" "minimal" 64 "False" "N1" "True" "True" "checkpoints_full_recovery"

echo "
🔬 PHASE 6: BEST COMBINATION EXPERIMENTS"
run_comprehensive_experiment "Optimal-Config" "minimal" 64 "True" "N1" "True" "True" "checkpoints_optimal"
run_comprehensive_experiment "Heavy-Config" "custom" 128 "False" "N1" "True" "True" "checkpoints_heavy"

echo "========================================="
echo "🎉 ALL EXPERIMENT SUITES COMPLETED! 🎉"
echo "Total Experiments Run: 14"
echo "Finished at: $(date)"
echo "========================================="
echo ""
echo "📊 EXPERIMENT SUMMARY:"
echo "• Phase 1: Baseline experiments (2)"
echo "• Phase 2: Observation space comparison (3)"
echo "• Phase 3: Action space comparison (3)"
echo "• Phase 4: Batch size comparison (2)"
echo "• Phase 5: Topology recovery comparison (2)"
echo "• Phase 6: Best combination experiments (2)"
echo ""
echo "📁 Checkpoint directories created:"
echo "   checkpoints_minimal_fast/"
echo "   checkpoints_custom_full/"
echo "   checkpoints_obs_*/"
echo "   checkpoints_act_*/"
echo "   checkpoints_batch*/"
echo "   checkpoints_*_recovery/"
echo "   checkpoints_optimal/"
echo "   checkpoints_heavy/"
echo ""
echo "🔍 Check individual experiment logs for detailed results."