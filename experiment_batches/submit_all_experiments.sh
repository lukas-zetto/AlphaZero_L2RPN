#!/bin/bash

# Master Experiment Launcher for AlphaZero L2RPN Training
# This script submits all experiment suites as separate SLURM jobs

set -e

SCRIPT_DIR="/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/queno-project-download/experiment_batches"
LOG_DIR="/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/queno-project-download/logs"

# Ensure directories exist
mkdir -p "$LOG_DIR"

echo "🚀 AlphaZero L2RPN Experiment Launcher"
echo "======================================"
echo "$(date): Starting experiment submission"
echo ""

# Function to submit job and capture job ID
submit_job() {
    local script="$1"
    local description="$2"
    
    echo "📤 Submitting: $description"
    echo "   Script: $script"
    
    if [[ -f "$SCRIPT_DIR/$script" ]]; then
        job_id=$(sbatch "$SCRIPT_DIR/$script" | grep -oE '[0-9]+')
        echo "   Job ID: $job_id"
        echo "   Status: SUBMITTED ✅"
        return 0
    else
        echo "   Status: SCRIPT NOT FOUND ❌"
        return 1
    fi
    echo ""
}

# Submit all experiment suites
echo "🧪 SUBMITTING EXPERIMENT SUITES"
echo "--------------------------------"

# Individual experiment suites
submit_job "run_batch_size_experiments.sbatch" "Batch Size Comparison (16, 32, 64, 128)"
submit_job "run_observation_space_experiments.sbatch" "Observation Space Comparison (minimal, custom, gym)"
submit_job "run_action_space_experiments.sbatch" "Action Space Comparison (full, reduced, N0, SYM)"
submit_job "run_topology_recovery_experiments.sbatch" "Topology Recovery Comparison (none, reconnect, reset, full)"
submit_job "run_method_comparison_experiments.sbatch" "Value Method Comparison (heuristic, mcts_root, mcts_all, binary)"

# Comprehensive suite (if you want to run everything in one job)
echo ""
echo "🎯 COMPREHENSIVE EXPERIMENT SUITE"
echo "---------------------------------"
submit_job "run_all_experiments.sbatch" "All Parameter Combinations (comprehensive, ~7 days)"

echo ""
echo "📋 EXPERIMENT SUBMISSION SUMMARY"
echo "================================"
echo "✅ Individual suites: 5 jobs submitted"
echo "✅ Comprehensive suite: 1 job submitted"
echo "📊 Total jobs: 6"
echo ""
echo "🔍 Monitor job status with:"
echo "   squeue -u $USER"
echo ""
echo "📄 Check individual logs in: $LOG_DIR/"
echo "   exp_batch_sizes_*.out"
echo "   exp_obs_spaces_*.out" 
echo "   exp_act_spaces_*.out"
echo "   exp_topology_*.out"
echo "   exp_heuristic_vs_mcts_*.out"
echo "   exp_all_suites_*.out"
echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "   • Individual suites: ~3 days each"
echo "   • Comprehensive suite: ~7 days total"
echo "   • Results saved to checkpoints_*/ directories"
echo "   • Each experiment creates separate checkpoint dirs"
echo ""
echo "🎉 All experiments submitted successfully!"
echo "$(date): Submission completed"