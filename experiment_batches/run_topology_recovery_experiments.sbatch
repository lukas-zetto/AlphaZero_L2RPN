#!/bin/bash
#SBATCH --job-name=exp_topology
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/queno-project-download/logs/exp_topology_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/queno-project-download/logs/exp_topology_%j.err

set -ex

# Create logs directory if it doesn't exist
mkdir -p /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/queno-project-download/logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "========================================="
echo "EXPERIMENT SUITE: Topology Recovery Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Container setup
CONTAINER_IMAGE="/home/ka/ka_iai/ka_jp7970/teacher_container.sqsh"

# Base configuration
export num_cycles=3
export episodes_per_iteration=50
export num_iterations=30
export batch_size=64
export obs_space_type="minimal"

# Function to run topology recovery experiment
run_topology_experiment() {
    local exp_name="$1"
    local reconnection_enabled="$2"
    local topology_reset_enabled="$3"
    local disconnect_enabled="$4"
    local auto_reconnect="$5"
    local checkpoint_dir="$6"
    
    echo ""
    echo "🧪 Starting Experiment: $exp_name"
    echo "   Reconnection Module: $reconnection_enabled"
    echo "   Topology Reset Module: $topology_reset_enabled"
    echo "   Disconnect Module: $disconnect_enabled"
    echo "   Auto Reconnect: $auto_reconnect"
    echo "   Checkpoint Dir: $checkpoint_dir"
    echo "   Time: $(date)"
    echo "----------------------------------------"
    
    # Set experiment-specific environment variables
    export reconnection_module_enabled=$reconnection_enabled
    export topology_reset_module_enabled=$topology_reset_enabled
    export disconnect_module_enabled=$disconnect_enabled
    export auto_reconnect=$auto_reconnect
    export checkpoint_dir=$checkpoint_dir
    
    # Run training
    enroot start --rw \
        --mount /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/queno-project-download:/workspace \
        --mount /home/ka/ka_iai/ka_jp7970/grid2op_data:/root/data_grid2op \
        ${CONTAINER_IMAGE} \
        bash -c "
            cd /workspace/scripts && \
            export PYTHONPATH=/workspace/src:\$PYTHONPATH && \
            python -u train_alphazero_v2.py \
                --method mcts_all_nodes \
                --num_iterations $num_iterations \
                --episodes_per_iteration $episodes_per_iteration \
                2>&1
        "
    
    echo "✅ Completed Experiment: $exp_name at $(date)"
    echo ""
}

# Run topology recovery experiments
run_topology_experiment "No Recovery (Baseline)" "False" "False" "False" "False" "checkpoints_topo_none"
run_topology_experiment "Reconnection Only" "True" "False" "False" "True" "checkpoints_topo_reconnect"
run_topology_experiment "Reset Only" "False" "True" "False" "False" "checkpoints_topo_reset"
run_topology_experiment "Full Recovery (All modules)" "True" "True" "True" "True" "checkpoints_topo_full"
run_topology_experiment "Reconnect + Reset (No Disconnect)" "True" "True" "False" "True" "checkpoints_topo_reco_reset"

echo "========================================="
echo "ALL TOPOLOGY RECOVERY EXPERIMENTS COMPLETED"
echo "Finished at: $(date)"
echo "========================================="