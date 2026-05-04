# AlphaZero L2RPN Experiment Suite

This directory contains comprehensive experiment scripts for testing different configurations of the AlphaZero power grid control agent.

## 📋 Overview

The experiment suite systematically tests the following parameters:

- **Batch Sizes**: 16, 32, 64, 128
- **Observation Spaces**: minimal (20 features), custom (117 features), gym (traditional RL)
- **Action Spaces**: full, reduced, N0/N1/SYM reductions
- **Topology Recovery**: none, reconnection only, reset only, full recovery
- **Value Methods**: heuristic, mcts_root, mcts_all_nodes, binary_root, binary_all_nodes

## 🚀 Quick Start

### Submit All Experiments
```bash
# Submit all experiment suites as separate jobs
./submit_all_experiments.sh
```

### Submit Individual Experiment Suites
```bash
# Batch size comparison
sbatch run_batch_size_experiments.sbatch

# Observation space comparison  
sbatch run_observation_space_experiments.sbatch

# Action space comparison
sbatch run_action_space_experiments.sbatch

# Topology recovery comparison
sbatch run_topology_recovery_experiments.sbatch

# Value method comparison
sbatch run_method_comparison_experiments.sbatch

# Comprehensive suite (all combinations)
sbatch run_all_experiments.sbatch
```

## 📊 Experiment Details

### 1. Batch Size Experiments (`run_batch_size_experiments.sbatch`)
- **Duration**: ~3 days
- **Configurations**: 4 different batch sizes
- **Other params**: Fixed to minimal obs, full actions, reconnection enabled
- **Checkpoints**: `checkpoints_batch{16,32,64,128}/`

### 2. Observation Space Experiments (`run_observation_space_experiments.sbatch`)
- **Duration**: ~3 days  
- **Configurations**: 4 different observation spaces
- **Other params**: Fixed batch size 64, full actions, reconnection enabled
- **Checkpoints**: `checkpoints_obs_{minimal,custom,gym,gym_unnorm}/`

### 3. Action Space Experiments (`run_action_space_experiments.sbatch`)
- **Duration**: ~3 days
- **Configurations**: 5 different action space setups
- **Other params**: Fixed minimal obs, batch size 64, reconnection enabled  
- **Checkpoints**: `checkpoints_act_{full,reduced,n0,sym,no_identity}/`

### 4. Topology Recovery Experiments (`run_topology_recovery_experiments.sbatch`)
- **Duration**: ~3 days
- **Configurations**: 5 different recovery module combinations
- **Other params**: Fixed minimal obs, batch size 64, full actions
- **Checkpoints**: `checkpoints_topo_{none,reconnect,reset,full,reco_reset}/`

### 5. Value Method Experiments (`run_method_comparison_experiments.sbatch`)
- **Duration**: ~2 days
- **Configurations**: 5 different value assignment methods
- **Other params**: Fixed minimal obs, batch size 64, full actions, reconnection enabled
- **Checkpoints**: `checkpoints_method_{heuristic,mcts_root,mcts_all,binary_root,binary_all}/`

### 6. Comprehensive Suite (`run_all_experiments.sbatch`)
- **Duration**: ~7 days
- **Configurations**: 14 systematic combinations covering all major parameter interactions
- **Phases**: Baseline → Obs Space → Action Space → Batch Size → Topology → Optimal
- **Checkpoints**: Various `checkpoints_*_sys/` directories

## 🔧 Configuration Parameters

All experiments support environment variable overrides:

### Core Training Parameters
```bash
export batch_size=64
export learning_rate=0.0001
export num_cycles=3
export episodes_per_iteration=50
export num_iterations=30
export mcts_simulations=400
```

### Observation Space Parameters  
```bash
export obs_space_type="minimal"     # "minimal", "custom", "gym"
export gym_obs_normalize="True"     # "True", "False"
```

### Action Space Parameters
```bash
export use_reduced_action_space="False"        # "True", "False"
export reduction="N1"                          # "N0", "N1", "SYM" 
export drop_identity="False"                   # "True", "False"
export reduced_actions="[0, 9, 17, 22, 26, 29, 54, 59]"
```

### Topology Recovery Parameters
```bash
export reconnection_module_enabled="True"     # "True", "False"
export topology_reset_module_enabled="True"   # "True", "False" 
export disconnect_module_enabled="False"      # "True", "False"
export auto_reconnect="True"                  # "True", "False"
```

### Advanced Parameters
```bash
export checkpoint_dir="checkpoints_experiment"
export gamma=0.95
export max_depth=25
export critical_threshold=0.98
export puct_c=1.624
export temperature=1.0
export dirichlet_alpha=0.15
export dirichlet_epsilon=0.05
```

## 📁 Output Structure

```
experiment_batches/
├── run_*.sbatch                    # Individual experiment scripts
├── submit_all_experiments.sh       # Master launcher
└── README.md                       # This file

logs/
├── exp_batch_sizes_*.out          # Batch size experiment logs
├── exp_obs_spaces_*.out           # Observation space experiment logs  
├── exp_act_spaces_*.out           # Action space experiment logs
├── exp_topology_*.out             # Topology recovery experiment logs
├── exp_heuristic_vs_mcts_*.out    # Method comparison experiment logs
└── exp_all_suites_*.out           # Comprehensive suite logs

checkpoints_*/                     # Model checkpoints for each experiment
├── model_checkpoint_*.pth         # Saved model weights
├── training_log.txt               # Training progress log
└── config.json                    # Experiment configuration
```

## 🔍 Monitoring Progress

### Check Job Status
```bash
# View all your jobs
squeue -u $USER

# Watch job progress  
watch squeue -u $USER

# Check specific job details
scontrol show job <job_id>
```

### Check Experiment Logs
```bash
# Tail latest output
tail -f logs/exp_*_<job_id>.out

# Check for errors
grep -i error logs/exp_*_<job_id>.err

# View experiment progress
grep -E "Starting|Completed|PHASE" logs/exp_all_suites_*.out
```

### Monitor Training Progress
```bash
# Check latest checkpoint creation
ls -la checkpoints_*/model_checkpoint_*.pth

# View training metrics (if logged)
grep -E "loss|reward|success" logs/exp_*_*.out
```

## ⚙️ Customization

### Modify Experiment Parameters
Edit the individual `.sbatch` files to change:
- Resource allocation (`--cpus-per-task`, `--mem`, `--time`)
- Experiment parameters (`num_cycles`, `episodes_per_iteration`)
- Configuration combinations

### Add New Experiments
1. Copy an existing `.sbatch` file
2. Modify the experiment function calls
3. Update job name and output paths
4. Add to `submit_all_experiments.sh`

### Change Container/Environment
Update the `CONTAINER_IMAGE` variable in each script:
```bash
CONTAINER_IMAGE="/path/to/your/container.sqsh"
```

## 🚨 Important Notes

1. **Resource Usage**: Each experiment suite uses 16-32 CPUs and 32-64G RAM
2. **Duration**: Individual suites: 2-3 days, Comprehensive: ~7 days  
3. **Storage**: Each experiment creates ~1-2GB of checkpoints and logs
4. **Dependencies**: Requires enroot container with Grid2Op, lightsim2grid, torch
5. **Reproducibility**: Uses fixed seeds for deterministic results

## 📈 Expected Results

After completion, you'll have systematic comparisons of:
- Training stability across different batch sizes
- Feature importance via observation space ablation  
- Action space complexity vs. performance tradeoffs
- Impact of topology recovery mechanisms
- Effectiveness of different value assignment methods

Use the checkpoint directories to:
- Load trained models for evaluation
- Continue training from specific configurations
- Analyze learning curves and convergence patterns
- Compare final performance across setups

## 🤝 Troubleshooting

**Container Issues**: Ensure the container path is correct and accessible
```bash
# Test container access
enroot list
enroot start --rw <container_name> echo "Container works"
```

**Memory Issues**: Reduce `batch_size` or `episodes_per_iteration` in experiments
```bash
export batch_size=32          # Reduce from 64
export episodes_per_iteration=25  # Reduce from 50
```

**Time Limits**: Extend wall time or reduce experiment scope
```bash
#SBATCH --time=96:00:00      # Extend to 4 days instead of 3
```

**Path Issues**: Verify all paths are absolute and accessible from compute nodes
```bash
# Check paths
ls -la /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/queno-project-download/scripts/
ls -la /home/ka/ka_iai/ka_jp7970/teacher_container.sqsh
```