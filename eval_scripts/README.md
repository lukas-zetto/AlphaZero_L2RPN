# Comprehensive Experiment Evaluation System

This evaluation system automatically handles multiple experiments with multiple runs, creating organized eval directories and comparison plots.

## Directory Structure

The system expects this structure:
```
checkpoints/
├── experiment1/           # e.g., horizon_experiments
│   ├── run1/             # e.g., checkpoints_horizon10  
│   │   └── *.pt          # checkpoint files
│   ├── run2/             # e.g., checkpoints_horizon30
│   │   └── *.pt
│   └── ...
├── experiment2/          # e.g., mcts_experiments
│   ├── run1/             # e.g., checkpoints_mcts50
│   └── run2/             # e.g., checkpoints_mcts200
└── ...
```

Creates:
```
eval/
├── experiment1/
│   ├── run1/
│   │   ├── checkpoint_evaluations.json
│   │   └── checkpoint_evaluations.csv
│   ├── run2/
│   │   ├── checkpoint_evaluations.json  
│   │   └── checkpoint_evaluations.csv
│   ├── experiment1_comparison.png        # All runs combined
│   ├── experiment1_avg_rewards.png       # Reward comparison
│   ├── experiment1_avg_steps.png         # Steps comparison
│   ├── experiment1_survival_rates.png    # Survival comparison
│   └── experiment_summary.json
├── experiment2/
│   └── ... (same structure)
└── ...
```

## Usage

### 1. Evaluate All Experiments
```bash
sbatch eval_scripts/run_all_experiments_eval.sbatch
```
- Evaluates all experiments in parallel
- Creates comparison plots for each experiment
- Resource intensive but comprehensive

### 2. Evaluate Single Experiment
```bash
# Method 1: Set environment variable
sbatch --export=EXPERIMENT_NAME=horizon_experiments3 eval_scripts/run_single_experiment_eval.sbatch

# Method 2: Edit the script to change EXPERIMENT_NAME variable
```

### 3. Plot Only (No Re-evaluation)
```bash
sbatch eval_scripts/run_plots_only.sbatch
```
- Uses existing evaluation results
- Regenerates all comparison plots
- Fast, useful for updating plot styles

### 4. Manual Python Execution
```bash
# Evaluate all experiments
python3 scripts/evaluate_all_experiments.py

# Evaluate specific experiment
python3 scripts/evaluate_all_experiments.py --experiment horizon_experiments3

# Plot only from existing results
python3 scripts/evaluate_all_experiments.py --plot-only

# Custom directories
python3 scripts/evaluate_all_experiments.py \
    --checkpoints-dir /path/to/checkpoints \
    --eval-dir /path/to/eval
```

## Output Files

### Per Run:
- `checkpoint_evaluations.json` - Detailed results for each checkpoint
- `checkpoint_evaluations.csv` - Same data in CSV format

### Per Experiment:
- `experiment_comparison.png` - 3-panel comparison (reward, steps, survival)
- `experiment_avg_rewards.png` - Reward comparison only
- `experiment_avg_steps.png` - Steps comparison only  
- `experiment_survival_rates.png` - Survival rate comparison only
- `experiment_summary.json` - Statistics summary

## Key Features

1. **Automatic Discovery** - Finds all experiments and runs automatically
2. **Incremental Evaluation** - Skips already evaluated checkpoints
3. **Parallel Processing** - Evaluates multiple checkpoints simultaneously
4. **Comparison Plots** - Combines all runs from same experiment
5. **Robust Error Handling** - Continues if individual evaluations fail
6. **Multiple Output Formats** - JSON, CSV, and PNG outputs

## Current Experiments Detected

Based on your current checkpoints directory:
- action_space_experiments
- batch_size_experiments  
- horizon_experiments
- horizon_experiments3
- mcts_experiments
- method_experiments
- observation_experiments
- reward_experiments

## Resource Requirements

- **Full evaluation**: 40 CPUs, 80GB RAM, 24h time limit
- **Single experiment**: 20 CPUs, 40GB RAM, 12h time limit  
- **Plotting only**: 4 CPUs, 8GB RAM, 2h time limit

## Monitoring Progress

Check logs in:
```
logs/eval/eval_all_experiments_<jobid>.out
logs/eval/eval_single_exp_<jobid>.out
logs/eval/plot_experiments_<jobid>.out
```

Monitor jobs with:
```bash
squeue -u $USER
```

## Troubleshooting

1. **No experiments found**: Check checkpoints directory structure
2. **Evaluation failures**: Check individual log files in `logs/eval/`
3. **Memory issues**: Reduce parallel workers in script
4. **Permission issues**: Ensure write access to eval directory

## Customization

Edit `scripts/evaluate_all_experiments.py` to:
- Change plot styles/colors
- Add new metrics
- Modify parallel processing limits
- Change output formats