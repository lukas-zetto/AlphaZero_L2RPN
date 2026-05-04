# Individual Experiment Evaluation Scripts

This directory contains dedicated sbatch scripts for evaluating each experiment individually.

## Available Experiments

| Script | Experiment | Resources | Time Limit |
|--------|------------|-----------|------------|
| `eval_horizon_experiments.sbatch` | Horror value horizons | 20 CPUs, 40GB | 12h |
| `eval_horizon_experiments3.sbatch` | Horizon experiments v3 | 15 CPUs, 30GB | 8h |
| `eval_mcts_experiments.sbatch` | MCTS simulation counts | 25 CPUs, 50GB | 15h |
| `eval_method_experiments.sbatch` | Training methods | 20 CPUs, 40GB | 12h |
| `eval_observation_experiments.sbatch` | Observation spaces | 20 CPUs, 40GB | 12h |
| `eval_reward_experiments.sbatch` | Reward functions | 25 CPUs, 50GB | 15h |
| `eval_action_space_experiments.sbatch` | Action spaces | 20 CPUs, 40GB | 12h |
| `eval_batch_size_experiments.sbatch` | Batch sizes | 25 CPUs, 50GB | 15h |

## Usage

### Submit Individual Experiment
```bash
# From the project root directory
sbatch eval_scripts/individual_experiments/eval_horizon_experiments.sbatch
sbatch eval_scripts/individual_experiments/eval_mcts_experiments.sbatch
# ... etc
```

### Submit Multiple Experiments
```bash
# Submit several experiments at once
sbatch eval_scripts/individual_experiments/eval_horizon_experiments.sbatch
sbatch eval_scripts/individual_experiments/eval_mcts_experiments.sbatch
sbatch eval_scripts/individual_experiments/eval_method_experiments.sbatch
```

### Monitor Jobs
```bash
squeue -u $USER
```

## Output Locations

Each script will create results in:
```
eval/{experiment_name}/
├── {run_name}/
│   ├── checkpoint_evaluations.json
│   └── checkpoint_evaluations.csv
├── {experiment_name}_comparison.png
├── {experiment_name}_avg_rewards.png
├── {experiment_name}_avg_steps.png
├── {experiment_name}_survival_rates.png
└── experiment_summary.json
```

## Advantages of Individual Scripts

1. **Targeted Evaluation** - Only evaluate the experiment you're interested in
2. **Optimized Resources** - Each script uses appropriate CPU/memory for that experiment
3. **Parallel Execution** - Can submit multiple experiments simultaneously
4. **Easy Management** - Simple filenames make it clear what each script does
5. **No Parameters** - No need to remember experiment names or use environment variables

## Resource Notes

- **Smaller experiments** (horizon_experiments3): Lighter resources
- **Larger experiments** (mcts_experiments, reward_experiments): More resources
- **Time limits** are conservative estimates based on typical evaluation times

## Logs

Check evaluation progress in:
```
logs/eval/eval_{experiment_name}_{job_id}.out
logs/eval/eval_{experiment_name}_{job_id}.err
```