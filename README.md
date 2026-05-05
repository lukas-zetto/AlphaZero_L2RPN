# AlphaZero L2RPN

AlphaZero MCTS agent for power grid topology optimization using the [Grid2Op](https://github.com/rte-france/Grid2Op) framework (l2rpn_case14_sandbox environment).

## Project Structure

```
AlphaZero_L2RPN/
├── src/                        # Core source code
│   ├── config.py               # Training configuration
│   ├── actions/                # Action space definitions
│   ├── agent/                  # Agent implementation
│   ├── networks/               # Neural network architectures
│   ├── rewards/                # Reward function implementations
│   └── training/               # AlphaZero MCTS training loop
├── scripts/                    # Training, evaluation, and plotting scripts
├── experiment_batches/         # SLURM sbatch files for running experiments
├── eval_scripts/               # SLURM sbatch files for evaluation
└── logs/                       # Log output directory
```

## Setup (HPC / SLURM with Enroot)

### Step 1: Import the container image

Submit the download job or run directly on a node:

```bash
sbatch download_container.sbatch
```

This runs:

```bash
module load pyxis enroot
enroot import -o ~/l2rpn_idf_2023.sqsh docker://bdonnot/l2rpn:idf.2023.4
```

### Step 2: Create the enroot container instance

```bash
enroot create --name dqn_idf_2023 ~/l2rpn_idf_2023.sqsh
```

You can use any name — just keep it consistent with the `CONTAINER_IMAGE` variable in the sbatch files.

### Step 3: Pre-download Grid2Op data (first time only)

```bash
enroot start --root --rw --mount $HOME:$HOME dqn_idf_2023 bash -c "
    python3 -c \"import grid2op; grid2op.make('l2rpn_case14_sandbox')\"
"
```

## Running Training

Each experiment has a corresponding sbatch file in `experiment_batches/`. For example, to run the horizon experiments:

```bash
sbatch experiment_batches/run_horizon_experiments.sbatch
```

The sbatch files launch training inside the container like this:

```bash
enroot start --root --rw \
    --mount "$HOME:$HOME" \
    dqn_idf_2023 \
    bash -c "
        cd /path/to/AlphaZero_L2RPN && \
        export PYTHONPATH=/path/to/AlphaZero_L2RPN/src:\$PYTHONPATH && \
        python -u scripts/train_alphazero_v2.py --method heuristic
    "
```

To submit all experiment suites at once:

```bash
bash experiment_batches/submit_all_experiments.sh
```

Monitor jobs with:

```bash
squeue -u $USER
```

## Running Evaluation

Evaluation sbatch files are in `eval_scripts/`. To evaluate all checkpoints for a specific experiment group:

```bash
sbatch eval_scripts/eval_all_action_space.sbatch
```

Or run evaluation and plotting directly (inside the container):

```bash
enroot start --root --rw --mount $HOME:$HOME dqn_idf_2023 bash -c "
    cd /path/to/AlphaZero_L2RPN
    export PYTHONPATH=/path/to/AlphaZero_L2RPN/src:\$PYTHONPATH
    python3 scripts/evaluate_all_experiments.py \
        --checkpoints-dir checkpoints \
        --eval-dir eval \
        --experiment horizon_experiments3_fresh
"
```
