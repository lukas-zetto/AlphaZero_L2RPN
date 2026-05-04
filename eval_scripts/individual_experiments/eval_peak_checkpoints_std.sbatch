#!/bin/bash
#SBATCH --job-name=eval-peak-std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --output=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_peak_checkpoints_std_%j.out
#SBATCH --error=/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/eval_peak_checkpoints_std_%j.err

set -euo pipefail

echo "Starting focused peak-checkpoint evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

export PATH=/usr/local/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export PYTHONUNBUFFERED=1
export AGENT_DEBUG=false

ROOT_DIR="/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN"
OUT_DIR="$ROOT_DIR/eval/peak_checkpoint_std"
mkdir -p "$ROOT_DIR/logs/eval"
mkdir -p "$OUT_DIR"

enroot start --root --rw --mount /home/ka/ka_iai/ka_jp7970:/work dqn_idf_2023 bash -lc '
set -euo pipefail

cd /work/PdF_L2RPN/AlphaZero_L2RPN
export PYTHONPATH="/workspace/src:/workspace:/work/PdF_L2RPN/AlphaZero_L2RPN/src:${PYTHONPATH:-}"

if [ ! -d "/workspace" ]; then
    mkdir -p /workspace
    ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN /workspace/AlphaZero_L2RPN
    ln -sfn /work /workspace
    ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/src /workspace/src
    mkdir -p /workspace/logs
    ln -sfn /work/PdF_L2RPN/AlphaZero_L2RPN/logs/eval /workspace/logs/
fi

export DATA_DIR="/work/data_grid2op"

OUT_DIR="/work/PdF_L2RPN/AlphaZero_L2RPN/eval/peak_checkpoint_std"
mkdir -p "$OUT_DIR"

STATS_CSV="$OUT_DIR/peak_checkpoint_std_stats.csv"
echo "label,checkpoint_path,episodes,avg_steps,std_steps,min_steps,max_steps,survived_episodes,survival_rate" > "$STATS_CSV"

evaluate_one() {
    local label="$1"
    local checkpoint_path="$2"

    echo "----------------------------------------"
    echo "Evaluating $label"
    echo "Checkpoint: $checkpoint_path"

    export MODEL_PATH="$checkpoint_path"
    export OBS_SPACE_TYPE="minimal"

    python3 scripts/evaluate_agent.py

    local cmp_json="/work/PdF_L2RPN/AlphaZero_L2RPN/comparison_results_l2rpn_case14_sandbox.json"
    local out_json="$OUT_DIR/${label}_comparison_results.json"
    cp "$cmp_json" "$out_json"

    python3 - "$label" "$checkpoint_path" "$out_json" "$STATS_CSV" <<'PY'
import csv
import json
import sys

label, checkpoint_path, json_path, csv_path = sys.argv[1:5]
with open(json_path, "r") as f:
    data = json.load(f)

stats = data["custom_agent"]["statistics"]
row = {
    "label": label,
    "checkpoint_path": checkpoint_path,
    "episodes": stats.get("episodes"),
    "avg_steps": stats.get("avg_steps"),
    "std_steps": stats.get("std_steps"),
    "min_steps": stats.get("min_steps"),
    "max_steps": stats.get("max_steps"),
    "survived_episodes": stats.get("survived_episodes"),
    "survival_rate": stats.get("survival_rate"),
}

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label",
            "checkpoint_path",
            "episodes",
            "avg_steps",
            "std_steps",
            "min_steps",
            "max_steps",
            "survived_episodes",
            "survival_rate",
        ],
    )
    writer.writerow(row)

print(
    f"Saved {label}: avg_steps={row['avg_steps']}, std_steps={row['std_steps']}, "
    f"survived={row['survived_episodes']}/{row['episodes']}"
)
PY
}

evaluate_one \
  "action_space_N1_peak_run6_ckpt17" \
  "/work/PdF_L2RPN/AlphaZero_L2RPN/checkpoints/action_space_experiments_fixed_6/checkpoints_action_n1/alphazero_v2_train17.pt"

evaluate_one \
  "reward_D3QN2020_peak_run8_ckpt32" \
  "/work/PdF_L2RPN/AlphaZero_L2RPN/checkpoints/reward_experimentsfixed_8/checkpoints_reward_d3qn2020/alphazero_v2_train32.pt"

echo "Done. Summary CSV: $STATS_CSV"
'

echo "Job completed at: $(date)"
echo "Results will be in: $OUT_DIR"
