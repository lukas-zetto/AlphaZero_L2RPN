#!/usr/bin/env python3
"""
Summarize peak performance using best individual run per variant.

Simple approach:
1. For each variant, find the run with the highest individual peak
2. Use that run's maximum avg_steps as the variant's peak
3. Calculate std deviation from individual episode survival times at that peak
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import numpy as np


EXPERIMENTS = {
    "action_space": {
        "results_file": "action_space_multi_run_results/action_space_multi_run_results.json",
        "variant_order": ["SYM", "N0", "N1"],
    },
    "observation": {
        "results_file": "observation_multi_run_results/observation_multi_run_results.json",
        "variant_order": ["Minimal", "Essential", "Custom", "Gym"],
    },
    "reward": {
        "results_file": "reward_multi_run_results/reward_multi_run_results.json",
        "variant_order": ["AlphaZero", "D3QN-2022", "D3QN-2020", "Loss", "MaxRho", "PPO", "LinesCapacity"],
    },
    "method": {
        "results_file": "method_multi_run_results/complete_method_results.json",
        "variant_order": ["MCTS_Root", "Heuristic", "No_Guidance"],
    },
}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    eval_dir = script_dir / "eval"
    output_dir = eval_dir / "best_run_stats"

    parser = argparse.ArgumentParser(
        description="Summarize peak performance using the best individual run per variant."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=eval_dir,
        help="Base eval directory containing the multi-run result folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=output_dir / "best_run_peaks.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=output_dir / "best_run_peaks.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--max-training-steps",
        type=float,
        default=None,
        help="Optional cap in millions of training steps when selecting peaks.",
    )
    return parser.parse_args()


def load_results(results_file):
    with open(results_file, "r") as handle:
        return json.load(handle)


def allowed_result(result, max_training_steps_millions):
    if not result.get("success", True):
        return False
    if max_training_steps_millions is None:
        return True

    training_steps = result.get("training_steps")
    if training_steps is None:
        # If training_steps is missing, we can't apply the cap, so allow it
        return True
    try:
        return float(training_steps) / 1e6 <= max_training_steps_millions
    except (TypeError, ValueError):
        return True


def numeric_or_none(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def calculate_variant_peak_stats(variant_all_run_data, max_training_steps_millions):
    """
    Calculate average and standard deviation of peak performances across all runs for this variant.
    
    Args:
        variant_all_run_data: Dict mapping run_name -> list of results for this variant
        max_training_steps_millions: Optional cap on training steps
    
    Returns:
        Dict with averaged peak performance statistics across runs
    """
    if not variant_all_run_data:
        return None
    
    # Find peak for each run
    run_peaks = {}
    run_peak_details = {}
    
    for run_name, run_results in variant_all_run_data.items():
        best_result = None
        best_avg_steps = 0
        
        for result in run_results:
            if not allowed_result(result, max_training_steps_millions):
                continue
            
            avg_steps = numeric_or_none(result.get("avg_steps"))
            if avg_steps is None:
                continue
                
            if avg_steps > best_avg_steps:
                best_avg_steps = avg_steps
                best_result = result
        
        if best_result is not None:
            run_peaks[run_name] = best_avg_steps
            run_peak_details[run_name] = {
                "peak_avg_steps": best_avg_steps,
                "result": best_result
            }
    
    if not run_peaks:
        return None
    
    # Calculate statistics across run peaks
    peak_values = list(run_peaks.values())
    avg_peak = mean(peak_values)
    std_peak = pstdev(peak_values) if len(peak_values) > 1 else 0.0
    
    # Find the run with the highest peak for additional details
    best_run_name = max(run_peaks.keys(), key=lambda r: run_peaks[r])
    best_run_detail = run_peak_details[best_run_name]
    best_result = best_run_detail["result"]
    
    return {
        "avg_peak_across_runs": avg_peak,
        "std_peak_across_runs": std_peak,
        "best_run_name": best_run_name,
        "best_run_peak": run_peaks[best_run_name],
        "checkpoint_num": best_result.get("checkpoint_num"),
        "training_steps": numeric_or_none(best_result.get("training_steps")),
        "checkpoint_path": best_result.get("checkpoint_path"),
        "total_episodes": best_result.get("total_episodes", 0),
        "survived_episodes": best_result.get("survived_episodes", 0),
        "all_run_peaks": run_peaks,
        "run_count": len(run_peaks)
    }


def ordered_variants(experiment_name, all_results):
    preferred = EXPERIMENTS[experiment_name]["variant_order"]
    present = set()
    for run_data in all_results.values():
        present.update(run_data.keys())

    ordered = [variant for variant in preferred if variant in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def summarize_experiment(experiment_name, all_results, max_training_steps_millions):
    variants = ordered_variants(experiment_name, all_results)
    variant_runs = {
        variant: sorted(run_name for run_name, run_data in all_results.items() if run_data.get(variant))
        for variant in variants
    }

    summaries = []

    for variant in variants:
        # Collect all run data for this variant 
        variant_all_run_data = {}
        for run_name in variant_runs[variant]:
            if variant in all_results[run_name]:
                variant_all_run_data[run_name] = all_results[run_name][variant]
        
        # Calculate peak statistics across runs for this variant
        peak_stats = calculate_variant_peak_stats(variant_all_run_data, max_training_steps_millions)
        if peak_stats is None:
            continue

        summaries.append({
            "experiment": experiment_name,
            "variant": variant,
            "avg_peak_across_runs": peak_stats["avg_peak_across_runs"],
            "std_peak_across_runs": peak_stats["std_peak_across_runs"],
            "best_run_name": peak_stats["best_run_name"],
            "best_run_peak": peak_stats["best_run_peak"],
            "checkpoint_num": peak_stats["checkpoint_num"],
            "training_steps": peak_stats["training_steps"],
            "checkpoint_path": peak_stats["checkpoint_path"],
            "total_episodes": peak_stats["total_episodes"],
            "survived_episodes": peak_stats["survived_episodes"],
            "survival_rate_percent": 100.0 * peak_stats["survived_episodes"] / max(1, peak_stats["total_episodes"]),
            "all_run_peaks": peak_stats["all_run_peaks"],
            "run_count": peak_stats["run_count"]
        })

    return summaries


def format_steps(value):
    return f"{value:.2f}"


def write_csv(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "variant",
        "avg_peak_across_runs",
        "std_peak_across_runs",
        "best_run_name",
        "best_run_peak",
        "survival_rate_percent",
        "checkpoint_num",
        "training_steps",
        "checkpoint_path",
        "all_run_peaks",
        "run_count"
    ]

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "experiment": row["experiment"],
                "variant": row["variant"],
                "avg_peak_across_runs": f"{row['avg_peak_across_runs']:.6f}",
                "std_peak_across_runs": f"{row['std_peak_across_runs']:.6f}",
                "best_run_name": row["best_run_name"],
                "best_run_peak": f"{row['best_run_peak']:.6f}",
                "survival_rate_percent": f"{row['survival_rate_percent']:.2f}",
                "checkpoint_num": row["checkpoint_num"] or "",
                "training_steps": f"{row['training_steps']:.0f}" if row['training_steps'] is not None else "",
                "checkpoint_path": row["checkpoint_path"] or "",
                "all_run_peaks": json.dumps(row["all_run_peaks"], sort_keys=True),
                "run_count": row["run_count"]
            })


def write_json(rows, output_json, max_training_steps_millions):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "max_training_steps_millions": max_training_steps_millions,
        "summary": rows,
    }
    with open(output_json, "w") as handle:
        json.dump(payload, handle, indent=2)


def print_summary(rows):
    current_experiment = None
    for row in rows:
        if row["experiment"] != current_experiment:
            current_experiment = row["experiment"]
            print(f"\n[{current_experiment}]")

        # Format training steps info
        if row["training_steps"] is not None:
            training_info = f" at {row['training_steps']/1e6:.1f}M steps"
        else:
            training_info = f" at checkpoint {row['checkpoint_num']}"

        per_run_list = ", ".join(
            f"{run}={format_steps(peak)}"
            for run, peak in sorted(row["all_run_peaks"].items())
        )
        
        print(f"- {row['variant']}:")
        print(f"    📈 Avg peak across runs: {format_steps(row['avg_peak_across_runs'])} ± {format_steps(row['std_peak_across_runs'])} avg steps")
        print(f"    🥇 Best run: {row['best_run_name']} with {format_steps(row['best_run_peak'])} avg steps{training_info}")
        print(f"    💯 Survival rate: {row['survival_rate_percent']:.1f}% ({row['survived_episodes']}/{row['total_episodes']} episodes)")
        print(f"    📊 All run peaks: [{per_run_list}]")


def main():
    args = parse_args()

    rows = []
    for experiment_name, config in EXPERIMENTS.items():
        results_file = args.eval_dir / config["results_file"]
        if not results_file.exists():
            print(f"Skipping {experiment_name}: missing {results_file}")
            continue

        all_results = load_results(results_file)
        rows.extend(summarize_experiment(experiment_name, all_results, args.max_training_steps))

    rows.sort(key=lambda row: (row["experiment"], row["variant"]))

    print_summary(rows)
    write_csv(rows, args.output_csv)
    write_json(rows, args.output_json, args.max_training_steps)
    
    print(f"\nCSV saved to {args.output_csv}")
    print(f"JSON saved to {args.output_json}")


if __name__ == "__main__":
    main()