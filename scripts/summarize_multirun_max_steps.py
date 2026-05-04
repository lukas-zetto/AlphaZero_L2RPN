#!/usr/bin/env python3
"""
Summarize max average steps survived for each multi-run experiment variant.

Modified to calculate maximum of averaged line and standard deviation across multiple runs:
1. Interpolate all runs to common time axis (training steps)  
2. Calculate mean and std across runs at each time point (the "averaged line" and "std line")
3. Find maximum of the averaged line and provide std line statistics
4. Also provides individual run maxima statistics for comparison
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import numpy as np
from scipy import interpolate


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
    output_dir = eval_dir / "multi_run_stats"

    parser = argparse.ArgumentParser(
        description="Summarize maximum of averaged line and standard deviation line across multi-run experiments plus individual run statistics."
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
        default=output_dir / "max_avg_steps_summary.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=output_dir / "max_avg_steps_summary.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--max-training-steps",
        type=float,
        default=None,
        help="Optional cap in millions of training steps when selecting maxima.",
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


def summarize_variant_averaged_line(all_run_data, max_training_steps_millions, max_steps_cap=40.0):
    """
    Calculate maximum of averaged line and standard deviation line across multiple runs for a single variant.
    Uses UNION of all run time ranges (earliest start to latest end) to capture all peaks.
    
    Args:
        all_run_data: Dict mapping run_name -> list of results for this variant
        max_training_steps_millions: Optional cap on training steps when selecting data
        max_steps_cap: Maximum training steps (in millions) for interpolation consistency
    
    Returns:
        Dict with averaged line maximum, std line statistics, and individual run maxima for comparison
    """
    if not all_run_data:
        return None
    
    # Collect interpolation data for all runs
    run_interpolations = {}
    individual_maxima = []
    use_checkpoint_numbers = False
    
    for run_name, run_results in all_run_data.items():
        # Filter and extract data points for this run
        run_points = []
        has_training_steps = False
        
        for result in run_results:
            if not allowed_result(result, max_training_steps_millions):
                continue
            avg_steps = numeric_or_none(result.get("avg_steps"))
            training_steps = numeric_or_none(result.get("training_steps"))
            checkpoint_num = result.get("checkpoint_num")
            
            if avg_steps is None:
                continue
            
            # Determine time axis - prefer training_steps, fallback to checkpoint_num
            if training_steps is not None:
                time_value = training_steps / 1e6  # Convert to millions
                has_training_steps = True
                # Apply cap for fair comparison across runs
                if time_value > max_steps_cap:
                    continue
            elif checkpoint_num is not None:
                time_value = float(checkpoint_num)  # Use checkpoint as time proxy
                use_checkpoint_numbers = True
            else:
                continue  # Skip if no time information available
            
            run_points.append({
                'time_value': time_value,
                'avg_steps': avg_steps,
                'checkpoint_num': checkpoint_num,
                'checkpoint_path': result.get("checkpoint_path"),
                'has_training_steps': has_training_steps
            })
        
        if len(run_points) < 2:  # Need at least 2 points for interpolation
            continue
            
        # Sort by time value
        run_points.sort(key=lambda x: x['time_value'])
        
        # Store the best individual run result for backwards compatibility
        best_point = max(run_points, key=lambda x: x['avg_steps'])
        individual_maxima.append({
            "max_avg_steps": best_point['avg_steps'],
            "checkpoint_num": best_point['checkpoint_num'], 
            "training_steps": best_point['time_value'] * 1e6 if best_point['has_training_steps'] else None,
            "checkpoint_path": best_point['checkpoint_path'],
            "run_name": run_name
        })
        
        # Prepare data for interpolation
        x_vals = [p['time_value'] for p in run_points]
        y_vals = [p['avg_steps'] for p in run_points]
        
        run_interpolations[run_name] = {
            'x': x_vals,
            'y': y_vals,
            'points': run_points,
            'has_training_steps': has_training_steps
        }
    
    if not run_interpolations:
        return None
    
    # Find union range - use all available data from any run (earliest start to latest end)
    x_min_values = []
    x_max_values = []
    
    for run_data in run_interpolations.values():
        x_min_values.append(min(run_data['x']))
        x_max_values.append(max(run_data['x']))
    
    if not x_min_values:
        return None
        
    # Use union range - earliest start and latest end across all runs  
    x_min = min(x_min_values)  # Earliest start time across all runs
    x_max = max(x_max_values)  # Latest end time across all runs
    
    # Only apply training steps cap if we have actual training steps
    if not use_checkpoint_numbers:
        x_max = min(x_max, max_steps_cap)
        if x_min >= x_max:
            print(f"Warning: Training steps cap {max_steps_cap}M too restrictive for available data")
            return None
    
    # Create common x-axis for interpolation (enough points for smooth curve)
    x_common = np.linspace(x_min, x_max, 150)
    
    # Interpolate all runs to common x-axis - use NaN for missing data
    interpolated_runs = []
    for run_name, run_data in run_interpolations.items():
        try:
            # Get this run's actual data range
            run_x_min, run_x_max = min(run_data['x']), max(run_data['x'])
            
            # Create mask for points within this run's actual data range
            valid_mask = (x_common >= run_x_min) & (x_common <= run_x_max)
            
            # Interpolate only within bounds, no extrapolation
            f_steps = interpolate.interp1d(run_data['x'], run_data['y'], 
                                         kind='linear', bounds_error=True)
            
            # Initialize with NaN and fill only valid range (no extrapolation)
            interpolated_steps = np.full(len(x_common), np.nan)
            if any(valid_mask):
                interpolated_steps[valid_mask] = f_steps(x_common[valid_mask])
            
            interpolated_runs.append(interpolated_steps)
            
        except Exception as e:
            print(f"Warning: Could not interpolate run {run_name}: {e}")
            continue
    
    if not interpolated_runs:
        return None
    
    # Calculate mean and std across runs, handling NaN values from non-overlapping regions
    interpolated_array = np.array(interpolated_runs)
    
    # Calculate mean and std ignoring NaN values
    with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
        mean_steps = np.nanmean(interpolated_array, axis=0)
        std_steps = np.nanstd(interpolated_array, axis=0) if len(interpolated_runs) > 1 else np.zeros_like(mean_steps)
    
    # Find valid (non-NaN) points for analysis
    valid_points = ~np.isnan(mean_steps)
    
    if not any(valid_points):
        print("Warning: No valid interpolated points found")
        return None
    
    # Only analyze the valid range
    valid_mean = mean_steps[valid_points]
    valid_std = std_steps[valid_points]
    valid_x = x_common[valid_points]
    
    # Find maximum of the averaged line (only in valid range)
    max_idx = np.argmax(valid_mean)
    averaged_line_max = float(valid_mean[max_idx])
    max_time_value = float(valid_x[max_idx])
    
    # Calculate statistics from the std line (only valid points)
    std_line_max = float(np.max(valid_std))
    std_line_mean = float(np.mean(valid_std))
    std_line_min = float(np.min(valid_std))
    std_at_max_point = float(valid_std[max_idx])  # Std deviation at the point where mean is maximum
    
    # Calculate statistics from individual run maxima
    individual_max_values = [run_max["max_avg_steps"] for run_max in individual_maxima]
    
    # Convert time back to training steps if applicable
    if use_checkpoint_numbers:
        max_training_steps_out = None
        time_axis_info = "checkpoint_numbers"
    else:
        max_training_steps_out = max_time_value * 1e6  # Convert back to actual steps
        time_axis_info = "training_steps_millions"
    
    return {
        "averaged_line_max": averaged_line_max,
        "averaged_line_max_at_training_steps": max_training_steps_out,
        "max_time_value": max_time_value,
        "time_axis_type": time_axis_info,
        "std_at_max_point": std_at_max_point,
        
        # Standard deviation line statistics
        "std_line_max": std_line_max,
        "std_line_mean": std_line_mean,
        "std_line_min": std_line_min,
        
        # Individual run statistics (legacy)
        "individual_run_maxima": individual_maxima,
        "individual_max_values": individual_max_values,
        "individual_max_mean": mean(individual_max_values) if individual_max_values else 0.0,
        "individual_max_std": pstdev(individual_max_values) if len(individual_max_values) > 1 else 0.0,
        "n_runs_used": len(interpolated_runs),
        "interpolation_range": {
            "x_min": float(x_min), 
            "x_max": float(x_max),
            "valid_points": int(sum(valid_points)),
            "total_points": len(x_common)
        },
        "common_x_points": len(x_common),
        "used_checkpoint_numbers": use_checkpoint_numbers
    }


def summarize_variant(run_results, max_training_steps_millions):
    """Legacy function kept for backwards compatibility"""
    candidates = []
    for result in run_results:
        if not allowed_result(result, max_training_steps_millions):
            continue
        avg_steps = numeric_or_none(result.get("avg_steps"))
        if avg_steps is None:
            continue
        candidates.append(result)

    if not candidates:
        return None

    best = max(candidates, key=lambda item: numeric_or_none(item.get("avg_steps")) or float("-inf"))
    return {
        "max_avg_steps": float(best["avg_steps"]),
        "checkpoint_num": best.get("checkpoint_num"),
        "training_steps": numeric_or_none(best.get("training_steps")),
        "checkpoint_path": best.get("checkpoint_path"),
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

    common_runs = sorted(set.intersection(*(set(runs) for runs in variant_runs.values()))) if variant_runs else []
    summaries = []

    for variant in variants:
        # Collect all run data for this variant 
        variant_all_run_data = {}
        for run_name in variant_runs[variant]:
            if variant in all_results[run_name]:
                variant_all_run_data[run_name] = all_results[run_name][variant]
        
        # Calculate averaged line maximum
        averaged_summary = summarize_variant_averaged_line(variant_all_run_data, max_training_steps_millions)
        if averaged_summary is None:
            continue
        
        # Also calculate per-run maxima for backwards compatibility/comparison
        per_run = {}
        for run_name in variant_runs[variant]:
            best = summarize_variant(all_results[run_name][variant], max_training_steps_millions)
            if best is not None:
                per_run[run_name] = best

        # Find overall best run from individual maxima
        if per_run:
            best_run_name, best_run_details = max(per_run.items(), key=lambda item: item[1]["max_avg_steps"])
        else:
            best_run_name, best_run_details = None, None

        summaries.append({
            "experiment": experiment_name,
            "variant": variant,
            "runs_used": sorted(variant_all_run_data.keys()),
            "common_runs_all_variants": common_runs,
            "run_count": averaged_summary["n_runs_used"],
            "common_run_count": len(common_runs),
            
            # New: Averaged line results (primary metric)
            "averaged_line_max": averaged_summary["averaged_line_max"],
            "averaged_line_max_at_training_steps": averaged_summary["averaged_line_max_at_training_steps"],
            "max_time_value": averaged_summary["max_time_value"],
            "time_axis_type": averaged_summary["time_axis_type"],
            "std_at_max_point": averaged_summary["std_at_max_point"],
            
            # New: Standard deviation line statistics
            "std_line_max": averaged_summary["std_line_max"],
            "std_line_mean": averaged_summary["std_line_mean"],
            "std_line_min": averaged_summary["std_line_min"],
            
            # Legacy: Individual run maxima (for comparison)
            "max_avg_steps_per_run": {run_name: details["max_avg_steps"] for run_name, details in per_run.items()},
            "max_avg_steps_mean": averaged_summary["individual_max_mean"],
            "max_avg_steps_std": averaged_summary["individual_max_std"],
            "overall_best_run": best_run_name,
            "overall_best_max_avg_steps": best_run_details["max_avg_steps"] if best_run_details else None,
            "overall_best_checkpoint_num": best_run_details["checkpoint_num"] if best_run_details else None,
            "overall_best_training_steps": best_run_details["training_steps"] if best_run_details else None,
            
            # Detailed breakdown
            "per_run_details": per_run,
            "individual_run_maxima_details": averaged_summary["individual_run_maxima"],
            "interpolation_info": {
                "range": averaged_summary["interpolation_range"],
                "common_x_points": averaged_summary["common_x_points"],
                "used_checkpoint_numbers": averaged_summary["used_checkpoint_numbers"]
            }
        })

    return summaries


def format_steps(value):
    return f"{value:.2f}"


def write_csv(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "variant",
        "run_count",
        "common_run_count",
        "runs_used",
        "common_runs_all_variants",
        # New: Averaged line metrics (primary)
        "averaged_line_max",
        "averaged_line_max_at_training_steps",
        "max_time_value",
        "time_axis_type",
        "std_at_max_point",
        # New: Standard deviation line metrics
        "std_line_max",
        "std_line_mean", 
        "std_line_min",
        # Legacy: Individual run maxima metrics
        "max_avg_steps_per_run",
        "max_avg_steps_mean",
        "max_avg_steps_std",
        "overall_best_run",
        "overall_best_max_avg_steps",
        "overall_best_checkpoint_num",
        "overall_best_training_steps",
    ]

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "experiment": row["experiment"],
                "variant": row["variant"],
                "run_count": row["run_count"],
                "common_run_count": row["common_run_count"],
                "runs_used": ",".join(row["runs_used"]),
                "common_runs_all_variants": ",".join(row["common_runs_all_variants"]),
                # New averaged line metrics
                "averaged_line_max": f"{row['averaged_line_max']:.6f}",
                "averaged_line_max_at_training_steps": f"{row['averaged_line_max_at_training_steps']:.0f}" if row['averaged_line_max_at_training_steps'] is not None else "",
                "max_time_value": f"{row['max_time_value']:.6f}",
                "time_axis_type": row["time_axis_type"],
                "std_at_max_point": f"{row['std_at_max_point']:.6f}",
                # New standard deviation line metrics
                "std_line_max": f"{row['std_line_max']:.6f}",
                "std_line_mean": f"{row['std_line_mean']:.6f}",
                "std_line_min": f"{row['std_line_min']:.6f}",
                # Legacy individual run metrics
                "max_avg_steps_per_run": json.dumps(row["max_avg_steps_per_run"], sort_keys=True),
                "max_avg_steps_mean": f"{row['max_avg_steps_mean']:.6f}",
                "max_avg_steps_std": f"{row['max_avg_steps_std']:.6f}",
                "overall_best_run": row["overall_best_run"] or "",
                "overall_best_max_avg_steps": f"{row['overall_best_max_avg_steps']:.6f}" if row['overall_best_max_avg_steps'] is not None else "",
                "overall_best_checkpoint_num": row["overall_best_checkpoint_num"] or "",
                "overall_best_training_steps": row["overall_best_training_steps"] or "",
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

        per_run_list = ", ".join(
            f"{run}={format_steps(value)}"
            for run, value in sorted(row["max_avg_steps_per_run"].items())
        )
        
        print(
            f"- {row['variant']}:"
        )
        print(
            f"    🎯 Averaged line max: {format_steps(row['averaged_line_max'])} "
            f"({format_time_info(row)})"
        )
        print(
            f"    📏 Std at max point: {format_steps(row['std_at_max_point'])}"
        )
        print(
            f"    📊 Std line stats: max={format_steps(row['std_line_max'])}, "
            f"mean={format_steps(row['std_line_mean'])}, "
            f"min={format_steps(row['std_line_min'])}"
        )
        print(
            f"    🔍 Individual runs: mean={format_steps(row['max_avg_steps_mean'])}, "
            f"std={format_steps(row['max_avg_steps_std'])}, best={row['overall_best_run'] or 'N/A'} "
            f"({format_steps(row['overall_best_max_avg_steps']) if row['overall_best_max_avg_steps'] is not None else 'N/A'})"
        )
        print(f"    📋 Per-run: [{per_run_list}]")
        
        # Add debug info about interpolation range and time axis
        interp_info = row.get("interpolation_info", {})
        if interp_info.get("used_checkpoint_numbers"):
            print(f"    ⚠️  Used checkpoint numbers as time axis (training_steps missing)")
        
        valid_points = interp_info.get("valid_points", 0)
        total_points = interp_info.get("total_points", 0)
        if valid_points < total_points:
            print(f"    📐 Interpolation: {valid_points}/{total_points} valid points (union range covers all peaks)"
                  f" - range: {interp_info.get('x_min', 0):.1f}-{interp_info.get('x_max', 0):.1f})")


def format_time_info(row):
    """Format time information based on available data."""
    if row["time_axis_type"] == "training_steps_millions" and row["averaged_line_max_at_training_steps"] is not None:
        return f"at {row['averaged_line_max_at_training_steps']/1e6:.1f}M steps"
    elif row["time_axis_type"] == "checkpoint_numbers":
        return f"at checkpoint {row['max_time_value']:.0f}"
    else:
        return f"at {row['max_time_value']:.1f}"


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

    write_csv(rows, args.output_csv)
    write_json(rows, args.output_json, args.max_training_steps)
    print_summary(rows)
    print(f"\nCSV saved to {args.output_csv}")
    print(f"JSON saved to {args.output_json}")


if __name__ == "__main__":
    main()