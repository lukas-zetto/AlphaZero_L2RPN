#!/usr/bin/env python3
"""
Multi-run evaluation for Action Space Experiments  
Evaluates runs _5, _6, _7, _8 and creates comparison plots with confidence bands
Creates organized subfolders for results and saves all data in JSON
"""

import os
import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from scipy import interpolate

# Set larger default font sizes
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

# Add the same path setup as working evaluation script
sys.path.append('/workspace/src')
sys.path.append('/workspace')
sys.path.append('/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src')

def get_available_cpus():
    """Get the number of available CPUs, respecting SLURM and environment settings."""
    # Check SLURM environment first
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        try:
            return int(slurm_cpus)
        except ValueError:
            pass
    
    # Check OMP_NUM_THREADS
    omp_threads = os.environ.get('OMP_NUM_THREADS')
    if omp_threads:
        try:
            return int(omp_threads)
        except ValueError:
            pass
    
    # Fall back to a SAFE default instead of using all node CPUs
    # This prevents overloading the node when SLURM vars aren't available
    return 20  # Safe default matching our SLURM allocation

def load_existing_results(results_file):
    """Load already evaluated results from JSON file."""
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                # Map checkpoint_num to result for quick lookup
                return {r.get('checkpoint_num'): r for r in results if r.get('checkpoint_num') is not None}
        except Exception as e:
            print(f"    ⚠️ Could not load existing results: {e}")
            return {}
    return {}


def save_partial_results(results_data, results_file):
    """Save partial results to JSON file for resume capability."""
    try:
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"    💾 Saved {len(results_data)} results to {results_file}")
    except Exception as e:
        print(f"    ⚠️ Could not save partial results: {e}")


# Import from working evaluation script  
from scripts.evaluate_all_experiments import evaluate_checkpoint, extract_checkpoint_number, find_checkpoints, extract_training_steps_from_logs


def create_action_space_plots_with_confidence_bands(all_results, output_dir):
    """Create plots with smooth interpolated confidence bands across runs"""
    
    print(f"\n📊 Creating action space comparison plots with smooth confidence bands...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Action space variants
    variants = ["SYM", "N0", "N1"]
    colors = {'SYM': '#F9A825', 'N0': '#1565C0', 'N1': '#00838F'}
    
    # Organize data by variant and run
    # Find runs that are available for ALL variants
    all_run_sets = []
    for variant in variants:
        variant_runs = set()
        for run_suffix, run_data in all_results.items():
            if variant in run_data and run_data[variant]:  # Check if variant has data
                variant_runs.add(run_suffix)
        all_run_sets.append(variant_runs)
    
    # Find intersection - runs available for all variants
    if all_run_sets:
        common_runs = set.intersection(*all_run_sets)
        print(f"  📊 Common runs available for all variants: {sorted(list(common_runs))}")
        
        if not common_runs:
            print("  ⚠️  No common runs found across all variants!")
            return
    else:
        print("  ⚠️  No data found!")
        return
    
    variant_run_data = {}
    
    for variant in variants:
        variant_run_data[variant] = {}
        
        for run_suffix, run_data in all_results.items():
            # Only process runs that are common to all variants
            if run_suffix not in common_runs:
                continue
            if variant in run_data:
                variant_results = run_data[variant]
                
                # Extract data for this run/variant
                run_points = []
                for result in variant_results:
                    if result.get('success', True) and result.get('training_steps') is not None:
                        training_steps_millions = result['training_steps'] / 1e6  # Convert to millions
                        # Only include data points within 40M steps for fair comparison  
                        if training_steps_millions <= 40.0:
                            run_points.append({
                                'training_steps': training_steps_millions,
                                'avg_steps': result.get('avg_steps', 0),
                                'avg_reward': result.get('avg_reward', 0),
                                'survival_rate': 100.0 * result.get('survived_episodes', 0) / result.get('total_episodes', 101)
                            })
                
                if run_points:
                    print(f"    {variant} run {run_suffix}: {len(run_points)} data points after 40M filter")
                    # Sort by training steps
                    run_points.sort(key=lambda x: x['training_steps'])
                    variant_run_data[variant][run_suffix] = run_points
                else:
                    print(f"    {variant} run {run_suffix}: 0 data points after 40M filter")
        
    print(f"  Total variants with data: {len([v for v in variant_run_data.values() if v])}")
    if not variant_run_data or not any(variant_run_data.values()):
        print("  No data available for plotting")
        return
    
    # Create interpolated data for smooth plotting
    print("  Creating interpolated data for smooth plotting...")
    
    # Determine global x-axis range
    all_x_values = []
    for variant_data in variant_run_data.values():
        for run_data in variant_data.values():
            all_x_values.extend([p['training_steps'] for p in run_data])
    
    if not all_x_values:
        print("  No training steps data found")
        return
    
    x_min, x_max = min(all_x_values), max(all_x_values)
    
    # Limit to 40M steps to ensure fair comparison across all runs
    x_max = min(x_max, 40.0)  # 40M steps
    print(f"  Training steps range (limited): {x_min:.1f}M - {x_max:.1f}M")
    
    # Create common x-axis for interpolation
    x_common = np.linspace(x_min, x_max, 150)
    
    # Interpolate data for each variant
    from scipy import interpolate
    
    interpolated_data = {}
    for variant in variants:
        if variant in variant_run_data and variant_run_data[variant]:
            print(f"    Processing {variant}: {len(variant_run_data[variant])} runs")
            
            # Collect interpolated data for all runs of this variant
            all_rewards = []
            all_steps = []
            all_survival_rates = []
            
            for run_suffix, run_data in variant_run_data[variant].items():
                if len(run_data) < 2:  # Need at least 2 points for interpolation
                    continue
                    
                x_vals = [p['training_steps'] for p in run_data]
                
                # Interpolate rewards
                y_rewards = [p['avg_reward'] for p in run_data]
                f_reward = interpolate.interp1d(x_vals, y_rewards, kind='linear', 
                                              fill_value='extrapolate', bounds_error=False)
                interpolated_rewards = f_reward(x_common)
                all_rewards.append(interpolated_rewards)
                
                # Interpolate steps
                y_steps = [p['avg_steps'] for p in run_data]
                f_steps = interpolate.interp1d(x_vals, y_steps, kind='linear',
                                             fill_value='extrapolate', bounds_error=False)
                interpolated_steps = f_steps(x_common)
                all_steps.append(interpolated_steps)
                
                # Interpolate survival rates
                y_survival = [p['survival_rate'] for p in run_data]
                f_survival = interpolate.interp1d(x_vals, y_survival, kind='linear',
                                                fill_value='extrapolate', bounds_error=False)
                interpolated_survival = f_survival(x_common)
                all_survival_rates.append(interpolated_survival)
            
            if all_rewards:
                # Calculate statistics
                mean_rewards = np.mean(all_rewards, axis=0)
                std_rewards = np.std(all_rewards, axis=0) if len(all_rewards) > 1 else np.zeros_like(mean_rewards)
                
                mean_steps = np.mean(all_steps, axis=0)
                std_steps = np.std(all_steps, axis=0) if len(all_steps) > 1 else np.zeros_like(mean_steps)
                
                mean_survival = np.mean(all_survival_rates, axis=0)
                std_survival = np.std(all_survival_rates, axis=0) if len(all_survival_rates) > 1 else np.zeros_like(mean_survival)
                
                interpolated_data[variant] = {
                    'x': x_common,
                    'rewards_mean': mean_rewards,
                    'rewards_std': std_rewards,
                    'steps_mean': mean_steps,
                    'steps_std': std_steps,
                    'survival_mean': mean_survival,
                    'survival_std': std_survival,
                    'n_runs': len(all_rewards)
                }
                
                print(f"      Interpolated {len(x_common)} points from {len(all_rewards)} runs")
    
    if not interpolated_data:
        print("  No interpolated data available for plotting")
        return
    
    # Create plots with original style + smooth confidence bands
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    
    # Plot 1: Average Reward
    axes[0].set_title('Action space comparison', fontsize=24, fontweight='bold')
    for variant in variants:
        if variant in interpolated_data:
            data = interpolated_data[variant]
            
            # Plot mean line (thick, no markers)
            axes[0].plot(data['x'], data['rewards_mean'], 
                        color=colors[variant], linewidth=3,
                        label=f"{variant}")
            
            # Plot confidence band
            axes[0].fill_between(data['x'], 
                               data['rewards_mean'] - data['rewards_std'],
                               data['rewards_mean'] + data['rewards_std'],
                               color=colors[variant], alpha=0.2)
    
    axes[0].set_xlabel('Training steps (millions)', fontsize=22)
    axes[0].set_ylabel('Average Reward', fontsize=22)
    axes[0].legend(fontsize=18, loc='lower right')
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average Steps
    axes[1].set_title('Action space comparison', fontsize=24, fontweight='bold')
    for variant in variants:
        if variant in interpolated_data:
            data = interpolated_data[variant]
            
            # Plot mean line (thick, no markers)
            axes[1].plot(data['x'], data['steps_mean'],
                        color=colors[variant], linewidth=3,
                        label=f"{variant}")
            
            # Plot confidence band
            axes[1].fill_between(data['x'],
                               data['steps_mean'] - data['steps_std'], 
                               data['steps_mean'] + data['steps_std'],
                               color=colors[variant], alpha=0.2)
    
    axes[1].set_xlabel('Training steps (millions)', fontsize=22)
    axes[1].set_ylabel('Average Steps Survived', fontsize=22)
    axes[1].legend(fontsize=18, loc='lower right')
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Survival Rate
    axes[2].set_title('Action space comparison', fontsize=24, fontweight='bold')
    for variant in variants:
        if variant in interpolated_data:
            data = interpolated_data[variant]
            
            # Plot mean line (thick, no markers)
            axes[2].plot(data['x'], data['survival_mean'],
                        color=colors[variant], linewidth=3,
                        label=f"{variant}")
            
            # Plot confidence band
            axes[2].fill_between(data['x'],
                               data['survival_mean'] - data['survival_std'],
                               data['survival_mean'] + data['survival_std'], 
                               color=colors[variant], alpha=0.2)
    
    axes[2].set_xlabel('Training steps (millions)', fontsize=22)
    axes[2].set_ylabel('Survival Rate (%)', fontsize=22)
    axes[2].legend(fontsize=18, loc='lower right')
    axes[2].tick_params(axis='both', labelsize=16)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as SVG like original
    svg_filename = output_dir / f"action_space_smooth_confidence_bands_{timestamp}.svg"
    plt.savefig(svg_filename, format='svg', bbox_inches='tight')
    print(f"  📈 Saved plot: {svg_filename}")
    
    # Save as PDF
    pdf_filename = output_dir / f"action_space_smooth_confidence_bands_{timestamp}.pdf"
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"  📈 Saved plot: {pdf_filename}")
    
    # Also save as PNG
    png_filename = output_dir / f"action_space_smooth_confidence_bands_{timestamp}.png"
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"  📈 Saved plot: {png_filename}")
    
    plt.close()
    
    return interpolated_data


def create_action_space_steps_only_plot(all_results, output_dir):
    """Create a plot showing only average steps survived comparison"""
    
    print("📊 Creating action space steps-only plot...")
    
    # Setup
    variants = ["SYM", "N0", "N1"]
    
    # Find runs that are available for ALL variants
    all_run_sets = []
    for variant in variants:
        variant_runs = set()
        for run_suffix, run_data in all_results.items():
            if variant in run_data and run_data[variant]:
                variant_runs.add(run_suffix)
        all_run_sets.append(variant_runs)
    
    if all_run_sets:
        common_runs = set.intersection(*all_run_sets)
        print(f"  📊 Using common runs: {sorted(list(common_runs))}")
    else:
        print("  ⚠️  No data found!")
        return
    
    # Process data for steps plotting
    variant_run_data = {}
    
    for variant in variants:
        variant_run_data[variant] = {}
        
        for run_suffix, run_data in all_results.items():
            if run_suffix not in common_runs:
                continue
                
            if variant in run_data:
                variant_results = run_data[variant]
                
                # Extract data for this run/variant
                run_points = []
                for result in variant_results:
                    if result.get('success', True) and result.get('training_steps') is not None:
                        training_steps_millions = result['training_steps'] / 1e6
                        if training_steps_millions <= 40.0:
                            run_points.append({
                                'training_steps': training_steps_millions,
                                'avg_steps': result.get('avg_steps', 0)
                            })
                
                if run_points:
                    run_points.sort(key=lambda x: x['training_steps'])
                    variant_run_data[variant][run_suffix] = run_points
    
    if not variant_run_data or not any(variant_run_data.values()):
        print("  No data available for steps plotting")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for each variant
    colors = {'SYM': '#F9A825', 'N0': '#1565C0', 'N1': '#00838F'}
    
    # Global x-axis range 
    all_x_values = []
    for variant_data in variant_run_data.values():
        for run_data in variant_data.values():
            all_x_values.extend([p['training_steps'] for p in run_data])
    
    x_min, x_max = min(all_x_values), max(all_x_values)
    x_max = min(x_max, 40.0)
    x_common = np.linspace(x_min, x_max, 150)
    
    from scipy import interpolate
    
    # Plot each variant
    for variant in variants:
        if variant in variant_run_data and variant_run_data[variant]:
            # Collect data for all runs
            all_steps = []
            
            for run_suffix, run_data in variant_run_data[variant].items():
                if len(run_data) < 2:
                    continue
                    
                x_vals = [p['training_steps'] for p in run_data]
                y_vals = [p['avg_steps'] for p in run_data]
                
                # Interpolate
                f = interpolate.interp1d(x_vals, y_vals, kind='linear', 
                                       fill_value='extrapolate', bounds_error=False)
                interpolated = f(x_common)
                all_steps.append(interpolated)
            
            if all_steps:
                # Calculate mean and std
                mean_steps = np.mean(all_steps, axis=0)
                std_steps = np.std(all_steps, axis=0) if len(all_steps) > 1 else np.zeros_like(mean_steps)
                
                # Plot
                ax.plot(x_common, mean_steps, color=colors[variant], linewidth=3, label=f'{variant}')
                ax.fill_between(x_common, mean_steps - std_steps, mean_steps + std_steps, 
                              color=colors[variant], alpha=0.2)
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=22)
    ax.set_ylabel('Average Steps Survived', fontsize=22)
    ax.set_title('Action Space Comparison: Average Steps Survived', fontsize=24, fontweight='bold')
    ax.legend(fontsize=18, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pdf_filename = output_dir / f"action_space_steps_only_{timestamp}.pdf"
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"  📈 Saved steps plot: {pdf_filename}")
    
    png_filename = output_dir / f"action_space_steps_only_{timestamp}.png"
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"  📈 Saved steps plot: {png_filename}")
    
    plt.close()


def main(plot_only=False, results_file=None): 
    """Multi-run evaluation for action space experiments with organized subfolders"""
    
    if plot_only:
        if not results_file or not Path(results_file).exists():
            print("❌ For plot-only mode, please provide a valid --results-file path")
            print("   Use a previously saved .json results file from eval output directory")
            return
        
        print("📊 Plot-Only Mode: Creating plots from existing results")
        print("=" * 60)
        
        # Load existing results
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        
        # Fix null training_steps in existing results
        print("  🔧 Fixing null training_steps in existing results...")
        fixed_count = 0
        
        for run_suffix, run_data in all_results.items():
            for exp_name, exp_results in run_data.items():
                experiment_name = f"action_space_experiments_fixed{run_suffix}"
                
                # Map experiment names to subdirs
                subdir_mapping = {"SYM": "checkpoints_action_sym", "N0": "checkpoints_action_n0", "N1": "checkpoints_action_n1"}
                exp_subdir = subdir_mapping.get(exp_name)
                
                if exp_subdir:
                    # Extract training steps for this experiment/run combination
                    steps_mapping = extract_training_steps_from_logs(experiment_name, exp_subdir)
                    
                    # Update results that have null training_steps
                    for result in exp_results:
                        if result.get('training_steps') is None:
                            checkpoint_num = result.get('checkpoint_num')
                            if checkpoint_num is not None and checkpoint_num in steps_mapping:
                                result['training_steps'] = steps_mapping[checkpoint_num]
                                fixed_count += 1
        
        print(f"  ✓ Fixed training_steps for {fixed_count} result entries")
        
        # Create output directory based on results file location
        results_dir = Path(results_file).parent
        
        # Create plots
        if all_results:
            plot_data = create_action_space_plots_with_confidence_bands(all_results, results_dir)
            
            # Create separate steps-only plot
            create_action_space_steps_only_plot(all_results, results_dir)
            
            print(f"📊 Plots created in: {results_dir}")
        else:
            print("❌ No data found in results file")
        return
    
    print("🚀 Multi-run Action Space Evaluation with Confidence Bands")
    print("=" * 70)
    
    # Auto-detect container vs host environment
    if os.path.exists("/work"):
        # Running inside container
        checkpoint_base = "/work/PdF_L2RPN/AlphaZero_L2RPN/checkpoints"
        eval_base = "/work/PdF_L2RPN/AlphaZero_L2RPN/eval"
        print("📦 Running inside container")
    else:
        # Running on host
        checkpoint_base = "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/checkpoints"
        eval_base = "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/eval"
        print("🖥️  Running on host")
    
    # Create consistent output directory (no timestamp for better resume functionality)
    results_dir = Path(eval_base) / "action_space_multi_run_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Runs to evaluate
    runs = ["_5", "_6", "_7", "_8"]
    
    # Action space experiments structure
    experiments = [
        {"name": "SYM", "subdir": "checkpoints_action_sym"},
        {"name": "N0", "subdir": "checkpoints_action_n0"}, 
        {"name": "N1", "subdir": "checkpoints_action_n1"}
    ]
    
    all_results = {}
    
    for run_suffix in runs:
        print(f"\n🔍 Evaluating run {run_suffix}")
        
        experiment_name = f"action_space_experiments_fixed{run_suffix}"
        experiment_dir = Path(checkpoint_base) / experiment_name
        
        if not experiment_dir.exists():
            print(f"⚠️  Experiment directory not found: {experiment_dir}")
            continue
            
        run_results = {}
        
        for exp in experiments:
            checkpoint_dir = experiment_dir / exp["subdir"]
            
            if not checkpoint_dir.exists():
                print(f"  ⚠️  Checkpoint dir not found: {checkpoint_dir}")
                continue
                
            # Find checkpoints
            checkpoints = find_checkpoints(checkpoint_dir)
            if not checkpoints:
                print(f"  ⚠️  No checkpoints in {checkpoint_dir}")
                continue
                
            print(f"  📁 Evaluating {exp['name']}: {len(checkpoints)} checkpoints")
            
            # Create results file path for this specific experiment + run
            exp_results_file = results_dir / f"{experiment_name}_{exp['name']}_results.json"
            
            # Load existing results for resume capability  
            existing_results = load_existing_results(exp_results_file)
            already_evaluated = set(existing_results.keys())
            
            if already_evaluated:
                print(f"    📋 Found {len(already_evaluated)} already evaluated checkpoints")
            
            # Extract training steps from logs (needed for both existing and new results)
            steps_mapping = extract_training_steps_from_logs(experiment_name, exp["subdir"])
            
            # Only evaluate missing checkpoints
            checkpoints_to_eval = [cp for cp in checkpoints if extract_checkpoint_number(cp) not in already_evaluated]
            
            if not checkpoints_to_eval:
                print(f"    ✓ All checkpoints already evaluated, loading existing results")
                results_data = list(existing_results.values())
                # Update existing results with training steps
                for result in results_data:
                    if result.get('training_steps') is None:
                        checkpoint_num = result.get('checkpoint_num')
                        if checkpoint_num is not None:
                            result['training_steps'] = steps_mapping.get(checkpoint_num)
            else:
                print(f"    🚀 Evaluating {len(checkpoints_to_eval)} new checkpoints (skipping {len(already_evaluated)} already done)")
                
                # Start with existing results and update them with training steps
                results_data = list(existing_results.values())
                for result in results_data:
                    if result.get('training_steps') is None:
                        checkpoint_num = result.get('checkpoint_num')
                        if checkpoint_num is not None:
                            result['training_steps'] = steps_mapping.get(checkpoint_num)
                
                # Use same parallelization pattern as working script
                available_cpus = get_available_cpus()
                max_workers = min(available_cpus, len(checkpoints_to_eval))
                print(f"    🚀 Using {max_workers} workers (detected {available_cpus} CPUs)")
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_checkpoint = {
                        executor.submit(evaluate_checkpoint, cp, f"{experiment_name}_{exp['name']}"): cp 
                        for cp in checkpoints_to_eval
                    }
                    
                    completed_count = 0
                    for future in as_completed(future_to_checkpoint):
                        checkpoint_path = future_to_checkpoint[future]
                        checkpoint_num = extract_checkpoint_number(checkpoint_path)
                        
                        try:
                            result = future.result()
                            if result and result.get('success', True):
                                # Add training steps information
                                result['checkpoint_num'] = checkpoint_num
                                result['training_steps'] = steps_mapping.get(checkpoint_num)
                                result['checkpoint_path'] = str(checkpoint_path)
                                
                                results_data.append(result)
                                completed_count += 1
                                print(f"    ✅ Checkpoint {checkpoint_num}: Reward {result.get('avg_reward', 0):.1f}, Steps {result.get('avg_steps', 0):.1f}")
                                
                                # Save results periodically (every 10 checkpoints)
                                if completed_count % 10 == 0:
                                    save_partial_results(results_data, exp_results_file)
                                    
                            else:
                                print(f"    ❌ Checkpoint {checkpoint_num}: evaluation failed")
                            
                        except Exception as e:
                            print(f"    ❌ Checkpoint {checkpoint_num} failed: {e}")
                            continue
                
                # Final save after all evaluations complete 
                save_partial_results(results_data, exp_results_file)
            
            if results_data:
                run_results[exp['name']] = results_data
                print(f"    ✅ Collected {len(results_data)} successful evaluations")
        
        if run_results:
            all_results[run_suffix] = run_results
    
    # Save all results to JSON
    json_file = results_dir / "action_space_multi_run_results.json"
    with open(json_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_serializable = {}
        for run, run_data in all_results.items():
            json_serializable[run] = {}
            for variant, results in run_data.items():
                json_serializable[run][variant] = []
                for result in results:
                    clean_result = {k: convert_for_json(v) for k, v in result.items()}
                    json_serializable[run][variant].append(clean_result)
        
        json.dump(json_serializable, f, indent=2, default=str)
    
    print(f"\n💾 Saved all results to: {json_file}")
    
    # Fix any null training_steps before plotting
    if all_results:
        print(f"\n🔧 Fixing null training_steps in results before plotting...")
        fixed_count = 0
        
        for run_suffix, run_data in all_results.items():
            for exp_name, exp_results in run_data.items():
                experiment_name = f"action_space_experiments_fixed{run_suffix}"
                
                # Map experiment names to subdirs
                subdir_mapping = {"SYM": "checkpoints_action_sym", "N0": "checkpoints_action_n0", "N1": "checkpoints_action_n1"}
                exp_subdir = subdir_mapping.get(exp_name)
                
                if exp_subdir:
                    # Extract training steps for this experiment/run combination
                    steps_mapping = extract_training_steps_from_logs(experiment_name, exp_subdir)
                    
                    # Update results that have null training_steps
                    for result in exp_results:
                        if result.get('training_steps') is None:
                            checkpoint_num = result.get('checkpoint_num')
                            if checkpoint_num is not None and checkpoint_num in steps_mapping:
                                result['training_steps'] = steps_mapping[checkpoint_num]
                                fixed_count += 1
        
        print(f"  ✓ Fixed training_steps for {fixed_count} result entries")
    
    # Create comparison plots with confidence bands
    if all_results:
        plot_data = create_action_space_plots_with_confidence_bands(all_results, results_dir)
        
        # Save plot data summary
        if plot_data:
            summary_file = results_dir / "plot_data_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(plot_data, f, indent=2)
            print(f"  💾 Saved plot data summary: {summary_file}")
    
    print(f"\n🎉 Action space evaluation complete!")
    print(f"📁 All results organized in: {results_dir}")
    print(f"📊 Graphs show training progress with confidence bands across {len(runs)} runs")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-run Action Space Experiment Evaluation')
    parser.add_argument('--plot-only', action='store_true', 
                       help='Only create plots from existing results (skip evaluation)')
    parser.add_argument('--results-file', type=str,
                       help='Path to existing results JSON file (required for plot-only mode)')
    
    args = parser.parse_args()
    main(plot_only=args.plot_only, results_file=args.results_file)
