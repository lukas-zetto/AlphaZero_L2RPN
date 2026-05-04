#!/usr/bin/env python3
"""
Multi-run evaluation for Method Comparison Experiments  
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


def create_method_plots_with_confidence_bands(all_results, output_dir):
    """Create plots with smooth interpolated confidence bands across runs"""
    
    print(f"\n📊 Creating method comparison plots with smooth confidence bands...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Method variants (all three methods)
    # Method comparison variants
    methods = ["MCTS_Root", "Heuristic", "No_Guidance"]
    colors = {'MCTS_Root': '#6A1B9A', 'Heuristic': '#00838F', 'No_Guidance': '#F4511E'}
    display_names = {'MCTS_Root': 'Learned Q-function', 'Heuristic': 'Heuristic', 'No_Guidance': 'No Guidance'}
    
    # Organize data by variant and run
    method_run_data = {}
    
    for method in methods:
        method_run_data[method] = {}
        
        for run_suffix, run_data in all_results.items():
            if method in run_data:
                method_results = run_data[method]
                
                # Extract data for this run/variant
                run_points = []
                for result in method_results:
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
                    # Sort by training steps
                    run_points.sort(key=lambda x: x['training_steps'])
                    method_run_data[method][run_suffix] = run_points
    
    # Use available runs for each method individually (no intersection requirement)
    print("  📊 Using available runs per method:")
    
    filtered_method_run_data = {}
    for method in methods:
        filtered_method_run_data[method] = {}
        for run_suffix, run_points in method_run_data.get(method, {}).items():
            filtered_method_run_data[method][run_suffix] = run_points
            print(f"    {method} run {run_suffix}: {len(run_points)} data points after 40M filter")
    
    method_run_data = filtered_method_run_data
    print(f"  Total methods with data: {len([m for m in method_run_data.values() if m])}")
    
    if not method_run_data or not any(method_run_data.values()):
        print("  No data available for plotting")
        return
    
    # Create interpolated data for smooth plotting
    print("  Creating interpolated data for smooth plotting...")
    
    # Determine global x-axis range
    all_x_values = []
    for method_data in method_run_data.values():
        for run_data in method_data.values():
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

    def summarize_curves(curves):
        curves_array = np.asarray(curves, dtype=float)
        valid_mask = np.any(~np.isnan(curves_array), axis=0)
        mean = np.full_like(x_common, np.nan, dtype=float)
        std = np.full_like(x_common, np.nan, dtype=float)
        if np.any(valid_mask):
            mean[valid_mask] = np.nanmean(curves_array[:, valid_mask], axis=0)
            if len(curves) > 1:
                std[valid_mask] = np.nanstd(curves_array[:, valid_mask], axis=0)
            else:
                std[valid_mask] = 0.0
        return mean, std, valid_mask
    
    interpolated_data = {}
    for method in methods:
        if method in method_run_data and method_run_data[method]:
            print(f"    Processing {method}: {len(method_run_data[method])} runs")
            
            # Collect interpolated data for all runs of this variant
            all_rewards = []
            all_steps = []
            all_survival_rates = []
            
            for run_suffix, run_data in method_run_data[method].items():
                if len(run_data) < 2:  # Need at least 2 points for interpolation
                    continue
                    
                x_vals = [p['training_steps'] for p in run_data]
                
                # Interpolate rewards
                y_rewards = [p['avg_reward'] for p in run_data]
                f_reward = interpolate.interp1d(x_vals, y_rewards, kind='linear', 
                                              fill_value=np.nan, bounds_error=False)
                interpolated_rewards = f_reward(x_common)
                all_rewards.append(interpolated_rewards)
                
                # Interpolate steps
                y_steps = [p['avg_steps'] for p in run_data]
                f_steps = interpolate.interp1d(x_vals, y_steps, kind='linear',
                                             fill_value=np.nan, bounds_error=False)
                interpolated_steps = f_steps(x_common)
                all_steps.append(interpolated_steps)
                
                # Interpolate survival rates
                y_survival = [p['survival_rate'] for p in run_data]
                f_survival = interpolate.interp1d(x_vals, y_survival, kind='linear',
                                                fill_value=np.nan, bounds_error=False)
                interpolated_survival = f_survival(x_common)
                all_survival_rates.append(interpolated_survival)
            
            if all_rewards:
                # Calculate statistics only where at least one run has real data.
                mean_rewards, std_rewards, rewards_valid = summarize_curves(all_rewards)
                mean_steps, std_steps, steps_valid = summarize_curves(all_steps)
                mean_survival, std_survival, survival_valid = summarize_curves(all_survival_rates)
                
                interpolated_data[method] = {
                    'x': x_common,
                    'rewards_mean': mean_rewards,
                    'rewards_std': std_rewards,
                    'rewards_valid': rewards_valid,
                    'steps_mean': mean_steps,
                    'steps_std': std_steps,
                    'steps_valid': steps_valid,
                    'survival_mean': mean_survival,
                    'survival_std': std_survival,
                    'survival_valid': survival_valid,
                    'n_runs': len(all_rewards)
                }
                
                print(f"      Interpolated {len(x_common)} points from {len(all_rewards)} runs")
    
    if not interpolated_data:
        print("  No interpolated data available for plotting")
        return
    
    # Create plots with original style + smooth confidence bands
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Plot 1: Average Reward
    axes[0].set_title('Method Comparison Experiments - Average Reward', fontsize=24, fontweight='bold')
    for variant in methods:
        if variant in interpolated_data:
            data = interpolated_data[variant]
            
            # Plot mean line (thick, no markers)
            axes[0].plot(data['x'][data['rewards_valid']], data['rewards_mean'][data['rewards_valid']], 
                        color=colors[variant], linewidth=3,
                        label=display_names.get(variant, variant.replace('_', ' ')))
            
            # Plot confidence band
            axes[0].fill_between(data['x'][data['rewards_valid']], 
                               (data['rewards_mean'] - data['rewards_std'])[data['rewards_valid']],
                               (data['rewards_mean'] + data['rewards_std'])[data['rewards_valid']],
                               color=colors[variant], alpha=0.2)
    
    axes[0].set_xlabel('Training Steps (Millions)', fontsize=22)
    axes[0].set_ylabel('Average Reward', fontsize=22)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average Steps
    axes[1].set_title('Method Comparison Experiments - Average Steps Survived', fontsize=24, fontweight='bold')
    for variant in methods:
        if variant in interpolated_data:
            data = interpolated_data[variant]
            
            # Plot mean line (thick, no markers)
            axes[1].plot(data['x'][data['steps_valid']], data['steps_mean'][data['steps_valid']],
                        color=colors[variant], linewidth=3,
                        label=display_names.get(variant, variant.replace('_', ' ')))
            
            # Plot confidence band
            axes[1].fill_between(data['x'][data['steps_valid']],
                               (data['steps_mean'] - data['steps_std'])[data['steps_valid']], 
                               (data['steps_mean'] + data['steps_std'])[data['steps_valid']],
                               color=colors[variant], alpha=0.2)
    
    axes[1].set_xlabel('Training Steps (Millions)', fontsize=22)
    axes[1].set_ylabel('Average Steps Survived', fontsize=22)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Survival Rate
    axes[2].set_title('Method Comparison Experiments - Survival Rate', fontsize=24, fontweight='bold')
    for variant in methods:
        if variant in interpolated_data:
            data = interpolated_data[variant]
            
            # Plot mean line (thick, no markers)
            axes[2].plot(data['x'][data['survival_valid']], data['survival_mean'][data['survival_valid']],
                        color=colors[variant], linewidth=3,
                        label=display_names.get(variant, variant.replace('_', ' ')))
            
            # Plot confidence band
            axes[2].fill_between(data['x'][data['survival_valid']],
                               (data['survival_mean'] - data['survival_std'])[data['survival_valid']],
                               (data['survival_mean'] + data['survival_std'])[data['survival_valid']], 
                               color=colors[variant], alpha=0.2)
    
    axes[2].set_xlabel('Training Steps (Millions)', fontsize=22)
    axes[2].set_ylabel('Survival Rate (%)', fontsize=22)
    axes[2].legend(loc='lower right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as PDF like original
    pdf_filename = output_dir / f"method_comparison_smooth_confidence_bands_{timestamp}.pdf"
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"  📈 Saved plot: {pdf_filename}")
    
    # Also save as PNG
    png_filename = output_dir / f"method_comparison_smooth_confidence_bands_{timestamp}.png"
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    print(f"  📈 Saved plot: {png_filename}")
    
    plt.close()
    
    return interpolated_data


def create_method_steps_only_plot(all_results, output_dir):
    """Create a plot showing only average steps survived comparison for methods"""
    
    print("📊 Creating method steps-only plot...")
    
    # Setup
    methods = ["MCTS_Root", "Heuristic", "No_Guidance"]
    display_names = {'MCTS_Root': 'Learned Q-function', 'Heuristic': 'Heuristic', 'No_Guidance': 'No Guidance'}
    
    # Find runs that are available for ALL methods
    # Use available runs for each method individually
    print("  📊 Using available runs per method:")
    
    # Process data for steps plotting
    method_run_data = {}
    
    for method in methods:
        method_run_data[method] = {}
        
        for run_suffix, run_data in all_results.items():
            if method in run_data:
                method_results = run_data[method]
                
                # Extract data for this run/method
                run_points = []
                for result in method_results:
                    if result.get('success', True) and result.get('training_steps') is not None:
                        training_steps_millions = result['training_steps'] / 1e6
                        if training_steps_millions <= 40.0:
                            run_points.append({
                                'training_steps': training_steps_millions,
                                'avg_steps': result.get('avg_steps', 0)
                            })
                
                if run_points:
                    run_points.sort(key=lambda x: x['training_steps'])
                    method_run_data[method][run_suffix] = run_points
    
    if not method_run_data or not any(method_run_data.values()):
        print("  No data available for steps plotting")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for each method
    colors = {'MCTS_Root': '#6A1B9A', 'Heuristic': '#00838F', 'No_Guidance': '#F4511E'}
    
    # Global x-axis range 
    all_x_values = []
    for method_data in method_run_data.values():
        for run_data in method_data.values():
            all_x_values.extend([p['training_steps'] for p in run_data])
    
    x_min, x_max = min(all_x_values), max(all_x_values)
    x_max = min(x_max, 40.0)
    x_common = np.linspace(x_min, x_max, 150)
    
    from scipy import interpolate

    def summarize_curves(curves):
        curves_array = np.asarray(curves, dtype=float)
        valid_mask = np.any(~np.isnan(curves_array), axis=0)
        mean = np.full_like(x_common, np.nan, dtype=float)
        std = np.full_like(x_common, np.nan, dtype=float)
        if np.any(valid_mask):
            mean[valid_mask] = np.nanmean(curves_array[:, valid_mask], axis=0)
            if len(curves) > 1:
                std[valid_mask] = np.nanstd(curves_array[:, valid_mask], axis=0)
            else:
                std[valid_mask] = 0.0
        return mean, std, valid_mask
    
    # Plot each method
    for method in methods:
        if method in method_run_data and method_run_data[method]:
            # Collect data for all runs
            all_steps = []
            
            for run_suffix, run_data in method_run_data[method].items():
                if len(run_data) < 2:
                    continue
                    
                x_vals = [p['training_steps'] for p in run_data]
                y_vals = [p['avg_steps'] for p in run_data]
                
                # Interpolate
                f = interpolate.interp1d(x_vals, y_vals, kind='linear', 
                                       fill_value=np.nan, bounds_error=False)
                interpolated = f(x_common)
                all_steps.append(interpolated)
            
            if all_steps:
                # Calculate mean and std only where runs have real data.
                mean_steps, std_steps, steps_valid = summarize_curves(all_steps)
                
                # Plot
                ax.plot(x_common[steps_valid], mean_steps[steps_valid], color=colors[method], linewidth=3,
                    label=display_names.get(method, method))
                ax.fill_between(x_common[steps_valid], mean_steps[steps_valid] - std_steps[steps_valid], mean_steps[steps_valid] + std_steps[steps_valid], 
                              color=colors[method], alpha=0.2)
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=22)
    ax.set_ylabel('Average Steps Survived', fontsize=22)
    ax.set_title('Method Comparison: Average Steps Survived', fontsize=24, fontweight='bold')
    ax.legend(fontsize=18, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pdf_filename = output_dir / f"method_steps_only_{timestamp}.pdf"
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"  📈 Saved steps plot: {pdf_filename}")
    
    png_filename = output_dir / f"method_steps_only_{timestamp}.png"
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"  📈 Saved steps plot: {png_filename}")
    
    plt.close()


def evaluate_experiments(): 
    """Multi-run evaluation for method comparison experiments with organized subfolders"""
    
    print("🚀 Multi-run Method Comparison Evaluation with Confidence Bands")
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
    results_dir = Path(eval_base) / "method_multi_run_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Runs to evaluate with their correct seeds (based on checkpoint sources)
    run_configs = {
        "_5": {
            "MCTS_Root": {"chronic_seed": 123, "main_seed": 2025},  # New training with obs_experiment seeds
            "Heuristic": {"chronic_seed": 123, "main_seed": 2025},  # From observation_experimentsfixed13_5
            "No_Guidance": {"chronic_seed": 123, "main_seed": 2024},  # From queno_finalX_4
        },
        "_6": {
            "MCTS_Root": {"chronic_seed": 271, "main_seed": 2718},  # Existing with obs_experiment seeds 
            "Heuristic": {"chronic_seed": 271, "main_seed": 2718},  # From observation_experimentsfixed13_6
            "No_Guidance": {"chronic_seed": 42, "main_seed": 3141},  # From queno_finalX_5
        },
        "_7": {
            "MCTS_Root": {"chronic_seed": 666, "main_seed": 1729},  # Existing with obs_experiment seeds
            "Heuristic": {"chronic_seed": 666, "main_seed": 1729},  # Existing from method_experiments
            "No_Guidance": {"chronic_seed": 314, "main_seed": 1618},  # From queno_finalX_6
        },
        "_8": {
            "MCTS_Root": {"chronic_seed": 666, "main_seed": 1729},  # New training with obs_experiment seeds
            "Heuristic": {"chronic_seed": 666, "main_seed": 1729},  # From observation_experimentsfixed13_7
            "No_Guidance": {"chronic_seed": 359, "main_seed": 1415},  # From queno_finalX_7
        }
    }
    
    # Note: Each method uses different seeds based on their checkpoint sources:
    # - Heuristic checkpoints: From observation experiments 
    # - No_Guidance checkpoints: From queno-final experiments with different seeds per run
    # - MCTS_Root checkpoints: From method experiments (existing or new training)
    # All methods use their original training seeds for fair evaluation comparison
    
    # Method comparison experiments structure (all three methods)
    experiments = [
        {"name": "MCTS_Root", "subdir": "checkpoints_method_mcts_root"},
        {"name": "Heuristic", "subdir": "checkpoints_method_heuristic"},
        {"name": "No_Guidance", "subdir": "checkpoints_method_no_guidance"}
    ]
    
    all_results = {}
    
    for run_suffix in run_configs.keys():
        print(f"\n🔍 Evaluating run {run_suffix}")
        
        experiment_name = f"method_experiments_fixed{run_suffix}"
        experiment_dir = Path(checkpoint_base) / experiment_name
        
        if not experiment_dir.exists():
            print(f"⚠️  Experiment directory not found: {experiment_dir}")
            continue
            
        run_results = {}
        
        for exp in experiments:
            if exp["name"] not in run_configs[run_suffix]:
                print(f"  ⚠️  No seed configuration for {exp['name']} in run {run_suffix}")
                continue
                
            checkpoint_dir = experiment_dir / exp["subdir"]
            
            if not checkpoint_dir.exists():
                print(f"  ⚠️  Checkpoint dir not found: {checkpoint_dir}")
                continue
                
            # Find checkpoints
            checkpoints = find_checkpoints(checkpoint_dir)
            if not checkpoints:
                print(f"  ⚠️  No checkpoints in {checkpoint_dir}")
                continue
                
            seeds = run_configs[run_suffix][exp["name"]]
            print(f"  📁 Evaluating {exp['name']}: {len(checkpoints)} checkpoints")
            print(f"    🎲 Using seeds: chronic={seeds['chronic_seed']}, main={seeds['main_seed']}")
            
            # Create results file path for this specific experiment + run
            exp_results_file = results_dir / f"{experiment_name}_{exp['name']}_results.json"
            
            # Load existing results for resume capability  
            existing_results = load_existing_results(exp_results_file)
            already_evaluated = set(existing_results.keys())
            
            if already_evaluated:
                print(f"    📋 Found {len(already_evaluated)} already evaluated checkpoints")
            
            # Extract training steps from logs (handle different log sources)
            if exp["name"] == "Heuristic":
                # Heuristic logs came from observation experiments with different naming
                obs_exp_name = f"observation_experimentsfixed13{run_suffix}"
                steps_mapping = extract_training_steps_from_observation_logs(obs_exp_name, "checkpoints_obs_gym")
            elif exp["name"] == "No_Guidance":
                # No_Guidance logs came from queno-final experiments with different naming
                queno_run_mapping = {"_5": "X_4", "_6": "X_5", "_7": "X_6", "_8": "X_7"}
                queno_exp_name = f"queno_final{queno_run_mapping[run_suffix]}"
                steps_mapping = extract_training_steps_from_queno_logs(queno_exp_name, "checkpoints_no_guidance")
            else:
                # MCTS Root logs are either existing or will be generated by new training
                steps_mapping = extract_training_steps_from_method_logs(experiment_name, exp["subdir"])
            
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
                    # Submit evaluation tasks with correct seeds
                    future_to_checkpoint = {}
                    for cp in checkpoints_to_eval:
                        future = executor.submit(
                            evaluate_checkpoint_with_seeds, 
                            cp, 
                            f"{experiment_name}_{exp['name']}",
                            seeds['chronic_seed'],
                            seeds['main_seed']
                        )
                        future_to_checkpoint[future] = cp
                    
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
                                result['chronic_seed'] = seeds['chronic_seed']
                                result['main_seed'] = seeds['main_seed']
                                
                                results_data.append(result)
                                completed_count += 1
                                print(f"    ✅ Checkpoint {checkpoint_num}: Reward {result.get('avg_reward', 0):.1f}, Steps {result.get('avg_steps', 0):.1f}")
                                
                                # Save results periodically (every 10 checkpoints)
                                if completed_count % 10 == 0:
                                    save_partial_results(results_data, exp_results_file)
                                    
                        except Exception as e:
                            print(f"    ❌ Checkpoint {checkpoint_num} evaluation failed: {e}")
            
            # Final save
            save_partial_results(results_data, exp_results_file)
                
            # Store results by experiment name
            if results_data:
                run_results[exp["name"]] = results_data
                print(f"  📊 {exp['name']}: {len(results_data)} evaluations completed")
            else:
                print(f"  📊 {exp['name']}: No successful evaluations")
        
        if run_results:
            all_results[run_suffix] = run_results
    
    # Create comparison plots
    if all_results:
        plot_data = create_method_plots_with_confidence_bands(all_results, results_dir)
        
        # Save complete results
        complete_results_file = results_dir / "complete_method_results.json"
        with open(complete_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Saved complete results to: {complete_results_file}")
        
        print(f"\n📊 Summary:")
        for run_suffix, run_data in all_results.items():
            print(f"  Run {run_suffix}:")
            for exp_name, exp_data in run_data.items():
                print(f"    {exp_name}: {len(exp_data)} checkpoints")
                
        print(f"\n✅ Method comparison evaluation completed!")
        print(f"📁 Results saved to: {results_dir}")
    else:
        print("\n❌ No results to process")


def evaluate_checkpoint_with_seeds(checkpoint_path, experiment_label, chronic_seed, main_seed):
    """Evaluate checkpoint with specific seeds (wrapper around evaluate_checkpoint)"""
    # Set environment variables for the evaluation
    os.environ['EVALUATION_CHRONIC_SEED'] = str(chronic_seed)
    os.environ['EVALUATION_MAIN_SEED'] = str(main_seed)
    
    try:
        result = evaluate_checkpoint(checkpoint_path, experiment_label)
        if result:
            result['evaluation_chronic_seed'] = chronic_seed
            result['evaluation_main_seed'] = main_seed
        return result
    except Exception as e:
        print(f"Error evaluating {checkpoint_path} with seeds chronic={chronic_seed}, main={main_seed}: {e}")
        return None


def extract_training_steps_from_observation_logs(experiment_name, checkpoint_subdir):
    """Extract training steps from observation experiment logs"""
    steps_mapping = {}
    
    log_dir = Path(f"/work/PdF_L2RPN/AlphaZero_L2RPN/logs/{experiment_name}")
    if not log_dir.exists():
        print(f"    ⚠️ Log directory not found: {log_dir}")
        return steps_mapping
    
    # Look for observation experiment log files
    log_files = list(log_dir.glob("exp_obs_space_*.out"))
    if not log_files:
        print(f"    ⚠️ No observation log files found in {log_dir}")
        return steps_mapping
    
    print(f"    📋 Extracting training steps from {len(log_files)} observation log files")
    
    try:
        import re
        for log_file in log_files:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for checkpoint save patterns in observation logs
            checkpoint_matches = re.findall(r'Saved checkpoint (\d+) after (\d+) steps', content)
            
            for checkpoint_str, steps_str in checkpoint_matches:
                checkpoint_num = int(checkpoint_str)
                training_steps = int(steps_str)
                steps_mapping[checkpoint_num] = training_steps
                
        print(f"    📊 Extracted training steps for {len(steps_mapping)} checkpoints from observation logs")
        
    except Exception as e:
        print(f"    ⚠️ Error extracting steps from observation logs: {e}")
    
    return steps_mapping


def extract_training_steps_from_queno_logs(experiment_name, checkpoint_subdir):
    """Extract training steps from queno-final experiment logs"""
    steps_mapping = {}
    
    log_dir = Path(f"/work/PdF_L2RPN/AlphaZero_L2RPN/logs/{experiment_name}")
    if not log_dir.exists():
        print(f"    ⚠️ Log directory not found: {log_dir}")
        return steps_mapping
    
    # Look for queno-final log file patterns
    log_patterns = ["queno_train_*.out", "exp_queno_*.out", "train_*.out", "alphazero_*.out"]
    log_files = []
    for pattern in log_patterns:
        log_files.extend(log_dir.glob(pattern))
    
    if not log_files:
        print(f"    ⚠️ No queno log files found in {log_dir}")
        return steps_mapping
    
    print(f"    📋 Extracting training steps from {len(log_files)} queno log files")
    
    try:
        import re
        for log_file in log_files:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for checkpoint save patterns
            checkpoint_matches = re.findall(r'Saved checkpoint (\d+) after (\d+) steps', content)
            
            for checkpoint_str, steps_str in checkpoint_matches:
                checkpoint_num = int(checkpoint_str)
                training_steps = int(steps_str)
                steps_mapping[checkpoint_num] = training_steps
                
        print(f"    📊 Extracted training steps for {len(steps_mapping)} checkpoints from queno logs")
        
    except Exception as e:
        print(f"    ⚠️ Error extracting steps from queno logs: {e}")
    
    return steps_mapping


def extract_training_steps_from_method_logs(experiment_name, checkpoint_subdir):
    """Extract training steps from method experiment logs"""
    steps_mapping = {}
    
    log_dir = Path(f"/work/PdF_L2RPN/AlphaZero_L2RPN/logs/{experiment_name}")
    if not log_dir.exists():
        print(f"    ⚠️ Log directory not found: {log_dir}")
        return steps_mapping
    
    # Look for various method log file patterns
    log_patterns = ["mcts_root_*.out", "exp_methods_*.out", "train_*.out"]
    log_files = []
    for pattern in log_patterns:
        log_files.extend(log_dir.glob(pattern))
    
    if not log_files:
        print(f"    ⚠️ No method log files found in {log_dir}")
        return steps_mapping
    
    print(f"    📋 Extracting training steps from {len(log_files)} method log files")
    
    try:
        import re
        for log_file in log_files:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for checkpoint save patterns
            checkpoint_matches = re.findall(r'Saved checkpoint (\d+) after (\d+) steps', content)
            
            for checkpoint_str, steps_str in checkpoint_matches:
                checkpoint_num = int(checkpoint_str)
                training_steps = int(steps_str)
                steps_mapping[checkpoint_num] = training_steps
                
        print(f"    📊 Extracted training steps for {len(steps_mapping)} checkpoints from method logs")
        
    except Exception as e:
        print(f"    ⚠️ Error extracting steps from method logs: {e}")
    
    return steps_mapping
def main(plot_only=False, results_file=None):
    """Multi-run evaluation for method comparison experiments with organized subfolders"""
    
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
            for method_name, exp_results in run_data.items():
                experiment_name = f"method_experiments_fixed{run_suffix}"
                
                # Extract training steps for this experiment/run combination  
                steps_mapping = extract_training_steps_from_logs(experiment_name, method_name)
                
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
            plot_data = create_method_plots_with_confidence_bands(all_results, results_dir)
            
            # Create separate steps-only plot
            create_method_steps_only_plot(all_results, results_dir)
            
            print(f"📊 Plots created in: {results_dir}")
        else:
            print("❌ No data found in results file")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-run Method Comparison Experiment Evaluation')
    parser.add_argument('--plot-only', action='store_true', 
                       help='Only create plots from existing results (skip evaluation)')
    parser.add_argument('--results-file', type=str,
                       help='Path to existing results JSON file (required for plot-only mode)')
    
    args = parser.parse_args()
    if args.plot_only:
        main(plot_only=args.plot_only, results_file=args.results_file)
    else:
        evaluate_experiments()