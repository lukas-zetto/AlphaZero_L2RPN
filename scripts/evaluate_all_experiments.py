#!/usr/bin/env python3
"""
Comprehensive evaluation system for all experiments.
Each subdirectory of checkpoints/ represents one experiment with multiple runs.
Creates corresponding eval/ subdirectories and evaluates all runs within each experiment.
Plots all runs from the same experiment on combined graphs.
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

# Add the same path setup as training script
sys.path.append('/workspace/src')
sys.path.append('/workspace')
sys.path.append('/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src')


def get_available_cpus():
    """
    Get the number of available CPUs, respecting SLURM and environment settings.
    Priority: SLURM_CPUS_PER_TASK > OMP_NUM_THREADS > os.cpu_count()
    """
    return 15
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
    
    # Fall back to system CPU count
    return os.cpu_count() or 4  # Default to 4 if detection fails

# Import evaluation function from existing script
from scripts.evaluate_all_checkpoints import evaluate_checkpoint, extract_checkpoint_number, find_checkpoints


MAX_CHECKPOINTS_PER_RUN = 30


def extract_training_steps_from_logs(experiment_name, run_name):
    """Extract training steps for each checkpoint from training logs."""
    steps_mapping = {}
    
    # Look for training log files in the logs directory
    # Auto-detect container vs host environment
    if os.path.exists("/work"):
        log_base_dir = Path("/work/PdF_L2RPN/AlphaZero_L2RPN/logs")
    else:
        log_base_dir = Path("/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs")
    
    # For aa_methods, look specifically in the aa_methods subdirectory
    if experiment_name.lower() == 'aa_methods':
        run_name_lower = run_name.lower()
        if 'mcts_root' in run_name_lower:
            log_patterns = [
                'aa_methods/train_mcts_root_*.out',
            ]
        elif 'heuristic' in run_name_lower:
            log_patterns = [
                'aa_methods/exp_obs_space_*.out',
                'method_experiments/exp_methods_*.out',
                '*/exp_methods_*.out',
            ]
        elif 'queno_final' in run_name_lower or 'finalx' in run_name_lower:
            log_patterns = [
                'aa_methods/train_*queno*final*.out',
            ]
        else:
            log_patterns = [
                'aa_methods/*.out',
            ]
    else:
        # Try different log naming patterns (generic + run-specific)
        log_patterns = [
            f"{experiment_name}/exp_*.out",
            f"{experiment_name}_*/exp_*.out",
            f"exp_*_{experiment_name}*.out",
        ]
        
        # Special patterns for specific experiment types
        if 'observation_experiments' in experiment_name:
            log_patterns.extend([
                f"{experiment_name}/exp_obs_space_*.out",
            ])
        elif 'reward_experiments' in experiment_name:
            log_patterns.extend([
                f"{experiment_name}/exp_reward_*.out",
                f"{experiment_name}/exp_*.out",
            ])
        elif 'action_space_experiments' in experiment_name:
            log_patterns.extend([
                f"{experiment_name}/exp_action_*.out",
            ])

        run_name_lower = run_name.lower()
        if 'mcts_root' in run_name_lower:
            log_patterns.extend([
                'train_mcts_root_*.out',
                '**/train_mcts_root_*.out',
            ])
        if 'heuristic' in run_name_lower:
            log_patterns.extend([
                'train_*heuristic*.out',
                '**/train_*heuristic*.out',
                '*obs*space*.out',
                '**/*obs*space*.out',
            ])
        if 'queno_final' in run_name_lower or 'finalx' in run_name_lower:
            log_patterns.extend([
                'train_*queno*final*.out',
                '**/train_*queno*final*.out',
                '*finalX*.out',
                '**/*finalX*.out',
                '*finalx*.out',
                '**/*finalx*.out',
            ])

    log_files = []
    seen = set()
    for pattern in log_patterns:
        matches = list(log_base_dir.glob(pattern))
        for lf in matches:
            lf_str = str(lf)
            if lf_str not in seen:
                seen.add(lf_str)
                log_files.append(lf)

    for log_file in log_files:
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Pattern: "💾 Checkpoint saved: .../run_name/alphazero_v2_trainX.pt\n   Steps: XXX,XXX | ..."
            # Handle both absolute paths (/work/.../checkpoints_run_name/) and relative (checkpoints_run_name/)
            
            # Try multiple patterns in order of specificity
            checkpoint_patterns = [
                # Exact run_name match in path
                r'💾\s*Checkpoint saved:\s*([^\s]*' + re.escape(run_name) + r'[^\s]*/alphazero_v2_train(\d+)\.pt)\s*\n\s*Steps:\s*([\d,]+)',
                # General pattern for any alphazero checkpoint, then filter by run_name
                r'💾\s*Checkpoint saved:\s*([^\s]*alphazero_v2_train(\d+)\.pt)\s*\n\s*Steps:\s*([\d,]+)',
            ]
            
            for pattern in checkpoint_patterns:
                matches = re.findall(pattern, content)
                
                if pattern == checkpoint_patterns[0]:  # Exact match pattern
                    # Use all matches since they already match run_name
                    checkpoint_matches = matches
                else:  # General pattern - filter by run_name
                    # Filter matches that contain the run_name in the path
                    checkpoint_matches = [(path, num, steps) for path, num, steps in matches if run_name in path]
                
                if checkpoint_matches:
                    matches = checkpoint_matches
                    break
            
            # Fallback: if no matches with run_name filter, try without run_name filtering
            # but only for action space experiments where we know the structure
            if not matches and 'action_space_experiments' in experiment_name:
                general_pattern = r'💾\s*Checkpoint saved:\s*([^\s]*alphazero_v2_train(\d+)\.pt)\s*\n\s*Steps:\s*([\d,]+)'
                all_matches = re.findall(general_pattern, content)
                
                # For action space experiments, try to match by action type in path
                if 'action_sym' in run_name.lower():
                    matches = [(path, num, steps) for path, num, steps in all_matches if 'action_sym' in path]
                elif 'action_n0' in run_name.lower():
                    matches = [(path, num, steps) for path, num, steps in all_matches if 'action_n0' in path]
                elif 'action_n1' in run_name.lower():
                    matches = [(path, num, steps) for path, num, steps in all_matches if 'action_n1' in path]
                else:
                    # Use all matches from this log file
                    matches = all_matches

            for match in matches:
                if len(match) == 3:  # (path, checkpoint_num_str, steps_str)
                    _, checkpoint_num_str, steps_str = match
                else:  # fallback format
                    checkpoint_num_str, steps_str = match[1], match[2]
                
                checkpoint_num = int(checkpoint_num_str)
                steps = int(steps_str.replace(',', ''))
                steps_mapping[checkpoint_num] = steps

        except Exception as e:
            print(f"Warning: Could not read log file {log_file}: {e}")
            continue
    
    if steps_mapping:
        print(f"  ✓ Extracted training steps for {len(steps_mapping)} checkpoints from logs (filtered for {run_name})")
    else:
        print(f"  ⚠️ Could not extract training steps from logs for {experiment_name}/{run_name}")
    
    return steps_mapping


def discover_experiments(checkpoints_base_dir):
    """Discover all experiment directories and their runs."""
    checkpoints_base = Path(checkpoints_base_dir)
    experiments = {}
    
    if not checkpoints_base.exists():
        print(f"Checkpoints directory {checkpoints_base} does not exist!")
        return experiments
    
    # Each subdirectory is an experiment
    for experiment_dir in checkpoints_base.iterdir():
        if experiment_dir.is_dir():
            experiment_name = experiment_dir.name
            runs = {}
            
            # Each subdirectory within the experiment is a run
            for run_dir in experiment_dir.iterdir():
                if run_dir.is_dir():
                    run_name = run_dir.name
                    checkpoints = find_checkpoints(run_dir)
                    if checkpoints:
                        runs[run_name] = {
                            'path': run_dir,
                            'checkpoints': checkpoints
                        }
            
            if runs:
                experiments[experiment_name] = runs
    
    return experiments


def evaluate_experiment_run(experiment_name, run_name, run_data, eval_base_dir, obs_space_type=None):
    """Evaluate all checkpoints in a single experiment run."""
    print(f"\n🔍 Evaluating {experiment_name}/{run_name}...")
    
    # Create eval directory for this run
    eval_run_dir = Path(eval_base_dir) / experiment_name / run_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Results file for this run
    results_file = eval_run_dir / "checkpoint_evaluations.json"
    
    checkpoints = sorted(
        run_data['checkpoints'],
        key=lambda cp: (extract_checkpoint_number(cp) is None, extract_checkpoint_number(cp))
    )[:MAX_CHECKPOINTS_PER_RUN]
    allowed_checkpoint_nums = {
        extract_checkpoint_number(cp)
        for cp in checkpoints
        if extract_checkpoint_number(cp) is not None
    }

    # Load existing results if available
    existing_results = {}
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results_list = json.load(f)
                existing_results = {
                    r['checkpoint_num']: r
                    for r in results_list
                    if r.get('checkpoint_num') is not None and r['checkpoint_num'] in allowed_checkpoint_nums
                }
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    # Find checkpoints to evaluate
    checkpoints_to_eval = []
    
    for cp in checkpoints:
        cp_num = extract_checkpoint_number(cp)
        if cp_num is not None and cp_num not in existing_results:
            checkpoints_to_eval.append(cp)
    
    total_available = len(run_data['checkpoints'])
    print(f"  Using first {len(checkpoints)} checkpoints out of {total_available} available")
    print(f"  Found {len(checkpoints)} selected checkpoints, {len(checkpoints_to_eval)} to evaluate")
    
    # Evaluate missing checkpoints
    results = list(existing_results.values())
    
    if checkpoints_to_eval:
        # Evaluate in parallel - respect SLURM/environment CPU allocation
        available_cpus = get_available_cpus()
        max_workers = min(available_cpus, len(checkpoints_to_eval))
        print(f"  Evaluating with {max_workers} parallel workers (detected {available_cpus} available CPUs)...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_checkpoint = {
                executor.submit(evaluate_checkpoint, cp, f"{experiment_name}_{run_name}", obs_space_type): cp 
                for cp in checkpoints_to_eval
            }
            
            for future in as_completed(future_to_checkpoint):
                checkpoint = future_to_checkpoint[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result['success'] and result['avg_reward'] is not None:
                        print(f"    ✓ Checkpoint {result['checkpoint_num']}: reward={result['avg_reward']:.2f}")
                    else:
                        print(f"    ✗ Checkpoint {result['checkpoint_num']}: evaluation failed")
                except Exception as e:
                    print(f"    ✗ Checkpoint {checkpoint.name}: {e}")
    
    # Extract training steps from logs
    steps_mapping = extract_training_steps_from_logs(experiment_name, run_name)
    
    # Add training steps to results
    for result in results:
        checkpoint_num = result.get('checkpoint_num')
        if checkpoint_num is not None and checkpoint_num in steps_mapping:
            result['training_steps'] = steps_mapping[checkpoint_num]
        else:
            result['training_steps'] = None
    
    # Sort and save results
    results_by_num = {}
    for r in results:
        if r.get('checkpoint_num') is not None:
            results_by_num[r['checkpoint_num']] = r
    
    final_results = list(results_by_num.values())
    final_results.sort(key=lambda x: x['checkpoint_num'])
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save CSV with all important fields
    csv_file = eval_run_dir / "checkpoint_evaluations.csv"
    df = pd.DataFrame(final_results)
    
    # Ensure key columns are present and properly formatted
    if 'training_steps' in df.columns:
        # Convert training_steps to numeric, handling None values
        df['training_steps'] = pd.to_numeric(df['training_steps'], errors='coerce')
    
    # Reorder columns for better readability
    preferred_columns = [
        'checkpoint_num', 'training_steps', 'success', 'avg_reward', 'avg_steps', 
        'total_episodes', 'survived_episodes', 'evaluation_time'
    ]
    available_columns = [col for col in preferred_columns if col in df.columns]
    other_columns = [col for col in df.columns if col not in preferred_columns]
    column_order = available_columns + other_columns
    
    df = df[column_order]
    df.to_csv(csv_file, index=False)
    
    print(f"  ✓ Saved {len(final_results)} results to {results_file}")
    print(f"  ✓ Saved CSV with training steps to {csv_file}")
    
    return final_results


def create_experiment_comparison_plots(experiment_name, all_runs_data, eval_experiment_dir, max_training_steps=None):
    """Create comparison plots for all runs within an experiment.
    
    Args:
        experiment_name: Name of the experiment
        all_runs_data: Dictionary of run data
        eval_experiment_dir: Directory to save plots
        max_training_steps: Maximum training steps to plot (None for no limit)
    """
    print(f"\n📊 Creating comparison plots for {experiment_name}...")
    if max_training_steps:
        print(f"  Limiting plot to {max_training_steps/1e6:.1f}M training steps")

    experiment_name_lower = experiment_name.lower()
    is_aa_methods = experiment_name_lower == 'aa_methods'
    if 'action_space' in experiment_name_lower:
        primary_title = 'Action Space comparison'
    elif 'observation' in experiment_name_lower:
        primary_title = 'Observation Space comparison'
    elif 'batch_size' in experiment_name_lower:
        primary_title = 'Batch size comparison'
    elif 'reward' in experiment_name_lower:
        primary_title = 'Reward comparison'
    elif 'safety' in experiment_name_lower:
        primary_title = 'Early stopping parameter comparison'
    elif 'horizon' in experiment_name_lower:
        primary_title = 'Horizon parameter comparison'
    elif 'topology_reset' in experiment_name_lower:
        primary_title = 'Topology Reset Threshold comparison'
    else:
        primary_title = 'Average Reward comparison'
    
    plt.style.use('default')
    
    # Prepare data for plotting
    plot_data = {}
    
    for run_name, results in all_runs_data.items():
        successful_results = [r for r in results if r['success'] and r['avg_reward'] is not None]
        if successful_results:
            # Use training steps if available, fall back to checkpoint numbers
            has_steps = any(r.get('training_steps') is not None for r in successful_results)
            
            if has_steps:
                # Filter out results without training steps and sort by steps
                results_with_steps = [r for r in successful_results if r.get('training_steps') is not None]
                
                # Apply max_training_steps filter if specified
                if max_training_steps:
                    results_with_steps = [r for r in results_with_steps if r['training_steps'] <= max_training_steps]
                
                results_with_steps.sort(key=lambda x: x['training_steps'])
                
                # Convert training steps to millions for better readability
                x_values = [r['training_steps'] / 1e6 for r in results_with_steps]
                avg_rewards = [r['avg_reward'] for r in results_with_steps]
                avg_steps = [r.get('avg_steps') or 0 for r in results_with_steps]
                survival_rates = [
                    (r['survived_episodes'] / r['total_episodes'] * 100) 
                    if r.get('survived_episodes') and r.get('total_episodes') 
                    else None
                    for r in results_with_steps
                ]
                x_label = 'Training Steps (Millions)'
            else:
                # Fallback to checkpoint numbers
                checkpoint_nums = [r['checkpoint_num'] for r in successful_results]
                avg_rewards = [r['avg_reward'] for r in successful_results]
                avg_steps = [r.get('avg_steps') or 0 for r in successful_results]
                survival_rates = [
                    (r['survived_episodes'] / r['total_episodes'] * 100) 
                    if r.get('survived_episodes') and r.get('total_episodes') 
                    else None
                    for r in successful_results
                ]
                x_values = checkpoint_nums
                x_label = 'Checkpoint Number (Training Iteration)'
            
            plot_data[run_name] = {
                'x_values': x_values,
                'x_label': x_label,
                'avg_rewards': avg_rewards,
                'avg_steps': avg_steps,
                'survival_rates': survival_rates
            }
    
    if not plot_data:
        print(f"  No successful results to plot for {experiment_name}")
        return
    
    def clean_label(name):
        """Clean legend labels and apply experiment-specific aliases."""
        if is_aa_methods:
            alias = {
                'checkpoints_queno_finalX': 'IL variant',
                'checkpoints_method_heuristic': 'Heuristic variant',
                'checkpoints_mcts_root': 'Learned Q-value Variant'
            }
            return alias.get(name, name.replace('checkpoints_', ''))
        if 'batch_size' in experiment_name_lower:
            match = re.search(r'episodes(\d+)', name)
            if match:
                return f'h = {int(match.group(1))}'
            return name.replace('checkpoints_', '').replace('episodes', 'h = ')
        if 'observation' in experiment_name_lower:
            alias = {
                'checkpoints_obs_essential': 'reduced',
                'checkpoints_obs_gym': 'complete'
            }
            default_name = name.replace('checkpoints_', '').replace('obs_', '')
            return alias.get(name, default_name)
        if 'safety' in experiment_name_lower:
            # Extract t_skipped and t_stopping from names like 'checkpoints_safety_200_50'
            match = re.search(r'safety_(\d+)_(\d+)', name)
            if match:
                t_skipped, t_stopping = match.groups()
                return f't_skipped={t_skipped}, t_stopping={t_stopping}'
            return name.replace('checkpoints_', '').replace('safety_', '')
        if 'horizon' in experiment_name_lower:
            # Extract horizon value from names like 'checkpoints_horizon30'
            match = re.search(r'horizon(\d+)', name)
            if match:
                horizon_val = match.group(1)
                return f'Horizon = {horizon_val}'
            return name.replace('checkpoints_', '').replace('horizon', '')
        if 'topology_reset' in experiment_name_lower:
            # Extract threshold value from names like 'checkpoints_topo_reset_085'
            match = re.search(r'topo_reset_(\d+)', name)
            if match:
                threshold_val = match.group(1)
                # Convert 085 -> 0.85
                threshold_float = int(threshold_val) / 100.0
                return f'Reset Threshold = {threshold_float}'
            return name.replace('checkpoints_', '').replace('topo_reset_', '')
        return name.replace('checkpoints_', '')
    
    # Sort plot_data items for experiments with numeric parameters to show in ascending order
    if 'batch_size' in experiment_name_lower:
        def get_batch_size_sort_key(item):
            run_name = item[0]
            match = re.search(r'episodes(\d+)', run_name)
            if match:
                return int(match.group(1))
            return 0
        plot_data_items = sorted(plot_data.items(), key=get_batch_size_sort_key)
    elif 'safety' in experiment_name_lower:
        def get_safety_sort_key(item):
            run_name = item[0]
            match = re.search(r'safety_(\d+)_(\d+)', run_name)
            if match:
                t_skipped, t_stopping = map(int, match.groups())
                return (t_skipped, t_stopping)  # Sort by t_skipped first, then t_stopping
            return (0, 0)  # Default if no match
        plot_data_items = sorted(plot_data.items(), key=get_safety_sort_key)
    elif 'horizon' in experiment_name_lower:
        def get_horizon_sort_key(item):
            run_name = item[0]
            match = re.search(r'horizon(\d+)', run_name)
            if match:
                horizon_val = int(match.group(1))
                return horizon_val
            return 0  # Default if no match
        plot_data_items = sorted(plot_data.items(), key=get_horizon_sort_key)
    elif 'topology_reset' in experiment_name_lower:
        def get_topology_reset_sort_key(item):
            run_name = item[0]
            match = re.search(r'topo_reset_(\d+)', run_name)
            if match:
                threshold_val = int(match.group(1))
                return threshold_val  # Sort by threshold value (075, 080, 085, etc.)
            return 0  # Default if no match
        plot_data_items = sorted(plot_data.items(), key=get_topology_reset_sort_key)
    else:
        plot_data_items = plot_data.items()

    def get_line_color(run_name, index):
        if is_aa_methods or 'method' in experiment_name_lower:
            method_colors = {
                'checkpoints_method_heuristic': '#00838F',
                'checkpoints_queno_finalX': '#1565C0',
                'checkpoints_mcts_root': '#F9A825',
            }
            if run_name in method_colors:
                return method_colors[run_name]
            if 'heuristic' in run_name.lower():
                return '#00838F'
        if 'batch_size' in experiment_name_lower:
            batch_colors = {
                50: '#1565C0',
                150: '#F9A825',
                300: '#00838F',
                400: '#6A1B9A',
                903: '#C62828',
            }
            match = re.search(r'episodes(\d+)', run_name)
            if match:
                return batch_colors.get(int(match.group(1)), '#444444')
        if 'safety' in experiment_name_lower:
            safety_colors = {
                (10, 5): '#1565C0',
                (50, 10): '#00838F',
                (100, 20): '#F9A825',
                (200, 20): '#6A1B9A',
                (200, 50): '#C62828',
            }
            match = re.search(r'safety_(\d+)_(\d+)', run_name)
            if match:
                return safety_colors.get((int(match.group(1)), int(match.group(2))), '#444444')
        if 'horizon' in experiment_name_lower:
            horizon_colors = {
                10: '#1565C0',
                30: '#00838F',
                100: '#F9A825',
                200: '#6A1B9A',
                500: '#C62828',
            }
            match = re.search(r'horizon(\d+)', run_name)
            if match:
                return horizon_colors.get(int(match.group(1)), '#444444')
        if 'topology_reset' in experiment_name_lower:
            reset_colors = {
                75: '#00838F',
                80: '#1565C0',
                85: '#F9A825',
                90: '#6A1B9A',
                95: '#C62828',
            }
            match = re.search(r'topo_reset_(\d+)', run_name)
            if match:
                return reset_colors.get(int(match.group(1)), '#444444')
        fallback_colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data_items)))
        return fallback_colors[index]
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    
    # Determine x-axis label (all runs should use the same type)
    x_label = list(plot_data.values())[0]['x_label'] if plot_data else 'Training Progress'
    
    # Plot 1: Average Reward Comparison
    axes[0].set_title(primary_title, fontsize=22, fontweight='bold')
    for i, (run_name, data) in enumerate(plot_data_items):
        line_color = get_line_color(run_name, i)
        axes[0].plot(data['x_values'], data['avg_rewards'], 
                    marker='o', linewidth=2, markersize=4, 
                    label=clean_label(run_name), color=line_color)
    axes[0].set_xlabel(x_label, fontsize=18)
    axes[0].set_ylabel('Average Reward', fontsize=18)
    axes[0].legend(fontsize=16, loc='lower right')
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average Steps Comparison
    if 'reward' in experiment_name.lower():
        steps_title = 'Reward comparison'
    elif 'action_space' in experiment_name.lower():
        steps_title = 'Action space comparison'
    elif 'observation' in experiment_name.lower():
        steps_title = 'Observation space comparison'
    elif 'batch_size' in experiment_name.lower():
        steps_title = 'Batch size comparison'
    elif 'safety' in experiment_name.lower():
        steps_title = 'Early stopping parameter comparison'
    elif 'horizon' in experiment_name.lower():
        steps_title = 'Horizon parameter comparison'
    elif 'topology_reset' in experiment_name.lower():
        steps_title = 'Topology reset threshold comparison'
    else:
        steps_title = 'Method comparison'
    axes[1].set_title(steps_title, fontsize=22, fontweight='bold')
    for i, (run_name, data) in enumerate(plot_data_items):
        if any(s > 0 for s in data['avg_steps']):
            line_color = get_line_color(run_name, i)
            axes[1].plot(data['x_values'], data['avg_steps'], 
                        marker='s', linewidth=2, markersize=4, 
                        label=clean_label(run_name), color=line_color)
    axes[1].set_xlabel(x_label, fontsize=18)
    axes[1].set_ylabel('Average Steps Survived', fontsize=18)
    axes[1].legend(fontsize=16, loc='lower right')
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Survival Rate Comparison  
    axes[2].set_title(primary_title, fontsize=22, fontweight='bold')
    for i, (run_name, data) in enumerate(plot_data_items):
        valid_survival = [(x, sr) for x, sr in zip(data['x_values'], data['survival_rates']) if sr is not None]
        if valid_survival:
            x_vals, rates = zip(*valid_survival)
            line_color = get_line_color(run_name, i)
            axes[2].plot(x_vals, rates, marker='^', linewidth=2, markersize=4, 
                        label=clean_label(run_name), color=line_color)
    axes[2].set_xlabel(x_label, fontsize=18)
    axes[2].set_ylabel('Survival Rate (%)', fontsize=18)
    axes[2].legend(fontsize=16, loc='lower right')
    axes[2].tick_params(axis='both', labelsize=16)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_file = eval_experiment_dir / f"{experiment_name}_comparison.svg"
    plt.savefig(comparison_file, format='svg', bbox_inches='tight')
    plt.savefig(comparison_file.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    print(f"  ✓ Saved comparison plot to {comparison_file}")
    plt.close()
    
    # Create individual metric plots
    for metric in ['avg_rewards', 'avg_steps', 'survival_rates']:
        plt.figure(figsize=(12, 8))
        
        metric_titles = {
            'avg_rewards': primary_title,
            'avg_steps': 'Method comparison',
            'survival_rates': primary_title
        }
        if 'reward' in experiment_name_lower:
            metric_titles['avg_steps'] = 'Reward comparison'
        if 'observation' in experiment_name_lower:
            metric_titles['avg_steps'] = 'Observation space comparison'
        if 'batch_size' in experiment_name_lower:
            metric_titles['avg_rewards'] = 'Batch size comparison'
            metric_titles['avg_steps'] = 'Batch size comparison'
            metric_titles['survival_rates'] = 'Batch size comparison'
        if 'action_space' in experiment_name_lower:
            metric_titles['avg_steps'] = 'Action space comparison'
            metric_titles['avg_rewards'] = 'Action Space comparison'
        if 'safety' in experiment_name_lower:
            metric_titles['avg_steps'] = 'Early stopping parameter comparison'
        if 'horizon' in experiment_name_lower:
            metric_titles['avg_steps'] = 'Horizon parameter comparison'
        if 'topology_reset' in experiment_name_lower:
            metric_titles['avg_steps'] = 'Topology reset threshold comparison'
        plt.title(metric_titles.get(metric, metric.replace('_', ' ').title()), fontsize=22, fontweight='bold')
        
        for i, (run_name, data) in enumerate(plot_data_items):
            line_color = get_line_color(run_name, i)
            if metric == 'avg_rewards':
                plt.plot(data['x_values'], data['avg_rewards'], 
                        marker='o', linewidth=3, markersize=6, 
                        label=clean_label(run_name), color=line_color)
                plt.ylabel('Average Reward', fontsize=18)
            elif metric == 'avg_steps':
                if any(s > 0 for s in data['avg_steps']):
                    plt.plot(data['x_values'], data['avg_steps'], 
                            marker='s', linewidth=3, markersize=6, 
                            label=clean_label(run_name), color=line_color)
                plt.ylabel('Average Steps Survived', fontsize=18)
            elif metric == 'survival_rates':
                valid_survival = [(x, sr) for x, sr in zip(data['x_values'], data['survival_rates']) if sr is not None]
                if valid_survival:
                    x_vals, rates = zip(*valid_survival)
                    plt.plot(x_vals, rates, marker='^', linewidth=3, markersize=6, 
                            label=clean_label(run_name), color=line_color)
                plt.ylabel('Survival Rate (%)', fontsize=18)
                plt.ylim([0, 105])
        
        plt.xlabel(x_label, fontsize=18)
        plt.legend(fontsize=16, loc='lower right')
        plt.tick_params(axis='both', labelsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save individual metric plot
        metric_file = eval_experiment_dir / f"{experiment_name}_{metric}.svg"
        plt.savefig(metric_file, format='svg', bbox_inches='tight')
        plt.savefig(metric_file.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
        print(f"  ✓ Saved {metric_titles.get(metric, metric)} plot to {metric_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate all experiments with multiple runs')
    parser.add_argument('--checkpoints-dir', type=str, 
                       default='/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/checkpoints',
                       help='Base checkpoints directory containing experiment subdirs')
    parser.add_argument('--eval-dir', type=str,
                       default='/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/eval',
                       help='Base eval directory to create experiment subdirs')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Evaluate only specific experiment (default: all experiments)')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only create plots from existing results, do not re-evaluate')
    parser.add_argument('--max-parallel-experiments', type=int, default=3,
                       help='Maximum number of experiments to evaluate in parallel')
    parser.add_argument('--obs_space_type', type=str, default=None,
                       help='Override observation space type (minimal, essential, custom, gym)')
    
    args = parser.parse_args()
    
    # Ensure base eval directory exists
    os.makedirs(args.eval_dir, exist_ok=True)
    try:
        os.makedirs("/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval", exist_ok=True)
    except PermissionError:
        print("Warning: Cannot create /home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/eval; continuing with local paths.")
    
    print("="*80)
    print("COMPREHENSIVE EXPERIMENT EVALUATION")
    print("="*80)
    print(f"Checkpoints directory: {args.checkpoints_dir}")
    print(f"Eval directory: {args.eval_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Discover all experiments
    experiments = discover_experiments(args.checkpoints_dir)
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"\nDiscovered {len(experiments)} experiments:")
    for exp_name, runs in experiments.items():
        print(f"  📁 {exp_name}: {len(runs)} runs")
        for run_name in runs.keys():
            print(f"    └── {run_name}")
    
    # Filter by specific experiment if requested
    if args.experiment:
        if args.experiment in experiments:
            experiments = {args.experiment: experiments[args.experiment]}
            print(f"\nEvaluating only experiment: {args.experiment}")
        else:
            print(f"\nExperiment '{args.experiment}' not found!")
            return
    
    # Process each experiment
    for experiment_name, runs in experiments.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING EXPERIMENT: {experiment_name}")
        print(f"{'='*60}")
        
        # Create experiment eval directory
        eval_experiment_dir = Path(args.eval_dir) / experiment_name
        eval_experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate or load results for each run
        all_runs_data = {}
        
        if not args.plot_only:
            # Evaluate all runs in this experiment
            for run_name, run_data in runs.items():
                results = evaluate_experiment_run(experiment_name, run_name, run_data, args.eval_dir, args.obs_space_type)
                all_runs_data[run_name] = results
        else:
            # Load existing results for plotting
            for run_name in runs.keys():
                results_file = eval_experiment_dir / run_name / "checkpoint_evaluations.json"
                csv_file = eval_experiment_dir / run_name / "checkpoint_evaluations.csv"
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            results = json.load(f)

                            # Backfill training_steps from logs when missing (common for older evals)
                            if any(r.get('training_steps') is None for r in results):
                                steps_mapping = extract_training_steps_from_logs(experiment_name, run_name)
                                if steps_mapping:
                                    for r in results:
                                        ckpt_num = r.get('checkpoint_num')
                                        if ckpt_num is not None and r.get('training_steps') is None:
                                            r['training_steps'] = steps_mapping.get(ckpt_num)

                            all_runs_data[run_name] = results
                    except Exception as e:
                        print(f"Warning: Could not load results for {experiment_name}/{run_name}: {e}")
                elif csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        results = df.to_dict(orient='records')
                        all_runs_data[run_name] = results
                    except Exception as e:
                        print(f"Warning: Could not load CSV for {experiment_name}/{run_name}: {e}")
        
        # Create comparison plots for this experiment
        if all_runs_data:
            # Set max training steps for specific experiments
            max_steps = None
            if 'reward' in experiment_name.lower():
                max_steps = 0.3 * 1e8  # 30 million steps for reward experiments
            elif 'action_space' in experiment_name.lower():
                max_steps = 0.8 * 1e8  # 80 million steps for action space experiments
            elif any(x in experiment_name.lower() for x in ['horizon', 'safety', 'topology_reset']):
                max_steps = 0.4 * 1e8  # 40 million steps for horizon, safety, topology reset experiments
            
            create_experiment_comparison_plots(experiment_name, all_runs_data, eval_experiment_dir, max_training_steps=max_steps)
            
            # Create comprehensive CSV combining all runs
            all_data_rows = []
            for run_name, results in all_runs_data.items():
                for result in results:
                    row = result.copy()
                    row['experiment_name'] = experiment_name
                    row['run_name'] = run_name
                    all_data_rows.append(row)
            
            if all_data_rows:
                combined_csv = eval_experiment_dir / f"{experiment_name}_all_runs_data.csv"
                combined_df = pd.DataFrame(all_data_rows)
                
                # Reorder columns for experiment-level CSV
                exp_preferred_columns = [
                    'experiment_name', 'run_name', 'checkpoint_num', 'training_steps', 
                    'success', 'avg_reward', 'avg_steps', 'total_episodes', 'survived_episodes',
                    'evaluation_time'
                ]
                exp_available_columns = [col for col in exp_preferred_columns if col in combined_df.columns]
                exp_other_columns = [col for col in combined_df.columns if col not in exp_preferred_columns]
                exp_column_order = exp_available_columns + exp_other_columns
                
                combined_df = combined_df[exp_column_order]
                combined_df.to_csv(combined_csv, index=False)
                print(f"  ✓ Saved combined runs data to {combined_csv}")
            
            # Create summary statistics
            summary_file = eval_experiment_dir / "experiment_summary.json"
            summary = {
                'experiment_name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'runs': {}
            }
            
            for run_name, results in all_runs_data.items():
                successful = [r for r in results if r['success'] and r['avg_reward'] is not None]
                if successful:
                    rewards = [r['avg_reward'] for r in successful]
                    steps_survived = [r.get('avg_steps', 0) for r in successful if r.get('avg_steps') is not None]
                    training_steps = [r.get('training_steps') for r in successful if r.get('training_steps') is not None]
                    
                    summary['runs'][run_name] = {
                        'total_checkpoints': len(results),
                        'successful_evaluations': len(successful),
                        'best_reward': max(rewards),
                        'final_reward': rewards[-1] if rewards else None,
                        'average_reward': np.mean(rewards),
                        'best_steps_survived': max(steps_survived) if steps_survived else None,
                        'final_steps_survived': steps_survived[-1] if steps_survived else None,
                        'average_steps_survived': np.mean(steps_survived) if steps_survived else None,
                        'max_training_steps': max(training_steps) if training_steps else None,
                        'total_training_steps_evaluated': len(training_steps)
                    }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"  ✓ Saved experiment summary to {summary_file}")
        
    # Create global summary CSV across all experiments
    global_summary_rows = []
    for experiment_name, runs in experiments.items():
        eval_experiment_dir = Path(args.eval_dir) / experiment_name
        
        # Try to load the combined CSV for this experiment
        combined_csv = eval_experiment_dir / f"{experiment_name}_all_runs_data.csv"
        if combined_csv.exists():
            try:
                exp_df = pd.read_csv(combined_csv)
                global_summary_rows.append(exp_df)
            except Exception as e:
                print(f"Warning: Could not load combined CSV for {experiment_name}: {e}")
    
    if global_summary_rows:
        global_df = pd.concat(global_summary_rows, ignore_index=True)
        global_csv = Path(args.eval_dir) / "all_experiments_summary.csv"
        global_df.to_csv(global_csv, index=False)
        print(f"\n  ✓ Created global summary CSV: {global_csv}")
        print(f"    Contains {len(global_df)} total evaluation records across all experiments")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {args.eval_dir}")
    print("Each experiment has its own subdirectory with:")
    print("  - Individual run evaluations (JSON + CSV)")
    print("  - Combined runs CSV with training steps")
    print("  - Comparison plots combining all runs")
    print("  - Summary statistics")
    print("\nGlobal files:")
    print("  - all_experiments_summary.csv: Complete dataset for analysis")


if __name__ == "__main__":
    main()