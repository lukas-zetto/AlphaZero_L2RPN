#!/usr/bin/env python3
"""
Multi-run evaluation script that:
1. Uses the correct test chronics based on random split with chronic_seed 
2. Evaluates multiple runs (_5, _6, _7) 
3. Computes statistics (mean, std) across runs
4. Creates plots with error bars/confidence bands
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import random
from concurrent.futures import ProcessPoolExecutor
import torch

# Path setup
sys.path.append('/workspace/src')
sys.path.append('/workspace')
sys.path.append('/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/src')

from src.agent.my_agent import MyCustomAgent
from src.config import AGENT_CONFIG, ENV_CONFIG

# Seed mappings for different runs
RUN_SEEDS = {
    '_5': {'main': 2025, 'chronic': 123},
    '_6': {'main': 2718, 'chronic': 271}, 
    '_7': {'main': 1729, 'chronic': 666}
}

def get_test_chronics_random_split(env, chronic_seed=None, train_test_split=0.9):
    """Get test chronics using NEW random split logic (same as updated training)"""
    import random
    
    total_chronics = len(env.chronics_handler.real_data.subpaths)
    all_chronics = list(range(total_chronics))
    
    # Use seed for deterministic random split if provided
    if chronic_seed is not None:
        random.seed(chronic_seed)
    
    # Randomly shuffle all chronics before splitting 
    random.shuffle(all_chronics)
    
    train_pool_size = int(total_chronics * train_test_split)
    train_chronics = all_chronics[:train_pool_size]  # First X% after shuffle
    test_chronics = all_chronics[train_pool_size:]   # Remaining after shuffle
    
    print(f"🎲 Random split (seed={chronic_seed}):")
    print(f"  Total chronics: {total_chronics}")
    print(f"  Train chronics: {len(train_chronics)} - {sorted(train_chronics)}")
    print(f"  Test chronics: {len(test_chronics)} - {sorted(test_chronics)}")
    
    return test_chronics


def evaluate_single_run(experiment_type, run_suffix, chronic_seed, max_episodes=None):
    """Evaluate a single run with specific chronic_seed"""
    print(f"\n🔍 Evaluating {experiment_type} run {run_suffix} (chronic_seed={chronic_seed})")
    
    # Setup environment
    import grid2op
    from lightsim2grid import LightSimBackend
    from grid2op.Parameters import Parameters
    
    env_name = "l2rpn_case14_sandbox"
    params = Parameters()
    params.MAX_SUB_CHANGED = 1
    params.MAX_LINE_STATUS_CHANGED = 1
    
    env = grid2op.make(
        env_name,
        backend=LightSimBackend(),
        param=params
    )
    
    # Get test chronics using random split with correct seed
    test_chronics = get_test_chronics_random_split(env, chronic_seed)
    
    if max_episodes and max_episodes < len(test_chronics):
        test_chronics = test_chronics[:max_episodes]
    
    # Find model checkpoint 
    checkpoint_base = f"/work/PdF_L2RPN/AlphaZero_L2RPN"
    
    # Different experiment patterns
    if experiment_type == "action_space":
        checkpoint_patterns = [
            f"{checkpoint_base}/checkpoints/action_space_experiments_fixed{run_suffix}/checkpoints_action_sym",
            f"{checkpoint_base}/checkpoints/action_space_experiments_fixed{run_suffix}/checkpoints_action_n0", 
            f"{checkpoint_base}/checkpoints/action_space_experiments_fixed{run_suffix}/checkpoints_action_n1"
        ]
        exp_names = ["SYM", "N0", "N1"]
    elif experiment_type == "reward":
        checkpoint_patterns = [
            f"{checkpoint_base}/checkpoints/reward_experiments_fixed{run_suffix}/checkpoints_reward_alphazero",
            f"{checkpoint_base}/checkpoints/reward_experiments_fixed{run_suffix}/checkpoints_reward_d3qn2022",
            f"{checkpoint_base}/checkpoints/reward_experiments_fixed{run_suffix}/checkpoints_reward_loss"
        ]
        exp_names = ["AlphaZero", "D3QN-2022", "Loss"]
    elif experiment_type == "observation": 
        checkpoint_patterns = [
            f"{checkpoint_base}/checkpoints/observation_experimentsfixed13{run_suffix}/checkpoints_obs_minimal",
            f"{checkpoint_base}/checkpoints/observation_experimentsfixed13{run_suffix}/checkpoints_obs_essential",
            f"{checkpoint_base}/checkpoints/observation_experimentsfixed13{run_suffix}/checkpoints_obs_custom"
        ]
        exp_names = ["Minimal", "Essential", "Custom"]
    elif experiment_type == "method":
        checkpoint_patterns = [
            f"{checkpoint_base}/checkpoints/method_experiments_fixed{run_suffix}/checkpoints_method_mcts_root",
            f"{checkpoint_base}/checkpoints/method_experiments_fixed{run_suffix}/checkpoints_method_heuristic"
        ]
        exp_names = ["MCTS Root", "Heuristic"]
    
    results = {}
    
    for checkpoint_dir, exp_name in zip(checkpoint_patterns, exp_names):
        if not Path(checkpoint_dir).exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_dir}")
            continue
            
        # Find latest checkpoint
        checkpoint_files = list(Path(checkpoint_dir).glob("alphazero_v2_train*.pt"))
        if not checkpoint_files:
            print(f"⚠️  No checkpoints found in {checkpoint_dir}")
            continue
            
        latest_checkpoint = max(checkpoint_files, 
                              key=lambda x: int(x.stem.split('train')[1]))
        
        print(f"📁 Evaluating {exp_name}: {latest_checkpoint}")
        
        # Load agent and evaluate
        try:
            agent = MyCustomAgent(env, str(latest_checkpoint))
            
            episode_scores = []
            episode_lengths = []
            survival_rates = []
            
            for chronic_id in test_chronics:
                env.set_id(chronic_id)
                obs = env.reset()
                
                done = False
                step_count = 0
                episode_score = 0
                
                while not done:
                    action = agent.act(obs, None, None)
                    obs, reward, done, info = env.step(action)
                    episode_score += reward
                    step_count += 1
                    
                    if step_count >= 8064:  # Max episode length
                        break
                
                episode_scores.append(episode_score)
                episode_lengths.append(step_count)
                survival_rates.append(1.0 if step_count >= 8064 else 0.0)
            
            results[exp_name] = {
                'scores': episode_scores,
                'lengths': episode_lengths, 
                'survival_rates': survival_rates,
                'mean_score': np.mean(episode_scores),
                'std_score': np.std(episode_scores),
                'mean_length': np.mean(episode_lengths),
                'survival_rate': np.mean(survival_rates)
            }
            
            print(f"✅ {exp_name}: mean_score={np.mean(episode_scores):.2f}±{np.std(episode_scores):.2f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {exp_name}: {e}")
            
    return results


def plot_multi_run_statistics(all_results, experiment_type, output_dir):
    """Create plots with mean and std across multiple runs"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize data for plotting
    experiments = set()
    for run in all_results:
        experiments.update(all_results[run].keys())
    
    experiments = sorted(list(experiments))
    
    # Collect statistics across runs
    stats_data = []
    
    for exp_name in experiments:
        scores_across_runs = []
        survival_across_runs = []
        lengths_across_runs = []
        
        for run_suffix in all_results:
            if exp_name in all_results[run_suffix]:
                data = all_results[run_suffix][exp_name]
                scores_across_runs.append(data['mean_score'])
                survival_across_runs.append(data['survival_rate']) 
                lengths_across_runs.append(data['mean_length'])
        
        if len(scores_across_runs) > 0:
            stats_data.append({
                'experiment': exp_name,
                'mean_score': np.mean(scores_across_runs),
                'std_score': np.std(scores_across_runs),
                'mean_survival': np.mean(survival_across_runs),
                'std_survival': np.std(survival_across_runs),
                'mean_length': np.mean(lengths_across_runs),
                'std_length': np.std(lengths_across_runs),
                'n_runs': len(scores_across_runs)
            })
    
    df = pd.DataFrame(stats_data)
    
    if len(df) == 0:
        print("⚠️  No evaluation results to plot")
        return None
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Scores with error bars
    axes[0].bar(df['experiment'], df['mean_score'], 
               yerr=df['std_score'], capsize=5, alpha=0.8)
    axes[0].set_title(f'{experiment_type.title()} Experiments - Mean Score')
    axes[0].set_ylabel('Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Survival rates
    axes[1].bar(df['experiment'], df['mean_survival'], 
               yerr=df['std_survival'], capsize=5, alpha=0.8, color='green')
    axes[1].set_title('Survival Rate')
    axes[1].set_ylabel('Survival Rate')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Episode lengths
    axes[2].bar(df['experiment'], df['mean_length'],
               yerr=df['std_length'], capsize=5, alpha=0.8, color='orange')
    axes[2].set_title('Episode Length')
    axes[2].set_ylabel('Steps')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{experiment_type}_multi_run_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics to CSV
    df.to_csv(output_dir / f'{experiment_type}_statistics.csv', index=False)
    
    print(f"📊 Statistics saved to {output_dir}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Multi-run evaluation with statistics')
    parser.add_argument('--experiment-type', required=True, 
                       choices=['action_space', 'reward', 'observation', 'method'])
    parser.add_argument('--runs', nargs='+', default=['_5', '_6', '_7'],
                       help='Run suffixes to evaluate')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='Max episodes per run')
    parser.add_argument('--output-dir', default='eval/multi_run_stats')
    
    args = parser.parse_args()
    
    print(f"🚀 Multi-run evaluation for {args.experiment_type}")
    print(f"📋 Runs: {args.runs}")
    
    all_results = {}
    
    # Evaluate each run
    for run_suffix in args.runs:
        if run_suffix not in RUN_SEEDS:
            print(f"⚠️  Unknown run suffix: {run_suffix}")
            continue
            
        chronic_seed = RUN_SEEDS[run_suffix]['chronic']
        all_results[run_suffix] = evaluate_single_run(
            args.experiment_type, run_suffix, chronic_seed, args.max_episodes
        )
    
    # Generate statistics and plots
    if all_results:
        stats_df = plot_multi_run_statistics(all_results, args.experiment_type, args.output_dir)
        if stats_df is not None:
            print("\\n📈 Multi-run statistics:")
            print(stats_df.to_string(index=False))
        else:
            print("⚠️  No valid results found for plotting")
    else:
        print("❌ No results to analyze")


if __name__ == "__main__":
    main()