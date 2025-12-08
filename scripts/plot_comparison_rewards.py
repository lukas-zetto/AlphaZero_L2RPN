#!/usr/bin/env python3
"""
Plot average rewards over iterations for all 5 value target methods.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_rewards_from_log(log_path):
    """Extract iteration average rewards from training log."""
    iterations = []
    avg_rewards = []
    
    with open(log_path, 'r') as f:
        current_iteration = None
        iteration_rewards = []
        
        for line in f:
            # Detect iteration start
            iter_match = re.search(r'ITERATION (\d+)/\d+', line)
            if iter_match:
                # Save previous iteration's data
                if current_iteration is not None and iteration_rewards:
                    iterations.append(current_iteration)
                    avg_rewards.append(np.mean(iteration_rewards))
                
                current_iteration = int(iter_match.group(1))
                iteration_rewards = []
            
            # Look for episode completion with rewards
            # Pattern: "Total reward: 1234.56"
            if current_iteration is not None:
                reward_match = re.search(r'Total reward:\s*([-+]?\d+\.?\d*)', line)
                if reward_match:
                    reward = float(reward_match.group(1))
                    iteration_rewards.append(reward)
        
        # Save last iteration
        if current_iteration is not None and iteration_rewards:
            iterations.append(current_iteration)
            avg_rewards.append(np.mean(iteration_rewards))
    
    return iterations, avg_rewards


def main():
    methods = ['mcts_root', 'binary_root', 'mcts_all_nodes', 'binary_all_nodes', 'heuristic']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    plt.figure(figsize=(12, 7))
    
    for method, color in zip(methods, colors):
        log_path = Path(f'/workspace/value_method_comparison_{method}/training.log')
        
        if not log_path.exists():
            print(f"Warning: Log file not found for {method}")
            continue
        
        print(f"Processing {method}...")
        iterations, rewards = extract_rewards_from_log(log_path)
        
        if iterations:
            plt.plot(iterations, rewards, marker='o', label=method, color=color, linewidth=2, markersize=4)
            print(f"  {method}: {len(iterations)} iterations, avg reward = {np.mean(rewards):.1f}")
        else:
            print(f"  {method}: No data found")
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Cumulative Reward', fontsize=12)
    plt.title('Value Target Method Comparison: Cumulative Reward Over Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = '/workspace/value_method_comparison_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
