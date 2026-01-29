#!/usr/bin/env python3
"""
Plot training progress from saved checkpoints.
Shows performance metrics vs environment steps.
"""

import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_checkpoint_metrics(checkpoint_path):
    """Load metrics from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract key metrics
        metrics = {
            'training_iteration': checkpoint.get('training_iteration', 0),
            'total_steps': checkpoint.get('total_steps', 0),
            'episodes_collected': checkpoint.get('episodes_collected', 0),
        }
        
        # Extract performance metrics if available
        if 'performance' in checkpoint:
            perf = checkpoint['performance']
            metrics['avg_reward'] = perf.get('avg_reward_recent', 0.0)
            metrics['success_rate'] = perf.get('success_rate_recent', 0.0)
            metrics['avg_episode_length'] = perf.get('avg_episode_length_recent', 0.0)
        else:
            # Fallback if performance not tracked
            metrics['avg_reward'] = None
            metrics['success_rate'] = None
            metrics['avg_episode_length'] = None
        
        return metrics
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def plot_training_progress(checkpoint_dir, output_file=None):
    """
    Plot training progress from all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_file: Path to save plot (if None, display interactively)
    """
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "alphazero_v2_train*.pt")
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Load metrics from all checkpoints
    all_metrics = []
    for cp_file in checkpoint_files:
        metrics = load_checkpoint_metrics(cp_file)
        if metrics is not None:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid metrics found in checkpoints")
        return
    
    # Sort by training iteration
    all_metrics.sort(key=lambda x: x['training_iteration'])
    
    # Extract data for plotting
    steps = [m['total_steps'] for m in all_metrics]
    episodes = [m['episodes_collected'] for m in all_metrics]
    avg_rewards = [m['avg_reward'] for m in all_metrics if m['avg_reward'] is not None]
    success_rates = [m['success_rate'] for m in all_metrics if m['success_rate'] is not None]
    avg_lengths = [m['avg_episode_length'] for m in all_metrics if m['avg_episode_length'] is not None]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Progress - {Path(checkpoint_dir).name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Average Reward vs Steps
    ax1 = axes[0, 0]
    if avg_rewards:
        steps_with_rewards = [m['total_steps'] for m in all_metrics if m['avg_reward'] is not None]
        ax1.plot(steps_with_rewards, avg_rewards, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Environment Steps', fontsize=12)
        ax1.set_ylabel('Avg Reward (last 100 episodes)', fontsize=12)
        ax1.set_title('Average Episode Reward', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='x')
    else:
        ax1.text(0.5, 0.5, 'No reward data available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Average Episode Reward', fontsize=14)
    
    # Plot 2: Success Rate vs Steps
    ax2 = axes[0, 1]
    if success_rates:
        steps_with_success = [m['total_steps'] for m in all_metrics if m['success_rate'] is not None]
        ax2.plot(steps_with_success, success_rates, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Environment Steps', fontsize=12)
        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.set_title('Episode Success Rate', fontsize=14)
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='plain', axis='x')
        # Add percentage on y-axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    else:
        ax2.text(0.5, 0.5, 'No success rate data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Episode Success Rate', fontsize=14)
    
    # Plot 3: Average Episode Length vs Steps
    ax3 = axes[1, 0]
    if avg_lengths:
        steps_with_lengths = [m['total_steps'] for m in all_metrics if m['avg_episode_length'] is not None]
        ax3.plot(steps_with_lengths, avg_lengths, 'r-', linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel('Environment Steps', fontsize=12)
        ax3.set_ylabel('Avg Episode Length', fontsize=12)
        ax3.set_title('Average Episode Length', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style='plain', axis='x')
    else:
        ax3.text(0.5, 0.5, 'No episode length data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Average Episode Length', fontsize=14)
    
    # Plot 4: Training Progress (Steps vs Episodes)
    ax4 = axes[1, 1]
    ax4.plot(episodes, steps, 'purple', linewidth=2, marker='o', markersize=4)
    ax4.set_xlabel('Episodes Collected', fontsize=12)
    ax4.set_ylabel('Environment Steps', fontsize=12)
    ax4.set_title('Training Progress (Steps vs Episodes)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='plain')
    
    # Add summary text
    if all_metrics:
        last_metrics = all_metrics[-1]
        summary_text = f"Latest: {last_metrics['total_steps']:,} steps | {last_metrics['episodes_collected']} episodes"
        if last_metrics['avg_reward'] is not None:
            summary_text += f" | Reward: {last_metrics['avg_reward']:.2f}"
        if last_metrics['success_rate'] is not None:
            summary_text += f" | Success: {last_metrics['success_rate']:.1%}"
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Checkpoints: {len(all_metrics)}")
    print(f"Total episodes: {episodes[-1]:,}")
    print(f"Total steps: {steps[-1]:,}")
    if avg_rewards:
        print(f"Final avg reward: {avg_rewards[-1]:.2f}")
        print(f"Best avg reward: {max(avg_rewards):.2f}")
    if success_rates:
        print(f"Final success rate: {success_rates[-1]:.1%}")
        print(f"Best success rate: {max(success_rates):.1%}")
    if avg_lengths and episodes:
        print(f"Avg steps per episode: {steps[-1] / episodes[-1]:.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot training progress from checkpoints')
    parser.add_argument('checkpoint_dir', type=str, 
                       help='Directory containing checkpoint files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: display interactively)')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Directory not found: {args.checkpoint_dir}")
        sys.exit(1)
    
    plot_training_progress(args.checkpoint_dir, args.output)


if __name__ == '__main__':
    main()
