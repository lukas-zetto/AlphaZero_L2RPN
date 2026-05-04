#!/usr/bin/env python3
"""
Evaluate all checkpoints in checkpoints_heuristic folder and create performance graphs.
Runs evaluations in parallel (up to 15 at a time).
"""

import os
import sys
# Add the same path setup as training script
sys.path.append('/workspace/src')  # For direct imports like 'from networks.neural_network_factory'
sys.path.append('/workspace')      # For src-prefixed imports like 'from src.networks.neural_network_factory'

import subprocess
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.append('/work/PdF_L2RPN/AlphaZero_L2RPN/src')

def find_checkpoints(checkpoint_dir):
    """Find all checkpoint files in the directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist!")
        return []
    
    checkpoints = sorted(checkpoint_dir.glob("alphazero_v2_train*.pt"))
    return checkpoints

def extract_checkpoint_number(checkpoint_path):
    """Extract the training iteration number from checkpoint filename."""
    # Format: alphazero_v2_train<N>.pt
    match = re.search(r'alphazero_v2_train(\d+)\.pt', checkpoint_path.name)
    if match:
        return int(match.group(1))
    return None

def evaluate_checkpoint(checkpoint_path, output_prefix="mcts_root", obs_space_type=None):
    """Run evaluation for a single checkpoint and parse results."""
    checkpoint_num = extract_checkpoint_number(checkpoint_path)
    print(f"Evaluating checkpoint {checkpoint_num}: {checkpoint_path.name}")
    print(f"🔍 DEBUG: Checkpoint path: {checkpoint_path}")
    print(f"🔍 DEBUG: File exists: {checkpoint_path.exists()}")
    print(f"🔍 DEBUG: File size: {checkpoint_path.stat().st_size if checkpoint_path.exists() else 'N/A'} bytes")
    
    # Create temporary config with this checkpoint
    # Create log subdirectory for this experiment
    log_subdir = f"/work/PdF_L2RPN/AlphaZero_L2RPN/logs/eval/{output_prefix}"
    os.makedirs(log_subdir, exist_ok=True)
    log_file = f"{log_subdir}/eval_checkpoint_{checkpoint_num}.log"
    
    # Run evaluation script with checkpoint path as model_path
    env = os.environ.copy()
    env['MODEL_PATH'] = str(checkpoint_path)
    if obs_space_type is not None:
        env['OBS_SPACE_TYPE'] = obs_space_type
    
    try:
        result = subprocess.run(
            ["python3", "/work/PdF_L2RPN/AlphaZero_L2RPN/scripts/evaluate_agent.py"],
            cwd="/work/PdF_L2RPN/AlphaZero_L2RPN",
            env=env,
            capture_output=True,
            text=True,
            timeout=None  # No timeout - let evaluations complete
        )
        
        # Save log
        with open(log_file, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)
        
        # Parse output for metrics
        output = result.stdout + result.stderr
        
        # Try to extract detailed episode results from log
        detailed_results = []
        try:
            # Look for JSON episode results in the output
            json_match = re.search(r'EPISODE_RESULTS_JSON:(.*?)END_EPISODE_RESULTS_JSON', output, re.DOTALL)
            if json_match:
                detailed_results = json.loads(json_match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Extract metrics using regex
        avg_reward = None
        avg_steps = None
        survived_episodes = None
        total_episodes = None
        
        # Look for patterns like:
        # "Average reward: 123.45"
        # "Average steps: 5678"
        # "Survived: 95/101"
        # OR table format:
        # "Avg Reward                819.58               1126.74"
        # "Avg Steps                 1103.1               1523.8"
        # "Survived Episodes         0                    0"
        
        reward_match = re.search(r'Average reward[:\s]+([0-9.]+)', output, re.IGNORECASE)
        if reward_match:
            avg_reward = float(reward_match.group(1))
        
        steps_match = re.search(r'Average steps[:\s]+([0-9.]+)', output, re.IGNORECASE)
        if steps_match:
            avg_steps = float(steps_match.group(1))
        
        survived_match = re.search(r'Survived[:\s]+(\d+)/(\d+)', output, re.IGNORECASE)
        if survived_match:
            survived_episodes = int(survived_match.group(1))
            total_episodes = int(survived_match.group(2))
        
        # Alternative patterns for table format
        if avg_reward is None:
            reward_match = re.search(r'Avg\s+Reward\s+([0-9.]+)', output, re.IGNORECASE)
            if reward_match:
                avg_reward = float(reward_match.group(1))
        
        if avg_steps is None:
            steps_match = re.search(r'Avg\s+Steps\s+([0-9.]+)', output, re.IGNORECASE)
            if steps_match:
                avg_steps = float(steps_match.group(1))
        
        if survived_episodes is None:
            survived_match = re.search(r'Survived\s+Episodes\s+(\d+)', output, re.IGNORECASE)
            if survived_match:
                survived_episodes = int(survived_match.group(1))
        
        # Try to get total episodes from "Episodes                  101"
        if total_episodes is None:
            episodes_match = re.search(r'^\s*Episodes\s+(\d+)', output, re.MULTILINE)
            if episodes_match:
                total_episodes = int(episodes_match.group(1))
        
        return {
            'checkpoint_num': checkpoint_num,
            'checkpoint_path': str(checkpoint_path),
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'survived_episodes': survived_episodes,
            'total_episodes': total_episodes,
            'episode_details': detailed_results,  # Add detailed episode results
            'log_file': log_file,
            'success': result.returncode == 0
        }
    
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Evaluation timeout for checkpoint {checkpoint_num}")
        return {
            'checkpoint_num': checkpoint_num,
            'checkpoint_path': str(checkpoint_path),
            'avg_reward': None,
            'avg_steps': None,
            'survived_episodes': None,
            'total_episodes': None,
            'episode_details': [],  # Empty list for failed evaluations
            'log_file': log_file,
            'success': False,
            'error': 'timeout'
        }
    except Exception as e:
        print(f"  ERROR evaluating checkpoint {checkpoint_num}: {e}")
        return {
            'checkpoint_num': checkpoint_num,
            'checkpoint_path': str(checkpoint_path),
            'avg_reward': None,
            'avg_steps': None,
            'survived_episodes': None,
            'total_episodes': None,
            'episode_details': [],  # Empty list for failed evaluations
            'log_file': log_file,
            'success': False,
            'error': str(e)
        }

def load_existing_results(results_file):
    """Load already evaluated results from JSON file."""
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            try:
                results = json.load(f)
                # Map checkpoint_num to result for quick lookup
                return {r['checkpoint_num']: r for r in results if r.get('checkpoint_num') is not None}
            except Exception:
                return {}
    return {}

def parse_metrics_from_log(log_file):
    """Parse survival rate, survived episodes, and avg steps from a log file."""
    import re
    survival_rate = None
    survived_episodes = None
    avg_steps = None
    total_episodes = 101  # Default test set size, adjust if needed
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Survival Rate (%)         3.0                  0.0                   +3.0%
                # Extract first number after "Survival Rate (%)"
                m = re.search(r'Survival Rate \(%\)\s+([0-9.]+)', line)
                if m:
                    survival_rate = float(m.group(1))
                # Survived Episodes         3                   0                    +3
                # Extract first number after "Survived Episodes"
                m = re.search(r'Survived Episodes\s+([0-9]+)', line)
                if m:
                    survived_episodes = int(m.group(1))
                # Avg Steps                 1852.5               1523.8               +328.71
                # Extract first number after "Avg Steps"
                m = re.search(r'Avg Steps\s+([0-9.]+)', line)
                if m:
                    avg_steps = float(m.group(1))
                # Using all 101 test scenarios
                m = re.search(r'Using all\s+([0-9]+)\s+test scenarios', line)
                if m:
                    total_episodes = int(m.group(1))
    except Exception as e:
        print(f"Warning: Could not parse metrics from {log_file}: {e}")
    return survival_rate, survived_episodes, avg_steps, total_episodes

def main(checkpoint_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Ensure logs directory exists
    os.makedirs("/work/PdF_L2RPN/AlphaZero_L2RPN/logs/eval", exist_ok=True)
    results_file = os.path.join(output_dir, "checkpoint_evaluations.json")

    # Find all checkpoints
    checkpoints = find_checkpoints(checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    if len(checkpoints) == 0:
        print("No checkpoints found!")
        return

    # Load existing results
    existing_results = load_existing_results(results_file)
    already_evaluated = set(existing_results.keys())
    print(f"{len(already_evaluated)} checkpoints already evaluated.")

    # Only evaluate missing checkpoints
    checkpoints_to_eval = [cp for cp in checkpoints if extract_checkpoint_number(cp) not in already_evaluated]
    print(f"{len(checkpoints_to_eval)} checkpoints to evaluate.")

    results = list(existing_results.values())
    max_workers = min(15, len(checkpoints_to_eval)) if checkpoints_to_eval else 0

    if checkpoints_to_eval:
        print(f"\nEvaluating with {max_workers} parallel workers...\n")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_checkpoint = {
                executor.submit(evaluate_checkpoint, cp, "full_run"): cp 
                for cp in checkpoints_to_eval
            }
            for future in as_completed(future_to_checkpoint):
                checkpoint = future_to_checkpoint[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result['success'] and result['avg_reward'] is not None:
                        print(f"  ✓ Checkpoint {result['checkpoint_num']}: "
                              f"reward={result['avg_reward']:.2f}, "
                              f"steps={result.get('avg_steps', 'N/A')}")
                    else:
                        print(f"  ✗ Checkpoint {result['checkpoint_num']}: evaluation failed")
                except Exception as e:
                    print(f"  ✗ Checkpoint {checkpoint.name}: {e}")
    else:
        print("No new checkpoints to evaluate.")

    # Remove duplicates (keep latest result for each checkpoint)
    results_by_num = {}
    for r in results:
        if r.get('checkpoint_num') is not None:
            results_by_num[r['checkpoint_num']] = r
    results = list(results_by_num.values())
    results.sort(key=lambda x: x['checkpoint_num'])

    # Save results to JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {results_file}")
    
    # Save results to CSV
    csv_file = os.path.join(output_dir, "checkpoint_evaluations.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved results to {csv_file}")
    
    # Create performance graphs
    successful_results = [r for r in results if r['success'] and r['avg_reward'] is not None]
    
    if len(successful_results) == 0:
        print("\nNo successful evaluations to plot!")
        return
    
    checkpoint_nums = [r['checkpoint_num'] for r in successful_results]
    avg_rewards = [r['avg_reward'] for r in successful_results]
    avg_steps = [r.get('avg_steps') or 0 for r in successful_results]
    survival_rates = [
        (r['survived_episodes'] / r['total_episodes'] * 100) 
        if r.get('survived_episodes') and r.get('total_episodes') 
        else None
        for r in successful_results
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Average Reward
    axes[0].plot(checkpoint_nums, avg_rewards, marker='o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Checkpoint Number (Training Iteration)')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Checkpoint Performance: Average Reward')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average Steps
    if any(s > 0 for s in avg_steps):
        axes[1].plot(checkpoint_nums, avg_steps, marker='s', linewidth=2, markersize=6, color='green')
        axes[1].set_xlabel('Checkpoint Number (Training Iteration)')
        axes[1].set_ylabel('Average Steps Survived')
        axes[1].set_title('Checkpoint Performance: Average Steps')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Survival Rate
    if any(sr is not None for sr in survival_rates):
        valid_survival = [(n, sr) for n, sr in zip(checkpoint_nums, survival_rates) if sr is not None]
        if valid_survival:
            nums, rates = zip(*valid_survival)
            axes[2].plot(nums, rates, marker='^', linewidth=2, markersize=6, color='red')
            axes[2].set_xlabel('Checkpoint Number (Training Iteration)')
            axes[2].set_ylabel('Survival Rate (%)')
            axes[2].set_title('Checkpoint Performance: Episode Survival Rate')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save figure
    fig_file = os.path.join(output_dir, "checkpoint_performance.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved performance graph to {fig_file}")
    
    # Also save individual reward plot
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoint_nums, avg_rewards, marker='o', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Checkpoint Number (Training Iteration)', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('AlphaZero Training Progress: Average Reward per Checkpoint', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    reward_fig_file = os.path.join(output_dir, "reward_vs_checkpoint.png")
    plt.savefig(reward_fig_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved reward graph to {reward_fig_file}")
    
    # Plot: Survived Episodes Percentage
    plt.figure(figsize=(10, 6))
    survival_rates = [
        (r['survived_episodes'] / r['total_episodes'] * 100)
        if r.get('survived_episodes') and r.get('total_episodes')
        else None
        for r in results
    ]
    valid_survival = [(n, sr) for n, sr in zip(checkpoint_nums, survival_rates) if sr is not None]
    if valid_survival:
        nums, rates = zip(*valid_survival)
        plt.plot(nums, rates, marker='^', linewidth=2, markersize=8, color='red')
        plt.xlabel('Checkpoint Number (Training Iteration)', fontsize=12)
        plt.ylabel('Survival Rate (%)', fontsize=12)
        plt.title('Survived Episodes Percentage per Checkpoint', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 105])
        plt.tight_layout()
        survival_fig_file = os.path.join(output_dir, "survival_rate_vs_checkpoint.png")
        plt.savefig(survival_fig_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved survival rate graph to {survival_fig_file}")
    else:
        print("No survival rate data to plot.")

    # Plot: Average Survived Steps
    plt.figure(figsize=(10, 6))
    avg_steps = [r.get('avg_steps') or 0 for r in results]
    if any(s > 0 for s in avg_steps):
        plt.plot(checkpoint_nums, avg_steps, marker='s', linewidth=2, markersize=8, color='green')
        plt.xlabel('Checkpoint Number (Training Iteration)', fontsize=12)
        plt.ylabel('Average Steps Survived', fontsize=12)
        plt.title('Average Survived Steps per Checkpoint', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        steps_fig_file = os.path.join(output_dir, "avg_steps_vs_checkpoint.png")
        plt.savefig(steps_fig_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved average steps graph to {steps_fig_file}")
    else:
        print("No average steps data to plot.")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Successful evaluations: {len(successful_results)}")
    print(f"Best checkpoint: {checkpoint_nums[avg_rewards.index(max(avg_rewards))]}")
    print(f"Best reward: {max(avg_rewards):.2f}")
    print(f"Latest checkpoint: {checkpoint_nums[-1]}")
    print(f"Latest reward: {avg_rewards[-1]:.2f}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true', help='Only plot from saved results, do not re-evaluate')
    parser.add_argument('--checkpoint-dir', type=str, default="/work/PdF_L2RPN/AlphaZero_L2RPN/checkpoints_heuristic_whole", help='Directory containing checkpoints')
    parser.add_argument('--results-dir', type=str, default="/work/PdF_L2RPN/AlphaZero_L2RPN/evaluation_results_heuristic_fresh", help='Directory to save evaluation results')
    args = parser.parse_args()

    if args.plot_only:
        # Only plot from saved results
        output_dir = args.results_dir
        results_file = os.path.join(output_dir, "checkpoint_evaluations.json")
        if not os.path.exists(results_file):
            print(f"No saved results found at {results_file}")
            sys.exit(1)
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Fill missing metrics from logs
        for r in results:
            if (r.get('survived_episodes') is None or r.get('avg_steps') is None or r.get('total_episodes') is None) and r.get('log_file'):
                sr, se, steps, te = parse_metrics_from_log(r['log_file'])
                if r.get('survived_episodes') is None and se is not None:
                    r['survived_episodes'] = se
                if r.get('total_episodes') is None and te is not None:
                    r['total_episodes'] = te
                if r.get('avg_steps') is None and steps is not None:
                    r['avg_steps'] = steps
                if r.get('survived_episodes') is None and sr is not None and r.get('total_episodes'):
                    r['survived_episodes'] = int(round(sr * r['total_episodes'] / 100))
                if r.get('survived_episodes') is not None and r.get('total_episodes'):
                    r['survival_rate'] = 100.0 * r['survived_episodes'] / r['total_episodes']
                elif sr is not None:
                    r['survival_rate'] = sr
        
        # Save updated results back to JSON
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Updated results with parsed metrics from logs")
        
        # Only keep successful results
        results = [r for r in results if r.get('success') and r.get('avg_reward') is not None]
        if not results:
            print("No successful results to plot!")
            sys.exit(1)
        # Reuse plotting code from main()
        checkpoint_nums = [r['checkpoint_num'] for r in results]
        avg_rewards = [r['avg_reward'] for r in results]
        avg_steps = [r.get('avg_steps') or 0 for r in results]
        survival_rates = [
            (r['survived_episodes'] / r['total_episodes'] * 100)
            if r.get('survived_episodes') and r.get('total_episodes')
            else None
            for r in results
        ]
        import matplotlib.pyplot as plt
        import pandas as pd
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes[0].plot(checkpoint_nums, avg_rewards, marker='o', linewidth=2, markersize=6)
        axes[0].set_xlabel('Checkpoint Number (Training Iteration)')
        axes[0].set_ylabel('Average Reward')
        axes[0].set_title('Checkpoint Performance: Average Reward')
        axes[0].grid(True, alpha=0.3)
        if any(s > 0 for s in avg_steps):
            axes[1].plot(checkpoint_nums, avg_steps, marker='s', linewidth=2, markersize=6, color='green')
            axes[1].set_xlabel('Checkpoint Number (Training Iteration)')
            axes[1].set_ylabel('Average Steps Survived')
            axes[1].set_title('Checkpoint Performance: Average Steps')
            axes[1].grid(True, alpha=0.3)
        if any(sr is not None for sr in survival_rates):
            valid_survival = [(n, sr) for n, sr in zip(checkpoint_nums, survival_rates) if sr is not None]
            if valid_survival:
                nums, rates = zip(*valid_survival)
                axes[2].plot(nums, rates, marker='^', linewidth=2, markersize=6, color='red')
                axes[2].set_xlabel('Checkpoint Number (Training Iteration)')
                axes[2].set_ylabel('Survival Rate (%)')
                axes[2].set_title('Checkpoint Performance: Episode Survival Rate')
                axes[2].grid(True, alpha=0.3)
                axes[2].set_ylim([0, 105])
        plt.tight_layout()
        fig_file = os.path.join(output_dir, "checkpoint_performance.png")
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved performance graph to {fig_file}")
        plt.figure(figsize=(10, 6))
        plt.plot(checkpoint_nums, avg_rewards, marker='o', linewidth=2, markersize=8, color='blue')
        plt.xlabel('Checkpoint Number (Training Iteration)', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title('AlphaZero Training Progress: Average Reward per Checkpoint', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        reward_fig_file = os.path.join(output_dir, "reward_vs_checkpoint.png")
        plt.savefig(reward_fig_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved reward graph to {reward_fig_file}")
        plt.figure(figsize=(10, 6))
        survival_rates = [
            (r['survived_episodes'] / r['total_episodes'] * 100)
            if r.get('survived_episodes') and r.get('total_episodes')
            else None
            for r in results
        ]
        valid_survival = [(n, sr) for n, sr in zip(checkpoint_nums, survival_rates) if sr is not None]
        if valid_survival:
            nums, rates = zip(*valid_survival)
            plt.plot(nums, rates, marker='^', linewidth=2, markersize=8, color='red')
            plt.xlabel('Checkpoint Number (Training Iteration)', fontsize=12)
            plt.ylabel('Survival Rate (%)', fontsize=12)
            plt.title('Survived Episodes Percentage per Checkpoint', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 105])
            plt.tight_layout()
            survival_fig_file = os.path.join(output_dir, "survival_rate_vs_checkpoint.png")
            plt.savefig(survival_fig_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved survival rate graph to {survival_fig_file}")
        else:
            print("No survival rate data to plot.")
        plt.figure(figsize=(10, 6))
        avg_steps = [r.get('avg_steps') or 0 for r in results]
        if any(s > 0 for s in avg_steps):
            plt.plot(checkpoint_nums, avg_steps, marker='s', linewidth=2, markersize=8, color='green')
            plt.xlabel('Checkpoint Number (Training Iteration)', fontsize=12)
            plt.ylabel('Average Steps Survived', fontsize=12)
            plt.title('Average Survived Steps per Checkpoint', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            steps_fig_file = os.path.join(output_dir, "avg_steps_vs_checkpoint.png")
            plt.savefig(steps_fig_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved average steps graph to {steps_fig_file}")
        else:
            print("No average steps data to plot.")
        sys.exit(0)

    main(args.checkpoint_dir, args.results_dir)
