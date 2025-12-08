"""
Compare 4 value assignment methods for AlphaZero training.

Methods:
1. mcts_root: MCTS Q-values from root node only (current default)
2. binary_root: Binary episode outcomes (+1/-1) from root node only
3. mcts_all_nodes: MCTS Q-values from ALL tree nodes (proper AlphaZero)
4. heuristic: Heuristic value function (discounted future rewards)

Each method trains for 18 iterations with 4 episodes per iteration (72 total episodes).
Uses 4 parallel workers to utilize all 16 cores efficiently.
"""

import sys
import os
sys.path.append('/workspace')

from src.config import AGENT_CONFIG, ENV_CONFIG, ACTIONS_CONFIG, USE_REDUCED_ACTION_SPACE, REDUCED_ACTIONS
from scripts.train_alphazero_v2 import AlphaZeroTrainerV2
from src.actions.action_catalog import build_action_catalog
import grid2op
import json
import time
from datetime import datetime

def run_single_method(method_name, output_dir):
    """Run training with a specific value assignment method."""
    print("="*80)
    print(f"TRAINING WITH METHOD: {method_name}")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Configuration:")
    print(f"  - Iterations: 18")
    print(f"  - Episodes per iteration: 4")
    print(f"  - Total episodes: 72")
    print(f"  - Parallel workers: 4")
    print(f"  - Value method: {method_name}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure for this method
    config = AGENT_CONFIG.copy()
    config['value_target_method'] = method_name
    config['episodes_per_iteration'] = 4
    config['parallel_workers'] = 4
    config['num_cycles'] = 1  # Single training cycle
    config['max_episode_steps'] = 10000
    
    # Create environment
    env = grid2op.make(
        ENV_CONFIG['name'],
        backend=grid2op.Backend.PandaPowerBackend(),
        reward_class=grid2op.Reward.FlatReward
    )
    
    # Create action catalog using the same method as train script
    full_catalog = build_action_catalog(
        env,
        substations=ACTIONS_CONFIG['substations'],
        reduction=ACTIONS_CONFIG['reduction'],
        drop_identity=ACTIONS_CONFIG['drop_identity'],
        include_do_nothing=ACTIONS_CONFIG.get('include_do_nothing', True)
    )
    
    # Apply reduced action space if enabled
    if USE_REDUCED_ACTION_SPACE:
        from dataclasses import replace
        reduced_actions = [full_catalog.actions[idx] for idx in REDUCED_ACTIONS if idx < len(full_catalog.actions)]
        action_catalog = replace(full_catalog, actions=reduced_actions)
    else:
        action_catalog = full_catalog
    
    print(f"Action catalog created: {len(action_catalog.actions)} actions")
    
    # Save config
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump({
            'method': method_name,
            'iterations': 18,
            'episodes_per_iteration': 4,
            'total_episodes': 72,
            'parallel_workers': 4,
            'num_actions': len(action_catalog.actions),
            'config': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                      for k, v in config.items()}
        }, f, indent=2)
    
    # Create trainer
    trainer = AlphaZeroTrainerV2(env, action_catalog, config)
    
    # Redirect logs
    log_file = f'{output_dir}/training.log'
    
    # Train for 18 iterations
    start_time = time.time()
    
    for iteration in range(1, 19):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/18")
        print(f"{'='*80}")
        
        # Collect episodes
        iteration_data = []
        for episode in range(config['episodes_per_iteration']):
            episode_data, outcome = trainer.collect_training_data_episode()
            if episode_data:
                iteration_data.extend(episode_data)
        
        print(f"\nCollected {len(iteration_data)} training samples")
        
        # Train network
        if iteration_data:
            trainer.train_network(iteration, iteration_data)
        
        # Save checkpoint
        checkpoint_path = f'{output_dir}/checkpoint_iter{iteration}.pt'
        trainer.save_model(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    elapsed = time.time() - start_time
    
    # Save final results
    results = {
        'method': method_name,
        'iterations': 18,
        'total_episodes': 72,
        'training_time_seconds': elapsed,
        'training_time_hours': elapsed / 3600,
        'completed_at': datetime.now().isoformat()
    }
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"METHOD {method_name} COMPLETE")
    print(f"{'='*80}")
    print(f"Training time: {elapsed/3600:.2f} hours")
    print(f"Results saved to: {output_dir}/results.json")
    print()
    
    return results


def main():
    """Run comparison across all 4 methods."""
    base_dir = '/workspace/value_method_comparison'
    os.makedirs(base_dir, exist_ok=True)
    
    methods = [
        'mcts_root',
        'binary_root', 
        'mcts_all_nodes',
        'heuristic'
    ]
    
    print("="*80)
    print("VALUE ASSIGNMENT METHOD COMPARISON")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base directory: {base_dir}")
    print(f"Methods to test: {', '.join(methods)}")
    print(f"Total training runs: {len(methods)}")
    print(f"Estimated time: ~{len(methods) * 3} hours (3 hours per method)")
    print("="*80)
    print()
    
    all_results = {}
    
    for i, method in enumerate(methods, 1):
        print(f"\n{'#'*80}")
        print(f"RUNNING METHOD {i}/{len(methods)}: {method}")
        print(f"{'#'*80}\n")
        
        output_dir = f'{base_dir}/{method}'
        
        try:
            results = run_single_method(method, output_dir)
            all_results[method] = results
        except Exception as e:
            print(f"\n❌ ERROR in method {method}: {e}")
            import traceback
            traceback.print_exc()
            all_results[method] = {'error': str(e), 'status': 'failed'}
    
    # Save comparison summary
    summary_path = f'{base_dir}/comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Summary saved to: {summary_path}")
    print("\nResults by method:")
    for method, result in all_results.items():
        if 'error' in result:
            print(f"  {method}: FAILED - {result['error']}")
        else:
            print(f"  {method}: {result['training_time_hours']:.2f} hours")
    print("="*80)


if __name__ == '__main__':
    main()
