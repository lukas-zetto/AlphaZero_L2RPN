#!/usr/bin/env python3
"""
Run 4 value target methods in parallel to compare their performance.
Each method runs with same config except for value_target_method.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config


def run_method(method_name):
    """Run training with specified value target method."""
    
    # Backup original config
    config_backup = config.AGENT_CONFIG.copy()
    
    try:
        # Update config for this method
        config.AGENT_CONFIG.update({
            'value_target_method': method_name,
            'max_training_iterations': 18,
            'episodes_per_iteration': 3,
            'parallel_workers': 3,
        })
        
        # Create output directory
        output_dir = Path(f'/workspace/value_method_comparison_{method_name}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Change to output directory
        os.chdir(output_dir)
        
        print(f"Starting training for method: {method_name}")
        print(f"Output directory: {output_dir}")
        print(f"Config: iterations={config.AGENT_CONFIG['max_training_iterations']}, "
              f"episodes={config.AGENT_CONFIG['episodes_per_iteration']}, "
              f"workers={config.AGENT_CONFIG['parallel_workers']}")
        print(f"Total episodes: 18 × 3 = 54 episodes")
        print()
        
        # Import and run training
        from scripts import train_alphazero_v2
        train_alphazero_v2.main()
        
        print(f"\n✅ Method {method_name} completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Method {method_name} failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore original config
        config.AGENT_CONFIG.clear()
        config.AGENT_CONFIG.update(config_backup)


def main():
    """Run all 5 methods in parallel."""
    
    methods = ['mcts_root', 'binary_root', 'mcts_all_nodes', 'binary_all_nodes', 'heuristic']
    
    print("=" * 80)
    print("VALUE TARGET METHOD COMPARISON")
    print("=" * 80)
    print(f"Methods to compare: {methods}")
    print(f"Config: 18 iterations × 3 episodes × 3 workers = 54 total episodes per method")
    print(f"Total cores used: 5 methods × 3 workers = 15 cores")
    print("=" * 80)
    print()
    
    # Launch all methods in parallel
    processes = []
    
    for method in methods:
        # Create output directory
        output_dir = f'/workspace/value_method_comparison_{method}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Launch subprocess for each method
        log_file = f'{output_dir}/training.log'
        err_file = f'{output_dir}/training_error.log'
        
        with open(log_file, 'w') as out_f, open(err_file, 'w') as err_f:
            proc = subprocess.Popen(
                [sys.executable, __file__, method],
                stdout=out_f,
                stderr=err_f,
                text=True
            )
            processes.append((method, proc))
            print(f"Started {method} training (PID: {proc.pid})")
    
    print(f"\nAll 5 methods started! PIDs: {[p.pid for _, p in processes]}")
    print("Monitor with: tail -f /workspace/value_method_comparison_*/training.log")
    print()
    
    # Wait for all to complete
    print("Waiting for all methods to complete...")
    for method, proc in processes:
        proc.wait()
        if proc.returncode == 0:
            print(f"✅ {method} completed successfully")
        else:
            print(f"❌ {method} failed with return code {proc.returncode}")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("Results saved to: /workspace/value_method_comparison_*/")
    print()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Child process - run single method
        method = sys.argv[1]
        run_method(method)
    else:
        # Parent process - launch all methods
        main()
