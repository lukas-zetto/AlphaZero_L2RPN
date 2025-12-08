#!/usr/bin/env python3
"""
Run 5 value target methods IN PARALLEL to compare their performance.
Like Optuna but all trials run simultaneously.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Run comparison trials for all 5 value target methods IN PARALLEL."""
    
    methods = ['mcts_root', 'binary_root', 'mcts_all_nodes', 'binary_all_nodes', 'heuristic']
    
    print("=" * 80)
    print("VALUE TARGET METHOD COMPARISON (PARALLEL)")
    print("=" * 80)
    print(f"Running {len(methods)} trials IN PARALLEL")
    print(f"Each trial: 18 iterations × 3 episodes with 3 parallel workers")
    print(f"Methods: {', '.join(methods)}")
    print(f"Total cores: 5 methods × 3 workers = 15 cores")
    print("=" * 80)
    print()
    
    processes = []
    
    # Launch all 5 methods in parallel
    for method in methods:
        trial_dir = Path(f'/workspace/value_method_comparison_{method}')
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        train_log = trial_dir / 'training.log'
        train_err = trial_dir / 'training_error.log'
        
        # Set environment variables
        env = os.environ.copy()
        env['VALUE_TARGET_METHOD'] = method
        env['MAX_TRAINING_ITERATIONS'] = '18'
        env['EPISODES_PER_ITERATION'] = '3'
        env['PARALLEL_WORKERS'] = '3'
        
        print(f"Starting {method}...")
        
        with open(train_log, 'w') as out_f, open(train_err, 'w') as err_f:
            proc = subprocess.Popen(
                [sys.executable, 'scripts/train_alphazero_v2.py'],
                cwd='/workspace',
                stdout=out_f,
                stderr=err_f,
                text=True,
                env=env
            )
            processes.append((method, proc))
            print(f"  ✓ Started {method} (PID: {proc.pid})")
    
    print(f"\n{'='*80}")
    print(f"All 5 methods started! PIDs: {[p.pid for _, p in processes]}")
    print("Monitor: docker exec topology_optimization tail -f /workspace/value_method_comparison_*/training.log")
    print(f"{'='*80}\n")
    
    # Wait for all to complete
    results = {}
    for method, proc in processes:
        print(f"Waiting for {method}...")
        proc.wait()
        if proc.returncode == 0:
            print(f"  ✅ {method} completed successfully")
            results[method] = 'SUCCESS'
        else:
            print(f"  ❌ {method} failed with return code {proc.returncode}")
            results[method] = 'FAILED'
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE - SUMMARY")
    print("=" * 80)
    for method, status in results.items():
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"{status_icon} {method:20s} : {status}")
    print("=" * 80)
    print("\nResults saved to: /workspace/value_method_comparison_*/")
    print()


if __name__ == '__main__':
    main()
