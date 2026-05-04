#!/usr/bin/env python3
"""
Fix training steps data in evaluation results by reading from log files
"""

import json
import re
from pathlib import Path

def extract_training_steps_for_observation(experiment_name, checkpoint_subdir):
    """Extract training steps from observation experiment logs"""
    steps_mapping = {}
    
    log_dir = Path(f"/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/logs/{experiment_name}")
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return steps_mapping
    
    log_files = list(log_dir.glob("exp_obs_space_*.out"))
    print(f"Found {len(log_files)} log files for {experiment_name}")
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Look for checkpoint save patterns that match this subdir
            # Pattern: checkpoints/observation_experimentsfixed13_5/checkpoints_obs_minimal/alphazero_v2_train0.pt
            pattern = rf'💾 Checkpoint saved: [^\s]*{re.escape(experiment_name)}/{re.escape(checkpoint_subdir)}/alphazero_v2_train(\d+)\.pt\s*\n\s*Steps:\s*([\d,]+)'
            matches = re.findall(pattern, content)
            
            for match in matches:
                checkpoint_num = int(match[0])
                steps = int(match[1].replace(',', ''))
                steps_mapping[checkpoint_num] = steps
                
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    print(f"Extracted training steps for {len(steps_mapping)} checkpoints from {experiment_name}/{checkpoint_subdir}")
    return steps_mapping

def fix_observation_results():
    """Fix training steps in the latest observation results"""
    results_file = "/home/ka/ka_iai/ka_jp7970/PdF_L2RPN/AlphaZero_L2RPN/eval/observation_multi_run_results/20260416_105806/observation_multi_run_results.json"
    
    print(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Observation space variant mapping
    variant_to_subdir = {
        "Minimal": "checkpoints_obs_minimal",
        "Essential": "checkpoints_obs_essential", 
        "Custom": "checkpoints_obs_custom",
        "Gym": "checkpoints_obs_gym"
    }
    
    for run_suffix, run_data in data.items():
        experiment_name = f"observation_experimentsfixed13{run_suffix}"
        print(f"\nProcessing run {run_suffix} ({experiment_name})")
        
        for variant_name, variant_results in run_data.items():
            if variant_name in variant_to_subdir:
                checkpoint_subdir = variant_to_subdir[variant_name]
                print(f"  Processing variant {variant_name} ({checkpoint_subdir})")
                
                # Extract training steps for this variant
                steps_mapping = extract_training_steps_for_observation(experiment_name, checkpoint_subdir)
                
                # Update results with training steps
                updated_count = 0
                for result in variant_results:
                    if result.get('training_steps') is None:
                        checkpoint_num = result.get('checkpoint_num')
                        if checkpoint_num is not None and checkpoint_num in steps_mapping:
                            result['training_steps'] = steps_mapping[checkpoint_num]
                            updated_count += 1
                
                print(f"    Updated {updated_count} results with training steps")
    
    # Save updated results
    backup_file = results_file.replace('.json', '_backup.json')
    print(f"\nSaving backup to {backup_file}")
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saving updated results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ Training steps data fixed!")

if __name__ == "__main__":
    fix_observation_results()