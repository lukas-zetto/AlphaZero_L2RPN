#!/usr/bin/env python3
"""
Evaluate only new checkpoints that haven't been evaluated yet
"""

import os
import sys
import numpy as np
import grid2op
import matplotlib.pyplot as plt
import glob

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agent.my_agent import MyCustomAgent
from src.config import AGENT_CONFIG, ENV_CONFIG, EVAL_CONFIG, ACTIONS_CONFIG


def get_test_scenarios(env, max_episodes=None):
    """Get test scenarios based on 90/10 split"""
    total_scenarios = len(env.chronics_handler.subpaths)
    train_size = int(total_scenarios * 0.9)
    test_scenarios = list(range(train_size, total_scenarios))
    
    if max_episodes is None:
        return test_scenarios
    else:
        return test_scenarios[:max_episodes]


def evaluate_on_scenarios(env, agent, scenarios):
    """Evaluate agent on specified scenarios"""
    results = []
    
    for i, scenario_id in enumerate(scenarios):
        print(f"\rScenario {i+1}/{len(scenarios)}: {scenario_id}", end='', flush=True)
        
        env.chronics_handler.tell_id(scenario_id)
        obs = env.reset()
        agent.reset(obs)
        
        episode_data = {
            'scenario_id': scenario_id,
            'steps': 0,
            'total_reward': 0,
            'survived': False,
            'final_step': 0
        }
        
        done = False
        step = 0
        max_steps = EVAL_CONFIG.get('max_steps', 8064)
        
        while not done and step < max_steps:
            try:
                action = agent.act(obs, reward if step > 0 else None, done)
                obs, reward, done, info = env.step(action)
                
                episode_data['total_reward'] += reward
                step += 1
                
            except Exception as e:
                print(f"\nError in scenario {scenario_id}, step {step}: {e}")
                break
        
        episode_data['steps'] = step
        episode_data['final_step'] = step
        episode_data['survived'] = (step >= max_steps)
        
        results.append(episode_data)
    
    print()
    return results


def main():
    # Read existing results
    results_file = "logs/checkpoint_results.txt"
    existing_results = {}
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    iteration = int(parts[0])
                    survival_rate = float(parts[1])
                    existing_results[iteration] = survival_rate
        print(f"Loaded {len(existing_results)} existing results")
    
    # Find all checkpoints
    checkpoint_files = glob.glob("checkpoints/alphazero_v2_iter*.pt")
    
    def get_iter_number(path):
        basename = os.path.basename(path)
        try:
            return int(basename.replace("alphazero_v2_iter", "").replace(".pt", ""))
        except:
            return 0
    
    checkpoint_files = sorted(checkpoint_files, key=get_iter_number)
    
    # Filter to only new checkpoints
    new_checkpoints = []
    for checkpoint_path in checkpoint_files:
        iteration = get_iter_number(checkpoint_path)
        if iteration not in existing_results:
            new_checkpoints.append((iteration, checkpoint_path))
    
    if not new_checkpoints:
        print("No new checkpoints to evaluate!")
        print("Regenerating plot from existing data...")
    else:
        print(f"\nFound {len(new_checkpoints)} new checkpoints to evaluate: {[it for it, _ in new_checkpoints]}")
        
        # Create environment
        print("\nCreating environment...")
        try:
            from lightsim2grid import LightSimBackend
            env = grid2op.make(ENV_CONFIG['name'], backend=LightSimBackend())
        except:
            env = grid2op.make(ENV_CONFIG['name'])
        
        test_scenarios = get_test_scenarios(env)
        print(f"Test scenarios: {len(test_scenarios)} scenarios")
        
        # Evaluate new checkpoints
        for iteration, checkpoint_path in new_checkpoints:
            print(f"\n{'='*60}")
            print(f"Evaluating Iteration {iteration}")
            print(f"{'='*60}")
            
            agent_config = AGENT_CONFIG.copy()
            agent_config['model_path'] = checkpoint_path
            full_config = {**agent_config, 'ACTIONS_CONFIG': ACTIONS_CONFIG}
            
            agent = MyCustomAgent(
                action_space=env.action_space,
                config=full_config
            )
            
            if os.path.exists(checkpoint_path):
                print(f"Loading model from: {checkpoint_path}")
                agent.load_model(checkpoint_path)
                print("✅ Model loaded")
            
            # Evaluate
            results = evaluate_on_scenarios(env, agent, test_scenarios)
            
            # Calculate survival rate
            survived_count = sum(1 for r in results if r['survived'])
            survival_rate = (survived_count / len(results)) * 100
            
            print(f"Iteration {iteration} survival rate: {survival_rate:.1f}%")
            
            # Add to existing results
            existing_results[iteration] = survival_rate
    
    # Save combined results
    print(f"\nSaving combined results...")
    iterations = sorted(existing_results.keys())
    survival_rates = [existing_results[it] for it in iterations]
    
    with open(results_file, 'w') as f:
        f.write("Iteration,Survival_Rate\n")
        for it, sr in zip(iterations, survival_rates):
            f.write(f"{it},{sr}\n")
    print(f"📁 Results saved to: {results_file}")
    
    # Create plot
    if len(iterations) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, survival_rates, marker='o', linewidth=2, markersize=6, label='Custom Agent', color='#2E86AB')
        plt.axhline(y=45.5, color='red', linestyle='--', linewidth=2, label='Do Nothing Baseline (45.5%)')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Survival Rate (%)', fontsize=12)
        plt.title('Agent Performance Over Training Iterations', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.ylim(0, 100)
        
        # Annotate best
        best_idx = survival_rates.index(max(survival_rates))
        plt.annotate(f'Best: {max(survival_rates):.1f}% (iter {iterations[best_idx]})', 
                    xy=(iterations[best_idx], survival_rates[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        output_path = "logs/checkpoints_performance.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {output_path}")
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total iterations evaluated: {len(iterations)}")
        print(f"Best survival rate: {max(survival_rates):.1f}% (iteration {iterations[best_idx]})")
        print(f"Latest survival rate: {survival_rates[-1]:.1f}% (iteration {iterations[-1]})")


if __name__ == "__main__":
    main()
