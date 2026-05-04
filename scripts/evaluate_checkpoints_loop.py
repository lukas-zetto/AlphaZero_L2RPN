#!/usr/bin/env python3
"""
Evaluation script for all checkpoints in checkpoints/ folder
Evaluates each checkpoint and plots survival rate over iterations
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
from config import AGENT_CONFIG, ENV_CONFIG, EVAL_CONFIG, ACTIONS_CONFIG, TRAINING_CONFIG
from config import USE_REDUCED_ACTION_SPACE, REDUCED_ACTIONS

# No additional config extraction needed - using direct imports


def get_test_scenarios(env, max_episodes=None):
    """Get test scenarios based on random selection from test pool (same as training)"""
    import random
    
    total_scenarios = len(env.chronics_handler.subpaths)
    
    # Use same split configuration as training
    train_test_split = TRAINING_CONFIG.get('train_test_split', 0.9)
    num_test_chronics = TRAINING_CONFIG.get('num_test_chronics', None)
    chronic_seed = TRAINING_CONFIG.get('chronic_seed', None)
    
    # Create split: first X% for training pool, remaining for test pool
    train_pool_size = int(total_scenarios * train_test_split)
    test_pool = list(range(train_pool_size, total_scenarios))
    
    # Use seed for deterministic selection if provided
    if chronic_seed is not None:
        random.seed(chronic_seed)
    
    # Select test chronics: either a random subset or all test chronics
    if num_test_chronics is None:
        # Use all chronics from test pool
        test_scenarios = test_pool
    elif len(test_pool) >= num_test_chronics:
        # Randomly select subset from test pool
        test_scenarios = sorted(random.sample(test_pool, num_test_chronics))
    else:
        # Test pool smaller than requested, use all
        test_scenarios = test_pool
        print(f"⚠️  Warning: Only {len(test_pool)} chronics available for testing (requested {num_test_chronics})")
    
    if max_episodes is None:
        return test_scenarios
    else:
        return test_scenarios[:max_episodes]


def evaluate_on_scenarios(env, agent, scenarios):
    """Evaluate agent on specified scenarios - EXACT COPY from evaluate_agent.py"""
    
    results = []
    
    for i, scenario_id in enumerate(scenarios):
        print(f"\rScenario {i+1}/{len(scenarios)}: {scenario_id}", end='', flush=True)
        
        # Set specific scenario
        env.chronics_handler.tell_id(scenario_id)
        
        # Reset environment and agent
        obs = env.reset()
        agent.reset(obs)
        
        # Run episode
        episode_data = {
            'scenario_id': scenario_id,
            'steps': 0,
            'total_reward': 0,
            'survived': False,
            'final_step': 0
        }
        
        done = False
        step = 0
        max_steps = EVAL_CONFIG.get('max_steps', 8064)  # ~1 week
        
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
    
    print()  # New line after progress
    return results


def main():
    # Create environment
    print("Creating environment...")
    try:
        from lightsim2grid import LightSimBackend
        env = grid2op.make(ENV_CONFIG['name'], backend=LightSimBackend())
    except:
        env = grid2op.make(ENV_CONFIG['name'])
    
    # Get test scenarios
    test_scenarios = get_test_scenarios(env)
    num_test = len(test_scenarios)
    
    print(f"Test scenarios: {len(test_scenarios)} scenarios")
    
    # Find all checkpoint files and sort numerically
    checkpoint_files = glob.glob("checkpoints/alphazero_v2_iter*.pt")
    
    # Sort by iteration number (extract number from filename)
    def get_iter_number(path):
        basename = os.path.basename(path)
        try:
            return int(basename.replace("alphazero_v2_iter", "").replace(".pt", ""))
        except:
            return 0
    
    checkpoint_files = sorted(checkpoint_files, key=get_iter_number)
    
    if not checkpoint_files:
        print("No checkpoints found in checkpoints/ folder!")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoints to evaluate")
    
    # Evaluate each checkpoint
    iterations = []
    survival_rates = []
    
    for checkpoint_path in checkpoint_files:
        # Extract iteration number from filename
        basename = os.path.basename(checkpoint_path)
        try:
            iteration = int(basename.replace("alphazero_v2_iter", "").replace(".pt", ""))
        except:
            print(f"Could not parse iteration from {basename}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating Iteration {iteration}")
        print(f"{'='*60}")
        
        # EXACT SAME LOGIC as evaluate_agent.py
        agent_config = AGENT_CONFIG.copy()
        agent_config['model_path'] = checkpoint_path
        
        # Merge ACTIONS_CONFIG exactly like evaluate_agent.py line 196
        full_config = {**agent_config, 'ACTIONS_CONFIG': ACTIONS_CONFIG}
        
        # Create NEW agent instance
        agent = MyCustomAgent(
            action_space=env.action_space,
            config=full_config
        )
        
        # Set agent to test mode for fast inference (no MCTS)
        agent.set_mode('test')
        
        # Load model EXPLICITLY like evaluate_agent.py lines 203-205
        if os.path.exists(checkpoint_path):
            print(f"Loading model from: {checkpoint_path}")
            
            # Check network state BEFORE loading
            import torch
            params_before = list(agent.neural_network.parameters())[0].clone() if agent.neural_network else None
            
            agent.load_model(checkpoint_path)
            
            # Check network state AFTER loading
            params_after = list(agent.neural_network.parameters())[0] if agent.neural_network else None
            
            if params_before is not None and params_after is not None:
                if torch.equal(params_before, params_after):
                    print(f"⚠️ WARNING: Parameters did NOT change after load_model!")
                else:
                    print(f"✅ Parameters successfully updated after load_model")
            
            print(f"✅ Model loaded for iteration {iteration}")
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
        
        # Evaluate - EXACT SAME FUNCTION as evaluate_agent.py
        results = evaluate_on_scenarios(env, agent, test_scenarios)
        
        # Compute survival rate
        survived = [r['survived'] for r in results]
        survival_rate = np.mean(survived) * 100
        
        iterations.append(iteration)
        survival_rates.append(survival_rate)
        
        print(f"✅ Iteration {iteration}: {survival_rate:.1f}% survival rate ({int(np.sum(survived))}/{num_test})")
    
    # Save results to file
    results_file = "logs/checkpoint_results.txt"
    with open(results_file, 'w') as f:
        f.write("Iteration,Survival_Rate\n")
        for it, sr in zip(iterations, survival_rates):
            f.write(f"{it},{sr}\n")
    print(f"\n📁 Results saved to: {results_file}")
    
    # Create plot
    if len(iterations) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, survival_rates, marker='o', linewidth=2, markersize=8, label='Custom Agent', color='#2E86AB')
        plt.axhline(y=45.5, color='red', linestyle='--', linewidth=2, label='Do Nothing Baseline (45.5%)')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Survival Rate (%)', fontsize=12)
        plt.title('Agent Performance Over Training Iterations', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.ylim(0, 100)
        
        # Add annotations for best and worst
        best_idx = survival_rates.index(max(survival_rates))
        plt.annotate(f'Best: {max(survival_rates):.1f}%', 
                    xy=(iterations[best_idx], survival_rates[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Save plot
        output_path = "logs/checkpoints_performance.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Best performance: {max(survival_rates):.1f}% at iteration {iterations[best_idx]}")
        print(f"Final performance (iter {iterations[-1]}): {survival_rates[-1]:.1f}%")
        print(f"Average performance: {sum(survival_rates)/len(survival_rates):.1f}%")
    else:
        print("\n❌ No data to plot!")

if __name__ == "__main__":
    main()
