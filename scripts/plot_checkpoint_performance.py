"""
Plot checkpoint performance over training iterations.
Shows survival rate (successful episodes) vs iteration number.
Uses exact same evaluation logic as evaluate_agent.py
"""

import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, '/workspace')

def extract_iteration_from_checkpoint(filename):
    """Extract iteration number from checkpoint filename."""
    match = re.search(r'iter(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def load_checkpoint_stats():
    """Load all checkpoint files and extract iteration numbers."""
    checkpoint_dir = "/workspace/checkpoints"
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt') and 'alphazero_v2' in filename:
            iteration = extract_iteration_from_checkpoint(filename)
            if iteration is not None:
                filepath = os.path.join(checkpoint_dir, filename)
                checkpoints.append({
                    'iteration': iteration,
                    'filename': filename,
                    'filepath': filepath
                })
    
    # Sort by iteration
    checkpoints.sort(key=lambda x: x['iteration'])
    return checkpoints

def evaluate_checkpoint(checkpoint_path, num_scenarios=101):
    """
    Evaluate a checkpoint on test scenarios using exact same logic as evaluate_agent.py
    Returns survival rate (fraction of episodes that reached max_steps).
    """
    import grid2op
    from grid2op.Parameters import Parameters
    from lightsim2grid import LightSimBackend
    import torch
    from src.agent.my_agent import MyCustomAgent
    from src.config import EVAL_CONFIG, AGENT_CONFIG, ACTIONS_CONFIG
    
    # Setup environment - EXACT same as evaluate_agent.py
    env = grid2op.make(
        "l2rpn_case14_sandbox",
        backend=LightSimBackend()
    )
    
    # Create config with ACTIONS_CONFIG included
    config = AGENT_CONFIG.copy()
    config['ACTIONS_CONFIG'] = ACTIONS_CONFIG
    
    # Create agent with full config
    agent = MyCustomAgent(env.action_space, config=config)
    
    # Manually load the checkpoint into the neural network
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    agent.neural_network.load_state_dict(checkpoint['model_state_dict'])
    agent.neural_network.eval()
    
    # Get test scenarios (last 10% of dataset)
    total_scenarios = len(env.chronics_handler.subpaths)
    train_size = int(total_scenarios * 0.9)
    test_scenarios = list(range(train_size, total_scenarios))  # All test scenarios
    
    # Use specified number of scenarios
    test_scenarios = test_scenarios[:num_scenarios]
    
    # Evaluate
    successful = 0
    total = len(test_scenarios)
    max_steps = EVAL_CONFIG.get('max_steps', 8064)  # Use same as evaluate_agent.py
    
    for scenario_id in test_scenarios:
        # Set specific scenario
        env.chronics_handler.tell_id(scenario_id)
        
        # Reset environment and agent
        obs = env.reset()
        agent.reset(obs)
        
        # Run episode
        done = False
        step = 0
        reward = 0
        
        while not done and step < max_steps:
            try:
                action = agent.act(obs, reward, done)
                obs, reward, done, info = env.step(action)
                step += 1
            except Exception as e:
                print(f"    Error in scenario {scenario_id}: {e}")
                break
        
        # Episode survived if reached max_steps
        if step >= max_steps:
            successful += 1
    
    env.close()
    survival_rate = successful / total if total > 0 else 0.0
    return survival_rate, successful, total

def main():
    print("Loading checkpoints...")
    checkpoints = load_checkpoint_stats()
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    print("Evaluating each checkpoint on ALL 101 test scenarios (this will take a while)...")
    
    iterations = []
    survival_rates = []
    
    for i, ckpt in enumerate(checkpoints):
        print(f"\n[{i+1}/{len(checkpoints)}] Evaluating iteration {ckpt['iteration']}...")
        try:
            survival_rate, successful, total = evaluate_checkpoint(ckpt['filepath'], num_scenarios=101)
            iterations.append(ckpt['iteration'])
            survival_rates.append(survival_rate * 100)  # Convert to percentage
            print(f"  Survived: {successful}/{total} ({survival_rate*100:.1f}%)")
        except Exception as e:
            import traceback
            print(f"  Failed: {e}")
            traceback.print_exc()
    
    if not iterations:
        print("No successful evaluations!")
        return
    
    # Create plot
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, survival_rates, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Training Iteration', fontsize=13)
    plt.ylabel('Survival Rate (%)', fontsize=13)
    plt.title('Checkpoint Performance Over Training\n(Evaluated on 101 Test Scenarios)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(-5, 105)
    
    # Add value labels on points
    for iter, rate in zip(iterations, survival_rates):
        plt.annotate(f'{rate:.0f}%', 
                    xy=(iter, rate), 
                    xytext=(0, 8), 
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    output_path = '/workspace/logs/checkpoint_performance.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Also show statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Checkpoints evaluated: {len(iterations)}")
    print(f"Iterations range: {min(iterations)} - {max(iterations)}")
    print(f"Best survival rate: {max(survival_rates):.1f}% (iteration {iterations[survival_rates.index(max(survival_rates))]})")
    print(f"Worst survival rate: {min(survival_rates):.1f}% (iteration {iterations[survival_rates.index(min(survival_rates))]})")
    print(f"Latest survival rate: {survival_rates[-1]:.1f}% (iteration {iterations[-1]})")
    print(f"Average survival rate: {np.mean(survival_rates):.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()
