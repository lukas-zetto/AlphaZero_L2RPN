#!/usr/bin/env python3
"""
Test script to verify initial MCTS action distribution is uniform
"""
import sys
import os
import numpy as np
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Parameters import Parameters

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import MASTER_CONFIG

# Extract configs from unified structure
AGENT_CONFIG = MASTER_CONFIG['core_agent']
TRAINING_CONFIG = MASTER_CONFIG['training']
ACTIONS_CONFIG = MASTER_CONFIG['actions']
USE_REDUCED_ACTION_SPACE = MASTER_CONFIG['actions']['use_reduced_action_space']
REDUCED_ACTIONS = MASTER_CONFIG['actions']['reduced_actions']
from src.actions.action_catalog import build_action_catalog
from src.networks.neural_network import create_neural_network
from src.training.alphazero_mcts_v2 import run_mcts

def test_initial_distribution():
    print("Testing initial MCTS distribution (no checkpoint loaded)...")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating Grid2Op environment...")
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = False
    env = grid2op.make(
        "l2rpn_case14_sandbox",
        backend=LightSimBackend(),
        param=params,
        test=True
    )
    
    # Build action catalog
    print("2. Building action catalog...")
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
    
    print(f"   Actions: {len(action_catalog.actions)}")
    
    # Create fresh network (no checkpoint loading)
    print("3. Creating fresh neural network (NO CHECKPOINT)...")
    # Dynamically determine input size from environment
    dummy_obs = env.reset()
    from networks.neural_network import encode_observation_simple
    dummy_encoded = encode_observation_simple(dummy_obs)
    input_size = len(dummy_encoded)
    num_actions = len(action_catalog.actions)
    print(f"🔍 TEST: Dynamic input_size={input_size}, num_actions={num_actions}")
    
    # Create network - this will start with random initialization
    neural_network = create_neural_network(
        input_size=input_size,
        num_actions=num_actions,
        config={}  # Empty config uses defaults
    )
    print(f"   ✅ Network created with random initialization")
    
    # Reset environment and run MCTS
    print("4. Running MCTS on initial state (100 simulations)...")
    obs = env.reset()
    
    # Run MCTS with policy_fn=None to use uniform priors
    root, stats = run_mcts(
        env,
        obs,
        action_catalog,
        num_simulations=100,
        c_puct=1.5,  # Standard value
        gamma=0.95,  # Standard discount
        max_depth=25,
        epsilon=0.0,
        policy_fn=None,  # ← This forces UNIFORM priors
        value_fn=None,   # Use heuristic value
        critical_threshold=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,  # No Dirichlet noise for testing
        penalty_for_failure=-5.0
    )
    
    # Extract visit counts from root node
    print("5. Analyzing action distribution...")
    visit_counts = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(num_actions)])
    total_visits = visit_counts.sum()
    action_probs = visit_counts / total_visits if total_visits > 0 else visit_counts
    
    # Analyze distribution
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    
    # Get non-zero probabilities
    non_zero = action_probs > 0
    num_non_zero = np.sum(non_zero)
    
    print(f"\nTotal actions: {len(action_probs)}")
    print(f"Actions explored (visit count > 0): {num_non_zero}")
    print(f"Total MCTS visits: {total_visits}")
    
    if num_non_zero > 0:
        non_zero_probs = action_probs[non_zero]
        print(f"\nVisit probability statistics:")
        print(f"  Mean: {non_zero_probs.mean():.6f}")
        print(f"  Std:  {non_zero_probs.std():.6f}")
        print(f"  Min:  {non_zero_probs.min():.6f}")
        print(f"  Max:  {non_zero_probs.max():.6f}")
        
        # Check if distribution is roughly uniform
        expected_uniform = 1.0 / num_non_zero
        print(f"\nExpected uniform probability: {expected_uniform:.6f}")
        
        # Calculate coefficient of variation (std/mean)
        cv = non_zero_probs.std() / non_zero_probs.mean()
        print(f"Coefficient of variation: {cv:.4f}")
        
        # Show top 10 actions
        print(f"\nTop 10 actions by visit count:")
        sorted_indices = np.argsort(visit_counts)[::-1]
        for i, idx in enumerate(sorted_indices[:10]):
            if visit_counts[idx] > 0:
                print(f"  {i+1}. Action {idx}: {visit_counts[idx]} visits ({action_probs[idx]*100:.2f}%)")
        
        # Conclusion
        print("\n" + "=" * 60)
        if cv < 0.5:  # Low coefficient of variation = uniform-ish
            print("✅ Distribution is ROUGHLY UNIFORM")
            print("   (Small variations expected due to MCTS exploration)")
        else:
            print("⚠️  Distribution shows significant bias")
            print("   (This may indicate a checkpoint is being loaded)")
        print("=" * 60)
    
    return action_probs

if __name__ == '__main__':
    test_initial_distribution()
