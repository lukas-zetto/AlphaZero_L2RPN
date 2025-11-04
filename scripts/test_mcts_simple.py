#!/usr/bin/env python3
"""
Simple MCTS diagnostic test - runs in Docker
"""
import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/src')

from grid2op import make
try:
    from lightsim2grid import LightSimBackend
    backend = LightSimBackend()
except:
    backend = None

# Import the training module
import train_agent
from src.actions.action_catalog import ActionCatalog, build_action_catalog, REDUCTION_N1

def analyze_mcts_decision(root, action_probs, starting_step):
    """Analyze MCTS decision details"""
    print("\n" + "="*80)
    print("🔍 MCTS DECISION ANALYSIS")
    print("="*80)
    
    # Collect child info
    child_info = []
    for action_str, child in root.children.items():
        q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
        child_info.append({
            'idx': child.action_idx,
            'max_steps': child.max_reachable_steps,
            'skipped': getattr(child, 'skipped_steps', 0),
            'visits': child.visit_count,
            'q_value': q_value,
            'recovery': child.is_recovery_node
        })
    
    child_info.sort(key=lambda x: x['max_steps'], reverse=True)
    
    print(f"\n🌳 Tree: {len(root.children)} children, root visits: {root.visit_count}")
    print(f"   Starting timestep: {starting_step}")
    
    # Check tree depth
    grandchild_count = 0
    great_grandchild_count = 0
    max_grandchild_steps = 0
    terminal_grandchildren = 0
    for child in root.children.values():
        if hasattr(child, 'children') and child.children:
            grandchild_count += len(child.children)
            for gc in child.children.values():
                max_grandchild_steps = max(max_grandchild_steps, gc.max_reachable_steps)
                if gc.is_terminal:
                    terminal_grandchildren += 1
                if hasattr(gc, 'children') and gc.children:
                    great_grandchild_count += len(gc.children)
    
    print(f"   Grandchildren (depth 2): {grandchild_count}")
    print(f"   Terminal grandchildren: {terminal_grandchildren}")
    print(f"   Great-grandchildren (depth 3): {great_grandchild_count}")
    if grandchild_count > 0:
        print(f"   Max steps at depth 2: {max_grandchild_steps}")
    
    # Show top 10
    print(f"\n📋 Top 10 by max_reachable_steps:")
    print(f"{'Rank':<6} {'Action':<8} {'MaxSteps':<12} {'Visits':<10} {'Q-Value':<12} {'Recovery'}")
    print("-" * 70)
    for i, c in enumerate(child_info[:10]):
        rec = "✓" if c['recovery'] else ""
        print(f"{i+1:<6} {c['idx']:<8} {c['max_steps']:<12} {c['visits']:<10} {c['q_value']:<12.4f} {rec}")
    
    # Check distribution
    max_steps_vals = [c['max_steps'] for c in child_info]
    unique = len(set(max_steps_vals))
    print(f"\n   Unique max_steps values: {unique} / {len(child_info)}")
    
    if unique <= 3:
        print(f"   ⚠️ WARNING: Most children have same max_reachable_steps!")
        from collections import Counter
        dist = Counter(max_steps_vals)
        print(f"   Distribution: {dict(dist.most_common(5))}")
    
    # Analyze policy entropy
    if action_probs:
        probs = np.array([p for _, p in action_probs])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_ent = np.log(len(probs))
        
        print(f"\n🎯 Policy Target:")
        print(f"   Entropy: {entropy:.3f} / {max_ent:.3f} ({100*entropy/max_ent:.1f}% of max)")
        
        sorted_p = sorted(probs, reverse=True)
        print(f"   Top 5 probs: {sorted_p[:5]}")
        
        if entropy / max_ent > 0.95:
            print(f"   ⚠️ NEARLY UNIFORM POLICY! This explains high cross-entropy loss.")
    
    print("="*80)


def main():
    print("🧪 MCTS Diagnostic Test")
    print("="*80)
    
    # Create environment
    env = make("l2rpn_case14_sandbox", backend=backend) if backend else make("l2rpn_case14_sandbox")
    print(f"✅ Environment: {env.name}")
    
    # Create action catalog BEFORE starting episode (catalog creation resets env!)
    all_subs = list(range(env.n_sub))
    action_catalog = build_action_catalog(env, all_subs, REDUCTION_N1, drop_identity=True)
    print(f"✅ Action catalog: {action_catalog.size} actions")
    
    # NOW start the episode
    env.set_id(5)  # Try chronic 5
    obs = env.reset()
    step = 0
    
    print("\n🔍 Searching for critical state (chronic 5, threshold 0.80)...")
    while step < 500:
        # Lower threshold to find states faster
        if np.max(obs.rho) > 0.80:
            print(f"✅ Found critical state at step {step}")
            print(f"   Max rho: {np.max(obs.rho):.3f}")
            
            # Run MCTS
            print(f"\n🌲 Running MCTS (1000 simulations)...")
            best_action, action_probs, stats = train_agent.run_alpha_zero_mcts(
                env,
                neural_network=None,
                num_simulations=1000,
                action_catalog=action_catalog,
                config={'mcts_simulations': 1000, 'puct_c': 1.4, 'gamma': 0.99, 't_skipped': 200, 't_stopping': 10},
                early_stop_recovery=True  # Enable early stopping with new t_stopping=10
            )
            
            # Analyze
            root = stats.get('root_node')
            starting_step = stats.get('starting_step', 0)
            
            if root:
                analyze_mcts_decision(root, action_probs, starting_step)
            
            print(f"\n📊 Basic stats:")
            print(f"   Simulations: {stats['simulations']}")
            print(f"   Recovery nodes: {stats['recovery_nodes_found']}")
            print(f"   Max episode steps: {stats['max_episode_steps']}")
            
            # Execute action
            print(f"\n⚡ Executing selected action...")
            obs_after, reward, done, info = env.step(best_action)
            print(f"   Rho before: {np.max(obs.rho):.3f}")
            print(f"   Rho after: {np.max(obs_after.rho):.3f}")
            print(f"   Change: {np.max(obs_after.rho) - np.max(obs.rho):.3f}")
            
            # Test with more simulations
            print(f"\n{'='*74}")
            print(f"🌲 Running MCTS with 2000 simulations...")
            print(f"{'='*74}")
            # Ensure environment is initialized for a new search
            if done:
                print("   Episode ended after first action. Resetting env before second MCTS...")
                env.set_id(5)
                obs = env.reset()
            else:
                obs = obs_after
            
            best_action2, action_probs2, stats2 = train_agent.run_alpha_zero_mcts(
                env,
                neural_network=None,
                num_simulations=2000,
                action_catalog=action_catalog,
                config={'mcts_simulations': 2000, 'puct_c': 1.4, 'gamma': 0.99, 't_skipped': 300, 't_stopping': 10},
                early_stop_recovery=True  # Enable early stopping with new t_stopping=10
            )
            
            root2 = stats2.get('root_node')
            if root2:
                analyze_mcts_decision(root2, action_probs2, starting_step)
            
            print(f"\n📊 Stats (2000 sims):")
            print(f"   Simulations: {stats2['simulations']}")
            print(f"   Recovery nodes: {stats2['recovery_nodes_found']}")
            
            obs_after2, reward2, done2, info2 = env.step(best_action2)
            print(f"\n⚡ Executing selected action (2000 sims)...")
            print(f"   Rho after: {np.max(obs_after2.rho):.3f}")
            print(f"   Change: {np.max(obs_after2.rho) - np.max(obs.rho):.3f}")
            
            break
        
        obs, _, done, _ = env.step(env.action_space())
        step += 1
        if done:
            print(f"Episode ended at step {step}, trying next chronic...")
            break
    
    if step >= 500:
        print("No critical state found in 500 steps")


if __name__ == "__main__":
    main()
