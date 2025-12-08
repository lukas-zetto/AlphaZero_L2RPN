"""
Visualize MCTS tree for the first critical state encountered.
Creates a graphical representation showing visit counts, Q-values, edge rewards, and rho.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import grid2op
from grid2op.Reward import BaseReward
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx

from src.config import AGENT_CONFIG, ENV_CONFIG, ACTIONS_CONFIG
from src.actions.action_catalog import ActionCatalog
from src.rewards.custom_reward import MyCustomReward
from src.training.alphazero_mcts_v2 import MCTSNodeV2, run_mcts, compute_heuristic_value


def visualize_tree(root: MCTSNodeV2, max_depth=3, filename='mcts_tree.png'):
    """
    Visualize MCTS tree using networkx and matplotlib.
    
    Args:
        root: Root node of MCTS tree
        max_depth: Maximum depth to visualize
        filename: Output filename for the visualization
    """
    G = nx.DiGraph()
    pos = {}
    labels = {}
    node_colors = []
    node_sizes = []
    
    # Counter for unique node IDs
    node_counter = [0]
    
    def add_nodes_recursive(node, depth=0, parent_id=None, x=0, width=1.0):
        if depth > max_depth:
            return
        
        # Create unique node ID
        node_id = node_counter[0]
        node_counter[0] += 1
        
        # Add node to graph
        G.add_node(node_id)
        
        # Position: x for horizontal, -depth for vertical (top to bottom)
        pos[node_id] = (x, -depth)
        
        # Node label with key information
        if node.observation is not None:
            rho = node.observation.rho.max()
            rho_str = f"{rho:.3f}"
        else:
            rho_str = "TERM"
        
        q_value = node.value()
        visits = node.visit_count
        edge_r = node.edge_reward
        
        label = f"V:{visits}\nQ:{q_value:.2f}\nE:{edge_r:.2f}\nρ:{rho_str}"
        labels[node_id] = label
        
        # Node color based on state
        if node.is_terminal:
            color = 'red'  # Terminal/failed
        elif node.observation is not None and node.observation.rho.max() > 0.98:
            color = 'orange'  # Critical
        elif node.skip_steps > 0:
            color = 'lightblue'  # Recovery node (skipped steps)
        else:
            color = 'lightgreen'  # Safe
        node_colors.append(color)
        
        # Node size based on visit count (logarithmic scale)
        size = 300 + 100 * np.log1p(visits)
        node_sizes.append(size)
        
        # Add edge from parent
        if parent_id is not None:
            G.add_edge(parent_id, node_id)
        
        # Recursively add children
        if depth < max_depth and node.children:
            # Sort children by visit count for better layout
            sorted_children = sorted(
                node.children.items(), 
                key=lambda x: x[1].visit_count, 
                reverse=True
            )
            
            # Only show top children at deeper levels to avoid clutter
            max_children = 10 if depth == 0 else 5 if depth == 1 else 3
            sorted_children = sorted_children[:max_children]
            
            n_children = len(sorted_children)
            if n_children > 0:
                child_width = width / n_children
                start_x = x - width / 2 + child_width / 2
                
                for i, (action_idx, child) in enumerate(sorted_children):
                    child_x = start_x + i * child_width
                    child_id = add_nodes_recursive(child, depth + 1, node_id, child_x, child_width)
                    
                    # Add action index to edge label
                    if child_id is not None:
                        G.edges[node_id, child_id]['label'] = f"A{action_idx}"
        
        return node_id
    
    # Build the graph
    add_nodes_recursive(root)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, 
                           arrowstyle='->', ax=ax, connectionstyle='arc3,rad=0.1')
    
    # Draw edge labels (action indices)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='lightgreen', label='Safe state'),
        mpatches.Patch(color='orange', label='Critical (ρ>0.98)'),
        mpatches.Patch(color='lightblue', label='Recovery node'),
        mpatches.Patch(color='red', label='Terminal/Failed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, label='Size = log(visits)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title('MCTS Tree Visualization\nV=visits, Q=Q-value, E=edge_reward, ρ=max_rho', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  📊 Tree visualization saved to {filename}")
    plt.close()


def main():
    print("=" * 60)
    print("MCTS TREE VISUALIZATION")
    print("=" * 60)
    
    # Set random seeds for reproducibility (same as training)
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed: {seed}")
    
    # Load environment
    env_name = ENV_CONFIG['name']
    print(f"Loading environment: {env_name}")
    
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(env_name, backend=backend)
    except:
        env = grid2op.make(env_name)
    
    # Build action catalog
    from src.actions.action_catalog import build_action_catalog
    action_catalog = build_action_catalog(
        env=env,
        substations=ACTIONS_CONFIG['substations'],
        reduction=ACTIONS_CONFIG['reduction'],
        drop_identity=ACTIONS_CONFIG['drop_identity'],
        include_do_nothing=ACTIONS_CONFIG['include_do_nothing']
    )
    print(f"Action catalog: {len(action_catalog.actions)} actions")
    
    # Custom reward function
    custom_reward_fn = MyCustomReward()
    
    # Heuristic value function
    def heuristic_value_fn(observation):
        # Create a dummy node to compute heuristic value
        dummy_node = MCTSNodeV2(
            env=None,
            observation=observation,
            parent=None,
            action_taken=None,
            action_idx=None,
            edge_reward=custom_reward_fn(
                action=env.action_space(),
                env=env,
                has_error=False,
                is_done=False,
                is_illegal=False,
                is_ambiguous=False,
                obs=observation
            ),
            steps_to_reach=0
        )
        return compute_heuristic_value(
            dummy_node,
            gamma=AGENT_CONFIG['gamma'],
            horizon=AGENT_CONFIG.get('heuristic_value_horizon', 20)
        )
    
    # Use the same chronic as training (chronic 92)
    # Set chronic to 92 to match the training log
    chronic_id = 92
    env.set_id(chronic_id)
    obs = env.reset()
    print(f"\nUsing chronic {chronic_id} to match training log...")
    print(f"Running until step 511 (first critical state in training)...")
    
    step = 0
    target_step = 511
    
    while step < target_step:
        obs, reward, done, info = env.step(env.action_space())
        step += 1
        
        if done:
            print(f"❌ Episode ended early at step {step}")
            return
    
    rho_max = obs.rho.max()
    print(f"\n✅ Reached step {step}: rho_max={rho_max:.3f}")
    
    if rho_max <= AGENT_CONFIG['critical_threshold']:
        print(f"⚠️ Warning: rho_max={rho_max:.3f} is not critical (threshold={AGENT_CONFIG['critical_threshold']})")
        print(f"   But proceeding with MCTS anyway to match training conditions...")
    
    # Reset random seed again right before MCTS (env.reset may have changed it)
    import random
    np.random.seed(42)
    random.seed(42)
    print(f"Re-seeded numpy random before MCTS: seed=42")
    
    # Run MCTS with 1000 simulations to match training
    print(f"\nRunning MCTS with 1000 simulations...")
    
    # Run MCTS
    root, stats = run_mcts(
        env=env,
        observation=obs,
        action_catalog=action_catalog,
        num_simulations=1000,  # 1000 simulations to match training
        c_puct=AGENT_CONFIG['puct_c'],
        gamma=AGENT_CONFIG['gamma'],
        max_depth=AGENT_CONFIG['max_depth'],
        value_fn=heuristic_value_fn,
        epsilon=AGENT_CONFIG['mcts_epsilon'],
        critical_threshold=AGENT_CONFIG['critical_threshold'],
        auto_reconnect=AGENT_CONFIG['auto_reconnect'],
        max_reconnections=AGENT_CONFIG['max_reconnections_per_action'],
        prefilter_rho_increase=AGENT_CONFIG['action_prefilter_rho_increase'],
        penalty_for_failure=AGENT_CONFIG['penalty_for_failure'],
        policy_fn=None,  # No neural network
        t_skipped=AGENT_CONFIG['t_skipped'],
        t_stopping=AGENT_CONFIG['t_stopping'],
        dirichlet_alpha=AGENT_CONFIG['dirichlet_alpha'],
        dirichlet_epsilon=AGENT_CONFIG['dirichlet_epsilon'],
        depth_bonus=AGENT_CONFIG.get('depth_bonus', 0.0),
        virtual_loss_weight=AGENT_CONFIG.get('virtual_loss_weight', 0.0),
        verbose=True
    )
    
    print(f"  Root visits: {root.visit_count}")
    print(f"  Root children: {len(root.children)}")
    
    # Visualize the tree
    print(f"\nGenerating tree visualization...")
    visualize_tree(root, max_depth=3, filename='logs/mcts_tree_visualization.png')
    
    # Print summary statistics
    print(f"\n📊 Tree Statistics:")
    print(f"  Total root children: {len(root.children)}")
    print(f"  Root visit count: {root.visit_count}")
    
    # Visit count distribution
    visit_counts = [child.visit_count for child in root.children.values()]
    print(f"  Visit distribution: min={min(visit_counts)}, max={max(visit_counts)}, mean={np.mean(visit_counts):.1f}")
    print(f"  Children with >0 visits: {sum(1 for v in visit_counts if v > 0)}/{len(visit_counts)}")
    
    # Top 5 actions by visit count
    sorted_children = sorted(
        root.children.items(),
        key=lambda x: x[1].visit_count,
        reverse=True
    )[:10]
    
    print(f"\n  Top 10 actions by visits:")
    for action_idx, child in sorted_children:
        rho_str = f"{child.observation.rho.max():.3f}" if child.observation else "TERM"
        print(f"    Action {action_idx}: V={child.visit_count}, Q={child.value():.2f}, "
              f"E={child.edge_reward:.2f}, ρ={rho_str}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
