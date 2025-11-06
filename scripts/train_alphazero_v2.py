"""
AlphaZero training with MCTS v2 (environment copy approach).
Trains a neural network using self-play with MCTS search.
"""

import sys
sys.path.append('/workspace/src')

import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
import numpy as np
import torch
import os
from collections import deque

from actions.action_catalog import build_action_catalog
from training.alphazero_mcts_v2 import run_mcts, select_action, MCTSNodeV2
from networks.neural_network import create_neural_network, encode_observation_simple, neural_network_forward, train_neural_network


class AlphaZeroTrainerV2:
    """AlphaZero self-play trainer using MCTS v2."""
    
    def __init__(self, env, action_catalog, config):
        self.env = env
        self.action_catalog = action_catalog
        self.config = config
        
        # Create neural network
        self.input_size = 83  # From encode_observation_simple
        self.num_actions = len(action_catalog.actions)
        
        self.neural_network = create_neural_network(
            input_size=self.input_size,
            num_actions=self.num_actions,
            config=config
        )
        
        # Training data buffer
        self.replay_buffer = deque(maxlen=config['replay_buffer_size'])
        
        print(f"Created neural network: input={self.input_size}, actions={self.num_actions}")
    
    def get_policy_value(self, observation):
        """Get policy and value from neural network."""
        if observation is None:
            # Return uniform policy and zero value for None observations
            uniform_policy = np.ones(self.num_actions) / self.num_actions
            return uniform_policy, 0.0
        
        policy, value = neural_network_forward(
            self.neural_network,
            observation,
            self.num_actions
        )
        return policy, value
    
    def self_play_episode(self, episode_id):
        """Run one self-play episode collecting training data."""
        print(f"\n{'='*60}")
        print(f"SELF-PLAY EPISODE {episode_id}")
        print(f"{'='*60}")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        step = 0
        episode_data = []
        
        while not done and step < self.config['max_episode_steps']:
            max_rho = np.max(obs.rho)
            critical_threshold = self.config.get('critical_threshold', 0.90)
            is_critical = max_rho > critical_threshold
            
            print(f"\nStep {step}: rho_max={max_rho:.3f} {'🔴 CRITICAL' if is_critical else '🟢 SAFE'} (threshold={critical_threshold})")
            
            # Only run MCTS if in critical state
            if not is_critical:
                print(f"  ⏭️ Skipping MCTS - grid is safe, taking do-nothing action")
                # Take do-nothing action
                obs, reward, done, info = self.env.step(self.env.action_space())
                step += 1
                continue
            
            # Run MCTS with neural network (only in critical states)
            epsilon = self.config.get('mcts_epsilon', 0.0)
            critical_threshold = self.config.get('critical_threshold', 0.90)
            print(f"  Running MCTS with {self.config['mcts_simulations']} simulations (epsilon={epsilon}, threshold={critical_threshold})...")
            
            root, stats = run_mcts(
                env=self.env,
                observation=obs,
                action_catalog=self.action_catalog,
                num_simulations=self.config['mcts_simulations'],
                c_puct=self.config['c_puct'],
                gamma=self.config['gamma'],
                value_fn=lambda o: self.get_policy_value(o)[1],  # Use NN value
                max_depth=self.config['max_depth'],
                epsilon=epsilon,
                verbose=False,
                critical_threshold=critical_threshold
            )
            
            # Print tree statistics for debugging
            print(f"  MCTS complete: {len(root.children)} root children, max_depth={stats['max_depth']}")
            print(f"    Tree structure:")
            for d in range(min(10, stats['max_depth']+1)):  # Show up to depth 10
                total = stats['nodes_by_depth'].get(d, 0)
                terminal = stats['terminal_by_depth'].get(d, 0)
                visits = stats['visits_by_depth'].get(d, 0)
                print(f"      Depth {d}: {total} nodes ({terminal} terminal, {visits} visits)")
            
            # Show root visit distribution and steps_to_reach
            if stats['root_children_visits']:
                visits = [c['visits'] for c in stats['root_children_visits']]
                values = [c['value'] for c in stats['root_children_visits']]
                steps = [c['steps_to_reach'] for c in stats['root_children_visits']]
                print(f"    Root visits: max={max(visits)}, mean={sum(visits)/len(visits):.1f}, >10={sum(1 for v in visits if v > 10)}")
                print(f"    Root Q-values: max={max(values):.3f}, min={min(values):.3f}, mean={sum(values)/len(values):.3f}")
                print(f"    Steps to reach: max={max(steps)}, mean={sum(steps)/len(steps):.1f}")
                
                # Show top 3 actions by steps_to_reach (most important metric now!)
                sorted_children = sorted(stats['root_children_visits'], key=lambda x: x['steps_to_reach'], reverse=True)[:3]
                print(f"    Top 3 actions by steps_to_reach:")
                for i, child_info in enumerate(sorted_children, 1):
                    action_idx = child_info['action_idx']
                    print(f"      {i}. Action {action_idx}: {child_info['steps_to_reach']} steps, {child_info['visits']} visits, Q={child_info['value']:.3f}")
            
            # Get action probabilities from visit counts
            visits = np.array([root.children.get(i, MCTSNodeV2(None, None)).visit_count 
                              for i in range(self.num_actions)])
            
            # Normalize to get policy
            total_visits = visits.sum()
            if total_visits > 0:
                mcts_policy = visits / total_visits
            else:
                mcts_policy = np.ones(self.num_actions) / self.num_actions
            
            # Calculate policy entropy
            policy_nonzero = mcts_policy[mcts_policy > 0]
            policy_entropy = -np.sum(policy_nonzero * np.log(policy_nonzero + 1e-10))
            max_entropy = np.log(self.num_actions)
            normalized_entropy = policy_entropy / max_entropy
            print(f"  🎯 Policy entropy: {policy_entropy:.3f} (normalized: {normalized_entropy:.3f})")
            if normalized_entropy > 0.9:
                print(f"    ⚠️ Warning: Policy is very uniform (entropy close to max)")
            
            # Select action using steps_to_reach (greedy, no epsilon here - epsilon only for MCTS exploration)
            from training.alphazero_mcts_v2 import select_action
            action_idx = select_action(root, temperature=0, epsilon=0.0)  # Greedy selection for actual action
            
            if action_idx in root.children:
                selected_steps = root.children[action_idx].steps_to_reach
                print(f"  ✅ Selected action {action_idx} (steps_to_reach={selected_steps}, visits={visits[action_idx]})")
            else:
                print(f"  ✅ Selected action {action_idx} (default/fallback)")
            
            # Store training example (state, policy, value placeholder)
            state_vector = encode_observation_simple(obs)
            episode_data.append({
                'state': state_vector,
                'policy': mcts_policy,
                'value': None,  # Will be filled with episode outcome
            })
            
            # Get line loads before action
            rho_before = obs.rho.copy()
            top_lines_before = np.argsort(rho_before)[-3:][::-1]
            
            # Take action in environment
            action = self.action_catalog.actions[action_idx]
            grid2op_action = action.apply(self.env.action_space)
            
            print(f"  📋 Action {action_idx}: {str(action)[:80]}")
            
            obs, reward, done, info = self.env.step(grid2op_action)
            step += 1
            
            # Show line load changes
            if not done:
                rho_after = obs.rho.copy()
                top_lines_after = np.argsort(rho_after)[-3:][::-1]
                print(f"  📉 Line loads:")
                print(f"    Before (top 3): " + ", ".join([f"L{i}={rho_before[i]:.2f}" for i in top_lines_before]))
                print(f"    After  (top 3): " + ", ".join([f"L{i}={rho_after[i]:.2f}" for i in top_lines_after]))
                max_change = np.max(np.abs(rho_after - rho_before))
                print(f"    Max change: {max_change:.3f}")
            
            if done:
                print(f"\n  ⚠️ Episode ended at step {step}")
                print(f"    Reason: {info.get('exception', 'Unknown')}")
                print(f"    Is illegal: {info.get('is_illegal', False)}")
                print(f"    Is ambiguous: {info.get('is_ambiguous', False)}")
                break
        
        # Assign values to all states based on episode outcome
        # Simple: +1 if survived, -1 if failed
        episode_value = 1.0 if step >= self.config['max_episode_steps'] else -1.0
        
        print(f"\n✅ Episode Summary:")
        print(f"  Steps: {step}/{self.config['max_episode_steps']}")
        print(f"  MCTS states collected: {len(episode_data)}")
        print(f"  Episode value: {episode_value:.1f}")
        print(f"  Success: {'Yes' if episode_value > 0 else 'No'}")
        
        for data in episode_data:
            data['value'] = episode_value
            self.replay_buffer.append(data)
        
        return len(episode_data), episode_value
    
    def train_network(self, iteration):
        """Train neural network on replay buffer."""
        if len(self.replay_buffer) == 0:
            print(f"Replay buffer is empty, skipping training")
            return None
        
        print(f"\n{'='*60}")
        print(f"TRAINING NEURAL NETWORK - Iteration {iteration}")
        print(f"{'='*60}")
        print(f"Training on {len(self.replay_buffer)} experiences")
        
        # Sample batch from replay buffer (or use all if smaller than batch_size)
        batch_size = min(self.config['batch_size'], len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        training_examples = []
        for idx in indices:
            data = self.replay_buffer[idx]
            training_examples.append({
                'state': data['state'],
                'mcts_policy': data['policy'],
                'value': data['value']
            })
        
        # Train
        loss_info = train_neural_network(
            self.neural_network,
            training_examples,
            self.config
        )
        
        print(f"Training losses:")
        print(f"  Policy loss: {loss_info['policy_loss']:.4f}")
        print(f"  Value loss: {loss_info['value_loss']:.4f}")
        print(f"  Total loss: {loss_info['total_loss']:.4f}")
        
        return loss_info
    
    def train(self, num_iterations):
        """Run full AlphaZero training loop."""
        print(f"\n{'='*60}")
        print(f"ALPHAZERO TRAINING V2")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Episodes per iteration: {self.config['episodes_per_iteration']}")
        print(f"MCTS simulations: {self.config['mcts_simulations']}")
        print()
        
        for iteration in range(num_iterations):
            print(f"\n{'#'*60}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'#'*60}")
            
            # Self-play phase
            for episode in range(self.config['episodes_per_iteration']):
                self.self_play_episode(episode)
            
            # Training phase
            self.train_network(iteration)
            
            # Save checkpoint
            if (iteration + 1) % self.config['save_every'] == 0:
                checkpoint_path = f"/workspace/checkpoints/alphazero_v2_iter{iteration+1}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': self.neural_network.state_dict(),
                    'replay_buffer_size': len(self.replay_buffer),
                }, checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")


def main():
    # Setup environment
    env_name = "l2rpn_case14_sandbox"
    print(f"Loading environment: {env_name}")
    
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    
    env = grid2op.make(
        env_name,
        backend=LightSimBackend(),
        param=params,
        test=True
    )
    
    # Build action catalog from config
    from config import ACTIONS_CONFIG
    substations = ACTIONS_CONFIG['substations']
    reduction = ACTIONS_CONFIG['reduction']
    drop_identity = ACTIONS_CONFIG['drop_identity']
    
    print(f"Building action catalog: substations={substations}, reduction={reduction}")
    catalog = build_action_catalog(env, substations=substations, reduction=reduction, drop_identity=drop_identity)
    print(f"Action catalog: {len(catalog.actions)} actions")
    
    # Load training configuration from config.py
    from config import AGENT_CONFIG
    
    config = {
        # MCTS parameters from config.py
        'mcts_simulations': AGENT_CONFIG['mcts_simulations'],
        'c_puct': AGENT_CONFIG['puct_c'],
        'gamma': AGENT_CONFIG['gamma'],
        'max_depth': AGENT_CONFIG['max_depth'],
        'temperature': AGENT_CONFIG['temperature'],
        'critical_threshold': AGENT_CONFIG['critical_threshold'],  # Only act when rho > threshold
        'mcts_epsilon': AGENT_CONFIG['mcts_epsilon'],  # Epsilon-greedy exploration in MCTS
        
        # Training parameters
        'episodes_per_iteration': 2,
        'max_episode_steps': 1000,  # Run full episodes (Grid2Op default is ~300-800 steps)
        'replay_buffer_size': 10000,
        'batch_size': AGENT_CONFIG['batch_size'],
        'learning_rate': AGENT_CONFIG['learning_rate'],
        'weight_decay': AGENT_CONFIG['weight_decay'],
        'epochs_per_iteration': AGENT_CONFIG['training_epochs'],
        
        # Checkpointing
        'save_every': 5,
        
        # Neural network architecture
        'hidden_sizes': [AGENT_CONFIG['hidden_size'], AGENT_CONFIG['hidden_size']],
        'dropout': 0.0,
    }
    
    print(f"\nTraining configuration:")
    print(f"  MCTS simulations: {config['mcts_simulations']}")
    print(f"  c_puct: {config['c_puct']}")
    print(f"  Max depth: {config['max_depth']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print()
    
    # Create trainer
    trainer = AlphaZeroTrainerV2(env, catalog, config)
    
    # Run training
    trainer.train(num_iterations=10)


if __name__ == "__main__":
    main()
