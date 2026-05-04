#!/usr/bin/env python3
"""
Evaluation script for MyCustomAgent using L2RPN scoring
"""

import os
import sys
import json
import numpy as np
import torch
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Agent import DoNothingAgent

# Add the same path setup as training script (this works reliably)
sys.path.append('/workspace/src')  # For direct imports like 'from networks.neural_network_factory'
sys.path.append('/workspace')      # For src-prefixed imports like 'from src.networks.neural_network_factory'

# Add project root to path (backup approach)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.agent.my_agent import MyCustomAgent
from src.rewards.custom_reward import MyCustomReward
from src.config import AGENT_CONFIG, ENV_CONFIG, EVAL_CONFIG, ACTIONS_CONFIG, TRAINING_CONFIG
from src.config import USE_REDUCED_ACTION_SPACE, REDUCED_ACTIONS

# No additional config extraction needed - using direct imports


def detect_model_config(model_path):
    """Auto-detect observation and action space configuration from model dimensions"""
    
    # Load model and get dimensions
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Try direct access to saved dimensions first (newer checkpoints)
        if 'input_size' in checkpoint and 'num_actions' in checkpoint:
            input_size = checkpoint['input_size']
            num_actions = checkpoint['num_actions']
        else:
            # Fallback to parsing model weights (older checkpoints)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']  
            else:
                raise ValueError("Cannot find model weights in checkpoint")
                
            input_size = state_dict['trunk.0.weight'].shape[1] 
            num_actions = state_dict['policy_head.weight'].shape[0]
            
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    # Complete observation space mapping
    OBS_SIZE_TO_TYPE = {
        20: 'minimal',     # Only rho values
        114: 'essential',  # Power + topology basics  
        117: 'custom',     # Full feature set
        200: 'gym',        # Traditional RL approach
    }
    
    # Complete action space mapping  
    ACTIONS_TO_REDUCTION = {
        82: 'N1',          # Most restrictive
        141: 'N0',         # Medium restrictive
        203: 'SYM',        # Least restrictive
    }
    
    obs_space_type = OBS_SIZE_TO_TYPE.get(input_size)
    reduction = ACTIONS_TO_REDUCTION.get(num_actions)
    
    if not obs_space_type:
        raise ValueError(f"Unknown observation size: {input_size}. Known sizes: {list(OBS_SIZE_TO_TYPE.keys())}")
    if not reduction:
        raise ValueError(f"Unknown action count: {num_actions}. Known counts: {list(ACTIONS_TO_REDUCTION.keys())}")
        
    detected_config = {
        'obs_space_type': obs_space_type,
        'reduction': reduction,
        'drop_identity': False,        
        'include_do_nothing': True,    
        'substations': list(range(14))
    }
    
    print(f"🔍 Auto-detected model configuration:")
    print(f"  📊 Observation space: {obs_space_type} ({input_size} features)")
    print(f"  🎯 Action space: {reduction} ({num_actions} actions)")
    
    return detected_config


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
    
    print(f"Total scenarios: {total_scenarios}")
    print(f"Training pool: 0-{train_pool_size-1} ({train_pool_size} scenarios)")
    print(f"Test pool: {train_pool_size}-{total_scenarios-1} ({len(test_pool)} scenarios)")
    print(f"Test scenarios selected (seed={chronic_seed}): {len(test_scenarios)} chronics - {test_scenarios}")
    
    if max_episodes is None:
        print(f"Using all {len(test_scenarios)} test scenarios")
        return test_scenarios
    else:
        print(f"Using first {max_episodes} test scenarios")
        return test_scenarios[:max_episodes]


def evaluate_on_scenarios(env, agent, scenarios=None, max_episodes=None):
    """Evaluate agent on specified scenarios"""
    
    if scenarios is None:
        # Use test scenarios from 90/10 split
        scenarios = get_test_scenarios(env, max_episodes)
    elif max_episodes is not None:
        scenarios = scenarios[:max_episodes]
    
    print(f"Evaluating on {len(scenarios)} scenarios...")
    
    results = []
    action_counts = {}  # Track action frequency
    
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
        max_steps = EVAL_CONFIG.get('max_steps', None)  # None = no limit
        
        while not done and (max_steps is None or step < max_steps):
            try:
                action = agent.act(obs, reward if step > 0 else None, done)
                obs, reward, done, info = env.step(action)
                
                episode_data['total_reward'] += reward
                step += 1
                
                # Track action taken (if agent has last_action attribute)
                if hasattr(agent, 'last_action_taken') and agent.last_action_taken is not None:
                    action_id = agent.last_action_taken
                    action_counts[action_id] = action_counts.get(action_id, 0) + 1
                
            except Exception as e:
                print(f"\nError in scenario {scenario_id}, step {step}: {e}")
                break
        
        episode_data['steps'] = step
        episode_data['final_step'] = step
        # Survived if: reached max_steps limit, OR completed most of chronic (>= 8064 steps)
        episode_data['survived'] = (max_steps is not None and step >= max_steps) or (max_steps is None and step >= 8064)
        
        results.append(episode_data)
    
    print()  # New line after progress
    
    # DEBUG: Check action_counts state
    print(f"DEBUG: action_counts has {len(action_counts)} unique actions, total {sum(action_counts.values())} actions")
    
    # Print action distribution summary
    if action_counts:
        print(f"\n{'='*60}")
        print("ACTION DISTRIBUTION")
        print(f"{'='*60}")
        total_actions = sum(action_counts.values())
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Total actions taken: {total_actions}")
        print(f"Unique actions used: {len(action_counts)}")
        print(f"\nTop actions:")
        for action_id, count in sorted_actions[:10]:
            percentage = (count / total_actions) * 100
            print(f"  Action {action_id}: {count} times ({percentage:.1f}%)")
        
        if len(sorted_actions) > 10:
            remaining_count = sum(count for _, count in sorted_actions[10:])
            remaining_pct = (remaining_count / total_actions) * 100
            print(f"  ... {len(sorted_actions) - 10} other actions: {remaining_count} times ({remaining_pct:.1f}%)")
        print()
    
    return results


import numpy as np

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def compute_statistics(episode_results):
    """Compute evaluation statistics"""
    
    total_episodes = len(episode_results)
    if total_episodes == 0:
        return {}
    
    # Basic statistics
    rewards = [r['total_reward'] for r in episode_results]
    steps = [r['steps'] for r in episode_results] 
    survived = [r['survived'] for r in episode_results]
    
    stats = {
        'episodes': total_episodes,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'avg_steps': np.mean(steps),
        'std_steps': np.std(steps),
        'min_steps': np.min(steps),
        'max_steps': np.max(steps),
        'survival_rate': np.mean(survived) * 100,
        'survived_episodes': np.sum(survived)
    }
    
    return stats


def print_evaluation_report(stats, results):
    """Print detailed evaluation report"""
    
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"\nOverall Performance:")
    print(f"  Episodes evaluated: {stats['episodes']}")
    print(f"  Average reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Reward range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    
    print(f"\nEpisode Length:")
    print(f"  Average steps: {stats['avg_steps']:.1f} ± {stats['std_steps']:.1f}")
    print(f"  Steps range: [{stats['min_steps']}, {stats['max_steps']}]")
    
    print(f"\nSurvival Analysis:")
    print(f"  Survival rate: {stats['survival_rate']:.1f}%")
    print(f"  Survived episodes: {stats['survived_episodes']}/{stats['episodes']}")
    
    # Reward distribution
    rewards = [r['total_reward'] for r in results]
    percentiles = [10, 25, 50, 75, 90]
    print(f"\nReward Distribution:")
    for p in percentiles:
        value = np.percentile(rewards, p)
        print(f"  {p}th percentile: {value:.2f}")
    
    # Failed episodes analysis
    failed = [r for r in results if not r['survived']]
    if failed:
        failed_steps = [r['steps'] for r in failed]
        print(f"\nFailed Episodes Analysis:")
        print(f"  Failed episodes: {len(failed)}")
        print(f"  Average failure step: {np.mean(failed_steps):.1f}")
        print(f"  Earliest failure: step {np.min(failed_steps)}")


def create_agent(env, agent_config):
    """Create agent based on configuration"""
    agent_type = agent_config.get('agent_type', 'custom')
    
    if agent_type == 'do_nothing':
        print("Creating DoNothingAgent (baseline)")
        return DoNothingAgent(env.action_space)
    elif agent_type == 'custom':
        print("Creating MyCustomAgent")
        
        # Auto-detect configuration from model if model_path is available
        model_path = agent_config.get('model_path')
        if model_path and os.path.exists(model_path):
            try:
                detected_config = detect_model_config(model_path)
                
                # Override ACTIONS_CONFIG with detected values
                detected_actions_config = {
                    **ACTIONS_CONFIG,
                    'reduction': detected_config['reduction'],
                    'drop_identity': detected_config['drop_identity'],
                    'include_do_nothing': detected_config['include_do_nothing'],
                    'substations': detected_config['substations']
                }
                
                # Override agent_config with detected observation space
                detected_agent_config = {
                    **agent_config,
                    'obs_space_type': detected_config['obs_space_type']
                }
                
                # Override with environment variable if provided
                env_obs_space_type = os.environ.get('OBS_SPACE_TYPE')
                if env_obs_space_type:
                    detected_agent_config['obs_space_type'] = env_obs_space_type
                    print(f"🔧 Overriding observation space with: {env_obs_space_type}")
                
                print(f"✅ Using auto-detected configuration")
                
                # Use detected configs
                full_config = {**detected_agent_config, 'ACTIONS_CONFIG': detected_actions_config}
                
            except Exception as e:
                print(f"⚠️ Auto-detection failed: {e}")
                print(f"📋 Falling back to default configuration")
                # Fall back to original config
                full_config = {**agent_config, 'ACTIONS_CONFIG': ACTIONS_CONFIG}
                
                # Override with environment variable if provided
                env_obs_space_type = os.environ.get('OBS_SPACE_TYPE')
                if env_obs_space_type:
                    full_config['obs_space_type'] = env_obs_space_type
                    print(f"🔧 Overriding observation space with: {env_obs_space_type}")
        else:
            # No model path or file doesn't exist - use default config
            full_config = {**agent_config, 'ACTIONS_CONFIG': ACTIONS_CONFIG}
            
            # Override with environment variable if provided
            env_obs_space_type = os.environ.get('OBS_SPACE_TYPE')
            if env_obs_space_type:
                full_config['obs_space_type'] = env_obs_space_type
                print(f"🔧 Overriding observation space with: {env_obs_space_type}")
        
        agent = MyCustomAgent(
            action_space=env.action_space,
            config=full_config
        )

        # Required for gym/essential observation encodings during inference.
        agent.set_env(env)
        
        # Set agent to test mode for fast inference (no MCTS)
        agent.set_mode('test')
        
        # Set MCTS config so the agent can use proper observation encoding
        agent.mcts_config = full_config
        
        # Set nn_funcs for fast neural network inference (required for test mode)
        # IMPORTANT: Use the final full_config here so nn_funcs gets the detected obs_space_type
        from src.networks.neural_network_factory import get_neural_network_functions
        agent.nn_funcs = get_neural_network_functions(full_config)
        
        # Load trained model if available
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            agent.load_model(model_path)
        else:
            print("No trained model found - using random/default policy")
        
        return agent
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}. Use 'custom' or 'do_nothing'")


def compare_agents(env, scenarios=None, max_episodes=None):
    """Compare custom agent vs do nothing agent on the same scenarios"""
    
    if scenarios is None:
        # Use test scenarios from 90/10 split
        scenarios = get_test_scenarios(env, max_episodes)
    elif max_episodes is not None:
        scenarios = scenarios[:max_episodes]
    
    print(f"Comparing agents on {len(scenarios)} scenarios...")
    
    # Create both agents
    custom_config = {**AGENT_CONFIG, 'agent_type': 'custom'}
    
    # Override model_path from environment variable if set
    if 'MODEL_PATH' in os.environ:
        custom_config['model_path'] = os.environ['MODEL_PATH']
        print(f"📦 Using model from environment: {custom_config['model_path']}")
    
    do_nothing_config = {**AGENT_CONFIG, 'agent_type': 'do_nothing'}
    
    custom_agent = create_agent(env, custom_config)
    do_nothing_agent = create_agent(env, do_nothing_config)
    
    custom_results = []
    do_nothing_results = []
    action_counts = {}  # Track custom agent action frequency
    
    for i, scenario_id in enumerate(scenarios):
        print(f"\rScenario {i+1}/{len(scenarios)}: {scenario_id}", end='', flush=True)
        
        # Test custom agent
        env.chronics_handler.tell_id(scenario_id)
        obs = env.reset()
        custom_agent.reset(obs)
        
        custom_episode = {
            'scenario_id': scenario_id,
            'steps': 0,
            'total_reward': 0,
            'survived': False,
        }
        
        done = False
        step = 0
        max_steps = EVAL_CONFIG.get('max_steps', None)
        
        while not done and (max_steps is None or step < max_steps):
            try:
                action = custom_agent.act(obs, reward if step > 0 else None, done)
                obs, reward, done, info = env.step(action)
                custom_episode['total_reward'] += reward
                step += 1
                
                # Track action taken
                if hasattr(custom_agent, 'last_action_taken') and custom_agent.last_action_taken is not None:
                    action_id = custom_agent.last_action_taken
                    action_counts[action_id] = action_counts.get(action_id, 0) + 1
            except Exception as e:
                break
        
        custom_episode['steps'] = step
        # Survived if: reached max_steps limit, OR completed most of chronic (>= 8064 steps)
        custom_episode['survived'] = (max_steps is not None and step >= max_steps) or (max_steps is None and step >= 8064)
        custom_results.append(custom_episode)
        
        # Test do nothing agent on same scenario
        env.chronics_handler.tell_id(scenario_id)
        obs = env.reset()
        
        do_nothing_episode = {
            'scenario_id': scenario_id,
            'steps': 0,
            'total_reward': 0,
            'survived': False,
        }
        
        done = False
        step = 0
        
        while not done and (max_steps is None or step < max_steps):
            try:
                action = do_nothing_agent.act(obs, reward if step > 0 else None, done)
                obs, reward, done, info = env.step(action)
                do_nothing_episode['total_reward'] += reward
                step += 1
            except Exception as e:
                break
        
        do_nothing_episode['steps'] = step
        # Survived if: reached max_steps limit, OR completed most of chronic (>= 8064 steps)
        do_nothing_episode['survived'] = (max_steps is not None and step >= max_steps) or (max_steps is None and step >= 8064)
        do_nothing_results.append(do_nothing_episode)
    
    print()  # New line after progress
    
    # Print action distribution summary for custom agent
    if action_counts:
        print(f"\n{'='*60}")
        print("CUSTOM AGENT - ACTION DISTRIBUTION")
        print(f"{'='*60}")
        total_actions = sum(action_counts.values())
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Total actions taken: {total_actions}")
        print(f"Unique actions used: {len(action_counts)}")
        print(f"\nTop actions:")
        for action_id, count in sorted_actions[:10]:
            percentage = (count / total_actions) * 100
            print(f"  Action {action_id}: {count} times ({percentage:.1f}%)")
        
        if len(sorted_actions) > 10:
            remaining_count = sum(count for _, count in sorted_actions[10:])
            remaining_pct = (remaining_count / total_actions) * 100
            print(f"  ... {len(sorted_actions) - 10} other actions: {remaining_count} times ({remaining_pct:.1f}%)")
        print()
    else:
        print("\n⚠️ No actions tracked (all do-nothing or tracking failed)\n")
    
    return custom_results, do_nothing_results


def print_comparison_report(custom_results, do_nothing_results):
    """Print comparison report between agents"""
    
    custom_stats = compute_statistics(custom_results)
    do_nothing_stats = compute_statistics(do_nothing_results)
    
    print("\n" + "="*80)
    print("AGENT COMPARISON REPORT")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Custom Agent':<20} {'Do Nothing Agent':<20} {'Difference':<15}")
    print("-" * 80)
    
    metrics = [
        ('Episodes', 'episodes', '{:.0f}'),
        ('Avg Reward', 'avg_reward', '{:.2f}'),
        ('Avg Steps', 'avg_steps', '{:.1f}'),
        ('Survival Rate (%)', 'survival_rate', '{:.1f}'),
        ('Survived Episodes', 'survived_episodes', '{:.0f}'),
    ]
    
    for metric_name, metric_key, fmt in metrics:
        custom_val = custom_stats.get(metric_key, 0)
        do_nothing_val = do_nothing_stats.get(metric_key, 0)
        
        if metric_key == 'survival_rate':
            diff = custom_val - do_nothing_val
            diff_str = f"{diff:+.1f}%"
        elif metric_key in ['avg_reward', 'avg_steps']:
            diff = custom_val - do_nothing_val
            diff_str = f"{diff:+.2f}"
        else:
            diff = custom_val - do_nothing_val
            diff_str = f"{diff:+.0f}"
        
        print(f"{metric_name:<25} {fmt.format(custom_val):<20} {fmt.format(do_nothing_val):<20} {diff_str:<15}")
    
    # Performance summary
    print(f"\n{'Performance Summary:'}")
    if custom_stats['avg_reward'] > do_nothing_stats['avg_reward']:
        print(f"  ✅ Custom agent outperforms do nothing by {custom_stats['avg_reward'] - do_nothing_stats['avg_reward']:.2f} reward points")
    else:
        print(f"  ❌ Custom agent underperforms do nothing by {do_nothing_stats['avg_reward'] - custom_stats['avg_reward']:.2f} reward points")
    
    if custom_stats['survival_rate'] > do_nothing_stats['survival_rate']:
        print(f"  ✅ Custom agent has {custom_stats['survival_rate'] - do_nothing_stats['survival_rate']:.1f}% better survival rate")
    else:
        print(f"  ❌ Custom agent has {do_nothing_stats['survival_rate'] - custom_stats['survival_rate']:.1f}% worse survival rate")


def main():
    """Main evaluation function"""
    
    print("MyCustomAgent Evaluation")
    print("=" * 30)
    
    # Check if we should run comparison
    compare_mode = AGENT_CONFIG.get('compare_with_baseline', False)
    
    # Setup environment
    print("Setting up environment...")
    
    # Get reward class from factory
    from src.rewards.reward_factory import get_reward_class
    reward_name = ENV_CONFIG.get('reward_class', 'AlphaZero')
    reward_class = get_reward_class(reward_name)
    print(f"🎯 Using reward: {reward_name}")
    
    env = grid2op.make(
        ENV_CONFIG['name'],
        backend=LightSimBackend(),
        reward_class=reward_class,
        other_rewards=ENV_CONFIG.get('other_rewards', {})
    )
    
    print(f"Environment: {env.name}")
    print(f"Available scenarios: {len(env.chronics_handler.subpaths)}")
    
    # Get evaluation episodes (None = all test scenarios, number = limit)
    max_episodes = EVAL_CONFIG.get('episodes', None)
    print(f"DEBUG: EVAL_CONFIG episodes setting: {max_episodes}")
    
    if compare_mode:
        print("\n=== COMPARISON MODE ===")
        print("Comparing Custom Agent vs Do Nothing Agent")
        
        # Run comparison
        custom_results, do_nothing_results = compare_agents(env, max_episodes=max_episodes)
        
        # Print comparison report
        print_comparison_report(custom_results, do_nothing_results)
        
        # Save comparison results (exclude non-serializable config parts)
        output_file = f"comparison_results_{ENV_CONFIG['name']}.json"
        
        # Create serializable config (exclude class objects)
        serializable_agent_config = {k: v for k, v in AGENT_CONFIG.items() 
                                   if not (hasattr(v, '__class__') and hasattr(v.__class__, '__name__'))}
        serializable_env_config = {k: v for k, v in ENV_CONFIG.items() 
                                 if k not in ['other_rewards', 'action_class', 'observation_class'] and 
                                    not (hasattr(v, '__class__') and hasattr(v.__class__, '__name__'))}
        
        output_data = {
            'config': {
                'agent_config': serializable_agent_config,
                'env_config': serializable_env_config,
                'eval_config': EVAL_CONFIG
            },
            'custom_agent': {
                'statistics': convert_numpy_types(compute_statistics(custom_results))
            },
            'do_nothing_agent': {
                'statistics': convert_numpy_types(compute_statistics(do_nothing_results))
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nComparison results saved to: {output_file}")
        
    else:
        print(f"\n=== SINGLE AGENT MODE ===")
        print(f"Agent type: {AGENT_CONFIG.get('agent_type', 'custom')}")
        
        # Create agent config and override model_path from environment variable if set
        agent_config = {**AGENT_CONFIG}
        if 'MODEL_PATH' in os.environ:
            agent_config['model_path'] = os.environ['MODEL_PATH']
            print(f"📦 Using model from environment: {agent_config['model_path']}")
            print(f"🔍 DEBUG: File exists: {os.path.exists(agent_config['model_path'])}")
        else:
            print("🔍 DEBUG: No MODEL_PATH in environment variables")
        
        # Create agent based on config
        print("Creating agent...")
        agent = create_agent(env, agent_config)
        
        # Run evaluation
        results = evaluate_on_scenarios(env, agent, max_episodes=max_episodes)
        
        # Compute and display statistics
        stats = compute_statistics(results)
        print_evaluation_report(stats, results)
        
        # Output detailed episode results for checkpoint evaluator
        print("\nEPISODE_RESULTS_JSON:")
        episode_results_json = []
        for r in results:
            episode_results_json.append({
                'scenario_id': r['scenario_id'],
                'steps': r['steps'],
                'total_reward': r['total_reward'],
                'survived': r['survived']
            })
        print(json.dumps(episode_results_json))
        print("END_EPISODE_RESULTS_JSON")
        
        # Save results
        agent_type = AGENT_CONFIG.get('agent_type', 'custom')
        output_file = f"evaluation_results_{agent_type}_{ENV_CONFIG['name']}.json"
        output_data = {
            'config': {
                'agent_config': AGENT_CONFIG,
                'env_config': ENV_CONFIG,
                'eval_config': EVAL_CONFIG
            },
            'statistics': convert_numpy_types(stats),
            'episodes': convert_numpy_types(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()