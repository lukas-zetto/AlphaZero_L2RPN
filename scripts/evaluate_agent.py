#!/usr/bin/env python3
"""
Evaluation script for MyCustomAgent using L2RPN scoring
"""

import os
import sys
import json
import numpy as np
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Agent import DoNothingAgent

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agent.my_agent import MyCustomAgent
from src.rewards.custom_reward import MyCustomReward
from src.config import AGENT_CONFIG, ENV_CONFIG, EVAL_CONFIG, ACTIONS_CONFIG


def get_test_scenarios(env, max_episodes=None):
    """Get test scenarios based on 90/10 split or configuration"""
    
    total_scenarios = len(env.chronics_handler.subpaths)
    
    # Calculate 90/10 split
    train_size = int(total_scenarios * 0.9)
    test_scenarios = list(range(train_size, total_scenarios))  # Last 10%
    
    print(f"Total scenarios: {total_scenarios}")
    print(f"Training scenarios: 0-{train_size-1} ({train_size} scenarios)")
    print(f"Test scenarios: {train_size}-{total_scenarios-1} ({len(test_scenarios)} scenarios)")
    
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
        # Merge ACTIONS_CONFIG into agent_config
        full_config = {**agent_config, 'ACTIONS_CONFIG': ACTIONS_CONFIG}
        agent = MyCustomAgent(
            action_space=env.action_space,
            config=full_config
        )
        
        # Load trained model if available
        model_path = agent_config.get('model_path')
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
    do_nothing_config = {**AGENT_CONFIG, 'agent_type': 'do_nothing'}
    
    custom_agent = create_agent(env, custom_config)
    do_nothing_agent = create_agent(env, do_nothing_config)
    
    custom_results = []
    do_nothing_results = []
    
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
        max_steps = EVAL_CONFIG.get('max_steps', 8064)
        
        while not done and step < max_steps:
            try:
                action = custom_agent.act(obs, reward if step > 0 else None, done)
                obs, reward, done, info = env.step(action)
                custom_episode['total_reward'] += reward
                step += 1
            except Exception as e:
                break
        
        custom_episode['steps'] = step
        custom_episode['survived'] = (step >= max_steps)
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
        
        while not done and step < max_steps:
            try:
                action = do_nothing_agent.act(obs, reward if step > 0 else None, done)
                obs, reward, done, info = env.step(action)
                do_nothing_episode['total_reward'] += reward
                step += 1
            except Exception as e:
                break
        
        do_nothing_episode['steps'] = step
        do_nothing_episode['survived'] = (step >= max_steps)
        do_nothing_results.append(do_nothing_episode)
    
    print()  # New line after progress
    
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
        
        # Create agent based on config
        print("Creating agent...")
        agent = create_agent(env, AGENT_CONFIG)
        
        # Run evaluation
        results = evaluate_on_scenarios(env, agent, max_episodes=max_episodes)
        
        # Compute and display statistics
        stats = compute_statistics(results)
        print_evaluation_report(stats, results)
        
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