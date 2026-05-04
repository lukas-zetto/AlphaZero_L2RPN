"""
Quick training script for Alpha Zero agent with auto-reconnection feature.
This version saves checkpoints with 'reconnect_' prefix to avoid conflicts.
"""

import os
import sys
import time
import warnings
import numpy as np
import json

# Suppress misleading scipy warnings
warnings.filterwarnings("ignore")

import grid2op

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agent.my_agent import MyCustomAgent
from src.config import MASTER_CONFIG
from src.training.line_load_logging import EpisodeLineLoadLogger
from src.training.train_agent import collect_training_data

# Extract configs from unified structure
AGENT_CONFIG = MASTER_CONFIG['core_agent']
ACTIONS_CONFIG = MASTER_CONFIG['actions']
ENV_CONFIG = MASTER_CONFIG['environment']
LINE_LOGGING_CONFIG = MASTER_CONFIG['line_logging']

# Merge ACTIONS_CONFIG into AGENT_CONFIG for catalog support
AGENT_CONFIG.update(ACTIONS_CONFIG)

# Scale value loss by 2x for better value head learning
AGENT_CONFIG['value_weight'] = 2.0
AGENT_CONFIG['policy_weight'] = 1.0


def get_chronic_split(env, train_ratio=0.9):
    """Split chronics into train/test sets using sequential order"""
    total_chronics = len(env.chronics_handler.subpaths)
    train_size = int(total_chronics * train_ratio)
    
    # Use chronics in sequential order (0, 1, 2, 3, ...)
    all_chronics = list(range(total_chronics))
    
    # Sequential split: first 90% for training, last 10% for testing
    train_chronics = all_chronics[:train_size]
    test_chronics = all_chronics[train_size:]
    
    print(f"📊 Chronic Split (Sequential Order):")
    print(f"   Total chronics: {total_chronics}")
    print(f"   Training chronics: {len(train_chronics)} (chronics 0-{train_size-1})")
    print(f"   Test chronics: {len(test_chronics)} (chronics {train_size}-{total_chronics-1})")
    
    return train_chronics, test_chronics


def quick_training():
    """
    Wrapper around episode execution that logs action impacts.
    
    This runs one episode and logs before/after state for each non-do-nothing action.
    """
    from src.training.train_agent import run_alpha_zero_mcts, is_critical_state
    from neural_network_factory import get_neural_network_functions
    
    episode_data = []
    obs = env.get_obs()
    done = False
    step = 0
    safe_steps = 0
    
    # Get neural network functions
    nn_funcs = get_neural_network_functions(AGENT_CONFIG)
    
    # Determine action space size
    if hasattr(agent, 'action_catalog') and agent.action_catalog is not None:
        num_actions = agent.action_catalog.size
    elif hasattr(agent, 'bus_actions') and agent.bus_actions is not None:
        num_actions = agent.bus_actions.get_action_space_size()
    else:
        num_actions = 21
    
    print(f"📊 Collecting episode 1/1")
    
    while not done:
        step += 1
        
        # PRIORITY 1: Check for auto-reconnect (before MCTS)
        disconnected = np.where(obs.line_status == False)[0]
        auto_reconnect_action = None
        line_reconnected = None
        
        for line_id in disconnected:
            if obs.time_before_cooldown_line[line_id] == 0:
                obs_before = obs
                auto_reconnect_action = env.action_space()
                auto_reconnect_action.line_set_status = [(line_id, 1)]
                line_reconnected = int(line_id)
                
                # Execute reconnect
                obs_after, reward, done, info = env.step(auto_reconnect_action)
                
                # Log impact
                impact_logger.log_action_impact(
                    obs_before, obs_after, auto_reconnect_action,
                    action_type='auto_reconnect',
                    line_reconnected=line_reconnected
                )
                
                if line_logger is not None:
                    line_logger.update(obs_after)
                
                obs = obs_after
                break  # Only one reconnect per step
        
        if auto_reconnect_action is not None:
            # Skip MCTS this step, move to next
            continue
        
        # Check if critical state (needs MCTS)
        if is_critical_state(obs, threshold=AGENT_CONFIG.get('critical_threshold', 0.95)):
            safe_steps = 0  # Reset safe counter
            # Store observation BEFORE action
            obs_before = obs
            
            # Run MCTS to select action
            try:
                best_action, action_probs, stats = run_alpha_zero_mcts(
                    env,
                    neural_network=agent.neural_network if hasattr(agent, 'neural_network') else None,
                    config=AGENT_CONFIG,
                    action_catalog=agent.action_catalog if hasattr(agent, 'action_catalog') else None
                )
                
                # Determine action details
                action_type = 'catalog'
                action_idx = None
                substation_id = None
                
                # Try to identify which action was selected
                if hasattr(agent, 'action_catalog') and agent.action_catalog is not None:
                    # Find matching catalog action
                    for idx, catalog_action in enumerate(agent.action_catalog.actions):
                        try:
                            test_action = catalog_action.apply(env.action_space)
                            if str(test_action) == str(best_action):
                                action_idx = idx
                                substation_id = catalog_action.sub_id
                                break
                        except:
                            pass
                
                # Execute action
                obs_after, reward, done, info = env.step(best_action)
                
                # Log action impact (skip do-nothing)
                if action_idx is not None or str(best_action) != str(env.action_space()):
                    impact_logger.log_action_impact(
                        obs_before, obs_after, best_action,
                        action_type=action_type,
                        action_idx=action_idx,
                        substation_id=substation_id
                    )
                
                # Update line logger
                if line_logger is not None:
                    line_logger.update(obs_after)
                
                # Store training data
                mcts_value = stats.get('root_value', 0.0)
                from src.training.train_agent import action_probs_to_fixed_policy
                probs_only = action_probs_to_fixed_policy(
                    action_probs, obs_before, num_actions=num_actions,
                    action_catalog=agent.action_catalog if hasattr(agent, 'action_catalog') else None
                )
                
                encoded_state = nn_funcs.encode_observation_simple(obs_before)
                episode_data.append({
                    'state': encoded_state,
                    'mcts_policy': probs_only,  # Changed from 'policy' to 'mcts_policy'
                    'value': mcts_value
                })
                
                obs = obs_after
                
            except Exception as e:
                print(f"    ❌ MCTS failed at step {step}: {e}")
                break
        else:
            # Safe state - do nothing (skip logging)
            safe_steps += 1
            obs, reward, done, info = env.step(env.action_space())
            if line_logger is not None:
                line_logger.update(obs)
        
        if done:
            break
    
    # Only print stats if we actually collected training data
    if len(episode_data) > 0:
        print(f"    Reached critical state after {safe_steps} safe steps (max_rho={obs.rho.max():.3f})")
    print(f"    Episode done at step {step}")
    avg_value = np.mean([ex['value'] for ex in episode_data]) if episode_data else 0.0
    print(f"    ✅ Episode: {len(episode_data)} steps, avg MCTS value: {avg_value:.3f}")
    return episode_data


def quick_training():
    """
    Alpha Zero training session with auto-reconnection feature
    """
    print("🏋️ ALPHA ZERO TRAINING (WITH AUTO-RECONNECT)")
    print("=" * 40)
    
    try:
        # Setup environment with reward factory
        print("🏗️ Setting up environment...")
        try:
            from lightsim2grid import LightSimBackend
            from src.rewards.reward_factory import get_reward_class
            
            reward_name = ENV_CONFIG.get('reward_class', 'AlphaZero')
            reward_class = get_reward_class(reward_name)
            print(f"🎯 Using reward: {reward_name}")
            
            env = grid2op.make(
                ENV_CONFIG['name'],
                reward_class=reward_class,
                backend=LightSimBackend()
            )
            print("   ✅ Environment created with LightSimBackend")
        except Exception as e:
            print(f"   ⚠️ LightSimBackend not available, using default: {e}")
            from src.rewards.reward_factory import get_reward_class
            reward_name = ENV_CONFIG.get('reward_class', 'AlphaZero')
            reward_class = get_reward_class(reward_name)
            env = grid2op.make(ENV_CONFIG['name'], reward_class=reward_class)
        
        # Create agent with auto-reconnect feature
        print("🤖 Creating agent...")
        merged_agent_config = {**AGENT_CONFIG, 'ACTIONS_CONFIG': ACTIONS_CONFIG}
        agent = MyCustomAgent(
            action_space=env.action_space,
            config=merged_agent_config
        )
        print("   ✅ Agent created with auto-reconnect feature")
        
        # Initialize line load logger
        line_logger = EpisodeLineLoadLogger(LINE_LOGGING_CONFIG)
        
        # Get chronic splits for 90/10 training/test
        train_chronics, test_chronics = get_chronic_split(env)
        
        # Save test chronics for later evaluation
        with open('data/test_chronics.json', 'w') as f:
            json.dump(test_chronics, f)
        print("💾 Test chronic indices saved to test_chronics.json")
        
        # Alpha Zero MCTS Training
        print(f"🎯 Starting Alpha Zero MCTS training...")
        print(f"🔄 Available training chronics: {len(train_chronics)} (90% of total)")
        max_chronics = AGENT_CONFIG.get('max_training_chronics', None)
        if max_chronics is not None:
            print(f"⚙️ Config limit: max_training_chronics = {max_chronics}")
        else:
            print(f"⚙️ Config: Using ALL training chronics (no limit set)")
        print(f"🌟 Focus: Critical states only (line load > 95%)")
        print(f"🔌 Auto-reconnect: Lines reconnected when cooldown expires")
        print()
        
        start_time = time.time()
        all_training_data = []
        
        # Use training chronics based on config setting
        if max_chronics is not None and max_chronics < len(train_chronics):
            chronics_to_use = train_chronics[:max_chronics]
            print(f"📊 Training on {len(chronics_to_use)} chronics (limited by config)...")
        else:
            chronics_to_use = train_chronics
            print(f"📊 Training on ALL {len(chronics_to_use)} training chronics...")
        
        print(f"    Using chronics in sequential order: {chronics_to_use[0]}-{chronics_to_use[-1]}")
        
        for i, chronic_id in enumerate(chronics_to_use):
            print(f"\n📊 Training on chronic {chronic_id} ({i+1}/{len(chronics_to_use)})")
            
            # Set specific chronic
            env.chronics_handler.tell_id(chronic_id)
            env.reset()
            
            # Collect data using standard function (has all the logging!)
            chronic_data = collect_training_data(
                env, 
                num_episodes=5,  # Multiple episodes per chronic for more data
                mcts_simulations=AGENT_CONFIG.get('mcts_simulations', 200),
                neural_network=agent.neural_network if hasattr(agent, 'neural_network') else None,
                config=AGENT_CONFIG,
                line_logger=line_logger,
                chronic_id=chronic_id,
                action_catalog=agent.action_catalog if hasattr(agent, 'action_catalog') else None
            )
            
            all_training_data.extend(chronic_data)
            
            print(f"📈 Collected {len(chronic_data)} training examples (total: {len(all_training_data)})")
            
            # Train neural network after each episode WITH EXPERIENCE REPLAY
            if chronic_data and hasattr(agent, 'neural_network') and agent.neural_network is not None:
                try:
                    print(f"   🧠 Training NN on {len(all_training_data)} examples (experience replay)...")
                    
                    from src.networks.neural_network_factory import get_neural_network_functions
                    nn_funcs = get_neural_network_functions(AGENT_CONFIG)
                    
                    loss_info = nn_funcs.train_neural_network(agent.neural_network, all_training_data, AGENT_CONFIG)
                    print(f"   ✅ NN updated with experience replay")
                    
                    # Save checkpoint every 50 episodes with unique timestamp
                    if (i + 1) % 50 == 0:
                        timestamp = int(time.time())
                        checkpoint_name = f'reconnect_v2x_checkpoint_ep{i+1}_t{timestamp}.pth'
                        agent.save_model(checkpoint_name)
                        print(f"   💾 Checkpoint saved: {checkpoint_name}")
                    
                except Exception as e:
                    print(f"   ⚠️ NN training failed: {e}")
            
            # Progress update every 10 chronics
            if (i + 1) % 10 == 0:
                print(f"🔄 Progress: {i+1}/{len(chronics_to_use)} chronics, {len(all_training_data)} total examples")
        
        print(f"\n✅ Alpha Zero training completed!")
        print(f"   📈 Training examples collected: {len(all_training_data)}")
        
        # Final neural network training
        if all_training_data and hasattr(agent, 'neural_network') and agent.neural_network is not None:
            print(f"   🧠 Final neural network training on {len(all_training_data)} examples...")
            
            try:
                from src.networks.neural_network_factory import get_neural_network_functions
                nn_funcs = get_neural_network_functions(AGENT_CONFIG)
                
                loss_info = nn_funcs.train_neural_network(agent.neural_network, all_training_data, AGENT_CONFIG)
                print(f"   Policy Loss: {loss_info['policy_loss']:.4f}, Value Loss: {loss_info['value_loss']:.4f}, Total Loss: {loss_info['total_loss']:.4f}")
                
                # Save final model with unique timestamp
                timestamp = int(time.time())
                final_model_name = f'reconnect_v2x_final_t{timestamp}.pth'
                agent.save_model(final_model_name)
                print(f"   ✅ Neural network training completed!")
                print(f"   💾 Model saved: {final_model_name}")
                
            except Exception as e:
                print(f"   ❌ Neural network training failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Training summary
        training_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("🎉 TRAINING SUMMARY (AUTO-RECONNECT)")
        print("=" * 50)
        print(f"✅ Training completed!")
        print(f"   ⏱️  Training time: {training_time:.1f} seconds")
        print(f"   📊 Training examples: {len(all_training_data)}")
        print(f"   🧠 MCTS simulations: {AGENT_CONFIG.get('mcts_simulations', 200)}")
        print(f"   🔌 Auto-reconnect: ✅ Enabled")
        print(f"   🌟 Recovery detection: ✅ Enabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Alpha Zero training with auto-reconnect"""
    print("🚀 ALPHA ZERO TRAINING (AUTO-RECONNECT)")
    print("=" * 60)
    
    training_success = quick_training()
    
    if not training_success:
        print("❌ Training failed")
        return False
    
    print(f"\n💡 Training completed with auto-reconnect!")
    print(f"📊 Ready for evaluation!")
    
    return True


if __name__ == "__main__":
    try:
        import sys
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

