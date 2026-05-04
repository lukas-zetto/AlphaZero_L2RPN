#!/usr/bin/env python3
"""
Test MCTS neural network functionality
"""

import os
import sys
import numpy as np
import grid2op
from lightsim2grid import LightSimBackend

# Add the same path setup as working evaluation script
sys.path.append('/workspace/src')  # For direct imports like 'from networks.neural_network_factory'
sys.path.append('/workspace')      # For src-prefixed imports like 'from src.networks.neural_network_factory'

# Add project root to path (backup approach)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.agent.my_agent import MyCustomAgent
from src.config import AGENT_CONFIG, ENV_CONFIG, EVAL_CONFIG, ACTIONS_CONFIG, TRAINING_CONFIG

def test_mcts_functionality():
    """Test if MCTS method works without explicit nn_funcs loading"""
    
    # Load agent in training mode to enable MCTS
    agent_config = AGENT_CONFIG.copy()
    agent_config['mode'] = 'training'  # Enable MCTS
    
    print("🔧 Creating agent...")
    agent = MyCustomAgent(agent_config)
    
    print("🔧 Loading environment...")
    env = grid2op.make('l2rpn_case14_sandbox')
    
    print("🔧 Loading neural network checkpoint...")
    agent.load('/workspace/saved_model_idf2023/checkpoint_70000.pkl', env)
    
    print("🔧 Resetting environment...")
    obs = env.reset()
    
    print(f"nn_funcs status before MCTS call: {agent.nn_funcs}")
    
    # Test MCTS directly (this is what would fail if nn_funcs is None)
    print("\n🧪 Testing MCTS method...")
    try:
        grid_state = {'critical': True}  # Dummy grid state
        result = agent._get_mcts_action(obs, grid_state)
        
        if result is not None:
            action, action_idx, training_data = result
            print(f"✅ MCTS SUCCESS!")
            print(f"   Action: {action is not None}")
            print(f"   Action index: {action_idx}")
            print(f"   Training data: {len(training_data) if training_data else 0} samples")
        else:
            print(f"❌ MCTS returned None")
            
    except Exception as e:
        print(f"❌ MCTS FAILED: {e}")
        if "'NoneType' object has no attribute 'neural_network_forward'" in str(e):
            print("🔍 This is the expected error - MCTS needs neural network functions loaded!")
        else:
            print(f"🔍 Unexpected error type: {type(e).__name__}")

if __name__ == "__main__":
    test_mcts_functionality()