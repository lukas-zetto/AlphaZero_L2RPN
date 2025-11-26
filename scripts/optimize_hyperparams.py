#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for AlphaZero agent.
Runs full training pipeline for each trial.
"""

import optuna
import sys
import os
import json
import subprocess
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config


def objective(trial):
    """
    Optuna objective function to minimize.
    Returns: negative average episode survival steps (we want to maximize).
    
    Runs full training pipeline with trial parameters.
    """
    
    # Suggest hyperparameters
    mcts_simulations = trial.suggest_int('mcts_simulations', 50, 400, step=50)
    puct_c = trial.suggest_float('puct_c', 1.0, 5.0)
    temperature = trial.suggest_float('temperature', 0.5, 3.0)
    temperature_decay = trial.suggest_float('temperature_decay', 0.90, 0.99)
    dirichlet_epsilon = trial.suggest_float('dirichlet_epsilon', 0.1, 0.5)
    dirichlet_alpha = trial.suggest_float('dirichlet_alpha', 0.1, 1.0)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    training_epochs = trial.suggest_int('training_epochs', 1, 5)
    replay_buffer_size = trial.suggest_int('replay_buffer_size', 30, 120, step=30)
    critical_threshold = trial.suggest_float('critical_threshold', 0.85, 0.98)
    
    # Policy target method with conditional selection_bias_weight
    policy_target_method = trial.suggest_categorical('policy_target_method', ['visits', 'visits_with_selection_bias'])
    if policy_target_method == 'visits_with_selection_bias':
        selection_bias_weight = trial.suggest_float('selection_bias_weight', 0.0, 1.0)
    else:
        selection_bias_weight = 0.0
    
    # PUCT modifications (depth bonus and virtual loss)
    use_depth_bonus = trial.suggest_categorical('use_depth_bonus', [True, False])
    depth_bonus = trial.suggest_float('depth_bonus', 0.01, 0.2) if use_depth_bonus else 0.0
    use_virtual_loss = trial.suggest_categorical('use_virtual_loss', [True, False])
    virtual_loss_weight = trial.suggest_float('virtual_loss_weight', 0.1, 1.0) if use_virtual_loss else 0.0
    
    # Quick training settings for optimization
    num_iterations = 10  # Short training run per trial
    episodes_per_iteration = 4
    parallel_workers = 0  # Sequential for stability
    
    params_to_update = {
        'mcts_simulations': mcts_simulations,
        'puct_c': puct_c,
        'temperature': temperature,
        'temperature_decay': temperature_decay,
        'dirichlet_epsilon': dirichlet_epsilon,
        'dirichlet_alpha': dirichlet_alpha,
        'learning_rate': learning_rate,
        'training_epochs': training_epochs,
        'replay_buffer_size': replay_buffer_size,
        'critical_threshold': critical_threshold,
        'intervention_threshold': critical_threshold,
        'policy_target_method': policy_target_method,
        'selection_bias_weight': selection_bias_weight,
        'use_depth_bonus': use_depth_bonus,
        'depth_bonus': depth_bonus,
        'use_virtual_loss': use_virtual_loss,
        'virtual_loss_weight': virtual_loss_weight,
        'episodes_per_iteration': episodes_per_iteration,
        'parallel_workers': parallel_workers,
        'num_cycles': 1,  # Only 1 cycle through chronics
    }
    
    # Save current config
    config_backup = config.AGENT_CONFIG.copy()
    
    try:
        # Update config
        config.AGENT_CONFIG.update(params_to_update)
        
        # Write config to temporary file
        trial_dir = Path(f'optuna_trials/trial_{trial.number}')
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = trial_dir / 'config_backup.json'
        with open(config_path, 'w') as f:
            json.dump(params_to_update, f, indent=2)
        
        # Clean up previous trial checkpoints
        checkpoint_dir = Path('checkpoints')
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob('alphazero_v2_iter*.pt'):
                ckpt.unlink()
        
        # Run training script
        print(f"\n{'='*80}")
        print(f"TRIAL {trial.number}: Training with parameters:")
        for key, val in params_to_update.items():
            print(f"  {key}: {val}")
        print(f"{'='*80}\n")
        
        result = subprocess.run(
            [sys.executable, 'scripts/train_alphazero_v2.py'],
            cwd='/workspace',
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max per trial
        )
        
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return float('inf')
        
        # Run evaluation
        eval_result = subprocess.run(
            [sys.executable, 'scripts/evaluate_agent.py'],
            cwd='/workspace',
            capture_output=True,
            text=True,
            timeout=600  # 10 min max for eval
        )
        
        if eval_result.returncode != 0:
            print(f"Evaluation failed: {eval_result.stderr}")
            return float('inf')
        
        # Parse evaluation results from output
        # Look for average survival steps in the output
        output = eval_result.stdout
        avg_steps = 0
        
        for line in output.split('\n'):
            if 'Average steps' in line or 'avg steps' in line.lower():
                try:
                    # Extract number from line
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        avg_steps = float(numbers[0])
                        break
                except:
                    pass
        
        if avg_steps == 0:
            # Fallback: check last checkpoint survival
            print("Could not parse evaluation output, using fallback score")
            avg_steps = 100  # Minimal score
        
        print(f"\nTrial {trial.number} completed: avg_steps = {avg_steps:.2f}\n")
        
        # Return negative (Optuna minimizes)
        return -avg_steps
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out")
        return float('inf')
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')
    finally:
        # Restore config
        config.AGENT_CONFIG.clear()
        config.AGENT_CONFIG.update(config_backup)


def main():
    """Run Optuna optimization."""
    
    print("=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Create study
    study = optuna.create_study(
        direction='minimize',  # Minimize negative avg steps = maximize avg steps
        study_name='alphazero_optimization',
        storage=None,  # Use in-memory storage (or specify SQLite/MySQL for persistence)
        sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)  # Prune bad trials early
    )
    
    # Run optimization
    n_trials = 50  # Number of hyperparameter combinations to try
    
    print(f"\nRunning {n_trials} optimization trials...")
    print("This will take a while. Each trial trains for 10 iterations.\n")
    
    study.optimize(objective, n_trials=n_trials, n_jobs=1)  # n_jobs=1 for sequential (or >1 for parallel)
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value (negative avg steps): {study.best_trial.value:.2f}")
    print(f"Best avg steps: {-study.best_trial.value:.2f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = Path('optimization_results')
    results_dir.mkdir(exist_ok=True)
    
    # Save best parameters to JSON
    best_params_path = results_dir / 'best_hyperparameters.json'
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"\n✅ Best parameters saved to: {best_params_path}")
    
    # Save optimization history
    history_path = results_dir / 'optimization_history.csv'
    df = study.trials_dataframe()
    df.to_csv(history_path, index=False)
    print(f"✅ Optimization history saved to: {history_path}")
    
    # Generate importance plot (if multiple trials)
    if len(study.trials) > 1:
        try:
            import plotly
            
            # Parameter importance
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_html(str(results_dir / 'param_importance.html'))
            
            # Optimization history
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.write_html(str(results_dir / 'optimization_history.html'))
            
            # Parallel coordinate plot
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
            fig_parallel.write_html(str(results_dir / 'parallel_coordinate.html'))
            
            print(f"✅ Visualization plots saved to: {results_dir}/")
        except ImportError:
            print("\n⚠️  Install plotly for visualizations: pip install plotly")
    
    print("\n" + "=" * 80)
    print("Optimization complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
