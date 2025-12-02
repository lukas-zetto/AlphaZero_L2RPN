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
    mcts_simulations = trial.suggest_int('mcts_simulations', 100, 1000, step=100)
    puct_c = trial.suggest_float('puct_c', 1.0, 5.0)
    temperature = trial.suggest_float('temperature', 0.5, 3.0)
    temperature_decay = trial.suggest_float('temperature_decay', 0.90, 0.99)
    dirichlet_epsilon = trial.suggest_float('dirichlet_epsilon', 0.1, 0.5)
    dirichlet_alpha = trial.suggest_float('dirichlet_alpha', 0.1, 1.0)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    training_epochs = trial.suggest_int('training_epochs', 1, 5)
    replay_buffer_size = trial.suggest_int('replay_buffer_size', 20, 80, step=20)  # With 90 episodes total (15×6), buffer of 20-80 keeps 22-89%
    critical_threshold = trial.suggest_float('critical_threshold', 0.95, 0.99)
    
    # Training schedule - fixed for speed
    episodes_per_iteration = 10  # Fixed at 10 episodes per iteration
    
    # Loss weighting hyperparameters
    policy_weight = trial.suggest_float('policy_weight', 0.5, 5.0)
    value_weight = trial.suggest_float('value_weight', 0.5, 3.0)
    
    # Penalty for failed episodes
    penalty_for_failure = trial.suggest_float('penalty_for_failure', -10.0, -1.0)
    
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
    num_cycles = 1  # 1 cycle through all training chronics (903 chronics)
    max_training_iterations = 15  # 15 iterations × 6 episodes = 90 total episodes per trial
    parallel_workers = 0  # Sequential episodes - each trial is single-threaded, so run many trials in parallel
    
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
        'policy_weight': policy_weight,
        'value_weight': value_weight,
        'penalty_for_failure': penalty_for_failure,
    }
    
    # Note: parallel_workers and num_cycles are not hyperparameters to optimize
    # They are fixed for all trials (configured in src/config.py manually if needed)
    
    # Create trial directory
    trial_dir = Path(f'optuna_trials/trial_{trial.number}')
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial parameters for reference
    config_path = trial_dir / 'trial_params.json'
    with open(config_path, 'w') as f:
        json.dump(params_to_update, f, indent=2)
    
    # Update config in memory (won't affect subprocess but good for tracking)
    config_backup = config.AGENT_CONFIG.copy()
    
    try:
        config.AGENT_CONFIG.update(params_to_update)
        
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
        
        # Write stdout/stderr to trial directory for debugging
        train_log = trial_dir / 'training.log'
        train_err = trial_dir / 'training_error.log'
        
        # Set environment variables for hyperparameters
        env = os.environ.copy()
        env['OPTUNA_MCTS_SIMULATIONS'] = str(mcts_simulations)
        env['OPTUNA_PUCT_C'] = str(puct_c)
        env['OPTUNA_TEMPERATURE'] = str(temperature)
        env['OPTUNA_TEMPERATURE_DECAY'] = str(temperature_decay)
        env['OPTUNA_DIRICHLET_EPSILON'] = str(dirichlet_epsilon)
        env['OPTUNA_DIRICHLET_ALPHA'] = str(dirichlet_alpha)
        env['OPTUNA_LEARNING_RATE'] = str(learning_rate)
        env['OPTUNA_LEARNING_RATE_DECAY'] = str(temperature_decay)
        env['OPTUNA_TRAINING_EPOCHS'] = str(training_epochs)
        env['OPTUNA_REPLAY_BUFFER_SIZE'] = str(replay_buffer_size)
        env['OPTUNA_CRITICAL_THRESHOLD'] = str(critical_threshold)
        env['OPTUNA_POLICY_TARGET_METHOD'] = policy_target_method
        env['OPTUNA_SELECTION_BIAS_WEIGHT'] = str(selection_bias_weight)
        env['OPTUNA_USE_DEPTH_BONUS'] = str(use_depth_bonus)
        env['OPTUNA_DEPTH_BONUS'] = str(depth_bonus)
        env['OPTUNA_USE_VIRTUAL_LOSS'] = str(use_virtual_loss)
        env['OPTUNA_VIRTUAL_LOSS_WEIGHT'] = str(virtual_loss_weight)
        env['OPTUNA_EPISODES_PER_ITERATION'] = str(episodes_per_iteration)
        env['OPTUNA_NUM_CYCLES'] = str(num_cycles)
        env['OPTUNA_MAX_TRAINING_ITERATIONS'] = str(max_training_iterations)
        env['OPTUNA_PARALLEL_WORKERS'] = str(parallel_workers)
        env['OPTUNA_POLICY_WEIGHT'] = str(policy_weight)
        env['OPTUNA_VALUE_WEIGHT'] = str(value_weight)
        env['OPTUNA_PENALTY_FOR_FAILURE'] = str(penalty_for_failure)
        
        with open(train_log, 'w') as out_f, open(train_err, 'w') as err_f:
            result = subprocess.run(
                [sys.executable, 'scripts/train_alphazero_v2.py'],
                cwd='/workspace',
                stdout=out_f,
                stderr=err_f,
                text=True,
                timeout=999999,  # 1 hour max per trial
                env=env  # Pass environment variables
            )
        
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            # Print last 50 lines of error log
            with open(train_err, 'r') as f:
                error_lines = f.readlines()
                print("Last 50 lines of training error:")
                print(''.join(error_lines[-50:]))
            return float('inf')
        
        # Run evaluation
        eval_log = trial_dir / 'evaluation.log'
        eval_err = trial_dir / 'evaluation_error.log'
        
        with open(eval_log, 'w') as out_f, open(eval_err, 'w') as err_f:
            eval_result = subprocess.run(
                [sys.executable, 'scripts/evaluate_agent.py'],
                cwd='/workspace',
                stdout=out_f,
                stderr=err_f,
                text=True,
                timeout=600  # 10 min max for eval
            )
        
        if eval_result.returncode != 0:
            print(f"Evaluation failed with return code {eval_result.returncode}")
            with open(eval_err, 'r') as f:
                error_lines = f.readlines()
                print("Last 50 lines of evaluation error:")
                print(''.join(error_lines[-50:]))
            return float('inf')
        
        # Parse evaluation results from output
        # Look for average survival steps in the output
        with open(eval_log, 'r') as f:
            output = f.read()
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
            # Fallback: check for any numeric value that could be avg_steps
            print("Could not parse 'Average steps' from evaluation output")
            print(f"Evaluation output saved to: {eval_log}")
            print(f"Evaluation output preview (first 1000 chars):")
            print(output[:1000])
            return float('inf')  # Don't accept trials without valid evaluation
        
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
    
    # Create study with persistent storage
    storage_path = 'sqlite:///optimization_results/optuna_study.db'
    study = optuna.create_study(
        direction='minimize',  # Minimize negative avg steps = maximize avg steps
        study_name='alphazero_optimization',
        storage=storage_path,  # Persistent SQLite storage - can resume later
        load_if_exists=True,  # Resume if study already exists
        sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)  # Prune bad trials early
    )
    
    print(f"\nStudy storage: {storage_path}")
    print(f"Existing trials: {len(study.trials)}")
    
    # Run optimization
    n_trials = 42  # Number of hyperparameter combinations to try (all run in parallel)
    
    print(f"\nRunning {n_trials} optimization trials...")
    print("This will take a while. Each trial trains for 15 iterations.\n")
    print("💡 Tip: You can interrupt (Ctrl+C) and resume later - progress is saved to SQLite DB\n")
    
    # Run 14 trials in parallel with sequential episodes in each trial
    # Each trial is single-threaded MCTS
    study.optimize(objective, n_trials=n_trials, n_jobs=14)
    
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
