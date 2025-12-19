#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for AlphaZero agent - Single Cycle Mode.
Runs one full cycle (903 episodes) per trial to optimize key hyperparameters.
Focuses on: puct_c, dirichlet noise, and heuristic value horizon.
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
    Returns: negative average survival rate (we want to maximize).
    
    Runs one full training cycle (903 episodes) with trial parameters.
    """
    
    # Suggest hyperparameters - focusing on key exploration/exploitation tradeoffs
    puct_c = trial.suggest_float('puct_c', 1.0, 5.0)
    dirichlet_epsilon = trial.suggest_float('dirichlet_epsilon', 0.0, 0.5)
    dirichlet_alpha = trial.suggest_float('dirichlet_alpha', 0.1, 1.0)
    heuristic_value_horizon = trial.suggest_int('heuristic_value_horizon', 5, 30)
    
    # Fixed training settings for consistency - FULL CYCLE
    episodes_per_iteration = 903  # Full training set
    training_epochs = 10  # Full training
    replay_buffer_size = 1806  # 2x episodes
    parallel_workers = 15  # Use 15 workers
    num_cycles = 1  # Just one cycle per trial
    
    params_to_update = {
        'puct_c': puct_c,
        'dirichlet_epsilon': dirichlet_epsilon,
        'dirichlet_alpha': dirichlet_alpha,
        'heuristic_value_horizon': heuristic_value_horizon,
    }
    
    # Create trial directory
    trial_dir = Path(f'optuna_trials_cycle_v2/trial_{trial.number}')
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial parameters for reference
    config_path = trial_dir / 'trial_params.json'
    with open(config_path, 'w') as f:
        json.dump(params_to_update, f, indent=2)
    
    # Update config in memory (won't affect subprocess but good for tracking)
    config_backup = config.AGENT_CONFIG.copy()
    
    try:
        config.AGENT_CONFIG.update(params_to_update)
        
        # Run training script
        print(f"\n{'='*80}")
        print(f"TRIAL {trial.number}: Training one cycle (903 episodes) with:")
        for key, val in params_to_update.items():
            print(f"  {key}: {val}")
        print(f"{'='*80}\n")
        
        # Write stdout/stderr to trial directory for debugging
        train_log = trial_dir / 'training.log'
        train_err = trial_dir / 'training_error.log'
        
        # Set environment variables for hyperparameters
        env = os.environ.copy()
        env['OPTUNA_PUCT_C'] = str(puct_c)
        env['OPTUNA_DIRICHLET_EPSILON'] = str(dirichlet_epsilon)
        env['OPTUNA_DIRICHLET_ALPHA'] = str(dirichlet_alpha)
        env['OPTUNA_HEURISTIC_VALUE_HORIZON'] = str(heuristic_value_horizon)
        env['OPTUNA_EPISODES_PER_ITERATION'] = str(episodes_per_iteration)
        env['OPTUNA_NUM_CYCLES'] = str(num_cycles)
        env['OPTUNA_TRAINING_EPOCHS'] = str(training_epochs)
        env['OPTUNA_REPLAY_BUFFER_SIZE'] = str(replay_buffer_size)
        env['OPTUNA_PARALLEL_WORKERS'] = str(parallel_workers)
        env['CHECKPOINT_DIR'] = f'checkpoints_heuristic_optimizing/trial_{trial.number}'
        
        with open(train_log, 'w') as out_f, open(train_err, 'w') as err_f:
            result = subprocess.run(
                [sys.executable, 'scripts/train_alphazero_v2.py', '--method', 'heuristic'],
                cwd='/workspace',
                stdout=out_f,
                stderr=err_f,
                text=True,
                timeout=28800000,  # 8 hours max per trial (6h collection + 2h training)
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
        
        # Find the checkpoint for this specific trial
        checkpoint_dir = Path(f'checkpoints_heuristic_optimizing/trial_{trial.number}')
        checkpoints = list(checkpoint_dir.glob('alphazero_v2_iter*.pt'))
        if not checkpoints:
            print(f"ERROR: No checkpoint found in {checkpoint_dir} after training!")
            return float('inf')
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        # Set model path for evaluation
        eval_env = env.copy()
        eval_env['MODEL_PATH'] = str(latest_checkpoint.absolute())
        print(f"Using checkpoint for evaluation: {latest_checkpoint}")
        
        with open(eval_log, 'w') as out_f, open(eval_err, 'w') as err_f:
            eval_result = subprocess.run(
                [sys.executable, 'scripts/evaluate_agent.py'],
                cwd='/workspace',
                stdout=out_f,
                stderr=err_f,
                text=True,
                timeout=18000,  # 30 min max for eval
                env=eval_env
            )
        
        if eval_result.returncode != 0:
            print(f"Evaluation failed with return code {eval_result.returncode}")
            with open(eval_err, 'r') as f:
                error_lines = f.readlines()
                print("Last 50 lines of evaluation error:")
                print(''.join(error_lines[-50:]))
            return float('inf')
        
        # Parse evaluation results from output
        # Look for survival rate (% episodes that survived)
        with open(eval_log, 'r') as f:
            output = f.read()
        
        survival_rate = 0.0
        avg_steps = 0.0
        avg_reward = 0.0
        
        for line in output.split('\n'):
            # Parse survival rate
            if 'survival rate' in line.lower() or '% survived' in line.lower():
                try:
                    import re
                    # Look for percentage
                    numbers = re.findall(r'(\d+\.?\d*)%', line)
                    if numbers:
                        survival_rate = float(numbers[0])
                except:
                    pass
            
            # Parse average steps as secondary metric
            if 'Average steps' in line or 'avg steps' in line.lower():
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        avg_steps = float(numbers[0])
                except:
                    pass
            
            # Parse average reward as tertiary metric
            if 'Average reward' in line or 'avg reward' in line.lower():
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        avg_reward = float(numbers[0])
                except:
                    pass
        
        if avg_steps == 0.0:
            # Fallback: check for any numeric value that could be metrics
            print("Could not parse average steps from evaluation output")
            print(f"Evaluation output saved to: {eval_log}")
            print(f"Evaluation output preview (first 2000 chars):")
            print(output[:2000])
            return float('inf')  # Don't accept trials without valid evaluation
        
        # Primary metric: avg steps (higher is better)
        score = avg_steps
        
        print(f"\nTrial {trial.number} completed:")
        print(f"  Survival rate: {survival_rate:.2f}%")
        print(f"  Avg steps: {avg_steps:.1f}")
        print(f"  Avg reward: {avg_reward:.2f}")
        print(f"  Score (avg steps): {score:.1f}\n")
        
        # Save metrics to trial directory
        metrics = {
            'survival_rate': survival_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'score': score
        }
        metrics_path = trial_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Return negative (Optuna minimizes, we want to maximize)
        return -score
        
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
    print("OPTUNA HYPERPARAMETER OPTIMIZATION - SINGLE CYCLE MODE")
    print("=" * 80)
    print("\nOptimizing: puct_c, dirichlet_epsilon, dirichlet_alpha, heuristic_value_horizon")
    print("Training: 1 full cycle (903 episodes) per trial")
    print("Evaluation: 101 test scenarios per trial")
    print("=" * 80)
    
    # Create study with persistent storage
    storage_path = 'sqlite:///optimization_results/optuna_cycle_study_v2.db'
    study = optuna.create_study(
        direction='minimize',  # Minimize negative avg steps = maximize avg steps
        study_name='alphazero_cycle_optimization_v2',
        storage=storage_path,  # Persistent SQLite storage - can resume later
        load_if_exists=True,  # Resume if study already exists
        sampler=optuna.samplers.TPESampler(),  # Tree-structured Parzen Estimator (no fixed seed for exploration)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)  # Prune bad trials early
    )
    
    print(f"\nStudy storage: {storage_path}")
    print(f"Existing trials: {len(study.trials)}")
    
    # Run optimization
    n_trials = 10  # Run 10 more trials
    
    print(f"\nRunning {n_trials} optimization trials (FULL MODE: 903 episodes/trial)...")
    print("Each trial trains for 903 episodes with 15 parallel workers.")
    print("Estimated time: ~3-4 hours per trial.\n")
    print("💡 Tip: You can interrupt (Ctrl+C) and resume later - progress is saved to SQLite DB\n")
    
    # Run trials sequentially (each trial already uses 14 parallel workers)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value (negative avg steps): {study.best_trial.value:.1f}")
    print(f"Best avg steps: {-study.best_trial.value:.1f}")
    
    # Load metrics from best trial
    best_trial_dir = Path(f'optuna_trials_cycle/trial_{study.best_trial.number}')
    metrics_path = best_trial_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print(f"\nBest trial metrics:")
        print(f"  Survival rate: {metrics['survival_rate']:.2f}%")
        print(f"  Avg steps: {metrics['avg_steps']:.1f}")
        print(f"  Avg reward: {metrics['avg_reward']:.2f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = Path('optimization_results')
    results_dir.mkdir(exist_ok=True)
    
    # Save best parameters to JSON
    best_params_path = results_dir / 'best_hyperparameters_cycle.json'
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"\n✅ Best parameters saved to: {best_params_path}")
    
    # Save optimization history
    history_path = results_dir / 'optimization_history_cycle.csv'
    df = study.trials_dataframe()
    df.to_csv(history_path, index=False)
    print(f"✅ Optimization history saved to: {history_path}")
    
    # Generate importance plot (if multiple trials)
    if len(study.trials) > 1:
        try:
            import plotly
            
            # Parameter importance
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_html(str(results_dir / 'param_importance_cycle.html'))
            
            # Optimization history
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.write_html(str(results_dir / 'optimization_history_cycle.html'))
            
            # Parallel coordinate plot
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
            fig_parallel.write_html(str(results_dir / 'parallel_coordinate_cycle.html'))
            
            # Contour plot for parameter interactions
            fig_contour = optuna.visualization.plot_contour(study, params=['puct_c', 'dirichlet_epsilon'])
            fig_contour.write_html(str(results_dir / 'contour_puct_dirichlet_cycle.html'))
            
            print(f"✅ Visualization plots saved to: {results_dir}/")
        except ImportError:
            print("\n⚠️  Install plotly for visualizations: pip install plotly")
    
    print("\n" + "=" * 80)
    print("Optimization complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
