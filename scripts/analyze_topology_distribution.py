#!/usr/bin/env python3
"""
Analyze topology parameter distribution from training data.

This script loads training data from the specified directory and analyzes
the distribution of topology parameters (which should be -1, 0, or 1).
"""

import numpy as np
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Setup logging to file
log_file = 'topology_analysis.log'
log = open(log_file, 'w')

def print_log(message):
    """Print to both console and log file"""
    print(message)
    log.write(message + '\n')
    log.flush()

def load_all_training_data(data_dir):
    """Load all training data files and extract states."""
    all_states = []
    
    # Get all training data files
    pattern = os.path.join(data_dir, 'training_data_iter_*.npz')
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print_log(f"Found {len(files)} training data files")
    
    for i, file_path in enumerate(files):
        print_log(f"Loading file {i+1}/{len(files)}: {os.path.basename(file_path)}")
        try:
            data = np.load(file_path)
            states = data['states']
            all_states.append(states)
            print_log(f"  Loaded {states.shape[0]} states from iteration {i}")
        except Exception as e:
            print_log(f"  Error loading {file_path}: {e}")
            continue
    
    if not all_states:
        raise ValueError("No training data could be loaded!")
    
    # Concatenate all states
    all_states = np.vstack(all_states)
    print_log(f"\nTotal states loaded: {all_states.shape[0]}")
    print_log(f"State dimension: {all_states.shape[1]}")
    
    return all_states

def analyze_topology_distribution(states):
    """Analyze the distribution of topology parameters."""
    
    # Based on config analysis:
    # - Line loads (rho): 20 features [0:20]  
    # - Line status: 20 features [20:40]
    # - Line cooldowns: 20 features [40:60]
    # - Bus topology: 57 features [60:117]
    
    print_log("=== STATE ENCODING ANALYSIS ===")
    print_log(f"Line loads (rho) [0:20]: min={states[:, 0:20].min():.3f}, max={states[:, 0:20].max():.3f}")
    print_log(f"Line status [20:40]: min={states[:, 20:40].min():.3f}, max={states[:, 20:40].max():.3f}")
    print_log(f"Line cooldowns [40:60]: min={states[:, 40:60].min():.3f}, max={states[:, 40:60].max():.3f}")
    print_log(f"Bus topology [60:117]: min={states[:, 60:117].min():.3f}, max={states[:, 60:117].max():.3f}")
    
    # Extract topology features (bus assignments)
    topology_features = states[:, 60:117]  # 57 topology features
    print_log(f"\n=== TOPOLOGY ANALYSIS ===")
    print_log(f"Topology features shape: {topology_features.shape}")
    
    # Check unique values in topology
    unique_values = np.unique(topology_features)
    print_log(f"Unique values in topology features: {unique_values}")
    
    # Count distribution of each value
    value_counts = Counter(topology_features.flatten())
    print_log(f"\n=== VALUE DISTRIBUTION ===")
    total_values = len(topology_features.flatten())
    
    for value in sorted(value_counts.keys()):
        count = value_counts[value]
        percentage = (count / total_values) * 100
        print_log(f"Value {value:4.1f}: {count:8d} occurrences ({percentage:5.2f}%)")
    
    # Analyze each topology parameter position
    print_log(f"\n=== PER-PARAMETER ANALYSIS ===")
    parameter_stats = []
    
    for param_idx in range(topology_features.shape[1]):
        param_values = topology_features[:, param_idx]
        unique_param_values = np.unique(param_values)
        param_counter = Counter(param_values)
        
        stats = {
            'parameter': param_idx,
            'unique_values': len(unique_param_values),
            'min_value': float(param_values.min()),
            'max_value': float(param_values.max()),
            'most_common_value': param_counter.most_common(1)[0][0],
            'most_common_count': param_counter.most_common(1)[0][1]
        }
        
        # Add counts for -1, 0, 1 specifically
        stats['count_minus_1'] = param_counter.get(-1.0, 0)
        stats['count_zero'] = param_counter.get(0.0, 0)
        stats['count_plus_1'] = param_counter.get(1.0, 0)
        
        parameter_stats.append(stats)
        
        if param_idx < 10:  # Show details for first 10 parameters
            print_log(f"Parameter {param_idx:2d}: values {unique_param_values}, most common: {stats['most_common_value']} ({stats['most_common_count']} times)")
    
    # Create summary DataFrame
    df = pd.DataFrame(parameter_stats)
    
    # Save detailed results
    df.to_csv('topology_parameter_analysis.csv', index=False)
    print_log(f"\nDetailed analysis saved to 'topology_parameter_analysis.csv'")
    
    # Summary statistics
    print_log(f"\n=== SUMMARY STATISTICS ===")
    print_log(f"Parameters with only -1, 0, 1 values: {np.sum((df['min_value'] >= -1) & (df['max_value'] <= 1))}")
    print_log(f"Parameters with -1 values: {np.sum(df['count_minus_1'] > 0)}")
    print_log(f"Parameters with 0 values: {np.sum(df['count_zero'] > 0)}")
    print_log(f"Parameters with +1 values: {np.sum(df['count_plus_1'] > 0)}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Overall value distribution
    plt.subplot(2, 2, 1)
    values = list(value_counts.keys())
    counts = list(value_counts.values())
    plt.bar(values, counts)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Overall Topology Value Distribution')
    plt.yscale('log')
    
    # Plot 2: -1, 0, +1 distribution per parameter
    plt.subplot(2, 2, 2)
    param_indices = df['parameter'].values
    plt.plot(param_indices, df['count_minus_1'], label='-1', alpha=0.7)
    plt.plot(param_indices, df['count_zero'], label='0', alpha=0.7)
    plt.plot(param_indices, df['count_plus_1'], label='+1', alpha=0.7)
    plt.xlabel('Parameter Index')
    plt.ylabel('Count')
    plt.title('Value Distribution per Parameter')
    plt.legend()
    plt.yscale('log')
    
    # Plot 3: Parameter value ranges
    plt.subplot(2, 2, 3)
    plt.plot(param_indices, df['min_value'], label='Min', alpha=0.7)
    plt.plot(param_indices, df['max_value'], label='Max', alpha=0.7)
    plt.axhline(-1, color='red', linestyle='--', alpha=0.5, label='-1')
    plt.axhline(0, color='green', linestyle='--', alpha=0.5, label='0')
    plt.axhline(1, color='blue', linestyle='--', alpha=0.5, label='+1')
    plt.xlabel('Parameter Index')
    plt.ylabel('Value')
    plt.title('Parameter Value Ranges')
    plt.legend()
    
    # Plot 4: Unique values per parameter
    plt.subplot(2, 2, 4)
    plt.bar(param_indices, df['unique_values'])
    plt.xlabel('Parameter Index')
    plt.ylabel('Number of Unique Values')
    plt.title('Unique Values per Parameter')
    
    plt.tight_layout()
    plt.savefig('topology_analysis.png', dpi=150, bbox_inches='tight')
    print_log(f"Visualization saved to 'topology_analysis.png'")
    
    return df

def main():
    # Configuration
    data_dir = 'training_data_fresh_with _modules_1'
    
    print_log("=" * 60)
    print_log("TOPOLOGY PARAMETER ANALYSIS STARTED")
    print_log(f"Timestamp: {pd.Timestamp.now()}")
    print_log("=" * 60)
    
    if not os.path.exists(data_dir):
        print_log(f"Error: Directory {data_dir} does not exist!")
        return
    
    print_log(f"Analyzing training data from: {data_dir}")
    
    try:
        # Load all training data
        states = load_all_training_data(data_dir)
        
        # Analyze topology distribution
        analysis_df = analyze_topology_distribution(states)
        
        print_log("\n=== ANALYSIS COMPLETE ===")
        print_log(f"Results saved to 'topology_parameter_analysis.csv' and 'topology_analysis.png'")
        
    except Exception as e:
        print_log(f"Error during analysis: {e}")
        import traceback
        print_log(traceback.format_exc())
    
    finally:
        # Close log file
        log.close()

if __name__ == "__main__":
    main()