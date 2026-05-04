#!/bin/bash
# Convenience script to list and optionally submit individual experiment evaluations

echo "=========================================="
echo "INDIVIDUAL EXPERIMENT EVALUATION SCRIPTS"
echo "=========================================="
echo ""

SCRIPTS_DIR="$(dirname "$0")"

# List all available evaluation scripts
echo "Available experiments:"
echo ""
for script in "$SCRIPTS_DIR"/eval_*.sbatch; do
    if [ -f "$script" ]; then
        # Extract experiment name from filename
        experiment=$(basename "$script" .sbatch | sed 's/eval_//')
        
        # Get resource info from the script
        cpus=$(grep "cpus-per-task" "$script" | head -n1 | sed -n 's/.*cpus-per-task=\([0-9]*\).*/\1/p')
        mem=$(grep "mem=" "$script" | head -n1 | sed -n 's/.*mem=\([0-9]*G\).*/\1/p')
        time=$(grep "time=" "$script" | head -n1 | sed -n 's/.*time=\([0-9:]*\).*/\1/p')
        
        printf "  %-30s %2s CPUs, %4s RAM, %8s time\n" "$experiment" "$cpus" "$mem" "$time"
        echo "    sbatch eval_scripts/individual_experiments/$(basename "$script")"
        echo ""
    fi
done

echo "=========================================="
echo ""

# Check if user wants to submit jobs
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage:"
    echo "  $0                     # List available experiments"
    echo "  $0 --submit <exp>      # Submit specific experiment"
    echo "  $0 --submit-all        # Submit all experiments"
    echo ""
    echo "Examples:"
    echo "  $0 --submit horizon_experiments"
    echo "  $0 --submit mcts_experiments"
    echo "  $0 --submit-all"
    echo ""
elif [ "$1" = "--submit" ] && [ ! -z "$2" ]; then
    experiment="$2"
    script_file="$SCRIPTS_DIR/eval_${experiment}.sbatch"
    
    if [ -f "$script_file" ]; then
        echo "Submitting evaluation for: $experiment"
        sbatch "$script_file"
    else
        echo "Error: Experiment '$experiment' not found!"
        echo "Available experiments: $(ls "$SCRIPTS_DIR"/eval_*.sbatch | sed 's/.*eval_//' | sed 's/.sbatch//' | tr '\n' ' ')"
    fi
elif [ "$1" = "--submit-all" ]; then
    echo "Submitting ALL experiment evaluations..."
    echo ""
    
    for script in "$SCRIPTS_DIR"/eval_*.sbatch; do
        if [ -f "$script" ]; then
            experiment=$(basename "$script" .sbatch | sed 's/eval_//')
            echo "Submitting: $experiment"
            sbatch "$script"
        fi
    done
    
    echo ""
    echo "All experiments submitted! Check with: squeue -u \$USER"
else
    echo "Use --help for usage information"
    echo "Use --submit <experiment_name> to submit a specific experiment"
    echo "Use --submit-all to submit all experiments"
fi