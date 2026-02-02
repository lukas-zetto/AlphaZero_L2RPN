# Training Progress Plotting

## Overview
Each checkpoint now saves:
- **Steps**: Total environment interactions (actual actions taken)
- **Performance metrics**: Average reward, success rate, episode length (last 100 episodes)

## Usage

### Plot training progress:
```bash
# Display interactive plot
python scripts/plot_training_progress.py checkpoints_heuristic_full_run/

# Save to file
python scripts/plot_training_progress.py checkpoints_heuristic_full_run/ --output training_progress.png
```

### What gets plotted:
1. **Average Reward vs Steps** - Shows learning progress
2. **Success Rate vs Steps** - Episode completion rate (0-100%)
3. **Average Episode Length vs Steps** - How long episodes run
4. **Steps vs Episodes** - Training progress overview

## Checkpoint Data Structure

Each checkpoint contains:
```python
{
    'total_steps': 45234,  # Environment steps at this checkpoint
    'episodes_collected': 903,
    'training_iteration': 0,
    'performance': {
        'avg_reward_recent': 145.2,  # Last 100 episodes
        'success_rate_recent': 0.85,  # 85% success rate
        'avg_episode_length_recent': 50.1,
        'all_episode_rewards': [...],  # Full history
        'all_episode_success': [...],
        'all_episode_lengths': [...],
    }
}
```

## Example Output

```
Found 50 checkpoints

============================================================
TRAINING SUMMARY
============================================================
Checkpoints: 50
Total episodes: 45,150
Total steps: 2,234,567
Final avg reward: 156.34
Best avg reward: 178.23
Final success rate: 87.5%
Best success rate: 91.2%
Avg steps per episode: 49.5
```

## Notes

- **Steps** = actual environment interactions (NOT MCTS simulations)
- Performance metrics use a rolling window of last 100 episodes
- All episode data is saved for later analysis
- Compatible with parallel and sequential training modes
