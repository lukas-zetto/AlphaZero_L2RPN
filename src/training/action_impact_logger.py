"""Action Impact Logger for tracking before/after grid stability."""

import csv
import numpy as np
from pathlib import Path


class ActionImpactLogger:
    """Logs grid state before and after each action to analyze impact."""
    
    def __init__(self, n_lines=20, csv_path='logs/action_impact_reconnect.csv'):
        self.n_lines = n_lines
        self.csv_path = csv_path
        self.chronic_id = None
        self.timestep = 0
        
        # Create logs directory if needed
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV with header
        self._write_header()
    
    def _write_header(self):
        """Write CSV header."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Build header
            header = [
                'chronic_id', 'timestep', 'action_type', 'action_idx', 
                'substation_id', 'line_reconnected',
                'before_max_rho', 'after_max_rho', 'delta_rho', 'improved',
                'before_overloads', 'after_overloads',
                'before_disconnected', 'after_disconnected'
            ]
            
            # Add individual line rhos
            for i in range(self.n_lines):
                header.append(f'before_rho_{i}')
            for i in range(self.n_lines):
                header.append(f'after_rho_{i}')
            
            writer.writerow(header)
    
    def set_chronic(self, chronic_id):
        """Set the current chronic being processed."""
        self.chronic_id = chronic_id
        self.timestep = 0
    
    def log_action_impact(self, obs_before, obs_after, action, action_type='catalog', 
                         action_idx=None, substation_id=None, line_reconnected=None):
        """
        Log the before/after state for an action.
        
        Parameters:
        -----------
        obs_before : grid2op.Observation
            Observation before action
        obs_after : grid2op.Observation
            Observation after action
        action : grid2op.Action
            The action taken
        action_type : str
            'auto_reconnect', 'catalog', or 'fallback'
        action_idx : int, optional
            Catalog action index
        substation_id : int, optional
            Which substation was modified (for catalog actions)
        line_reconnected : int, optional
            Which line was reconnected (for auto_reconnect)
        """
        self.timestep += 1
        
        # Extract before state
        before_rho = obs_before.rho
        before_max_rho = float(np.max(before_rho))
        before_overloads = int(np.sum(before_rho > 1.0))
        before_disconnected = int(np.sum(~obs_before.line_status))
        
        # Extract after state
        after_rho = obs_after.rho
        after_max_rho = float(np.max(after_rho))
        after_overloads = int(np.sum(after_rho > 1.0))
        after_disconnected = int(np.sum(~obs_after.line_status))
        
        # Calculate improvement
        delta_rho = after_max_rho - before_max_rho
        improved = delta_rho < -0.01  # Reduced max rho by at least 0.01
        
        # Build row
        row = [
            self.chronic_id,
            self.timestep,
            action_type,
            action_idx if action_idx is not None else '',
            substation_id if substation_id is not None else '',
            line_reconnected if line_reconnected is not None else '',
            f'{before_max_rho:.4f}',
            f'{after_max_rho:.4f}',
            f'{delta_rho:.4f}',
            improved,
            before_overloads,
            after_overloads,
            before_disconnected,
            after_disconnected
        ]
        
        # Add all before rho values
        for rho in before_rho:
            row.append(f'{rho:.4f}')
        
        # Add all after rho values
        for rho in after_rho:
            row.append(f'{rho:.4f}')
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
