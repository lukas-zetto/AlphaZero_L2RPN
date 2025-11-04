#!/usr/bin/env python3
"""Episode line load logging utilities.

Records per-episode summary of line loading extremes to aid analysis of
catalog-based action impact. Designed to be lightweight and optional.

Outputs a CSV with columns:
    chronic,episode,steps,peak_rho,steps_over_1,first_over_1_step,top_k_ids,top_k_rhos

Where:
    - top_k_ids: semicolon-separated line indices of highest per-episode max rho
    - top_k_rhos: semicolon-separated rho values aligned with top_k_ids

Configuration keys expected (LINE_LOGGING_CONFIG):
    enabled (bool)
    top_k (int)
    episode_csv (str)
    ensure_dir (bool)
    print_summary (bool)
"""
from __future__ import annotations
import os
import csv
import numpy as np
from typing import Optional, List, Dict, Any

class EpisodeLineLoadLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        self.top_k = int(self.config.get('top_k', 5))
        self.episode_csv = self.config.get('episode_csv', 'logs/episode_line_summary.csv')
        self.ensure_dir = self.config.get('ensure_dir', True)
        self.print_summary = self.config.get('print_summary', True)

        self._current_chronic: Optional[int] = None
        self._current_episode: Optional[int] = None
        self._step: int = 0
        self._max_rho_per_line: Optional[np.ndarray] = None
        self._line_over_1_counts: Optional[np.ndarray] = None
        self._steps_over_1: int = 0
        self._first_over_1_step: Optional[int] = None

        if self.enabled and self.ensure_dir:
            os.makedirs(os.path.dirname(self.episode_csv), exist_ok=True)
            if not os.path.exists(self.episode_csv):
                with open(self.episode_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'chronic','episode','steps','peak_rho','steps_over_1',
                        'first_over_1_step','top_k_ids','top_k_rhos'
                    ])

    def start_episode(self, chronic_id: int, episode_id: int):
        if not self.enabled:
            return
        self._current_chronic = chronic_id
        self._current_episode = episode_id
        self._step = 0
        self._max_rho_per_line = None
        self._line_over_1_counts = None
        self._steps_over_1 = 0
        self._first_over_1_step = None

    def update(self, observation):
        if not self.enabled:
            return
        if observation is None:
            return
        rho = getattr(observation, 'rho', None)
        if rho is None:
            return
        rho = np.asarray(rho, dtype=float)
        if self._max_rho_per_line is None:
            self._max_rho_per_line = rho.copy()
            self._line_over_1_counts = (rho > 1.0).astype(int)
        else:
            self._max_rho_per_line = np.maximum(self._max_rho_per_line, rho)
            self._line_over_1_counts += (rho > 1.0).astype(int)
        if rho.max() > 1.0:
            self._steps_over_1 += 1
            if self._first_over_1_step is None:
                self._first_over_1_step = self._step
        self._step += 1

    def end_episode(self):
        if not self.enabled:
            return None
        if self._max_rho_per_line is None:
            return None
        peak_rho = float(self._max_rho_per_line.max())
        k = min(self.top_k, self._max_rho_per_line.size)
        top_indices = np.argsort(self._max_rho_per_line)[-k:][::-1]
        top_rhos = self._max_rho_per_line[top_indices]
        row = [
            self._current_chronic,
            self._current_episode,
            self._step,
            f"{peak_rho:.5f}",
            self._steps_over_1,
            self._first_over_1_step if self._first_over_1_step is not None else '',
            ';'.join(map(str, top_indices.tolist())),
            ';'.join(f"{v:.5f}" for v in top_rhos.tolist())
        ]
        with open(self.episode_csv, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        if self.print_summary:
            print(f"[EpisodeSummary] chronic={self._current_chronic} ep={self._current_episode} steps={self._step} peak={peak_rho:.3f} top{self.top_k}={(list(zip(top_indices.tolist(), [round(x,3) for x in top_rhos.tolist()]))) }")
        summary = {
            'chronic': self._current_chronic,
            'episode': self._current_episode,
            'steps': self._step,
            'peak_rho': peak_rho,
            'top_k_indices': top_indices.tolist(),
            'top_k_rhos': top_rhos.tolist(),
            'steps_over_1': self._steps_over_1,
            'first_over_1_step': self._first_over_1_step,
        }
        # Reset current episode references
        self._current_chronic = None
        self._current_episode = None
        return summary

__all__ = ["EpisodeLineLoadLogger"]
