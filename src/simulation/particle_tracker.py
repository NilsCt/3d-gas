import numpy as np
from collections import deque
from typing import Dict, Set

class ParticleTracker:
      
    def __init__(self, max_points: int = 100):
        self.max_points = max_points
        self.trajectories: Dict[int, deque] = {} # int for particle index

    def add_particle(self, idx: int, position: np.ndarray):
        if idx not in self.trajectories:
            self.trajectories[idx] = deque(maxlen=self.max_points)
            self.trajectories[idx].append(position.copy())

    def remove_particle(self, idx: int):
        self.trajectories.pop(idx, None)

    def toggle(self, idx: int, position: np.ndarray):
        if idx in self.trajectories:
            self.remove_particle(idx)
        else:
            self.add_particle(idx, position)

    def record(self, positions: np.ndarray):
        for idx, trajectory in self.trajectories.items():
            trajectory.append(positions[idx].copy())

    def record_change(self, positions: np.ndarray, collided: Set[int]):
        for idx in self.trajectories:
            if idx in collided:
                self.trajectories[idx].append(positions[idx].copy())