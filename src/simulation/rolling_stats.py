import numpy as np


class RollingStats:

    def __init__(self, window_size: int = 100):
        self._window_size: int = window_size
        self._buffer: np.ndarray = np.zeros(self._window_size, dtype=np.float64)
        self._index: int = 0
        self.count: int = 0
        self.sum: float = 0.0

    @property
    def is_full(self) -> bool:
        return self.count >= self._window_size
    
    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def add(self, value: float):
        if self.is_full:
            old_value = self._buffer[self._index]
            self.sum -= old_value

        self._buffer[self._index] = value
        self.sum += value
        self._index = (self._index + 1) % self._window_size
        self.count = min(self.count + 1, self._window_size)

    def clear(self):
        self._buffer.fill(0.0)
        self._index = 0
        self.count = 0
        self.sum = 0.0

    def get_values(self) -> np.ndarray:
        if not self.is_full:
            return self._buffer[:self.count].copy()
        return np.concatenate([
            self._buffer[self._index:], # return in chronological order
            self._buffer[:self._index]
        ])
    

class RollingCalculator:

    def __init__(self, pressure_window: int = 100):
        self._impulse_buffer = RollingStats(pressure_window) # to compute pressure
        self._dt_buffer = RollingStats(pressure_window)

    def reset(self):
        self._impulse_buffer.clear()
        self._dt_buffer.clear()

    def record_step(self, wall_impulses: np.ndarray, dt: float):
        self._impulse_buffer.add(np.sum(wall_impulses))
        self._dt_buffer.add(dt)

    def compute_pressure(self, area: float) -> float:
        total_dt = self._dt_buffer.sum
        if total_dt <= 0:
            return 0.0
        total_impulse = self._impulse_buffer.sum
        return total_impulse / (total_dt * area)

    