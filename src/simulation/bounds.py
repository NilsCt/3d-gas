from dataclasses import dataclass
import numpy as np

@dataclass
class Bounds:
    """
    Represents a 3D rectangular region in space

    shrink(amount: float) -> Bounds: create a new bounds object shrunk by the given amount in all directions
        example to get the possible positions of the center of a particle with radius r
    is_valid -> bool: returns True if the bounds are valid (xmin < xmax, ymin < ymax, zmin < zmax)
    uniform_positions(rng: np.random.Generator, count: int | None = None) -> np.ndarray
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def shrink(self, amount: float) -> 'Bounds':
        return Bounds(
            xmin=self.xmin + amount,
            xmax=self.xmax - amount,
            ymin=self.ymin + amount,
            ymax=self.ymax - amount,
            zmin=self.zmin + amount,
            zmax=self.zmax - amount
        )
    
    @property
    def is_valid(self) -> bool:
        return self.xmin < self.xmax and self.ymin < self.ymax and self.zmin < self.zmax
    
    def uniform_positions(self, rng: np.random.Generator, count: int | None = None) -> np.ndarray:
        low = np.array([self.xmin, self.ymin, self.zmin], dtype=np.float64)
        high = np.array([self.xmax, self.ymax, self.zmax], dtype=np.float64)

        if count is None:
            return rng.uniform(low, high)
        return rng.uniform(low, high, size=(count, 3))

    def is_in_mask(self, positions: np.ndarray) -> np.ndarray:
        return (
            (positions[:, 0] >= self.xmin) & (positions[:, 0] <= self.xmax) &
            (positions[:, 1] >= self.ymin) & (positions[:, 1] <= self.ymax) &
            (positions[:, 2] >= self.zmin) & (positions[:, 2] <= self.zmax)
        )
        