import numpy as np
from dataclasses import dataclass
from .bounds import Bounds

@dataclass
class Container:
    """
    Represents a 3D rectangular container for particles

    API
    lx
    ly
    lz
    dimensions: array (lx, ly, lz)
    volume
    surface_area
    bounds
    center
    scale_all(factor: float)
    resize_all(new_dimensions: np.ndarray)
    clamp_positions(positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        moves the center of particles back inside the container if they are outside, taking into account their radii (so that the particles don't overlap with the walls)
    """

    lx: float
    ly: float
    lz: float

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.lx, self.ly, self.lz], dtype=np.float64)
    
    @property
    def volume(self) -> float:
        return self.lx * self.ly * self.lz
    
    @property
    def surface_area(self) -> float:
        return 2 * (self.lx * self.ly + self.lx * self.lz + self.ly * self.lz)
    
    @property
    def bounds(self) -> Bounds:
        return Bounds(
            xmin=0.0,
            xmax=self.lx,
            ymin=0.0,
            ymax=self.ly,
            zmin=0.0,
            zmax=self.lz
        )
    
    @property
    def center(self) -> np.ndarray:
        return self.dimensions / 2
    
    def scale_all(self, factor: float):
        self.lx *= factor
        self.ly *= factor
        self.lz *= factor

    def resize_all(self, new_dimensions: np.ndarray):
        self.lx = new_dimensions[0]
        self.ly = new_dimensions[1]
        self.lz = new_dimensions[2]

    def clamp_positions(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        # Used after compression to push particles back inside.
        clamped = positions.copy()
        for axis in range(3):
            min_bound = radii
            max_bound = self.dimensions[axis] - radii
            clamped[:, axis] = np.clip(clamped[:, axis], min_bound, max_bound)
        return clamped