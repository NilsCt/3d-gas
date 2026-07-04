import numpy as np
from typing import List

from ..utils import (
    ParticleType,
    concatenate_vectors,
    k_b
)
from .bounds import Bounds

class Gas:
    """
    Represents a collection of particles
    
    API
    positions
    velocities
    type_indices: indices of the particle types in self.types
    types: list of ParticleType objects, describing physical properties of the particles
    count
    masses
    radii
    colors
    kinetic_energies
    speeds
    momenta
    max_speed
    total_kinetic_energy
    mean_kinetic_energy
    total_momentum
    rms_speed

    clear(): clears all particles
    add_particles(particle_type: ParticleType, count: int, bounds: Bounds, temperature: float, check_overlap: bool = True, max_retries: int = 1000) -> int
        adds particles of the given type, within the given bounds, with velocities corresponding to the given temperature. Returns the number of particles actually added 
        (may be less than count if check_overlap is True and there is not enough space)
    """

    def __init__(self):
        self.positions: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.velocities: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.type_indices: np.ndarray = np.empty(0, dtype=np.int32)
        self.types: List[ParticleType] = []

        self._rng = np.random.default_rng()

    # Properties

    @property
    def count(self) -> int:
        return len(self.positions)
    
    @property
    def masses(self) -> np.ndarray:
        type_masses = np.array([t.mass for t in self.types], dtype=np.float64)
        return type_masses[self.type_indices]
    
    @property
    def radii(self) -> np.ndarray:
        type_radii = np.array([t.radius for t in self.types], dtype=np.float64)
        return type_radii[self.type_indices]
    
    @property
    def colors(self) -> np.ndarray:
        type_colors = np.array([t.color.to_4_array() for t in self.types], dtype=np.float64)
        return type_colors[self.type_indices]

    @property
    def kinetic_energies(self) -> np.ndarray:
        speeds_sq = np.sum(self.velocities**2, axis=1)
        return 0.5 * self.masses * speeds_sq

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)

    @property
    def momenta(self) -> np.ndarray:
        return self.velocities * self.masses[:, np.newaxis]
    
    @property
    def max_speed(self) -> float:
        if self.count == 0:
            return 0.0
        return np.max(self.speeds)
    
    @property
    def max_radius(self) -> float:
        if self.count == 0:
            return 0.0
        return np.max(self.radii)
    
    @property
    def total_kinetic_energy(self) -> float:
        return np.sum(self.kinetic_energies)
    
    @property
    def mean_kinetic_energy(self) -> float:
        if self.count == 0:
            return 0.0
        return np.mean(self.kinetic_energies)
    
    @property
    def total_momentum(self) -> np.ndarray:
        return np.sum(self.momenta, axis=0)
    
    @property
    def rms_speed(self) -> float:
        if self.count == 0:
            return 0.0
        return np.sqrt(np.mean(self.speeds**2))


    # Methods
    def _get_type_index(self, particle_type: ParticleType) -> int:
        # creates a new index for an unregistered particle type 
        for idx, t in enumerate(self.types):
            if t.name == particle_type.name:
                return idx
        idx = len(self.types)
        self.types.append(particle_type)
        return idx
    
    def clear(self):
        self.positions = np.empty((0, 3), dtype=np.float64)
        self.velocities = np.empty((0, 3), dtype=np.float64)
        self.type_indices = np.empty(0, dtype=np.int32)
        self.types = []

    def add_particles(
        self,
        particle_type: ParticleType,
        count: int,
        bounds: Bounds, # positions
        temperature: float,
        check_overlap: bool = True, # positions
        max_retries: int = 100,
    ) -> int: # number effectively added
        type_idx = self._get_type_index(particle_type)

        radius = particle_type.radius

        # Adjust bounds to keep particles inside
        bounds = bounds.shrink(radius)

        if not bounds.is_valid:
            raise ValueError(
                f"Bounds too small for particle radius {radius}. "
                f"Need at least {2*radius}m in each dimension."
            )

        # Generate velocities
        velocities = self._generate_maxwell_boltzmann_velocities(count, temperature, particle_type.mass)

        # Generate positions
        if check_overlap:
            positions = self._generate_non_overlapping_positions(
                count, bounds,
                radius, max_retries
            )
        else:
            positions = bounds.uniform_positions(self._rng, count)   
        actual_count = len(positions)
        if actual_count < count:
            velocities = velocities[:actual_count]

        if actual_count == 0:
            return 0

        # Append to arrays
        self.positions = concatenate_vectors(self.positions, positions)
        self.velocities = concatenate_vectors(self.velocities, velocities)
        self.type_indices = np.concatenate([
            self.type_indices,
            np.full(actual_count, type_idx, dtype=np.int32)
        ])

        return actual_count
    
    def _generate_non_overlapping_positions(
        self,
        count: int,
        bounds: Bounds,
        new_radius: float,
        max_retries: int,
    ) -> np.ndarray:
        """Generate positions that don't overlap with existing particles."""

        new_positions : np.ndarray = np.empty((0, 3), dtype=np.float64)
        for _ in range(count):
            placed = False
            for _ in range(max_retries):
                pos = bounds.uniform_positions(self._rng, count=1)

                # Check against existing particles
                if self.count > 0:
                    distances = np.linalg.norm(self.positions - pos, axis=1)
                    min_distances = self.radii + new_radius
                    if np.any(distances < min_distances):
                        continue

                # Check against already placed new particles
                if new_positions.size > 0:
                    distances = np.linalg.norm(new_positions - pos, axis=1)
                    if np.any(distances < 2 * new_radius):
                        continue

                new_positions = concatenate_vectors(new_positions, pos)
                placed = True
                break

            if not placed:
                break  # Could not place more particles

        return new_positions if new_positions.size > 0 else np.empty((0, 3), dtype=np.float64)


    def _generate_maxwell_boltzmann_velocities(
        self,
        count: int,
        temperature: float,
        mass: float
    ) -> np.ndarray:
        sigma = np.sqrt(k_b * temperature / mass)
        return self._rng.normal(0, sigma, size=(count, 3))

        