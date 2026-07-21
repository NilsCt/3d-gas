import numpy as np
from typing import Set

from .gas import Gas
from .container import Container
from ..utils import k_b

class Physics:
    """
    Physics engine for simulating gas behavior in a container.
    """

    @staticmethod
    def integrate(gas: Gas, dt: float):
        gas.positions += gas.velocities * dt # Euler explicit

    @staticmethod
    def resolve_wall_collisions(container: Container, gas: Gas, piston_velocity: np.ndarray) -> tuple[Set[int], np.ndarray]: # set of particles that bounced, wall impulses
        """
        Resolve collisions with container walls

        For each wall collision:
        - Reflect velocity component perpendicular to wall
        - Clamp position to be inside container
        - Record impulse for pressure calculation
        """
        positions = gas.positions
        velocities = gas.velocities
        radii = gas.radii
        masses = gas.masses
        wall_impulses = np.zeros(6, dtype=np.float64)
        total_collisions = 0
        bounced = set()

        for axis in range(3):
            min_bound = radii # x = 0
            below_min = positions[:, axis] < min_bound
            if np.any(below_min):
                colliding = np.where(below_min)[0]
                total_collisions += len(colliding)

                impulses = 2 * masses[colliding] * np.abs(velocities[colliding, axis])
                wall_impulses[2 * axis + 1] += np.sum(impulses)  # +1 to get -x -y or -z
                velocities[colliding, axis] = np.abs(velocities[colliding, axis]) # these walls don't move, so no piston velocity is added
                positions[colliding, axis] = min_bound[colliding]
                bounced.update(colliding)

            max_bound = container.dimensions[axis] - radii # x = lx
            above_max = positions[:, axis] > max_bound
            if np.any(above_max):
                colliding = np.where(above_max)[0]
                total_collisions += len(colliding)

                impulses = 2 * masses[colliding] * np.abs(velocities[colliding, axis])
                wall_impulses[2 * axis] += np.sum(impulses)  # +x, +y, +z
                velocities[colliding, axis] = -np.abs(velocities[colliding, axis]) + piston_velocity[axis]
                positions[colliding, axis] = max_bound[colliding]
                bounced.update(colliding)
        return bounced, wall_impulses

    @staticmethod
    def resolve_particle_collisions(gas: Gas, potential_pairs: np.ndarray) -> tuple[Set[int], int]: # set of particles that collided, total number of collisions
        # potential_pairs array of shape (N, 2) containing pairs of particle indices that are potentially colliding
        # typically obtained from the spatial grid
        if len(potential_pairs) == 0:
            return set(), 0
        positions = gas.positions
        velocities = gas.velocities
        radii = gas.radii
        masses = gas.masses
        collided = set()
        collision_count = 0

        for i, j in potential_pairs:
            r_ij = positions[j] - positions[i]
            dist_sq = np.dot(r_ij, r_ij)
            min_dist = radii[i] + radii[j]

            if dist_sq < min_dist**2 and dist_sq > 1e-30:
                dist = np.sqrt(dist_sq)
                n = r_ij / dist # unit vector from i to j
                v_rel = velocities[i] - velocities[j]
                v_rel_n = np.dot(v_rel, n) # relative velocity along n

                if v_rel_n > 0: # only do the collision if the particles are moving towards each other
                    collision_count += 1
                    collided.add(i)
                    collided.add(j)
                    m1, m2 = masses[i], masses[j]
                    mass_sum = m1 + m2
                    velocities[i] -= (2 * m2 / mass_sum) * v_rel_n * n # elastic
                    velocities[j] += (2 * m1 / mass_sum) * v_rel_n * n

                    overlap = min_dist - dist # separate particles if they are too close
                    if overlap > 0:
                        separation = (overlap / 2 + 1e-12) * n
                        positions[i] -= separation
                        positions[j] += separation
        return collided, collision_count

    @staticmethod
    def maxwell_boltzmann_speed_pdf(velocities: np.ndarray, temperature: float, mass: float) -> np.ndarray:
        a = mass / (2 * k_b * temperature)
        coeff = 4 * np.pi * (a / np.pi) ** 1.5
        return coeff * velocities**2 * np.exp(-a * velocities**2)

    @staticmethod
    def temperature_to_mean_speed(temperature: float, mass: float) -> float:
        return np.sqrt(8 * k_b * temperature / (np.pi * mass))

    @staticmethod
    def temperature_to_rms_speed(temperature: float, mass: float) -> float:
        return np.sqrt(3 * k_b * temperature / mass)

    @staticmethod
    def temperature_to_mean_energy(temperature: float) -> float:
        return (3 / 2) * k_b * temperature

    @staticmethod
    def mean_free_path(n_density: float, diameter: float) -> float:
        return 1.0 / (np.sqrt(2) * np.pi * diameter**2 * n_density)