import numpy as np

from ..utils import Config, ParticleType, k_b
from .container import Container
from .gas import Gas
from .physics import Physics
from .spatial_grid import SpatialGrid
from .bounds import Bounds
from .thermodynamics import ThermodynamicsState

class Simulation:
    """
    Core simulation class that manages the gas, container, and physics engine
    """

    def __init__(self, config: Config):
        self.config: Config = config
        self.container: Container = Container(config.lx, config.ly, config.lz)
        self.gas: Gas = Gas()
        self.grid: SpatialGrid = SpatialGrid(self.container, 1) 
        self._current_max_radius: float = 0.0
        self.cfl_factor: float = config.cfl_factor
        # arbitrary cell size for now, will be updated to config.cell_size_factor * largest particle radius

        self.time: float = 0
        self.dt: float = config.initial_dt

        self.thermodynamics_state: ThermodynamicsState = ThermodynamicsState()

    def _rebuild_grid(self):
        if self.gas.count == 0:
            return
        cell_size = self.gas.max_radius * self.config.cell_size_factor
        self.grid.build_grid(cell_size)
        self.grid.update(self.gas.positions)

    def _ensure_grid(self, gas: Gas):
        if self.gas.count == 0:
            return
        max_radius = self.gas.max_radius
        if max_radius > self._current_max_radius * 1.1:
            # if bigger particles are added, we might want to increase the cell size to avoid particles spanning multiple cells
            self._current_max_radius = max_radius
            self._rebuild_grid()

    def _compute_adaptive_dt(self) -> float:
        """
        Compute adaptive time step based on CFL condition
        dt <= cfl_factor * cell_size / v_max
        Ensures particles don't travel more than a fraction of a cell per step
        """
        dt_min = self.config.dt_min
        dt_max = self.config.dt_max
        if self.gas.count == 0:
            return dt_max
        v_max = self.gas.max_speed
        if v_max < 1e-10:  # avoid division by zero
            return dt_max

        cell_size = self.config.cell_size_factor * self.gas.max_radius
        dt = self.config.cfl_factor * cell_size / v_max
        return np.clip(dt, dt_min, dt_max)
    
    def gas_step(self, dt: float):
        """
        Advance simulation by one time step

        Order:
        1. Integrate positions (Euler explicit)
        2. Resolve wall collisions
        3. Update spatial grid
        4. Resolve particle-particle collisions
        """
        self.time += dt
        if self.gas.count == 0:
            return 
        Physics.integrate(self.gas, dt)
        Physics.resolve_wall_collisions(self.container, self.gas)
        self._ensure_grid(self.gas)
        potential_pairs = self.grid.get_potential_collision_pairs(self.gas.positions)
        Physics.resolve_particle_collisions(self.gas, potential_pairs)

    def add_particles(
        self,
        particle_type: ParticleType,
        count: int,
        temperature: float,
        bounds: Bounds | None = None,
    ) -> int:
        """
        If bounds is None, particles are added in the entire container
        Returns the number of particles successfully added
        """
        if bounds is None:
            bounds = self.container.bounds

        added = self.gas.add_particles(
            particle_type=particle_type,
            count=count,
            bounds=bounds,
            temperature=temperature,
            check_overlap=self.config.check_overlap,
            max_retries=self.config.max_overlap_retries,
        )
        self._ensure_grid(self.gas)  # Rebuild grid if needed
        self.grid.update(self.gas.positions)  # Update grid with new particle positions
        return added

    def clear_particles(self):
        self.container: Container = Container(self.config.lx, self.config.ly, self.config.lz)
        self.gas: Gas = Gas()
        self.grid: SpatialGrid = SpatialGrid(self.container, 1) 
        self._current_max_radius: float = 0.0
        self.time: float = 0
        self.dt: float = self.config.initial_dt

    def complete_step(self, max_dt: float | None = None) -> float: 
        # TODO will include measurements and transformations of the environment
        if self.config.dt_mode == "adaptive":
            self.dt = self._compute_adaptive_dt()
        if max_dt is not None:
            self.dt = min(self.dt, max_dt)
        self.gas_step(self.dt)
        self.update_thermodynamics_state()
        return self.dt

    def run(self, duration: float):
        """
        Run the simulation for a given duration
        """
        start_time = self.time
        while self.time - start_time < duration:
            result = self.complete_step()

    def update_thermodynamics_state(self):
        state = self.thermodynamics_state
        state.temperature = self.gas.temperature
        state.pressure = 0 # TODO
        state.volume = self.container.volume
        state.n_particles = self.gas.count
        state.total_kinetic_energy = self.gas.total_kinetic_energy
        state.mean_kinetic_energy = self.gas.mean_kinetic_energy
        state.pv_nkt = state.pressure * state.volume / (state.n_particles * k_b * state.temperature) if state.n_particles > 0 else 0
        state.rms_speed = self.gas.rms_speed
        state.momentum = self.gas.total_momentum
        state.mean_free_path = 0 # TODO
