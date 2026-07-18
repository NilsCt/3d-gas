from dataclasses import dataclass
from collections.abc import Callable
import numpy as np

from src.simulation.gas import Gas
from src.simulation.container import Container

@dataclass
class Thermostat:
    target_temperature: float
    tau: float = 1e-12
    active: bool = True
    duration: float | None = None # None = infinite duration
    elapsed_time: float = 0.0
    # bounds TODO thermostat only in specific bounds


class Deformation:

    def __init__(
        self, 
        initial_dimensions: np.ndarray, 
        final_dimensions: np.ndarray,  
        duration: float, 
        scale_particle_positions: bool = False, 
        piston_give_velocity: bool = False,
        clamp_positions: bool = False,
        progress: Callable[[float], float] | None = None
    ):
        self.initial_dimensions: np.ndarray = initial_dimensions
        self.final_dimensions: np.ndarray = final_dimensions
        self.elapsed_time: float = 0.0
        self.duration: float = duration
        self.scale_particle_positions: bool = scale_particle_positions
        self.piston_give_velocity: bool = piston_give_velocity
        self.clamp_positions: bool = clamp_positions
        if progress is None:
            self.progress: Callable[[float], float] = lambda t: t / self.duration # output in [0,1] to decide the velocity of the deformation
        else:
            self.progress: Callable[[float], float] = progress

    def get_current_dimensions(self) -> np.ndarray:
        current_time = self.elapsed_time
        if self.elapsed_time > self.duration:
            return self.final_dimensions
        t = self.progress(current_time)
        return (1 - t) * self.initial_dimensions + t * self.final_dimensions 
        
    def get_current_velocity(self) -> np.ndarray:
        current_time = self.elapsed_time
        if current_time < self.elapsed_time or current_time > self.elapsed_time + self.duration:
            return np.zeros(3, dtype=np.float64)
        else:
            t = self.progress(current_time)
            dt = 1e-16
            dimensions_before = (1 - (t - dt)) * self.initial_dimensions + (t - dt) * self.final_dimensions
            dimensions_after = (1 - (t + dt)) * self.initial_dimensions + (t + dt) * self.final_dimensions
            return (dimensions_after - dimensions_before) / (2 * dt)


class Transformations:

    def __init__(self):
        self.thermostats: list[Thermostat] = [] 
        self.deformation: Deformation | None = None # one at a time

    def add_thermostat(
        self,
        target_temperature: float,
        tau: float = 1e-12, 
        active: bool = True, 
        duration: float | None = None
    ):
        self.thermostats.append(Thermostat(target_temperature, tau, active, duration))

    def remove_thermostat(self, index: int):
        if 0 <= index < len(self.thermostats):
            del self.thermostats[index]

    def toggle_thermostat(self, index: int):
        if 0 <= index < len(self.thermostats):
            self.thermostats[index].active = not self.thermostats[index].active

    def _change_velocities(self, particles: Gas, factor: float):
        particles.velocities = particles.center_of_mass_velocity + (particles.velocities - particles.center_of_mass_velocity) * factor

    def apply_instantaneous_thermostat(self, particles: Gas, target_temperature: float):
        current_temperature = particles.temperature
        factor = np.sqrt(target_temperature / current_temperature)
        self._change_velocities(particles, factor)

    def _apply_thermostats(self, particles: Gas, dt: float):
        thermostats_to_remove = []
        for i, thermostat in enumerate(self.thermostats):
            thermostat.elapsed_time += dt
            if thermostat.duration is not None and thermostat.elapsed_time > thermostat.duration:
                thermostats_to_remove.append(i)
                continue
            if thermostat.active:
                current_temperature = particles.temperature
                target_temperature = thermostat.target_temperature
                factor = np.sqrt(1 + (target_temperature / current_temperature - 1) * dt / thermostat.tau) # Berendsen thermostat scaling factor
                self._change_velocities(particles, factor)
        for i in reversed(thermostats_to_remove):
            self.remove_thermostat(i)

    
    def set_deformation(
        self,
        initial_dimensions: np.ndarray, 
        final_dimensions: np.ndarray, 
        duration: float, 
        scale_particle_positions: bool = False, 
        piston_give_velocity: bool = False,
        clamp_positions: bool = False,
        progress: Callable[[float], float] | None = None
    ):
        self.deformation = Deformation(
            initial_dimensions, 
            final_dimensions,
            duration, 
            scale_particle_positions, 
            piston_give_velocity,
            clamp_positions, 
            progress
        )

    def stop_deformation(self):
        self.deformation = None

    def _apply_deformation(self, container: Container, particles: Gas, dt: float) -> bool:
        if self.deformation is None:
            return False
        self.deformation.elapsed_time += dt
        if self.deformation.elapsed_time > self.deformation.duration:
            self.stop_deformation()
            return False
        new_dimensions = self.deformation.get_current_dimensions()

        if self.deformation.scale_particle_positions: # scale particles so that they don't touch the walls
            scale_factors = new_dimensions / container.dimensions
            center = container.center
            particles.positions = center + (particles.positions - center) * scale_factors

        if self.deformation.clamp_positions: # clamp particles inside the container
            particles.positions = container.clamp_positions(particles.positions, particles.radii)

        container.resize_all(new_dimensions)
        return True # grid needs to be rebuilt

    def apply(self, container: Container, particles: Gas, dt: float):
        self._apply_thermostats(particles, dt)
        return self._apply_deformation(container, particles, dt)

    def get_piston_velocity(self) -> np.ndarray:
        if self.deformation is None:
            return np.zeros(3, dtype=np.float64)
        return self.deformation.get_current_velocity()
    
