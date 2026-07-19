from dataclasses import dataclass
from collections.abc import Callable
import numpy as np

from src.simulation.gas import Gas
from src.simulation.container import Container
from src.simulation.bounds import Bounds

@dataclass
class Thermostat:
    target_temperature: float
    tau: float = 1e-12
    bounds: Bounds | None = None # thermostat only in specific bounds
    active: bool = True
    duration: float | None = None # None = infinite duration
    action_after: Action | None = None # do something after the thermostat is done
    elapsed_time: float = 0.0


class Deformation:

    def __init__(
        self, 
        initial_dimensions: np.ndarray, 
        final_dimensions: np.ndarray,  
        duration: float, 
        scale_particle_positions: bool = False, 
        piston_give_velocity: bool = False,
        clamp_positions: bool = False,
        progress: Callable[[float], float] | None = None,
        action_after: Action | None = None
    ):
        self.initial_dimensions: np.ndarray = initial_dimensions
        self.final_dimensions: np.ndarray = final_dimensions
        self.elapsed_time: float = 0.0
        self.duration: float = duration
        self.scale_particle_positions: bool = scale_particle_positions
        self.piston_give_velocity: bool = piston_give_velocity
        self.clamp_positions: bool = clamp_positions
        self.action_after: Action | None = action_after
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


type Action = Callable[[], None]

@dataclass
class WaitingAction:
    duration: float
    action: Action
    elapsed_time: float = 0.0


class Transformations:

    def __init__(self):
        self.thermostats: list[Thermostat] = [] 
        self.deformation: Deformation | None = None # one at a time
        self.waiting_actions: list[WaitingAction] = []

    def add_thermostat(
        self,
        target_temperature: float,
        tau: float = 1e-12, 
        bounds: Bounds | None = None,
        active: bool = True, 
        duration: float | None = None,
        action_after: Action | None = None
    ):
        self.thermostats.append(Thermostat(target_temperature, tau, bounds, active, duration, action_after))

    def remove_thermostat(self, index: int):
        if 0 <= index < len(self.thermostats):
            del self.thermostats[index]

    def toggle_thermostat(self, index: int):
        if 0 <= index < len(self.thermostats):
            self.thermostats[index].active = not self.thermostats[index].active

    def _change_velocities(self, particles: Gas, factor: float, mask: np.ndarray | None = None):
        if mask is None:
            com_velocity = particles.center_of_mass_velocity
            particles.velocities = com_velocity + (particles.velocities - com_velocity) * factor
        else:
            com_velocity = particles.center_of_mass_with_mask(mask)
            particles.velocities[mask] = com_velocity + (particles.velocities[mask] - com_velocity) * factor

    def apply_instantaneous_thermostat(self, particles: Gas, target_temperature: float, bounds: Bounds | None = None):
        current_temperature = particles.temperature
        factor = np.sqrt(target_temperature / current_temperature)
        if bounds is None:
            mask = None
        else:
            mask = bounds.is_in_mask(particles.positions)
        self._change_velocities(particles, factor, mask)

    def _apply_thermostats(self, particles: Gas, dt: float):
        thermostats_to_remove = []
        actions_to_execute = []
        for i, thermostat in enumerate(self.thermostats):
            thermostat.elapsed_time += dt
            if thermostat.duration is not None and thermostat.elapsed_time > thermostat.duration:
                thermostats_to_remove.append(i)
                if thermostat.action_after is not None:
                    actions_to_execute.append(thermostat.action_after)
                continue
            if thermostat.active:
                current_temperature = particles.temperature
                target_temperature = thermostat.target_temperature
                factor = np.sqrt(1 + (target_temperature / current_temperature - 1) * dt / thermostat.tau) # Berendsen thermostat scaling factor
                if thermostat.bounds is None:
                    mask = None
                else:
                    mask = thermostat.bounds.is_in_mask(particles.positions)
                self._change_velocities(particles, factor, mask)
        for i in reversed(thermostats_to_remove):
            self.remove_thermostat(i)
        for action in actions_to_execute: # always execute after removing the thermostat, in case the action adds a new thermostat
            action()

    
    def set_deformation( # with no thermostat, corresponds to an adiabatic transformation
        self,
        initial_dimensions: np.ndarray, 
        final_dimensions: np.ndarray, 
        duration: float, 
        scale_particle_positions: bool = False, 
        piston_give_velocity: bool = False,
        clamp_positions: bool = False,
        progress: Callable[[float], float] | None = None,
        action_after: Action | None = None
    ):
        self.deformation = Deformation(
            initial_dimensions, 
            final_dimensions,
            duration, 
            scale_particle_positions, 
            piston_give_velocity,
            clamp_positions, 
            progress,
            action_after
        )

    def set_deformation_with_thermostat( # with thermostat, corresponds to an isothermal transformation (if starts with same temperature)
        self,
        initial_dimensions: np.ndarray,
        final_dimensions: np.ndarray,
        duration: float,
        target_temperature: float,
        tau: float = 1e-12,
        bounds: Bounds | None = None,
        scale_particle_positions: bool = False,
        piston_give_velocity: bool = False,
        clamp_positions: bool = False,
        progress: Callable[[float], float] | None = None,
        action_after: Action | None = None
    ):
        self.add_thermostat(
            target_temperature,
            tau=tau, 
            bounds=bounds, 
            duration=duration, 
            action_after=action_after
        )
        self.deformation = Deformation(
            initial_dimensions, 
            final_dimensions,
            duration, 
            scale_particle_positions, 
            piston_give_velocity,
            clamp_positions, 
            progress,
            None # do not duplicate the action_after
        )

    def stop_deformation(self):
        self.deformation = None

    def _apply_deformation(self, container: Container, particles: Gas, dt: float):
        if self.deformation is None:
            return False
        self.deformation.elapsed_time += dt
        if self.deformation.elapsed_time > self.deformation.duration:
            action = self.deformation.action_after
            self.stop_deformation() 
            if action is not None: # always execute after stopping the deformation, in case the action adds a new deformation
                action()
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
    
    def get_piston_velocity(self) -> np.ndarray:
        if self.deformation is None:
            return np.zeros(3, dtype=np.float64)
        return self.deformation.get_current_velocity()
    

    def add_waiting_action(self, duration: float, action: Action):
        self.waiting_actions.append(WaitingAction(duration, action))

    def _apply_waiting_actions(self, dt: float):
        actions_to_remove = []
        for i, waiting_action in enumerate(self.waiting_actions):
            waiting_action.elapsed_time += dt
            if waiting_action.elapsed_time >= waiting_action.duration:
                waiting_action.action()
                actions_to_remove.append(i)
        for i in actions_to_remove:
            self.waiting_actions.pop(i)


    def apply(self, container: Container, particles: Gas, dt: float):
        self._apply_thermostats(particles, dt)
        self._apply_deformation(container, particles, dt)
        self._apply_waiting_actions(dt)
    
    @staticmethod
    def _dims(V:float, V_base:float, dims_base:np.ndarray) -> np.ndarray:
        scale = (V / V_base) ** (1 / 3)
        return dims_base.copy() * scale

    def stirling_cycle(
        self,
        container: Container, # doesn't need to be a cube
        T_hot: float,
        T_cold: float,
        V_max: float,
        V_min: float,
        step_duration: float,
        prep_duration: float
    ):
        dims_max = Transformations._dims(V_max, container.volume, container.dimensions)
        dims_min = Transformations._dims(V_min, container.volume, container.dimensions)
        tau = step_duration / 5

        def step1_isothermal_compression():
            # V_max -> V_min at T_cold
            self.set_deformation_with_thermostat(
                initial_dimensions=dims_max,
                final_dimensions=dims_min,
                duration=step_duration,
                target_temperature=T_cold,
                tau=tau,
                scale_particle_positions=True,
                action_after=step2_isochoric_heating
            )

        def step2_isochoric_heating():
            # T_cold -> T_hot at V_min
            self.add_thermostat(T_hot, tau=tau, duration=step_duration, action_after=step3_isothermal_expansion)

        def step3_isothermal_expansion():
            # V_min -> V_max at T_hot
            self.set_deformation_with_thermostat(
                initial_dimensions=dims_min,
                final_dimensions=dims_max,
                duration=step_duration,
                target_temperature=T_hot,
                tau=tau,
                scale_particle_positions=True,
                action_after=step4_isochoric_cooling
            )

        def step4_isochoric_cooling():
            # T_hot -> T_cold at V_max
            self.add_thermostat(T_cold, tau=tau, duration=step_duration, action_after=step1_isothermal_compression)

        # Bring system to V_max and T_cold
        prep_tau = prep_duration / 5
        self.set_deformation_with_thermostat(
            initial_dimensions=container.dimensions.copy(),
            final_dimensions=dims_max,
            duration=prep_duration,
            target_temperature=T_cold,
            tau=prep_tau,
            scale_particle_positions=True,
            action_after=step1_isothermal_compression
        )

    # ici je triche, j'utilise gamma 
    # si je voulais faire sans il faut compresser et calculer T en continu pour arreter la compression
    # quand on atteint T_hot, mais on ne sais pas combien de temps cela peut durer
    # et il faut reprogrammer pas mal
    # mais comme pour les collisions on assume déja que les particules sont monoatomiques
    # (pas de rotation), cela est peut être acceptable
    def carnot_cycle(
        self,
        container: Container,
        T_hot: float,
        T_cold: float,
        V_max: float,
        V_min: float,
        step_duration: float,
        prep_duration: float,
        gamma: float = 5 / 3  # monoatomic gas
    ):
        temp_ratio = T_hot / T_cold
        adiabatic_exp = 1 / (gamma - 1)
        V_2 = V_min * (temp_ratio ** adiabatic_exp)
        V_4 = V_max * ((1 / temp_ratio) ** adiabatic_exp)

        dims_1 = Transformations._dims(V_max, container.volume, container.dimensions)  # A T_cold, V_max
        dims_2 = Transformations._dims(V_2, container.volume, container.dimensions)    # B T_cold, V_2
        dims_3 = Transformations._dims(V_min, container.volume, container.dimensions)  # C T_hot, V_min
        dims_4 = Transformations._dims(V_4, container.volume, container.dimensions)    # D T_hot, V_4
        tau = step_duration / 5

        def step1_isothermal_compression():
            # V_max -> V_2 at T_cold
            self.set_deformation_with_thermostat(
                initial_dimensions=dims_1,
                final_dimensions=dims_2,
                duration=step_duration,
                target_temperature=T_cold,
                tau=tau,
                scale_particle_positions=True,
                action_after=step2_adiabatic_compression
            )

        def step2_adiabatic_compression():
            # V_2 -> V_min, T_cold -> T_hot, No thermostat
            self.deformation = Deformation(
                initial_dimensions=dims_2,
                final_dimensions=dims_3,
                duration=step_duration,
                scale_particle_positions=True,
                action_after=step3_isothermal_expansion,
            )

        def step3_isothermal_expansion():
            # V_min -> V_4 at T_hot
            self.set_deformation_with_thermostat(
                initial_dimensions=dims_3,
                final_dimensions=dims_4,
                duration=step_duration,
                target_temperature=T_hot,
                tau=tau,
                scale_particle_positions=True,
                action_after=step4_adiabatic_expansion
            )

        def step4_adiabatic_expansion():
            # V_4 -> V_max, T_hot -> T_cold, No thermostat
            self.deformation = Deformation(
                initial_dimensions=dims_4,
                final_dimensions=dims_1,
                duration=step_duration,
                scale_particle_positions=True,
                action_after=step1_isothermal_compression,
            )

        # Bring system to V_max and T_cold
        prep_tau = prep_duration / 5
        self.set_deformation_with_thermostat(
            initial_dimensions=container.dimensions.copy(),
            final_dimensions=dims_1,
            duration=prep_duration,
            target_temperature=T_cold,
            tau=prep_tau,
            scale_particle_positions=True,
            action_after=step1_isothermal_compression
        )


    # je n'ai pas implémenter les isobares, il faudrait encore tricher ou alors c'est beaucoup plus complexe
    
