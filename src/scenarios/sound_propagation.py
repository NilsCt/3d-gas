import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, LIGHT_BLUE, ParticleType
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CameraConfig, Renderer

from typing import override


class SoundPropagationScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.lx = 40e-9
        self.ly = 4e-9
        self.lz = 4e-9

        base_particle = PARTICLE_PRESETS["N2"]
        self.particle_radius = 4.5e-10 # make particles bigger
        self.particle_type = ParticleType(
            name="N2_large",
            mass=base_particle.mass,
            radius=self.particle_radius,
            color=LIGHT_BLUE,
        )
        self.temperature = 200
        self.particle_count = 800

        self.pulse_amplitude = 10e-9 # how much the wall moves
        self.pulse_duration = 0.2 # real seconds
        self.wait_before_pulse = 3
        self.wait_between_pulses = 10

    @property
    @override
    def name(self) -> str:
        return "sound_propagation"

    @override
    def setup_system(self):
        config = Config(
            lx=self.lx,
            ly=self.ly,
            lz=self.lz,
            check_overlap=False, # important to put it to false to have a high particle density
            pressure_window=200,
        )
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type,
            count=self.particle_count,
            temperature=self.temperature,
        )

        camera_config = CameraConfig(azimuth=180, elevation=45, distance=1)
        renderer_config = RendererConfig(
            camera_config=camera_config,
            #color_mode="by_type"
            color_mode="by_energy"
            #color_mode="by_relative_energy"
        )
        renderer = Renderer(simulation=sim, config=renderer_config)
        energies = sim.gas.kinetic_energies
        renderer.color_picker.set_range(
            min=energies.min(),
            max=energies.max()*1.25,
            variation_strength=1
        )
        return sim, renderer

    @override
    def run(self):
        simulation = self.simulation

        def create_pulse():
            initial_dims = simulation.container.dimensions.copy()
            compressed_dims = initial_dims.copy()
            compressed_dims[0] = initial_dims[0] - self.pulse_amplitude

            def start_extension():
                def restart():
                    simulation.transformations.add_waiting_action(
                        self.wait_between_pulses * self.time_ratio,
                        create_pulse
                    )

                simulation.transformations.set_deformation(
                    initial_dimensions=compressed_dims,
                    final_dimensions=initial_dims,
                    duration=self.pulse_duration * self.time_ratio,
                    piston_give_velocity=True,
                    clamp_positions=False,
                    action_after=restart()
                )

            simulation.transformations.set_deformation(
                initial_dimensions=initial_dims,
                final_dimensions=compressed_dims,
                duration=self.pulse_duration * self.time_ratio,
                piston_give_velocity=True,
                clamp_positions=True,
                action_after=start_extension,
            )

        simulation.transformations.add_waiting_action(
            self.wait_before_pulse * self.time_ratio,
            create_pulse
        )


if __name__ == "__main__":
    scenario = SoundPropagationScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()
