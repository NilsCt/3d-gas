import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS
from src.simulation import Simulation
from src.visualization.renderer import Renderer, RendererConfig
from src.simulation.physics import Physics

from typing import override

class CarnotCycleScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count = 500
        self.particle_type = PARTICLE_PRESETS["H2"]
        self.T_cold = 300
        self.T_hot = 600
        self.l_max = 10e-9
        self.l_min = self.l_max * np.pow(0.125, 1/3)
        self.seconds_per_step = 5

    @property
    @override
    def name(self) -> str:
        return "carnot_cycle"

    @override
    def setup_system(self):
        config = Config(lx=self.l_max, ly=self.l_max, lz=self.l_max)
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type,
            count=self.particle_count,
            temperature=self.T_cold,
        )

        renderer_config = RendererConfig(color_mode="by_relative_energy")
        renderer = Renderer(simulation=sim, config=renderer_config)
        renderer.color_picker.set_range(
            min=Physics.temperature_to_mean_energy(self.T_cold + 50), 
            max=Physics.temperature_to_mean_energy(self.T_hot - 50), 
            variation_strength=1
        )
        return sim, renderer

    @override
    def run(self):
        simulation = self.simulation
        simulation.transformations.add_waiting_action(self.time_ratio * 2, 
            lambda: simulation.transformations.carnot_cycle(
                container=simulation.container,
                T_hot=self.T_hot,
                T_cold=self.T_cold,
                V_max=simulation.container.volume,
                V_min=self.l_min**3,
                step_duration=self.time_ratio * self.seconds_per_step,
                prep_duration=self.time_ratio,
            )             
        )

if __name__ == "__main__":
    scenario = CarnotCycleScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()