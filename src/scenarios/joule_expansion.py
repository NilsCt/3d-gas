import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, LIGHT_BLUE
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CAMERA_FACING_X, Renderer

from typing import override

class JouleExpansionScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count = 500
        self.temperature = 300
        self.particle_type = PARTICLE_PRESETS["H2"].new_color(LIGHT_BLUE)
        self.small_axis = 3e-9
        self.large_axis_min = 3e-9
        self.large_axis_max = self.small_axis * 3
        self.seconds_for_expansion = 1

    @property
    @override
    def name(self) -> str:
        return "joule_expansion"

    @property
    @override
    def time_ratio(self) -> float:
        return 5e-13

    @override
    def setup_system(self):
        config = Config(
            lx=self.large_axis_min, 
            ly=self.small_axis, 
            lz=self.small_axis,
            max_lx=self.large_axis_max,
            max_ly=self.small_axis,
            max_lz=self.small_axis
        )
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type,
            count=self.particle_count,
            temperature=self.temperature,
        )

        renderer_config = RendererConfig(
            camera_config=CAMERA_FACING_X.new_distance(1.2),
            max_dims=np.array([self.large_axis_max, self.small_axis, self.small_axis])
        )
        renderer = Renderer(simulation=sim, config=renderer_config)
        return sim, renderer

    @override
    def run(self):
        simulation = self.simulation
        simulation.transformations.add_waiting_action(self.time_ratio * 2, 
            lambda: simulation.transformations.set_deformation(
                initial_dimensions=np.array([self.large_axis_min, self.small_axis, self.small_axis]),
                final_dimensions=np.array([self.large_axis_max, self.small_axis, self.small_axis]),
                duration=self.time_ratio * self.seconds_for_expansion
            )
        )

if __name__ == "__main__":
    scenario = JouleExpansionScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()