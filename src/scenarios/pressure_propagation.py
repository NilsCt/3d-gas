import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, LIGHT_BLUE
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CAMERA_FACING_X, Renderer
from src.simulation.bounds import Bounds
from src.simulation.physics import Physics

from typing import override

class PressurePropagationScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count = 500
        self.particle_more = 200
        self.temperature = 300
        self.particle_type = PARTICLE_PRESETS["H2"].new_color(LIGHT_BLUE)
        self.lx = 10e-9
        self.dense_proportion = 1/10
        self.lx_dense = self.lx * self.dense_proportion
        self.lyz = 1.5e-9
        self.seconds_before = 2
        self.is_video = False

    @property
    @override
    def name(self) -> str:
        return "pressure_propagation"

    @property
    @override
    def time_ratio(self) -> float:
        return 5e-13

    @override
    def setup_system(self):
        config = Config(lx=self.lx, ly=self.lyz, lz=self.lyz, check_overlap=False)
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type,
            count=int(self.particle_count*self.dense_proportion) + self.particle_more,
            temperature=self.temperature,
            bounds=Bounds(xmin=0, xmax=self.lx_dense, ymin=0, ymax=self.lyz, zmin=0, zmax=self.lyz)
        )
        sim.add_particles(
            particle_type=self.particle_type,
            count=int(self.particle_count*(1-self.dense_proportion)),
            temperature=self.temperature,
            bounds=Bounds(xmin=self.lx_dense, xmax=self.lx, ymin=0, ymax=self.lyz, zmin=0, zmax=self.lyz)
        )

        renderer_config = RendererConfig(camera_config=CAMERA_FACING_X.new_distance(1.2), color_mode="by_speed", color_map_name="plasma")
        renderer = Renderer(simulation=sim, config=renderer_config)
        values = sim.gas.speeds
        renderer.color_picker.set_range(values.min()*1.2, values.max()*0.8)
        if not self.is_video:
            renderer.paused = True
        return sim, renderer

if __name__ == "__main__":
    scenario = PressurePropagationScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.is_video = True
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()