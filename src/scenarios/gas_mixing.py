import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, LIGHT_BLUE, LIGHT_GREEN
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CAMERA_FACING_X, Renderer
from src.simulation.physics import Physics
from src.simulation.bounds import Bounds

from typing import override

class GasMixingScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count_each = 600
        self.temperature = 300
        self.particle_type_1 = PARTICLE_PRESETS["H2"].new_color(LIGHT_BLUE)
        self.particle_type_2 = PARTICLE_PRESETS["He"].new_color(LIGHT_GREEN)
        self.lx = 6e-9
        self.lyz = 3e-9

    @property
    @override
    def name(self) -> str:
        return "gas_mixing"
    
    @property
    @override
    def time_ratio(self) -> float:
        return super().time_ratio /2

    @override
    def setup_system(self):
        config = Config(lx=self.lx, ly=self.lyz, lz=self.lyz)
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type_1,
            count=self.particle_count_each,
            temperature=self.temperature,
            bounds=Bounds(xmin=0, xmax=self.lx/2, ymin=0, ymax=self.lyz, zmin=0, zmax=self.lyz)
        )
        sim.add_particles(
            particle_type=self.particle_type_2,
            count=self.particle_count_each,
            temperature=self.temperature,
            bounds=Bounds(xmin=self.lx/2, xmax=self.lx, ymin=0, ymax=self.lyz, zmin=0, zmax=self.lyz)
        )

        renderer_config = RendererConfig(camera_config=CAMERA_FACING_X.new_distance(1.5))
        renderer = Renderer(simulation=sim, config=renderer_config)
        return sim, renderer    

if __name__ == "__main__":
    scenario = GasMixingScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()