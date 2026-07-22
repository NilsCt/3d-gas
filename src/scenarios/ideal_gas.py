import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, PRETTY_RED, LIGHT_GREEN
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CAMERA_FACING_X, Renderer
from src.simulation.physics import Physics
from src.simulation.bounds import Bounds

from typing import override

class IdealGasScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count_each = 300
        self.temperature = 300
        self.particle_type_1 = PARTICLE_PRESETS["H2"].new_color(PRETTY_RED)
        self.particle_type_2 = PARTICLE_PRESETS["He"].new_color(LIGHT_GREEN)
        self.particle_type_3 = PARTICLE_PRESETS["N2"]
        self.l = 10e-9

    @property
    @override
    def name(self) -> str:
        return "ideal_gas"
    
    @property
    @override
    def time_ratio(self) -> float:
        return super().time_ratio / 1.2

    @override
    def setup_system(self):
        config = Config(lx=self.l, ly=self.l, lz=self.l)
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type_1,
            count=self.particle_count_each,
            temperature=self.temperature,
        )
        sim.add_particles(
            particle_type=self.particle_type_2,
            count=self.particle_count_each,
            temperature=self.temperature,
        )
        sim.add_particles(
            particle_type=self.particle_type_3,
            count=self.particle_count_each,
            temperature=self.temperature,
        )

        renderer_config = RendererConfig(render_mode="spheres")
        renderer = Renderer(simulation=sim, config=renderer_config)
        return sim, renderer    

if __name__ == "__main__":
    scenario = IdealGasScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()