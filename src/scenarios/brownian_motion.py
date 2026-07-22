import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, LIGHT_BLUE, PRETTY_RED
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CAMERA_FACING_X, Renderer
from src.simulation.physics import Physics

from typing import override

class BrownianMotionScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count = 500
        self.temperature = 300
        self.particle_type = PARTICLE_PRESETS["H2"]
        self.l = 5e-9
        self.big_particle_type = PARTICLE_PRESETS["H2"].bigger(4, "Big", PRETTY_RED)
        self.trajectory_points = 10000

    @property
    @override
    def name(self) -> str:
        return "brownian_motion"
    
    @property
    @override
    def time_ratio(self) -> float:
        return super().time_ratio / 2

    @override
    def setup_system(self):
        config = Config(lx=self.l, ly=self.l, lz=self.l, trajectory_max_points=self.trajectory_points)
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.big_particle_type,
            count=1,
            temperature=self.temperature,
        )
        sim.add_particles(
            particle_type=self.particle_type,
            count=self.particle_count,
            temperature=self.temperature,
        )
        position = np.array([self.l/2, self.l/2, self.l/2])
        sim.gas.positions[0] = position  # place the big particle in the center
    
        render_config = RendererConfig(render_mode="spheres")
        renderer = Renderer(simulation=sim, config=render_config)
        return sim, renderer   
    
    @override
    def run(self):
        self.renderer.toggle_trajectory(0)

if __name__ == "__main__":
    scenario = BrownianMotionScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()