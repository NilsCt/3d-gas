import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, LIGHT_BLUE
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CAMERA_FACING_X, Renderer
from src.simulation.physics import Physics

from typing import override

class WeirdVelocitiesScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count = 500
        self.temperature = 300
        self.particle_type = PARTICLE_PRESETS["H2"].new_color(LIGHT_BLUE)
        self.l = 5e-9
        self.velocity = np.array(
            [Physics.temperature_to_mean_speed(self.temperature, self.particle_type.mass), 0, 0]
        )
        self.wait_time = 2.0  # seconds to wait before setting the velocities

    @property
    @override
    def name(self) -> str:
        return "weird_velocities"
    
    @property
    @override
    def time_ratio(self) -> float:
        return super().time_ratio /2

    @override
    def setup_system(self):
        config = Config(lx=self.l, ly=self.l, lz=self.l)
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type,
            count=self.particle_count,
            temperature=self.temperature,
        )
        sim.gas.velocities = np.zeros((self.particle_count, 3))

        renderer = Renderer(simulation=sim)
        return sim, renderer
    
    @override
    def run(self):
        time.sleep(self.wait_time)
        self.simulation.gas.velocities = -np.tile(self.velocity, (self.particle_count, 1))  # same velocity for all particles
        

if __name__ == "__main__":
    scenario = WeirdVelocitiesScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()