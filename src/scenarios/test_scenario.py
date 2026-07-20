import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS
from src.simulation import Simulation
from src.visualization.renderer import Renderer

from typing import override

class TestScenario(Scenario):

    @property
    @override
    def name(self) -> str:
        return "test_scenario"

    @override
    def setup_system(self):
        size = 10e-9
        config = Config(lx=size, ly=size, lz=size)
        sim = Simulation(config)

        temperature = 300
        sim.add_particles(
            particle_type=PARTICLE_PRESETS["He"],
            count=100,
            temperature=temperature,
        )
        sim.add_particles(
            particle_type=PARTICLE_PRESETS["N2"],
            count=10,
            temperature=temperature,
        )
        sim.particle_tracker.add_particle(0, sim.gas.positions[0])

        print(f"\nTotal: {sim.gas.count} particules")
        print(f"Enceinte: {size*1e9:.1f} nm x {size*1e9:.1f} nm x {size*1e9:.1f} nm")
        print(f"Température: {temperature} K")

        renderer = Renderer(simulation=sim)
        return sim, renderer

    @override
    def run(self):
        print("alala", self.simulation.gas.count)

if __name__ == "__main__":
    scenario = TestScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()