import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Config, PARTICLE_PRESETS
from src.simulation import Simulation
from src.visualization.renderer import Renderer
from src.visualization.live_viewer import LiveViewer


def main():
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
    live_viewer = LiveViewer(simulation=sim, renderer=renderer)
    live_viewer.start()


if __name__ == "__main__":
    main()
