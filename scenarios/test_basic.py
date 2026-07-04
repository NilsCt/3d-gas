import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Config, PARTICLE_PRESETS
from src.simulation import Simulation
from src.visualization.live_renderer import LiveRenderer


def main():
    size = 10e-9  # 10 nanomètres
    config = Config(
        lx=size,
        ly=size,
        lz=size,
        dt_mode="adaptive",
        target_fps=60,
    )
    sim = Simulation(config)

    temperature = 300
    n_particles = 100
    sim.add_particles(
        particle_type=PARTICLE_PRESETS["He"],
        count=n_particles,
        temperature=temperature,
    )
    sim.add_particles(
        particle_type=PARTICLE_PRESETS["N2"],
        count=10,
        temperature=temperature,
    )

    print(f"\nTotal: {sim.gas.count} particules")
    print(f"Enceinte: {size*1e9:.1f} nm x {size*1e9:.1f} nm x {size*1e9:.1f} nm")
    print(f"Température: {temperature} K")

    renderer = LiveRenderer(
        simulation=sim,
        render_mode="spheres",
        window_size=(1024, 768),
        title="Test - Simulation de gaz",
        auto_rotate=True,
    )
    renderer.start()


if __name__ == "__main__":
    main()
