import sys
import time
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

    # sim.transformations.add_thermostat(100, tau=1e-11, duration=1e-9)

    # sim.transformations.apply_instantaneous_thermostat(sim.gas, target_temperature=100)

    sim.transformations.set_deformation(
        initial_dimensions=sim.container.dimensions.copy(),
        final_dimensions=sim.container.dimensions.copy() / 2,
        duration=1e-11,
        #scale_particle_positions=True,
        #piston_give_velocity=True,
        #clamp_positions=True,
    )

    def scenario():
        #sim.transformations.stirling_cycle(
        #    container=sim.container,
        #    V_min=sim.container.volume / 8,
        #    V_max=sim.container.volume,
        #    T_cold=300,
        #    T_hot=400,
        #    step_duration=5e-12,
        #    prep_duration=1e-12,
        #)

        #sim.transformations.carnot_cycle(
        #    container=sim.container,
        #    V_min=sim.container.volume / 8,
        #    V_max=sim.container.volume,
        #    T_cold=300,
        #    T_hot=400,
        #    step_duration=5e-12,
        #    prep_duration=1e-12,
        #)

        #print("hello")
        #time.sleep(5)
        #print("bonjour")
        #sim.transformations.add_waiting_action(
        #    duration=1e-11,
        #    action=lambda: print("finito", sim.gas.count),
        #)
        print("hello")

    live_viewer.start(scenario=scenario)




if __name__ == "__main__":
    main()
