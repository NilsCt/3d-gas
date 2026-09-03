import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, LIGHT_BLUE, PRETTY_RED
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CAMERA_FACING_X, Renderer
from src.visualization.chart_overlay import ChartOverlay, ChartConfig, Chart
from src.simulation.physics import Physics

from typing import override


class SpeedDistributionChart(Chart):

    def __init__(self, max_speed: float, temperature: float, mass: float, color, n_bins: int = 40):
        super().__init__(title="Speed Distribution")
        self.max_speed = max_speed
        self.temperature = temperature
        self.mass = mass
        self.color = (color.red, color.green, color.blue) if hasattr(color, 'red') else color[:3]
        self.n_bins = n_bins
        self.speeds: np.ndarray = np.array([])

    def update(self, simulation: Simulation) -> None:
        self.speeds = simulation.gas.speeds.copy()

    def draw(self, ax, scale: float = 1.0) -> None:
        ax.set_xlim(0, self.max_speed)
        ax.set_xlabel("v [m/s]", fontsize=20 * scale, color='white')
        ax.set_ylabel("Density", fontsize=20 * scale, color='white')
        ax.tick_params(colors='white', labelsize=16 * scale)
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_color('white')
        ax.yaxis.get_offset_text().set_color('white')
        ax.xaxis.get_offset_text().set_fontsize(16 * scale)
        ax.yaxis.get_offset_text().set_fontsize(16 * scale)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5 * scale)

        v = np.linspace(0, self.max_speed, 200)
        pdf = Physics.maxwell_boltzmann_speed_pdf(v, self.temperature, self.mass)
        ax.plot(v, pdf, color=(PRETTY_RED.red, PRETTY_RED.green, PRETTY_RED.blue), linewidth=2 * scale)

        if len(self.speeds) > 0:
            bins = np.linspace(0, self.max_speed, self.n_bins + 1)
            ax.hist(self.speeds, bins=bins, density=True, color=self.color, alpha=0.7, edgecolor='white', linewidth=0.5 * scale)

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
    def setup_charts(self) -> ChartOverlay:
        mean_speed = Physics.temperature_to_mean_speed(self.temperature, self.particle_type.mass)
        chart_config = ChartConfig(position="top-right", size=(450, 350), expanded_size=(700, 550))
        overlay = ChartOverlay(config=chart_config)
        overlay.add_chart(SpeedDistributionChart(
            max_speed=mean_speed * 3.5,
            temperature=self.temperature,
            mass=self.particle_type.mass,
            color=self.particle_type.color,
        ))
        return overlay

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