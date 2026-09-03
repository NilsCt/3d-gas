import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, PRETTY_RED
from src.simulation import Simulation
from src.visualization.renderer import Renderer, RendererConfig
from src.visualization.chart_overlay import ChartOverlay, ChartConfig, Chart
from src.visualization.video_exporter import VideoConfig, VIDEOS_DIR
from src.simulation.physics import Physics

from typing import override


class PVDiagramChart(Chart):

    def __init__(self, start_delay: float = 0):
        super().__init__(title="p-V Diagram")
        self.start_delay = start_delay
        self.pressures: list[float] = []
        self.volumes: list[float] = []

    def update(self, simulation: Simulation) -> None:
        if simulation.time < self.start_delay:
            return
        self.pressures.append(simulation.thermodynamics_state.pressure)
        self.volumes.append(simulation.thermodynamics_state.volume)

    def draw(self, ax, scale: float = 1.0) -> None:
        ax.set_xlabel("V [m³]", fontsize=20 * scale, color='white')
        ax.set_ylabel("p [Pa]", fontsize=20 * scale, color='white')
        ax.tick_params(colors='white', labelsize=16 * scale)
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_color('white')
        ax.yaxis.get_offset_text().set_color('white')
        ax.xaxis.get_offset_text().set_fontsize(16 * scale)
        ax.yaxis.get_offset_text().set_fontsize(16 * scale)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5 * scale)

        if len(self.pressures) > 1:
            ax.plot(self.volumes, self.pressures, color='white', linewidth=1.5 * scale)
        if len(self.pressures) > 0:
            ax.scatter([self.volumes[-1]], [self.pressures[-1]],
                       color=(PRETTY_RED.red, PRETTY_RED.green, PRETTY_RED.blue),
                       s=150 * scale, zorder=10)

class StirlingCycleSlowScenario(Scenario):
    """
    Slow Stirling cycle with quasi-static transformations.
    Piston velocity is ~10x slower than particle velocity for more accurate thermodynamics.
    """

    def __init__(self):
        super().__init__()
        self.particle_count = 500
        self.particle_type = PARTICLE_PRESETS["H2"]
        self.T_cold = 300
        self.T_hot = 700
        self.l_max = 10e-9
        self.l_min = self.l_max / 2
        self.seconds_per_step = 50  # 10x slower than normal

    @property
    @override
    def name(self) -> str:
        return "stirling_cycle_slow"

    @property
    @override
    def video_config(self) -> VideoConfig:
        # Duration for at least one full cycle:
        # waiting (2s) + prep (1s) + 4 steps (4 * 50s) = 203s
        # Add margin for a complete cycle
        return VideoConfig(
            output_path=VIDEOS_DIR / f"{self.name}.mp4",
            duration=220.0
        )

    @override
    def setup_system(self):
        config = Config(lx=self.l_max, ly=self.l_max, lz=self.l_max, pressure_window=200)
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type,
            count=self.particle_count,
            temperature=self.T_cold,
        )

        renderer_config = RendererConfig(color_mode="by_relative_energy")
        renderer = Renderer(simulation=sim, config=renderer_config)
        renderer.color_picker.set_range(
            min=Physics.temperature_to_mean_energy(self.T_cold + 50),
            max=Physics.temperature_to_mean_energy(self.T_hot - 50),
            variation_strength=1
        )
        return sim, renderer

    @override
    def setup_charts(self) -> ChartOverlay:
        chart_config = ChartConfig(position="top-right", size=(450, 400), expanded_size=(700, 620))
        overlay = ChartOverlay(config=chart_config)
        # Start recording after waiting + prep phase
        overlay.add_chart(PVDiagramChart(start_delay=self.time_ratio * 4))
        return overlay

    @override
    def run(self):
        simulation = self.simulation
        simulation.transformations.add_waiting_action(self.time_ratio * 2,
            lambda: simulation.transformations.stirling_cycle(
                container=simulation.container,
                T_hot=self.T_hot,
                T_cold=self.T_cold,
                V_max=simulation.container.volume,
                V_min=self.l_min**3,
                step_duration=self.time_ratio * self.seconds_per_step,
                prep_duration=self.time_ratio * 2,  # longer prep for slow version
            )
        )

if __name__ == "__main__":
    scenario = StirlingCycleSlowScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()
