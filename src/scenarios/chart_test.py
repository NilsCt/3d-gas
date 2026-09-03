import sys
from pathlib import Path
from collections import deque
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, Renderer
from src.visualization.chart_overlay import ChartOverlay, ChartConfig, Chart

from typing import override


class PressureTimeChart(Chart):

    def __init__(self, max_points: int = 200):
        super().__init__(title="Pressure (Pa)")
        self.times: deque[float] = deque(maxlen=max_points)
        self.pressures: deque[float] = deque(maxlen=max_points)

    def update(self, simulation: Simulation) -> None:
        self.times.append(simulation.time)
        self.pressures.append(simulation.thermodynamics_state.pressure)

    def draw(self, ax, scale: float = 1.0) -> None:
        if len(self.times) > 1:
            ax.plot(list(self.times), list(self.pressures), 'c-', linewidth=1.5 * scale)
            ax.set_xlabel("Time (s)", fontsize=10 * scale, color='white')
            ax.set_ylabel("P (Pa)", fontsize=10 * scale, color='white')
            ax.tick_params(colors='white', labelsize=9 * scale)
            ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))


class SpeedDistributionChart(Chart):

    def __init__(self, n_bins: int = 30):
        super().__init__(title="Speed Distribution")
        self.n_bins = n_bins
        self.speeds: np.ndarray = np.array([])

    def update(self, simulation: Simulation) -> None:
        self.speeds = simulation.gas.speeds.copy()

    def draw(self, ax, scale: float = 1.0) -> None:
        if len(self.speeds) > 0:
            ax.hist(self.speeds, bins=self.n_bins, color='orange', alpha=0.7, edgecolor='white', linewidth=0.5 * scale)
            ax.set_xlabel("Speed (m/s)", fontsize=10 * scale, color='white')
            ax.set_ylabel("Count", fontsize=10 * scale, color='white')
            ax.tick_params(colors='white', labelsize=9 * scale)


class IdealGasLawChart(Chart):

    def __init__(self, max_points: int = 200):
        super().__init__(title="pV/nkT")
        self.times: deque[float] = deque(maxlen=max_points)
        self.ratios: deque[float] = deque(maxlen=max_points)

    def update(self, simulation: Simulation) -> None:
        self.times.append(simulation.time)
        self.ratios.append(simulation.thermodynamics_state.pv_nkt)

    def draw(self, ax, scale: float = 1.0) -> None:
        if len(self.times) > 1:
            ax.plot(list(self.times), list(self.ratios), 'lime', linewidth=1.5 * scale)
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1 * scale, alpha=0.7)
            ax.set_xlabel("Time (s)", fontsize=10 * scale, color='white')
            ax.set_ylabel("pV/nkT", fontsize=10 * scale, color='white')
            ax.tick_params(colors='white', labelsize=9 * scale)
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
            if len(self.ratios) > 0:
                min_r, max_r = min(self.ratios), max(self.ratios)
                margin = max(0.1, (max_r - min_r) * 0.2)
                ax.set_ylim(min(0.8, min_r - margin), max(1.2, max_r + margin))


class PVDiagramChart(Chart):

    def __init__(self, max_points: int = 500):
        super().__init__(title="P-V Diagram")
        self.pressures: deque[float] = deque(maxlen=max_points)
        self.volumes: deque[float] = deque(maxlen=max_points)

    def update(self, simulation: Simulation) -> None:
        self.pressures.append(simulation.thermodynamics_state.pressure)
        self.volumes.append(simulation.thermodynamics_state.volume)

    def draw(self, ax, scale: float = 1.0) -> None:
        if len(self.pressures) > 1:
            ax.plot(list(self.volumes), list(self.pressures), 'magenta', linewidth=1.5 * scale)
            ax.set_xlabel("V (m³)", fontsize=10 * scale, color='white')
            ax.set_ylabel("P (Pa)", fontsize=10 * scale, color='white')
            ax.tick_params(colors='white', labelsize=9 * scale)
            ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))


class ChartTestScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count = 500
        self.temperature = 300
        self.particle_type = PARTICLE_PRESETS["N2"]
        self.l = 10e-9

    @property
    @override
    def name(self) -> str:
        return "chart_test"

    @property
    @override
    def time_ratio(self) -> float:
        return 1e-12 / 2

    @override
    def setup_system(self):
        config = Config(lx=self.l, ly=self.l, lz=self.l)
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type,
            count=self.particle_count,
            temperature=self.temperature,
        )

        renderer_config = RendererConfig(render_mode="spheres")
        renderer = Renderer(simulation=sim, config=renderer_config)
        return sim, renderer

    @override
    def setup_charts(self) -> ChartOverlay:
        chart_config = ChartConfig(
            position="top-right",
            size=(350, 280),
            expanded_size=(700, 560),
        )
        overlay = ChartOverlay(config=chart_config)

        # Add multiple charts
        overlay.add_chart(PressureTimeChart())
        overlay.add_chart(SpeedDistributionChart())
        overlay.add_chart(IdealGasLawChart())

        return overlay


if __name__ == "__main__":
    scenario = ChartTestScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()
