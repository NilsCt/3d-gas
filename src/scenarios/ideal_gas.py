import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, PRETTY_RED, LIGHT_GREEN
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CAMERA_FACING_X, Renderer
from src.visualization.chart_overlay import ChartOverlay, ChartConfig, Chart
from src.simulation.physics import Physics
from src.simulation.bounds import Bounds

from typing import override


class IdealGasLawChart(Chart):

    def __init__(self):
        super().__init__(title="Ideal Gas Law: pV/nRT")
        self.times: list[float] = []
        self.ratios: list[float] = []

    def update(self, simulation: Simulation) -> None:
        self.times.append(simulation.time)
        self.ratios.append(simulation.thermodynamics_state.pv_nkt)

    def draw(self, ax, scale: float = 1.0) -> None:
        ax.set_xlabel("t [s]", fontsize=20 * scale, color='white')
        ax.set_ylabel("pV/nRT", fontsize=20 * scale, color='white')
        ax.tick_params(colors='white', labelsize=16 * scale)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_color('white')
        ax.xaxis.get_offset_text().set_fontsize(16 * scale)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5 * scale)

        ax.axhline(y=1.0, color=(PRETTY_RED.red, PRETTY_RED.green, PRETTY_RED.blue), linestyle='--', linewidth=2 * scale)

        if len(self.times) > 1:
            ax.plot(list(self.times), list(self.ratios), color='lime', linewidth=1.5 * scale)
            min_r, max_r = min(self.ratios), max(self.ratios)
            margin = max(0.1, (max_r - min_r) * 0.3)
            ax.set_ylim(min(0.8, min_r - margin), max(1.2, max_r + margin))

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
        return super().time_ratio / 1.8

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

    @override
    def setup_charts(self) -> ChartOverlay:
        chart_config = ChartConfig(position="top-right", size=(450, 350), expanded_size=(700, 550))
        overlay = ChartOverlay(config=chart_config)
        overlay.add_chart(IdealGasLawChart())
        return overlay

if __name__ == "__main__":
    scenario = IdealGasScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()