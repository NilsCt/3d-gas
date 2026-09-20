import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.scenario import Scenario
from src.utils import Config, PARTICLE_PRESETS, LIGHT_BLUE, LIGHT_GREEN, ParticleType
from src.simulation import Simulation
from src.visualization.renderer import RendererConfig, CAMERA_FACING_X, Renderer
from src.simulation.physics import Physics
from src.simulation.bounds import Bounds
from src.visualization.chart_overlay import Chart, ChartOverlay, ChartConfig

from typing import override, TYPE_CHECKING, List

if TYPE_CHECKING: # for cyclic imports...
    from src.simulation.simulation import Simulation as SimulationType


class MixingProportionChart(Chart):

    def __init__(self, lx: float, ly: float, lz: float, particle_types: List[ParticleType]):
        super().__init__(title="Mixing Progress")
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.particle_types = particle_types
        self.times: list[float] = []
        self.left_proportions: dict[int, list[float]] = {}
        self.right_proportions: dict[int, list[float]] = {}

    def update(self, simulation: "SimulationType") -> None:
        self.times.append(simulation.time)

        left_bounds = Bounds(xmin=0, xmax=self.lx/2, ymin=0, ymax=self.ly, zmin=0, zmax=self.lz)
        right_bounds = Bounds(xmin=self.lx/2, xmax=self.lx, ymin=0, ymax=self.ly, zmin=0, zmax=self.lz)
        left_counts = simulation.gas.count_by_type_in_bounds(left_bounds)
        right_counts = simulation.gas.count_by_type_in_bounds(right_bounds)

        for type_idx in range(len(simulation.gas.types)):
            if type_idx not in self.left_proportions:
                self.left_proportions[type_idx] = []
                self.right_proportions[type_idx] = []

            left_total = sum(left_counts.values())
            right_total = sum(right_counts.values())
            left_prop = left_counts.get(type_idx, 0) / left_total if left_total > 0 else 0
            right_prop = right_counts.get(type_idx, 0) / right_total if right_total > 0 else 0
            self.left_proportions[type_idx].append(left_prop)
            self.right_proportions[type_idx].append(right_prop)

    def draw(self, ax, scale: float = 1.0) -> None:
        ax.set_xlabel("Time [s]", fontsize=14 * scale, color='white')
        ax.set_ylabel("Proportion", fontsize=14 * scale, color='white')
        ax.tick_params(colors='white', labelsize=12 * scale)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_color('white')
        ax.xaxis.get_offset_text().set_fontsize(12 * scale)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, color='white')

        labels = [p.name for p in self.particle_types]
        colors = [p.color.to_3_tuple() for p in self.particle_types]

        ax.axhline(y=0.5, color=(0.8, 0.3, 0.3), linestyle='-', linewidth=1.5 * scale, alpha=0.4)

        for type_idx, props in self.left_proportions.items():
            if len(props) > 0:
                ax.plot(self.times, props, color=colors[type_idx % len(colors)],
                       linestyle='-', linewidth=2 * scale, label=f'{labels[type_idx]} (left)')

        ax.legend(fontsize=10 * scale, loc='upper right', facecolor=(0.18, 0.18, 0.18), edgecolor=(0.3, 0.3, 0.3), labelcolor='white', framealpha=0.9)


class GasMixingScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count_each = 600
        self.temperature = 300
        self.particle_types = [
            PARTICLE_PRESETS["H2"].new_color(LIGHT_BLUE),
            PARTICLE_PRESETS["He"].new_color(LIGHT_GREEN)
        ]
        self.particle_type_1 = self.particle_types[0]
        self.particle_type_2 = self.particle_types[1]
        self.lx = 6e-9
        self.lyz = 3e-9

    @property
    @override
    def name(self) -> str:
        return "gas_mixing"
    
    @property
    @override
    def time_ratio(self) -> float:
        return super().time_ratio /3

    @override
    def setup_system(self):
        config = Config(
            lx=self.lx, ly=self.lyz, lz=self.lyz,
            check_overlap=False # to make sure both particle types have exaclty the same number of particles
        )
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.particle_type_1,
            count=self.particle_count_each,
            temperature=self.temperature,
            bounds=Bounds(xmin=0, xmax=self.lx/2, ymin=0, ymax=self.lyz, zmin=0, zmax=self.lyz),
        )
        sim.add_particles(
            particle_type=self.particle_type_2,
            count=self.particle_count_each,
            temperature=self.temperature,
            bounds=Bounds(xmin=self.lx/2, xmax=self.lx, ymin=0, ymax=self.lyz, zmin=0, zmax=self.lyz)
        )

        renderer_config = RendererConfig(camera_config=CAMERA_FACING_X.new_distance(1.5))
        renderer = Renderer(simulation=sim, config=renderer_config)
        return sim, renderer

    @override
    def setup_charts(self) -> ChartOverlay:
        chart_config = ChartConfig(position="top-right", size=(400, 300), expanded_size=(600, 450))
        overlay = ChartOverlay(config=chart_config)
        overlay.add_chart(MixingProportionChart(lx=self.lx, ly=self.lyz, lz=self.lyz, particle_types=self.particle_types))
        return overlay


if __name__ == "__main__":
    scenario = GasMixingScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()