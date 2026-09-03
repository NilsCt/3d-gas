import sys
from pathlib import Path
from collections import deque
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


class TrajectoryXYChart(Chart):

    def __init__(self, particle_index: int, container_size: float, color, max_points: int = 2000):
        super().__init__(title="Trajectory XY")
        self.particle_index = particle_index
        self.container_size = container_size
        self.color = (color.red, color.green, color.blue) if hasattr(color, 'red') else color[:3]
        self.xs: deque[float] = deque(maxlen=max_points)
        self.ys: deque[float] = deque(maxlen=max_points)

    def update(self, simulation: Simulation) -> None:
        pos = simulation.gas.positions[self.particle_index]
        self.xs.append(pos[0])
        self.ys.append(pos[1])

    def draw(self, ax, scale: float = 1.0) -> None:
        ax.set_xlim(0, self.container_size)
        ax.set_ylim(0, self.container_size)
        ax.set_aspect('equal')
        ax.set_xlabel("x [m]", fontsize=20 * scale, color='white')
        ax.set_ylabel("y [m]", fontsize=20 * scale, color='white')
        ax.tick_params(colors='white', labelsize=16 * scale)
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_color('white')
        ax.yaxis.get_offset_text().set_color('white')
        ax.xaxis.get_offset_text().set_fontsize(16 * scale)
        ax.yaxis.get_offset_text().set_fontsize(16 * scale)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5 * scale)

        if len(self.xs) > 1:
            ax.plot(list(self.xs), list(self.ys), color='white', linewidth=1.5 * scale)
        if len(self.xs) > 0:
            ax.scatter([self.xs[-1]], [self.ys[-1]], color=self.color, s=250 * scale, zorder=10)

class BrownianMotionScenario(Scenario):

    def __init__(self):
        super().__init__()
        self.particle_count = 500
        self.temperature = 300
        self.particle_type = PARTICLE_PRESETS["H2"]
        self.l = 5e-9
        self.big_particle_type = PARTICLE_PRESETS["H2"].bigger(4, "Big", PRETTY_RED)
        self.trajectory_points = 10000

    @property
    @override
    def name(self) -> str:
        return "brownian_motion"
    
    @property
    @override
    def time_ratio(self) -> float:
        return super().time_ratio / 2

    @override
    def setup_system(self):
        config = Config(lx=self.l, ly=self.l, lz=self.l, trajectory_max_points=self.trajectory_points)
        sim = Simulation(config)

        sim.add_particles(
            particle_type=self.big_particle_type,
            count=1,
            temperature=self.temperature,
        )
        sim.add_particles(
            particle_type=self.particle_type,
            count=self.particle_count,
            temperature=self.temperature,
        )
        position = np.array([self.l/2, self.l/2, self.l/2])
        sim.gas.positions[0] = position  # place the big particle in the center
    
        render_config = RendererConfig(render_mode="spheres")
        renderer = Renderer(simulation=sim, config=render_config)
        return sim, renderer

    @override
    def setup_charts(self) -> ChartOverlay:
        chart_config = ChartConfig(position="top-right", size=(450, 450), expanded_size=(700, 700))
        overlay = ChartOverlay(config=chart_config)
        overlay.add_chart(TrajectoryXYChart(
            particle_index=0,
            container_size=self.l,
            color=PRETTY_RED,
        ))
        return overlay

    @override
    def run(self):
        self.renderer.toggle_trajectory(0)

if __name__ == "__main__":
    scenario = BrownianMotionScenario()
    args = scenario.parse_args()
    if args.video:
        scenario.launch_video_exporter()
    else:
        scenario.launch_live_viewer()