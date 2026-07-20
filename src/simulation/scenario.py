from abc import ABC, abstractmethod
import argparse

from src.simulation import Simulation
from src.visualization.renderer import Renderer
from src.visualization.live_viewer import LiveViewer
from src.visualization.video_exporter import VideoExporter, VideoConfig, VIDEOS_DIR

class Scenario(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(description=f"Run {self.name}")
        parser.add_argument(
            "--video",
            action="store_true",
            help="Export the simulation as a video instead of running the live viewer."
        )
        return parser.parse_args()

    @abstractmethod
    def setup_system(self) -> tuple[Simulation, Renderer]:
        pass

    @property
    def time_ratio(self) -> float:
        return 1e-12
    
    @property
    def video_config(self) -> VideoConfig:
        return VideoConfig(output_path=VIDEOS_DIR / f"{self.name}.mp4")

    def run(self):
        pass

    def launch_live_viewer(self):
        simulation, renderer = self.setup_system()
        self.simulation = simulation
        self.renderer = renderer
        live_viewer = LiveViewer(simulation=simulation, renderer=renderer, time_ratio=self.time_ratio)
        self.live_viewer = live_viewer
        live_viewer.start(self.run)

    def launch_video_exporter(self):
        simulation, renderer = self.setup_system()
        self.simulation = simulation
        self.renderer = renderer
        video_exporter = VideoExporter(
            simulation=simulation, 
            renderer=renderer, 
            config=self.video_config, 
            time_ratio=self.time_ratio
        )
        self.video_exporter = video_exporter
        video_exporter.export(self.run)