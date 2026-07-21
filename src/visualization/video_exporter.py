import imageio
from dataclasses import dataclass
from typing import Callable
import threading
from pathlib import Path

from src.simulation.simulation import Simulation
from .renderer import Renderer

VIDEOS_DIR = Path(__file__).parent.parent.parent / "videos"

@dataclass
class VideoConfig:
    output_path: Path = VIDEOS_DIR / "output.mp4"
    fps: int = 60
    duration: float = 30.0
    resolution: tuple = (1920, 1088)
    codec: str = "libx264"
    quality: int = 8  # imageio quality (higher = better)


class VideoExporter:

    def __init__(
        self,
        simulation: Simulation,
        renderer: Renderer,
        config: VideoConfig = VideoConfig(),
        time_ratio: float = 1e-12,
    ):
        self.simulation = simulation
        self.renderer = renderer
        self.config = config
        self.time_ratio = time_ratio

    def export(self, scenario: Callable[[], None] | None = None):
        config = self.config
        video_dt = 1.0 / config.fps
        sim_dt_per_frame = video_dt * self.time_ratio
        total_frames = int(config.duration * config.fps)

        self.renderer.init_offscreen(config.resolution)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            config.output_path,
            fps=config.fps,
            codec=config.codec,
            quality=config.quality,
        )

        if scenario is not None:
            self._scenario_thread = threading.Thread(target=scenario, daemon=True)
            self._scenario_thread.start()

        for frame_idx in range(total_frames):
            remaining_dt = sim_dt_per_frame
            while remaining_dt > 1e-16:
                remaining_dt -= self.simulation.complete_step(max_dt=remaining_dt)

            self.renderer.advance_rotation(video_dt)
            frame = self.renderer.render_offscreen_frame()
            writer.append_data(frame)

            if frame_idx % config.fps == 0:
                progress = 100 * frame_idx / total_frames
                print(f"Frame {frame_idx}/{total_frames} ({progress:.1f}%)")

        writer.close()
        self.renderer.cleanup_offscreen()
        print(f"Video saved to {config.output_path}")