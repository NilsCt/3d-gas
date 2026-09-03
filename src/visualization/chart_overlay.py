import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal, List, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from src.simulation.simulation import Simulation


class ChartDisplayMode(Enum):
    NORMAL = "normal"
    EXPANDED = "expanded"
    HIDDEN = "hidden"


@dataclass
class ChartConfig:
    position: Literal["top-right", "top-left", "bottom-right", "bottom-left"] = "top-right"
    size: tuple[int, int] = (320, 240)  # width, height in pixels (collapsed)
    expanded_size: tuple[int, int] = (800, 600)  # width, height in pixels (expanded)
    margin: int = 15  # Margin from edges in pixels
    background_alpha: float = 0.85
    border_color: tuple[float, float, float] = (0.3, 0.3, 0.3)
    border_width: int = 2
    style: str = "dark_background"


class Chart(ABC):
    """
    Abstract base class for charts.
    Subclass this to create specific chart types.
    """

    def __init__(self, title: str = ""):
        self.title = title

    @abstractmethod
    def update(self, simulation: "Simulation") -> None:
        """
        Update chart data from simulation state.
        Called every frame before drawing.
        """
        pass

    @abstractmethod
    def draw(self, ax: plt.Axes) -> None:
        """
        Draw the chart on the given matplotlib axes.
        """
        pass

    def setup_axes(self, ax: plt.Axes) -> None:
        """
        Optional: Configure axes appearance (labels, limits, etc.)
        Called once when the chart is added.
        """
        if self.title:
            ax.set_title(self.title, fontsize=10, color='white')


class ChartOverlay:
    """
    Manages matplotlib chart overlay rendering and composition.
    Works with both live viewer and video exporter.
    """

    def __init__(self, config: ChartConfig = ChartConfig()):
        self.config = config
        self.charts: List[Chart] = []
        self.display_mode: ChartDisplayMode = ChartDisplayMode.NORMAL
        self._figure: Figure | None = None
        self._axes: List[plt.Axes] = []
        self._canvas: FigureCanvasAgg | None = None
        self._needs_rebuild: bool = True

    def add_chart(self, chart: Chart) -> None:
        """Add a chart to the overlay."""
        self.charts.append(chart)
        self._needs_rebuild = True

    def remove_chart(self, chart: Chart) -> None:
        """Remove a chart from the overlay."""
        if chart in self.charts:
            self.charts.remove(chart)
            self._needs_rebuild = True

    def clear_charts(self) -> None:
        """Remove all charts."""
        self.charts.clear()
        self._needs_rebuild = True

    def cycle_display_mode(self) -> None:
        """Cycle through display modes: normal -> expanded -> hidden -> normal."""
        if self.display_mode == ChartDisplayMode.NORMAL:
            self.display_mode = ChartDisplayMode.EXPANDED
        elif self.display_mode == ChartDisplayMode.EXPANDED:
            self.display_mode = ChartDisplayMode.HIDDEN
        else:
            self.display_mode = ChartDisplayMode.NORMAL
        self._needs_rebuild = True

    def is_visible(self) -> bool:
        """Return True if charts should be rendered."""
        return self.display_mode != ChartDisplayMode.HIDDEN

    def _get_current_size(self) -> tuple[int, int]:
        """Get current size based on display mode."""
        if self.display_mode == ChartDisplayMode.EXPANDED:
            return self.config.expanded_size
        return self.config.size

    @property
    def expanded_size(self) -> tuple[int, int]:
        return self.config.expanded_size

    def _build_figure(self) -> None:
        """Build or rebuild the matplotlib figure with current charts."""
        if self._figure is not None:
            plt.close(self._figure)

        n_charts = len(self.charts)
        if n_charts == 0:
            self._figure = None
            self._canvas = None
            self._axes = []
            return

        size = self._get_current_size()
        dpi = 100
        figsize = (size[0] / dpi, size[1] / dpi)

        with plt.style.context(self.config.style):
            self._figure = Figure(figsize=figsize, dpi=dpi, facecolor='black')
            self._figure.patch.set_alpha(self.config.background_alpha)
            self._canvas = FigureCanvasAgg(self._figure)

            # Create subplot grid based on number of charts
            if n_charts == 1:
                rows, cols = 1, 1
            elif n_charts == 2:
                rows, cols = 2, 1
            elif n_charts <= 4:
                rows, cols = 2, 2
            else:
                rows = (n_charts + 1) // 2
                cols = 2

            self._axes = []
            for i, chart in enumerate(self.charts):
                ax = self._figure.add_subplot(rows, cols, i + 1)
                ax.set_facecolor((0, 0, 0, self.config.background_alpha))
                chart.setup_axes(ax)
                self._axes.append(ax)

            self._figure.tight_layout(pad=0.5)

        self._needs_rebuild = False

    def update(self, simulation: "Simulation") -> None:
        """Update all charts with current simulation data (skipped if hidden)."""
        if not self.is_visible():
            return
        for chart in self.charts:
            chart.update(simulation)

    def render_to_image(self) -> np.ndarray | None:
        """
        Render the charts to an RGBA numpy array.
        Returns None if no charts are configured or if hidden.
        """
        if len(self.charts) == 0 or not self.is_visible():
            return None

        if self._needs_rebuild or self._figure is None:
            self._build_figure()

        if self._figure is None:
            return None

        # Clear and redraw all axes
        for ax, chart in zip(self._axes, self.charts):
            ax.clear()
            chart.setup_axes(ax)
            chart.draw(ax)

        self._figure.tight_layout(pad=0.5)

        # Render to numpy array
        self._canvas.draw()
        buf = self._canvas.buffer_rgba()
        image = np.asarray(buf, dtype=np.uint8).copy()

        return image

    def _compute_position(self, frame_shape: tuple[int, int, int], overlay_shape: tuple[int, int, int]) -> tuple[int, int]:
        """
        Compute top-left corner position for overlay on frame.
        frame_shape: (height, width, channels)
        overlay_shape: (height, width, channels)
        Returns: (y, x) position
        """
        frame_h, frame_w = frame_shape[:2]
        overlay_h, overlay_w = overlay_shape[:2]
        margin = self.config.margin

        if self.config.position == "top-right":
            x = frame_w - overlay_w - margin
            y = margin
        elif self.config.position == "top-left":
            x = margin
            y = margin
        elif self.config.position == "bottom-right":
            x = frame_w - overlay_w - margin
            y = frame_h - overlay_h - margin
        elif self.config.position == "bottom-left":
            x = margin
            y = frame_h - overlay_h - margin
        else:
            x = frame_w - overlay_w - margin
            y = margin

        return int(y), int(x)

    def compose_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Compose the chart overlay onto the frame using alpha blending.
        frame: RGB or RGBA numpy array (height, width, channels)
        Returns: RGB numpy array with overlay composited
        """
        overlay = self.render_to_image()
        if overlay is None:
            return frame

        # Ensure frame is writable
        frame = frame.copy()

        y, x = self._compute_position(frame.shape, overlay.shape)
        overlay_h, overlay_w = overlay.shape[:2]

        # Clamp to frame bounds
        y_end = min(y + overlay_h, frame.shape[0])
        x_end = min(x + overlay_w, frame.shape[1])
        y_start = max(y, 0)
        x_start = max(x, 0)

        # Corresponding overlay region
        oy_start = y_start - y
        ox_start = x_start - x
        oy_end = oy_start + (y_end - y_start)
        ox_end = ox_start + (x_end - x_start)

        if y_end <= y_start or x_end <= x_start:
            return frame

        # Extract regions
        frame_region = frame[y_start:y_end, x_start:x_end]
        overlay_region = overlay[oy_start:oy_end, ox_start:ox_end]

        # Alpha blending
        alpha = overlay_region[:, :, 3:4].astype(np.float32) / 255.0
        overlay_rgb = overlay_region[:, :, :3].astype(np.float32)

        if frame_region.shape[2] == 4:
            frame_rgb = frame_region[:, :, :3].astype(np.float32)
        else:
            frame_rgb = frame_region.astype(np.float32)

        blended = (overlay_rgb * alpha + frame_rgb * (1 - alpha)).astype(np.uint8)

        if frame.shape[2] == 4:
            frame[y_start:y_end, x_start:x_end, :3] = blended
        else:
            frame[y_start:y_end, x_start:x_end] = blended

        return frame

    def get_overlay_bounds(self, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
        """
        Get the bounding box of the overlay on the frame.
        Returns (x_min, y_min, x_max, y_max) or None if no charts.
        """
        if len(self.charts) == 0:
            return None

        size = self._get_current_size()
        overlay_shape = (size[1], size[0], 4)  # (height, width, channels)
        y, x = self._compute_position(frame_shape, overlay_shape)

        return (x, y, x + size[0], y + size[1])

    def handle_click(self, click_x: int, click_y: int, frame_shape: tuple[int, int, int]) -> bool:
        """
        Check if click is within overlay bounds and toggle expanded if so.
        Returns True if click was handled (within overlay bounds).
        """
        bounds = self.get_overlay_bounds(frame_shape)
        if bounds is None:
            return False

        x_min, y_min, x_max, y_max = bounds
        if x_min <= click_x <= x_max and y_min <= click_y <= y_max:
            self.toggle_expanded()
            return True

        return False

    def cleanup(self) -> None:
        """Clean up matplotlib resources."""
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
            self._canvas = None
            self._axes = []
