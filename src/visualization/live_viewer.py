import time
import numpy as np
from typing import Callable

from src.simulation.simulation import Simulation
from .renderer import Renderer

class LiveViewer:

    def __init__(
        self,
        simulation: Simulation,
        renderer: Renderer,
        time_ratio: float = 1e-12,
        initial_speed_factor: float = 1.0
    ):
        self.simulation = simulation
        self.renderer = renderer
        self.time_ratio = time_ratio
        self.speed_factor = initial_speed_factor

        self._mouse_interacting: bool = False
        self._last_simulation_update: float = time.perf_counter()
        self._loop_count: int = 0

    def _on_timer(self, event):
        renderer = self.renderer
        if self._loop_count <= 1:
            self._last_simulation_update = time.perf_counter()
        elif not renderer.paused:
            now = time.perf_counter()
            delta_t = (now - self._last_simulation_update) * self.time_ratio * self.speed_factor
            while delta_t > 1e-16:
                delta_t -= self.simulation.complete_step(max_dt=delta_t)
            self._last_simulation_update = now
        self._loop_count += 1
        renderer.update_all()

    def _on_key_press(self, event):
        renderer = self.renderer
        if event.key == "Space":
            renderer.paused = not renderer.paused
            self._last_simulation_update = time.perf_counter() # avoid jump
        elif event.key == "+" or event.key == "=":
            self.speed_factor *= 1.5
        elif event.key == "-":
            self.speed_factor /= 1.5
        elif event.key == "R":
            renderer.cycle_render_mode()
        elif event.key == "C":
            renderer.color_picker.cycle_mode()
            renderer.update_particles()
            renderer.update_trajectories()
        elif event.key == "M":
            renderer.color_picker.cycle_colormap()
            renderer.update_particles()
            renderer.update_trajectories()
        elif event.key == "I":
            renderer.toggle_show_info()
        elif event.key == "A":
            renderer.toggle_auto_rotate(mouse_interacting=self._mouse_interacting)
        elif event.key == "Escape":
            self.stop()
        renderer.update_canvas()

    def _on_mouse_press(self, event):
        renderer = self.renderer
        self._mouse_interacting = True
        renderer.block_rotation()

        if event.button != 1: # only left click
            return
        click_pos = np.asarray(event.pos, dtype=np.float64)
        particle_idx = renderer.pick_particle(click_pos)
        if particle_idx is not None:
            renderer.toggle_trajectory(particle_idx)

    def _on_mouse_release(self, event):
        self._mouse_interacting = False
        self.renderer.unblock_rotation()

    def start(self, scenario: Callable[[], None] | None = None):
        self._last_simulation_update = time.perf_counter()
        self._loop_count = 0
        self.renderer.start(
            on_key_press=self._on_key_press,
            on_mouse_press=self._on_mouse_press,
            on_mouse_release=self._on_mouse_release,
            on_timer=self._on_timer,
            scenario=scenario,
        )

    def stop(self):
        self.renderer.stop()