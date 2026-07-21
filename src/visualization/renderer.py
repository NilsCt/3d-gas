import time
import numpy as np
from vispy import app, scene
from vispy.scene.visuals import Text, Markers, Mesh, Line
from vispy.geometry import create_sphere
from typing import Callable, Literal, Dict, Any
import threading
from dataclasses import dataclass, field

from src.simulation.simulation import Simulation
from src.utils import LIGHTER_GRAY
from src.simulation.simulation import ThermodynamicsState
from .color_picker import ColorPicker

@dataclass
class CameraConfig:
    azimuth: float = 30.0
    elevation: float = 23.0
    auto_rotate: bool = False
    rotation_speed: float = 0.10
    distance: float = 2.5

    def new_distance(self, new_distance: float) -> "CameraConfig":
        return CameraConfig(
            azimuth=self.azimuth,
            elevation=self.elevation,
            auto_rotate=self.auto_rotate,
            rotation_speed=self.rotation_speed,
            distance=new_distance
        )

CAMERA_DEFAULT = CameraConfig()
CAMERA_FACING_X = CameraConfig(azimuth=0.0, elevation=0.0)

@dataclass
class RendererConfig:
    target_fps: float = 60.0
    title: str = "Ideal Gas Simulation"
    window_size: tuple = (1024, 768)
    render_mode: Literal["points", "spheres"] = "points"
    color_mode: Literal["by_type", "by_energy", "by_speed", "by_relative_energy"] = "by_type"
    color_map_name: str = "coolwarm"
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    max_dims: np.ndarray | None = None
    show_info: bool = True

class Renderer:

    def __init__(   
        self,
        simulation: Simulation,
        config: RendererConfig = RendererConfig(),
    ):
        self.simulation = simulation
        self.target_fps = config.target_fps
        self.title = config.title
        self.window_size = config.window_size
        self.render_mode = config.render_mode
        self.color_picker = ColorPicker(mode=config.color_mode, colormap_name=config.color_map_name)

        # State
        self.paused = False
        self.show_info = config.show_info

        # Auto-rotation
        self.auto_rotate = config.camera_config.auto_rotate
        self._auto_rotate_enabled = config.camera_config.auto_rotate # paused when mouse interacting
        self.rotation_speed = config.camera_config.rotation_speed  # Degrees per second (real time)
        self.azimuth = config.camera_config.azimuth
        self.elevation = config.camera_config.elevation
        self.camera_distance = config.camera_config.distance
        self._last_rotation_time: float | None = None

        # VisPy objects
        self._trajectory_lines: Dict[int, Line] = {}

        # Normalization for display
        if config.max_dims is None:
            max_dims = self.simulation.container.dimensions
        else:
            max_dims = config.max_dims
        self._fixed_max_dims = max_dims.copy()
        max_dim = np.max(max_dims)
        self._scale_factor = 1.0 / max_dim
        self._offset = -max_dims / 2

    MODES = ["points", "spheres"]

    @staticmethod
    def create_spheres_mesh(positions: np.ndarray, radii: np.ndarray, colors: np.ndarray, subdivisions: int = 1) -> tuple:
        # returns (vertices, faces, vertex_colors) for Mesh visual
        sphere_data = create_sphere(subdivisions=subdivisions, method='ico')
        template_verts = sphere_data.get_vertices()
        template_faces = sphere_data.get_faces()
        n_verts_per_sphere = len(template_verts)
        n_faces_per_sphere = len(template_faces)
        n_spheres = len(positions)
        all_verts = np.zeros((n_spheres * n_verts_per_sphere, 3), dtype=np.float32)
        all_faces = np.zeros((n_spheres * n_faces_per_sphere, 3), dtype=np.uint32)
        all_colors = np.zeros((n_spheres * n_verts_per_sphere, 4), dtype=np.float32)

        for i in range(n_spheres):
            v_start = i * n_verts_per_sphere
            v_end = v_start + n_verts_per_sphere
            f_start = i * n_faces_per_sphere
            f_end = f_start + n_faces_per_sphere

            all_verts[v_start:v_end] = template_verts * radii[i] + positions[i] # Scale and translate vertices
            all_faces[f_start:f_end] = template_faces + v_start # pour ne pas relier des points de sphères différentes
            all_colors[v_start:v_end] = colors[i]
        return all_verts, all_faces, all_colors

    @staticmethod
    def create_box_edges(dims: np.ndarray) -> np.ndarray:
        #returns array of shape (24, 3) with line segment endpoints
        w, h, d = dims / 2
        vertices = np.array([
            [-w, -h, -d], [+w, -h, -d], [+w, +h, -d], [-w, +h, -d],  # back face
            [-w, -h, +d], [+w, -h, +d], [+w, +h, +d], [-w, +h, +d],  # front face
        ])
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # front face
            (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
        ]

        segments = []
        for i, j in edges:
            segments.append(vertices[i]) # début d'une arrête (convention)
            segments.append(vertices[j]) # fin d'une arrête
        return np.array(segments, dtype=np.float32)

    def _update_scaling_offset(self):
        self._offset = -self.simulation.container.dimensions / 2

    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        # Update offset for current container size (keeps particles centered)
        self._update_scaling_offset()
        return (positions + self._offset) * self._scale_factor

    def cycle_render_mode(self):
        idx = Renderer.MODES.index(self.render_mode)
        self.render_mode = Renderer.MODES[(idx + 1) % len(Renderer.MODES)]
        self.update_particles()

    def toggle_show_info(self):
        self.show_info = not self.show_info
        if self._info_text:
            self._info_text.visible = self.show_info

    def toggle_auto_rotate(self, mouse_interacting: bool = False):
        self.auto_rotate = not self.auto_rotate
        self._auto_rotate_enabled = self.auto_rotate
        self._last_rotation_time = None # Reset rotation time to avoid jump
        if self._view is not None:
            self.azimuth = self._view.camera.azimuth
        if mouse_interacting:
            self._auto_rotate_enabled = False

    def block_rotation(self): # because of mouse interaction
        if self.auto_rotate:
            self._auto_rotate_enabled = False
            if self._view is not None:
                self.azimuth = self._view.camera.azimuth

    def unblock_rotation(self): # because of mouse interaction
        if self.auto_rotate:
            if self._view is not None: # Sync azimuth with current camera position before resuming
                self.azimuth = self._view.camera.azimuth
            self._last_rotation_time = None # Reset rotation time to avoid jump
            self._auto_rotate_enabled = True

    def create_visuals(self):
        self._update_scaling_offset()

        # Container wireframe (just edges, no faces)
        dims = self.simulation.container.dimensions * self._scale_factor
        edge_points = Renderer.create_box_edges(dims)
        self._container_lines = Line(
            pos=edge_points,
            color=LIGHTER_GRAY.to_4_tuple(0.8),
            width=2,
            connect='segments',
            parent=self._view.scene,
        )

        self._info_text = Text(
            "Initializing...",
            color=(0.7, 0.7, 0.7, 1.0),
            font_size=13,
            anchor_x="left",
            anchor_y="bottom",
            parent=self._canvas.scene,
        )
        self._info_text.pos = (10, self.window_size[1] - 15) # near bottom left

        self._particles_markers = Markers(parent=self._view.scene)
        self._particles_mesh = Mesh(parent=self._view.scene, shading='smooth')
        self.update_particles()

    def update_canvas(self):
        self._canvas.update()

    def _sphere_pixel_sizes(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        if len(positions) == 0:
            return np.empty(0, dtype=np.float32)
        tr = self._particles_markers.transforms.get_transform(
            map_from="visual",
            map_to="canvas",
        )
        centers_screen = tr.map(positions)
        if centers_screen.shape[1] == 4:
            centers_screen = centers_screen[:, :3] / centers_screen[:, 3:4]
        offsets = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        pixel_radii_candidates = []
        for offset in offsets:
            shifted_positions = positions + radii[:, None] * offset[None, :]
            shifted_screen = tr.map(shifted_positions)
            if shifted_screen.shape[1] == 4:
                shifted_screen = shifted_screen[:, :3] / shifted_screen[:, 3:4]
            pixel_radius = np.linalg.norm(
                shifted_screen[:, :2] - centers_screen[:, :2],
                axis=1,
            )
            pixel_radii_candidates.append(pixel_radius)
        pixel_radii = np.max(np.stack(pixel_radii_candidates, axis=0), axis=0)
        pixel_diameters = 2.0 * pixel_radii
        return pixel_diameters.astype(np.float32)

    def update_particles(self):
        gas = self.simulation.gas
        if gas.count == 0:
            if self._particles_markers is not None:
                self._particles_markers.visible = False
            if self._particles_mesh is not None:
                self._particles_mesh.visible = False
            return

        positions = self._normalize_positions(gas.positions)
        radii = gas.radii * self._scale_factor
        colors = self.color_picker.get_colors(gas)

        if self.render_mode == "points":
            if self._particles_mesh is not None:
                self._particles_mesh.visible = False
            if self._particles_markers is not None:
                self._particles_markers.visible = True
                sizes = self._sphere_pixel_sizes(positions, radii)
                self._particles_markers.set_data(
                    positions,
                    face_color=colors,
                    edge_color=None,
                    size=sizes,
                    edge_width=0,
                )
        else:
            if self._particles_markers is not None:
                self._particles_markers.visible = False
            if self._particles_mesh is not None:
                self._particles_mesh.visible = True
                verts, faces, vert_colors = Renderer.create_spheres_mesh(
                    positions, radii, colors, subdivisions=1
                )
                self._particles_mesh.set_data(
                    vertices=verts,
                    faces=faces,
                    vertex_colors=vert_colors,
                )
                self._particles_mesh.shading_filter.light_dir = (0.0, 0, -1.0)

    def update_container(self):
        if self._container_lines is None:
            return
        self._update_scaling_offset()
        dims = self.simulation.container.dimensions * self._scale_factor
        edge_points = Renderer.create_box_edges(dims)
        self._container_lines.set_data(pos=edge_points)

    def update_info(self, state: ThermodynamicsState):
        if self._info_text is None or not self.show_info:
            return

        rotate_status = "ON" if self.auto_rotate else "OFF"
        color_mode = self.color_picker.mode
        if color_mode == "by_type":
            color_desc = "type"
        elif color_mode == "by_energy":
            color_desc = "energy"
        elif color_mode == "by_relative_energy":
            color_desc = "relative energy"
        else:
            color_desc = "speed"
        text = (
            f"Particles: {state.n_particles}\n"
            f"t = {self.simulation.time:.2e} s\n"
            f"T = {state.temperature:.1f} K\n"
            f"p = {state.pressure:.2e} Pa\n"
            f"MFP = {state.mean_free_path:.2e} m\n"
            f"pV/nRT = {state.pv_nkt:.3f}\n"
            f"\n"
            f"Render: {self.render_mode} (R to cycle)\n"
            f"Color: {color_desc} (C to cycle)\n"
            f"Rotation: {rotate_status} (A to toggle)"
        )
        self._info_text.text = text

    def update_camera_rotation(self):
        if self._auto_rotate_enabled and self._view is not None:
            current_time = time.time()
            if self._last_rotation_time is not None:
                # Compute rotation based on real elapsed time (0.15 deg/frame * 60 frames/sec = 9 deg/sec)
                elapsed = current_time - self._last_rotation_time
                degrees_per_second = self.rotation_speed * self.target_fps
                self.azimuth = (self.azimuth + degrees_per_second * elapsed) % 360
                self._view.camera.azimuth = self.azimuth
            self._last_rotation_time = current_time

    def toggle_trajectory(self, idx: int):
        self.simulation.particle_tracker.toggle(idx, self.simulation.gas.positions[idx])
        if idx not in self.simulation.particle_tracker.trajectories:
            line = self._trajectory_lines.pop(idx, None)
            if line:
                line.parent = None

    def update_trajectories(self):
        current_positions = self.simulation.gas.positions
        colors = self.color_picker.get_colors(self.simulation.gas)
        for idx, trajectory in self.simulation.particle_tracker.trajectories.items():
            if idx not in self._trajectory_lines:
                line = Line(
                    parent=self._view.scene,
                    width=2,
                    method='gl',
                )
                self._trajectory_lines[idx] = line
            if len(trajectory) >= 1:
                points = list(trajectory) + [current_positions[idx]]
                points = np.array(points)
                points_normalized = self._normalize_positions(points)
                self._trajectory_lines[idx].set_data(pos=points_normalized, color=colors[idx])

    def update_all(self):
        self.update_camera_rotation()
        self.update_particles()
        if not self.paused:
            self.update_container()
            self.update_trajectories()
            self.update_info(self.simulation.thermodynamics_state)
        self.update_canvas()

    def start(
        self, 
        on_key_press: Callable[[Any], None], 
        on_mouse_press: Callable[[Any], None],
        on_mouse_release: Callable[[Any], None],
        on_timer: Callable[[Any], None],
        scenario: Callable[[], None] | None = None
    ):
        self._canvas = scene.SceneCanvas(
            keys=None, # disable default key handlers
            size=self.window_size,
            title=self.title,
            show=True,
            bgcolor='black',
        )
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.TurntableCamera(
            fov=45,
            distance=self.camera_distance,
            elevation=self.elevation,
            azimuth=self.azimuth,
            up="+z",
        )

        self.create_visuals()
        self._canvas.events.key_press.connect(on_key_press)
        self._canvas.events.mouse_press.connect(on_mouse_press)
        self._canvas.events.mouse_release.connect(on_mouse_release)
        if scenario is not None:
            self._scenario_thread = threading.Thread(target=scenario, daemon=True)
            self._scenario_thread.start()
        self._timer = app.Timer(
            interval=1.0 / self.target_fps,
            connect=on_timer,
            start=True,
        )
        app.run()

    def stop(self):
        if self._timer is not None:
            self._timer.stop()
        if self._canvas is not None:
            self._canvas.close()
        app.quit()

    def pick_particle(self, click_pos: np.ndarray) -> int | None:
        # click_pos: 2d position in canvas coordinates (pixels)
        # returns a potential particle near this position, or None if no particle is close enough
        positions_3d = self._normalize_positions(self.simulation.gas.positions)
        tr = self._particles_mesh.transforms.get_transform(
            map_from="visual",
            map_to="canvas",
        )
        positions_screen = tr.map(positions_3d)
        if positions_screen.shape[1] == 4:
            positions_screen = positions_screen[:, :3] / positions_screen[:, 3:4]
        positions_2d = positions_screen[:, :2]
        distances = np.linalg.norm(positions_2d - click_pos, axis=1)

        threshold = 12.0
        candidates = np.where(distances < threshold)[0]
        if len(candidates) == 0:
            return None
        return candidates[np.argmin(distances[candidates])].item()

    # Offscreen

    def init_offscreen(self, resolution: tuple):
        self._offscreen_canvas = scene.SceneCanvas(
            keys=None,
            size=resolution,
            show=False,
            bgcolor='black',
        )
        self._offscreen_view = self._offscreen_canvas.central_widget.add_view()
        self._offscreen_view.camera = scene.TurntableCamera(
            fov=45,
            distance=self.camera_distance,
            elevation=self.elevation,
            azimuth=self.azimuth,
            up="+z",
        )
        # Temporarily swap _canvas and _view to reuse create_visuals
        self._canvas = self._offscreen_canvas
        self._view = self._offscreen_view
        self.create_visuals()

    def render_offscreen_frame(self) -> np.ndarray:
        self.update_container()
        self.update_particles()
        self.update_trajectories()
        self.update_info(self.simulation.thermodynamics_state)
        self._offscreen_canvas.update()
        self._offscreen_canvas.app.process_events()
        return self._offscreen_canvas.render()

    def advance_rotation(self, dt: float): 
        # for offscreen view
        # advance by dt instead of elapsed time
        if self.auto_rotate and self._offscreen_view is not None:
            degrees_per_second = self.rotation_speed * self.target_fps
            self.azimuth = (self.azimuth + degrees_per_second * dt) % 360
            self._offscreen_view.camera.azimuth = self.azimuth

    def cleanup_offscreen(self):
        if self._offscreen_canvas is not None:
            self._offscreen_canvas.close()
            self._offscreen_canvas = None
            self._offscreen_view = None