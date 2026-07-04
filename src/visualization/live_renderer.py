import time
import numpy as np
from vispy import app, scene
from vispy.scene import visuals
from vispy.geometry import create_sphere
from typing import Optional, Callable, Literal

from src.simulation.simulation import Simulation
from src.utils import Color, LIGHT_GRAY, LIGHTER_GRAY

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

class LiveRenderer:
    """
    Real-time 3D renderer for gas simulation using VisPy.
    """

    def __init__(
        self,
        simulation: Simulation,
        render_mode: Literal["points", "spheres"] = "points",
        color_mode: Literal["by_type", "by_energy", "by_speed"] = "by_type",
        window_size: tuple = (1024, 768),
        title: str = "Ideal Gas Simulation",
        auto_rotate: bool = True,
        rotation_speed: float = 0.10,
        camera_distance: float = 2.5,
        initial_speed_factor: float = 1.0,
        camera_azimuth: float = 45.0,
        camera_elevation: float = 25.0,
        max_dims: np.ndarray | None = None,

    ):
        self.simulation = simulation
        self.render_mode = render_mode
        self.window_size = window_size
        self.title = title

        # Color mapping
        #self.color_mapper = ColorMapper(mode=color_mode)

        # State
        self.paused = False
        self.speed_factor = initial_speed_factor
        self.show_info = True

        # Auto-rotation
        self.auto_rotate = auto_rotate
        self._auto_rotate_enabled = auto_rotate # paused when mouse interacting
        self._rotation_speed = rotation_speed  # Degrees per second (real time)
        self.azimuth = camera_azimuth
        self.elevation = camera_elevation
        self.camera_distance = camera_distance
        self._last_rotation_time: Optional[float] = None

        # VisPy objects
        self._canvas: Optional[scene.SceneCanvas] = None
        self._view: Optional[scene.ViewBox] = None
        self._particles_markers: Optional[visuals.Markers] = None  # For points mode
        self._particles_mesh: Optional[visuals.Mesh] = None  # For spheres mode
        self._container_lines: Optional[visuals.Line] = None
        #self._info_text: Optional[visuals.Text] = None
        #self._timer: Optional[app.Timer] = None

        # Offscreen rendering (reused to avoid context leaks)
        self._offscreen_canvas: Optional[scene.SceneCanvas] = None
        self._offscreen_view: Optional[scene.ViewBox] = None
        self._offscreen_size: Optional[tuple] = None

        # Callbacks
        self._container_lineson_frame: Optional[Callable[[int], None]] = None

        # Normalization for display
        if max_dims is None:
            max_dims = self.simulation.container.dimensions
        self._fixed_max_dims = max_dims.copy()
        max_dim = np.max(max_dims)
        self._scale_factor = 1.0 / max_dim
        self._offset = -max_dims / 2

    def update_scaling_offset(self):
        self._offset = -self.simulation.container.dimensions / 2

    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to display coordinates."""
        # Update offset for current container size (keeps particles centered)
        self.update_scaling_offset()
        return (positions + self._offset) * self._scale_factor
    
    def _create_visuals(self):
        """Create VisPy visual objects."""
        self.update_scaling_offset()

        # Container wireframe (just edges, no faces)
        dims = self.simulation.container.dimensions * self._scale_factor
        edge_points = create_box_edges(dims)
        self._container_lines = visuals.Line(
            pos=edge_points,
            color=LIGHTER_GRAY.to_4_tuple(0.8),
            width=2,
            connect='segments',
            parent=self._view.scene,
        )

        self._particles_markers = visuals.Markers(parent=self._view.scene)
        self._particles_mesh = visuals.Mesh(parent=self._view.scene, shading='smooth')
        self._update_particles()

        self._particles_mesh.shading_filter.light_dir = (0, 1, -1) # lighting for better colors
        self._particles_mesh.shading_filter.ambient_coefficient = 0.6

    def _update_particles(self):
        gas = self.simulation.gas
        if gas.count == 0:
            if self._particles_markers is not None:
                self._particles_markers.visible = False
            if self._particles_mesh is not None:
                self._particles_mesh.visible = False
            return

        positions = self._normalize_positions(gas.positions)
        radii = gas.radii * self._scale_factor
        colors = gas.colors

        if self.render_mode == "points":
            if self._particles_mesh is not None:
                self._particles_mesh.visible = False
            if self._particles_markers is not None:
                self._particles_markers.visible = True
                # Convert radii to pixel sizes (approximate)
                sizes = radii * 500
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
                verts, faces, vert_colors = create_spheres_mesh(
                    positions, radii, colors, subdivisions=1
                )
                self._particles_mesh.set_data(
                    vertices=verts,
                    faces=faces,
                    vertex_colors=vert_colors,
                )

    def _update_container(self):
        if self._container_lines is None:
            return
        self.update_scaling_offset()
        dims = self.simulation.container.dimensions * self._scale_factor
        edge_points = create_box_edges(dims)
        self._container_lines.set_data(pos=edge_points)

    def _update_camera_rotation(self):
        if self._auto_rotate_enabled and self._view is not None:
            current_time = time.time()
            if self._last_rotation_time is not None:
                # Compute rotation based on real elapsed time (0.15 deg/frame * 60 frames/sec = 9 deg/sec)
                elapsed = current_time - self._last_rotation_time
                degrees_per_second = self._rotation_speed * 60.0
                self.azimuth = (self.azimuth + degrees_per_second * elapsed) % 360
                self._view.camera.azimuth = self.azimuth
            self._last_rotation_time = current_time

    def _on_timer(self, event):
        # time callback, advance simulation and update display
        self._update_camera_rotation() # Update camera rotation (even when paused)

        if self.paused:
            self._canvas.update()
            return

        steps = max(1, int(self.speed_factor)) # Run multiple steps based on speed factor
        for _ in range(steps):
            self.simulation.complete_step() # TODO step en fonction du real elapsed time

        self._update_particles() # Update visuals
        self._canvas.update() # Request redraw

    def _on_key_press(self, event):
        if event.key == "Space":
            self.paused = not self.paused
        elif event.key == "+":
            self.speed_factor *= 1.5
        elif event.key == "-":
            self.speed_factor /= 1.5
        elif event.key == "R":
            modes = ["points", "spheres"]
            idx = modes.index(self.render_mode)
            self.render_mode = modes[(idx + 1) % len(modes)]
            self._update_particles()
        elif event.key == "A":
            self.auto_rotate = not self.auto_rotate
            self._auto_rotate_enabled = self.auto_rotate
        elif event.key == "Escape":
            self.stop()
        self._canvas.update()

    def _on_mouse_press(self, event):
        self._mouse_interacting = True
        if self.auto_rotate:
            self._auto_rotate_enabled = False
            if self._view is not None: # Sync azimuth with current camera position
                self.azimuth = self._view.camera.azimuth

    def _on_mouse_release(self, event):
        self._mouse_interacting = False
        if self.auto_rotate:
            if self._view is not None: # Sync azimuth with current camera position before resuming
                self.azimuth = self._view.camera.azimuth
            self._last_rotation_time = None # Reset rotation time to avoid jump
            self._auto_rotate_enabled = True

    def start(self):
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

        self._create_visuals()
        self._canvas.events.key_press.connect(self._on_key_press)
        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self._canvas.events.mouse_release.connect(self._on_mouse_release)
        self._timer = app.Timer(
            interval=1.0 / self.simulation.config.target_fps,
            connect=self._on_timer,
            start=True,
        )
        app.run() # blocking

    def stop(self):
        if self._timer is not None:
            self._timer.stop()
        if self._canvas is not None:
            self._canvas.close()
        app.quit()