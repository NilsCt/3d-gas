import numpy as np
from typing import Literal

from ..simulation.gas import Gas

# Predefined colormaps (cold to hot)
COLORMAPS = {
    "plasma": np.array([
        [0.050383, 0.029803, 0.527975],
        [0.254627, 0.013882, 0.615419],
        [0.417642, 0.000564, 0.653657],
        [0.562738, 0.051545, 0.641509],
        [0.692840, 0.165141, 0.564522],
        [0.798216, 0.280197, 0.469538],
        [0.881443, 0.392529, 0.383229],
        [0.949217, 0.517763, 0.295662],
        [0.988260, 0.652325, 0.211364],
        [0.988648, 0.809579, 0.145357],
        [0.940015, 0.975158, 0.131326],
    ]),
    "viridis": np.array([
        [0.267004, 0.004874, 0.329415],
        [0.282327, 0.140926, 0.457517],
        [0.253935, 0.265254, 0.529983],
        [0.206756, 0.371758, 0.553117],
        [0.163625, 0.471133, 0.558148],
        [0.127568, 0.566949, 0.550556],
        [0.134692, 0.658636, 0.517649],
        [0.266941, 0.748751, 0.440573],
        [0.477504, 0.821444, 0.318195],
        [0.741388, 0.873449, 0.149561],
        [0.993248, 0.906157, 0.143936],
    ]),
    "coolwarm": np.array([
        [0.230, 0.299, 0.754],  # Deep blue
        [0.280, 0.360, 0.790],
        [0.330, 0.420, 0.825],
        [0.385, 0.480, 0.855],
        [0.440, 0.540, 0.880],
        [0.500, 0.600, 0.900],
        [0.565, 0.665, 0.915],
        [0.635, 0.735, 0.930],
        [0.710, 0.800, 0.945],
        [0.800, 0.870, 0.960],
        [0.950, 0.950, 0.950],  # White (middle)
        [0.960, 0.870, 0.800],
        [0.945, 0.790, 0.700],
        [0.925, 0.710, 0.600],
        [0.900, 0.630, 0.505],
        [0.875, 0.550, 0.415],
        [0.845, 0.470, 0.340],
        [0.815, 0.400, 0.280],
        [0.785, 0.345, 0.240],
        [0.760, 0.305, 0.225],
        [0.750, 0.300, 0.230],  # Deep red
    ]),
    "hot": np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.6, 0.0, 0.0],
        [0.9, 0.0, 0.0],
        [1.0, 0.3, 0.0],
        [1.0, 0.6, 0.0],
        [1.0, 0.9, 0.0],
        [1.0, 1.0, 0.3],
        [1.0, 1.0, 0.6],
        [1.0, 1.0, 1.0],
    ]),
}
type ColorMode = Literal["by_type", "by_energy", "by_speed", "by_relative_energy"]
MODES : list[ColorMode] = ["by_type", "by_energy", "by_speed", "by_relative_energy"]

class ColorPicker:

    def __init__(
        self,
        mode: ColorMode = "by_type",
        colormap_name: str = "coolwarm",
    ):
        self.mode: ColorMode = mode
        self.colormap_name: str = colormap_name
        self.colormap: np.ndarray = COLORMAPS.get(colormap_name, COLORMAPS["coolwarm"])

        self._auto_range: bool = True
        self._min: float | None = None # can be speed, energy, temperature
        self._max: float | None = None
        self._variation_strength: float = 1 # only for by_relative_energy mode
        # how much the color can vary around the base color
        # 0: all particle same color, 1: full variation
        self.easier_coefficient: float = 0.0

    def set_range(self, min: float, max: float, variation_strength: float = 1.0):
        self._min = min
        self._max = max
        self._auto_range = False
        self._variation_strength = np.clip(variation_strength, 0.0, 1.0)

    def set_auto_range(self, auto: bool = True):
        self._auto_range = auto
        if auto:
            self._min = None
            self._max = None

    def cycle_mode(self):
        idx = MODES.index(self.mode)
        self.mode = MODES[(idx + 1) % len(MODES)]

    def cycle_colormap(self):
        names = list(COLORMAPS.keys())
        idx = names.index(self.colormap_name)
        self.colormap_name = names[(idx + 1) % len(names)]
        self.colormap = COLORMAPS[self.colormap_name]

    def _add_alpha_channel(self, colors: np.ndarray) -> np.ndarray:
        rgba = np.ones((colors.shape[0], 4), dtype=np.float32)
        rgba[:, :3] = colors
        return rgba
    
    def get_colors(self, particles: Gas) -> np.ndarray:
        n = particles.count
        if n == 0:
            return np.empty((0, 4), dtype=np.float32)
        if self.mode == "by_type":
            return particles.colors
        if self.mode == "by_relative_energy":
            return self._relative_values_to_colors(particles.kinetic_energies)
        
        if self.mode == "by_energy":
            values = particles.kinetic_energies
        else:  # by_speed
            values = particles.speeds
        return self._values_to_colors(values)

    def _values_to_colors(self, values: np.ndarray) -> np.ndarray:
        # color determined independently for each particle based on its value between min and max
        if len(values) == 0:
            return np.empty((0, 4), dtype=np.float32)
        if self._auto_range or self._min is None or self._max is None:
            min = np.min(values) * (1+self.easier_coefficient)
            max = np.max(values) * (1-self.easier_coefficient)
        else:
            min = self._min
            max = self._max

        normalized = np.clip((values - min) / (max - min), 0, 1) # normalize to [0, 1]
        indices = normalized * (len(self.colormap) - 1) #  map to colormap indices
        idx_low = np.floor(indices).astype(int) # interpolate colors
        idx_high = np.ceil(indices).astype(int)
        idx_high = np.minimum(idx_high, len(self.colormap) - 1)
        t = np.expand_dims(indices - idx_low, axis=1)
        colors = (1 - t) * self.colormap[idx_low] + t * self.colormap[idx_high]
        return self._add_alpha_channel(colors.astype(np.float32))

    def _relative_values_to_colors(self, values: np.ndarray) -> np.ndarray:
        # color determined by global mean and local deviation
        n = len(values)
        if n == 0:
            return np.empty((0, 4), dtype=np.float32)
        if self._auto_range or self._min is None or self._max is None:
            min = np.min(values)
            max = np.max(values)
        else:
            min = self._min
            max = self._max
        mean = np.mean(values)

        global_score = np.clip((mean - min) / (max - min), 0, 1)
        sigma = np.std(values)
        deviations = (values - mean) / (3 * sigma)  # Normalize to ~[-1, 1]
        deviations = np.clip(deviations, -1, 1)
        combined_score = global_score + self._variation_strength * deviations * 0.5
        combined_score = np.clip(combined_score, 0, 1)

        indices = combined_score * (len(self.colormap) - 1)
        idx_low = np.floor(indices).astype(int)
        idx_high = np.minimum(idx_low + 1, len(self.colormap) - 1)
        t = np.expand_dims(indices - idx_low, axis=1)
        colors = (1 - t) * self.colormap[idx_low] + t * self.colormap[idx_high]
        return self._add_alpha_channel(colors.astype(np.float32))