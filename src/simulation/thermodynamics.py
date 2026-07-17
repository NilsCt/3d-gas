from dataclasses import dataclass
import numpy as np

@dataclass
class ThermodynamicsState:

    temperature: float = 0
    pressure: float = 0
    volume: float = 0
    n_particles: int = 0
    total_kinetic_energy: float = 0
    mean_kinetic_energy: float = 0
    pv_nkt: float = 0  # roughly 1 if ideal gas law holds
    rms_speed: float = 0
    momentum: np.ndarray | None = None
    mean_free_path: float = 0