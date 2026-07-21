from dataclasses import dataclass
from typing import Tuple, Literal
import numpy as np

# All units are SI

k_b: float = 1.380649e-23  # Boltzmann constant (J/K)
N_A: float = 6.02214076e23  # Avogadro's number (mol^-1)

@dataclass(frozen=True)
class Color:
    red: float
    green: float
    blue: float

    def to_4_tuple(self, alpha: float = 1.0) -> Tuple[float, float, float, float]:
        return (self.red, self.green, self.blue, alpha)
    
    def to_4_array(self, alpha: float = 1.0) -> np.ndarray:
        return np.array([self.red, self.green, self.blue, alpha], dtype=np.float64)

WHITE: Color = Color(1.0, 1.0, 1.0)
PINK: Color = Color(1.0, 0.8, 0.8)
LIGHT_BLUE: Color = Color(0.3, 0.3, 0.8)
PRETTY_RED: Color = Color(1.0, 0.3, 0.3)
LIGHT_GREEN: Color = Color(0.5, 1.0, 0.5)
GRAY: Color = Color(0.6, 0.6, 0.6)
LIGHT_GRAY: Color = Color(0.7, 0.7, 0.7)
LIGHTER_GRAY: Color = Color(0.8, 0.8, 0.8)
ORANGE: Color = Color(1.0, 0.5, 0.0)
YELLOW: Color = Color(0.8, 0.8, 0.2)


@dataclass(frozen=True)
class ParticleType:
    name: str
    mass: float
    radius: float
    color: Color

    def new_color(self, color: Color) -> "ParticleType":
        return ParticleType(
            name=self.name,
            mass=self.mass,
            radius=self.radius,
            color=color
        )
    
    def bigger(self, factor: float, name: str, color: Color) -> "ParticleType":
        return ParticleType(
            name=name,
            mass=self.mass * factor**3,
            radius=self.radius * factor,
            color=color
        )

def _atomic_mass_to_kg(atomic_mass: float) -> float:
    return atomic_mass * 1.66053906660e-27

PARTICLE_PRESETS: dict[str, ParticleType] = {
    "H2": ParticleType(
        name="H2",
        mass=_atomic_mass_to_kg(2.016),
        radius=1.20e-10,  # VDW radius
        color=WHITE,
    ),
    "He": ParticleType(
        name="He",
        mass=_atomic_mass_to_kg(4.003),
        radius=1.40e-10,
        color=PINK,
    ),
    "N2": ParticleType(
        name="N2",
        mass=_atomic_mass_to_kg(28.014),
        radius=1.55e-10,
        color=LIGHT_BLUE,
    ),
    "O2": ParticleType(
        name="O2",
        mass=_atomic_mass_to_kg(31.998),
        radius=1.52e-10,
        color=PRETTY_RED,
    ),
    "Ar": ParticleType(
        name="Ar",
        mass=_atomic_mass_to_kg(39.948),
        radius=1.88e-10,
        color=LIGHT_GREEN,
    ),
    "CO2": ParticleType(
        name="CO2",
        mass=_atomic_mass_to_kg(44.01),
        radius=1.65e-10,
        color=GRAY,
    ),
    "Ne": ParticleType(
        name="Ne",
        mass=_atomic_mass_to_kg(20.180),
        radius=1.54e-10,
        color=ORANGE,
    ),
    "Kr": ParticleType(
        name="Kr",
        mass=_atomic_mass_to_kg(83.798),
        radius=2.02e-10,
        color=YELLOW,
    ),
}


@dataclass
class Config:
    lx: float
    ly: float
    lz: float
    max_lx: float | None = None
    max_ly: float | None = None
    max_lz: float | None = None
    cell_size_factor: float = 2.5
    cfl_factor: float = 0.3

    initial_dt: float = 1e-14
    dt_mode: Literal["fixed", "adaptive"] = "adaptive"
    dt_min: float = 1e-16
    dt_max: float = 1e-12
    check_overlap: bool = True
    max_overlap_retries: int = 100
    trajectory_max_points: int = 5
    pressure_window: int = 400
    mean_free_path_window: int = 400


def concatenate_vectors(va : np.ndarray, vb : np.ndarray) -> np.ndarray:
    return np.vstack([va, vb]) if len(va) > 0 else vb