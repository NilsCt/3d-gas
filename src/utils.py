from dataclasses import dataclass
from typing import Tuple
import numpy as np


# All units are SI

k_b: float = 1.380649e-23  # Boltzmann constant (J/K)
N_A: float = 6.02214076e23  # Avogadro's number (mol^-1)


@dataclass(frozen=True)
class Color:
    red: float
    green: float
    blue: float

WHITE: Color = Color(1.0, 1.0, 1.0)
PINK: Color = Color(1.0, 0.8, 0.8)
LIGHT_BLUE: Color = Color(0.5, 0.5, 1.0)
PRETTY_RED: Color = Color(1.0, 0.3, 0.3)
LIGHT_GREEN: Color = Color(0.5, 1.0, 0.5)
GRAY: Color = Color(0.6, 0.6, 0.6)
ORANGE: Color = Color(1.0, 0.5, 0.0)
YELLOW: Color = Color(0.8, 0.8, 0.2)


@dataclass(frozen=True)
class ParticleType:
    name: str
    mass: float
    radius: float
    color: Color

def _atomic_mass_to_kg(atomic_mass: float) -> float:
    return atomic_mass * 1.66053906660e-27

PARTICLE_PRESETS: dict[str, ParticleType] = {
    "H2": ParticleType(
        name="H2",
        mass=_atomic_mass_to_kg(2.016),
        radius=1.20e-10,  # Van der Waals radius
        color=WHITE,  # White
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
    bo:str
    #TODO


def concatenate_vectors(va : np.ndarray, vb : np.ndarray) -> np.ndarray:
    return np.vstack([va, vb]) if len(va) > 0 else vb

def maxwell_boltzmann_speed_pdf(velocities: np.ndarray, temperature: float, mass: float) -> np.ndarray:
    a = mass / (2 * k_b * temperature)
    coeff = 4 * np.pi * (a / np.pi) ** 1.5
    return coeff * velocities**2 * np.exp(-a * velocities**2)

def temperature_to_mean_speed(temperature: float, mass: float) -> float:
    return np.sqrt(8 * k_b * temperature / (np.pi * mass))

def temperature_to_rms_speed(temperature: float, mass: float) -> float:
    return np.sqrt(3 * k_b * temperature / mass)

def mean_free_path(n_density: float, diameter: float) -> float:
    return 1.0 / (np.sqrt(2) * np.pi * diameter**2 * n_density)