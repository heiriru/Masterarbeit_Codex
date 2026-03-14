"""Default simulation parameters for the 2D Schrodinger equation."""

from dataclasses import dataclass


@dataclass
class SimulationParameters:
    nx: int = 128
    ny: int = 128
    lx: float = 20.0
    ly: float = 20.0
    dt: float = 0.001
    t_final: float = 0.1
    sigma: float = 1.0
    x0: float = -3.0
    y0: float = 0.0
    kx: float = 1.0
    ky: float = 0.0
    potential_type: str = "free"
    omega: float = 0.15
    log_every: int = 10
