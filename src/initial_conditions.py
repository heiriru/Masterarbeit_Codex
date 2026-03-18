"""Initial-condition helpers for wavefunction-based simulations."""

from __future__ import annotations

import numpy as np

from src.boundary import DirichletBoundary2D
from src.grid import Grid2D


def make_gaussian_packet(
    grid: Grid2D,
    boundary: DirichletBoundary2D,
    sigma: float,
    x0: float = 0.0,
    y0: float = 0.0,
    kx: float = 0.0,
    ky: float = 0.0,
    target_particle_number: float = 1.0,
) -> np.ndarray:
    """Create a Gaussian wave packet with configurable particle number."""

    x, y = grid.mesh
    envelope = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma**2))
    phase = np.exp(1j * (kx * x + ky * y))
    psi = boundary.apply(envelope * phase)
    particle_number = np.sum(np.abs(psi) ** 2) * grid.dx * grid.dy
    if particle_number <= 0.0:
        raise ValueError("Initial condition has zero particle number after applying the boundary.")
    normalized = psi / np.sqrt(particle_number)
    return np.sqrt(target_particle_number) * normalized
