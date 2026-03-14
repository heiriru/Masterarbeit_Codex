"""Main simulation workflow for the 2D Schrodinger equation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.grid import Grid2D
from src.laplacian import laplacian_2d_periodic
from src.olver import rk4_step

logger = logging.getLogger(__name__)


@dataclass
class Diagnostics:
    times: list[float] = field(default_factory=list)
    norms: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)


class SchrodingerSimulation2D:
    """2D time-dependent Schrodinger equation solver in normalized units."""

    def __init__(self, grid: Grid2D, potential: np.ndarray, dt: float) -> None:
        self.grid = grid
        self.potential = potential
        self.dt = dt

    def rhs(self, psi: np.ndarray) -> np.ndarray:
        kinetic = -0.5 * laplacian_2d_periodic(psi, self.grid.dx, self.grid.dy)
        h_psi = kinetic + self.potential * psi
        return -1j * h_psi

    def norm(self, psi: np.ndarray) -> float:
        density = np.abs(psi) ** 2
        return float(np.sum(density) * self.grid.dx * self.grid.dy)

    def energy(self, psi: np.ndarray) -> float:
        h_psi = -0.5 * laplacian_2d_periodic(psi, self.grid.dx, self.grid.dy) + self.potential * psi
        val = np.vdot(psi, h_psi) * self.grid.dx * self.grid.dy
        return float(np.real(val))

    def run(
        self,
        psi0: np.ndarray,
        t_final: float,
        log_every: int = 10,
        diagnostics: Diagnostics | None = None,
    ) -> tuple[np.ndarray, Diagnostics]:
        n_steps = int(round(t_final / self.dt))
        psi = psi0.copy()
        diag = diagnostics or Diagnostics()

        for step in range(n_steps + 1):
            time = step * self.dt
            if step % log_every == 0 or step == n_steps:
                nrm = self.norm(psi)
                eng = self.energy(psi)
                diag.times.append(time)
                diag.norms.append(nrm)
                diag.energies.append(eng)
                logger.info("step=%d t=%.5f norm=%.8f energy=%.8f", step, time, nrm, eng)

            if step < n_steps:
                psi = rk4_step(psi, self.dt, self.rhs)

        return psi, diag


def make_gaussian_packet(
    grid: Grid2D,
    sigma: float,
    x0: float = 0.0,
    y0: float = 0.0,
    kx: float = 0.0,
    ky: float = 0.0,
) -> np.ndarray:
    """Create a normalized Gaussian wave packet."""

    x, y = grid.mesh
    envelope = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma**2))
    phase = np.exp(1j * (kx * x + ky * y))
    psi = envelope * phase
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * grid.dx * grid.dy)
    return psi / norm
