"""Physical model definitions for Schrödinger-type simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.boundary import DirichletBoundary2D
from src.grid import Grid2D
from src.operators import kinetic_operator


@dataclass
class LinearSchrodingerModel:
    """Linear 2D Schrödinger model with fixed potential and boundary condition."""

    grid: Grid2D
    potential: np.ndarray
    boundary: DirichletBoundary2D

    def apply_boundary(self, psi: np.ndarray) -> np.ndarray:
        return self.boundary.apply(psi)

    def rhs(self, psi: np.ndarray, t: float) -> np.ndarray:
        del t
        psi = self.apply_boundary(psi)
        h_psi = kinetic_operator(psi, self.grid.dx, self.grid.dy, self.boundary) + self.potential * psi
        return self.apply_boundary(-1j * h_psi)

    def particle_number(self, psi: np.ndarray) -> float:
        psi = self.apply_boundary(psi)
        density = np.abs(psi) ** 2
        return float(np.sum(density) * self.grid.dx * self.grid.dy)

    def energy(self, psi: np.ndarray) -> float:
        psi = self.apply_boundary(psi)
        h_psi = kinetic_operator(psi, self.grid.dx, self.grid.dy, self.boundary) + self.potential * psi
        value = np.vdot(psi, h_psi) * self.grid.dx * self.grid.dy
        return float(np.real(value))

    def expectation_values(self, psi: np.ndarray) -> tuple[float, float]:
        psi = self.apply_boundary(psi)
        density = np.abs(psi) ** 2
        cell_area = self.grid.dx * self.grid.dy
        x, y = self.grid.mesh
        x_mean = float(np.sum(x * density) * cell_area)
        y_mean = float(np.sum(y * density) * cell_area)
        return x_mean, y_mean
