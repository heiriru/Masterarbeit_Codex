"""Spatial operators for 2D Schrödinger-type simulations."""

from __future__ import annotations

import numpy as np

from src.boundary import DirichletBoundary2D


def laplacian_2d_dirichlet(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute the 2D Laplacian with homogeneous Dirichlet boundary conditions."""

    laplacian = np.zeros_like(field, dtype=field.dtype)
    laplacian[1:-1, 1:-1] = (
        (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dx * dx)
        + (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dy * dy)
    )
    return laplacian


def kinetic_operator(
    psi: np.ndarray,
    dx: float,
    dy: float,
    boundary: DirichletBoundary2D,
) -> np.ndarray:
    """Apply the kinetic operator -1/2 Laplacian to a field."""

    psi = boundary.apply(psi)
    return -0.5 * laplacian_2d_dirichlet(psi, dx, dy)
