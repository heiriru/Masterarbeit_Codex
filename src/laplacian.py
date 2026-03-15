"""Finite-difference Laplacian operators."""

from __future__ import annotations

import numpy as np


def laplacian_2d_dirichlet(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute the 2D Laplacian with homogeneous Dirichlet boundary conditions.

    The wavefunction is assumed to vanish on the outer boundary.
    """

    laplacian = np.zeros_like(field, dtype=field.dtype)
    laplacian[1:-1, 1:-1] = (
        (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dx * dx)
        + (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dy * dy)
    )
    return laplacian
