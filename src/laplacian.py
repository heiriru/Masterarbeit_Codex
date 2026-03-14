"""Finite-difference Laplacian operators."""

from __future__ import annotations

import numpy as np


def laplacian_2d_periodic(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute the 2D Laplacian using second-order centered differences.

    The operator assumes periodic boundary conditions in both directions.
    """

    d2x = (np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0)) / (dx * dx)
    d2y = (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)) / (dy * dy)
    return d2x + d2y
