"""Boundary-condition helpers for 2D simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirichletBoundary2D:
    """Homogeneous Dirichlet boundary condition with psi = 0 on the edge."""

    def apply(self, field: np.ndarray) -> np.ndarray:
        bounded = field.copy()
        bounded[0, :] = 0.0
        bounded[-1, :] = 0.0
        bounded[:, 0] = 0.0
        bounded[:, -1] = 0.0
        return bounded
