"""Grid utilities for 2D simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    """Uniform 2D Cartesian grid."""

    nx: int
    ny: int
    lx: float
    ly: float

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def x(self) -> np.ndarray:
        return np.linspace(-self.lx / 2, self.lx / 2, self.nx, endpoint=False)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(-self.ly / 2, self.ly / 2, self.ny, endpoint=False)

    @property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)
