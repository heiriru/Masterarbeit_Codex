"""Potential models for the 2D Schrodinger equation."""

from __future__ import annotations

import numpy as np

from src.grid import Grid2D


def free_potential(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def harmonic_potential(x: np.ndarray, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
    return 0.5 * (omega**2) * (x**2 + y**2)


def build_potential(grid: Grid2D, params: object) -> np.ndarray:
    """Build a potential field from a model-parameter object."""

    x, y = grid.mesh
    potential_type = getattr(params, "potential_type", "free")
    if potential_type == "free":
        return free_potential(x, y)
    if potential_type == "harmonic":
        omega = getattr(params, "omega", 1.0)
        return harmonic_potential(x, y, omega=omega)
    raise ValueError(f"Unsupported potential type: {potential_type}")
