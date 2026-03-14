"""Potential models for the 2D Schrodinger equation."""

from __future__ import annotations

import numpy as np


def free_potential(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def harmonic_potential(x: np.ndarray, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
    return 0.5 * (omega**2) * (x**2 + y**2)
