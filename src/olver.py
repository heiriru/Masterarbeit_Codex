"""Time integration solvers."""

from __future__ import annotations

from typing import Callable

import numpy as np

Array = np.ndarray


def rk4_step(state: Array, dt: float, rhs: Callable[[Array], Array]) -> Array:
    """Advance one Runge-Kutta 4 step."""

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
