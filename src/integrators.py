"""Time integration methods for wavefunction simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models import LinearSchrodingerModel


@dataclass(frozen=True)
class RK4Integrator:
    """Classical 4th-order Runge-Kutta integrator."""

    name: str = "rk4"

    def step(self, model: LinearSchrodingerModel, psi: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = model.rhs(psi, t)
        k2 = model.rhs(psi + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = model.rhs(psi + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = model.rhs(psi + dt * k3, t + dt)
        psi_next = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return model.apply_boundary(psi_next)


@dataclass(frozen=True)
class CrankNicolsonIntegrator:
    """Placeholder for a future Crank-Nicolson integrator."""

    name: str = "crank_nicolson"

    def step(self, model: LinearSchrodingerModel, psi: np.ndarray, t: float, dt: float) -> np.ndarray:
        del model, psi, t, dt
        raise NotImplementedError("Crank-Nicolson is planned but not implemented yet.")


def build_integrator(name: str) -> RK4Integrator | CrankNicolsonIntegrator:
    normalized_name = name.lower()
    if normalized_name == "rk4":
        return RK4Integrator()
    if normalized_name in {"crank_nicolson", "cn"}:
        return CrankNicolsonIntegrator()
    raise ValueError(f"Unsupported integrator: {name}")
