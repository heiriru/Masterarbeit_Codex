"""Simulation orchestration for 2D wavefunction models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.diagnostics import Diagnostics
from src.integrators import CrankNicolsonIntegrator, RK4Integrator
from src.models import LinearSchrodingerModel

logger = logging.getLogger(__name__)


@dataclass
class SimulationRunner2D:
    """Coordinate time integration, diagnostics, and snapshot collection."""

    model: LinearSchrodingerModel
    integrator: RK4Integrator | CrankNicolsonIntegrator
    dt: float

    def run(
        self,
        psi0: np.ndarray,
        t_final: float,
        log_every: int = 10,
        diagnostics: Diagnostics | None = None,
        snapshot_every: int | None = None,
    ) -> tuple[np.ndarray, Diagnostics]:
        n_steps = int(round(t_final / self.dt))
        psi = self.model.apply_boundary(psi0)
        diag = diagnostics or Diagnostics()

        for step in range(n_steps + 1):
            time = step * self.dt
            if step % log_every == 0 or step == n_steps:
                particle_number = self.model.particle_number(psi)
                energy = self.model.energy(psi)
                x_mean, y_mean = self.model.expectation_values(psi)
                diag.times.append(time)
                diag.particle_numbers.append(particle_number)
                diag.energies.append(energy)
                diag.x_expectation.append(x_mean)
                diag.y_expectation.append(y_mean)
                logger.info(
                    "step=%d t=%.5f particle_number=%.8f energy=%.8f <x>=%.5f <y>=%.5f",
                    step,
                    time,
                    particle_number,
                    energy,
                    x_mean,
                    y_mean,
                )

            if snapshot_every is not None and (step % snapshot_every == 0 or step == n_steps):
                diag.snapshot_times.append(time)
                diag.snapshots.append(psi.copy())

            if step < n_steps:
                psi = self.model.apply_boundary(self.integrator.step(self.model, psi, time, self.dt))

        return psi, diag


# Backward-compatible alias while the repo migrates to the new naming.
SchrodingerSimulation2D = SimulationRunner2D
