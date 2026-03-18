"""Diagnostics storage and helper views for time-dependent simulations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Diagnostics:
    times: list[float] = field(default_factory=list)
    particle_numbers: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    x_expectation: list[float] = field(default_factory=list)
    y_expectation: list[float] = field(default_factory=list)
    snapshot_times: list[float] = field(default_factory=list)
    snapshots: list[np.ndarray] = field(default_factory=list)

    @property
    def norms(self) -> list[float]:
        """Backward-compatible alias for particle number history."""

        return self.particle_numbers

    @property
    def particle_number_drift(self) -> list[float]:
        if not self.particle_numbers:
            return []
        reference = self.particle_numbers[0]
        return [value - reference for value in self.particle_numbers]

    @property
    def energy_drift(self) -> list[float]:
        if not self.energies:
            return []
        reference = self.energies[0]
        return [value - reference for value in self.energies]
