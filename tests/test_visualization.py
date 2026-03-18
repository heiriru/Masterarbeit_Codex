import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.diagnostics import Diagnostics
from src.grid import Grid2D
from src.visualization import plot_simulation_summary


def test_plot_simulation_summary_uses_drift_without_offset_ambiguity():
    grid = Grid2D(nx=16, ny=16, lx=4.0, ly=4.0)
    potential = np.zeros(grid.shape)
    diagnostics = Diagnostics(
        times=[0.0, 1.0, 2.0],
        particle_numbers=[1.0, 1.0001, 0.9999],
        energies=[2.0, 2.01, 1.99],
        x_expectation=[0.0, 0.1, 0.2],
        y_expectation=[0.0, -0.1, -0.2],
        snapshot_times=[0.0, 1.0, 2.0],
        snapshots=[np.ones(grid.shape, dtype=complex) for _ in range(3)],
    )

    _, diagnostics_fig, _ = plot_simulation_summary(
        grid=grid,
        potential=potential,
        diagnostics=diagnostics,
        output_path="Results/test_summary.png",
    )

    diag_axes = diagnostics_fig.axes[:3]
    assert diag_axes[0].get_ylabel() == "N(t) - N(0)"
    assert diag_axes[1].get_ylabel() == "E(t) - E(0)"
    plt.close("all")
