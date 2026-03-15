"""Run a free-particle Gaussian wave packet simulation."""

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.simulation_parameters import SimulationParameters
from src.grid import Grid2D
from src.potential import free_potential
from src.simulation import SchrodingerSimulation2D, make_gaussian_packet
from src.visualization import create_density_gif, create_phase_gif, plot_simulation_summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    params = SimulationParameters()
    results_dir = PROJECT_ROOT / "Results"
    grid = Grid2D(params.nx, params.ny, params.lx, params.ly)
    x, y = grid.mesh

    potential = free_potential(x, y)
    psi0 = make_gaussian_packet(
        grid,
        sigma=params.sigma,
        x0=params.x0,
        y0=params.y0,
        kx=params.kx,
        ky=params.ky,
    )

    sim = SchrodingerSimulation2D(grid=grid, potential=potential, dt=params.dt)
    n_steps = int(round(params.t_final / params.dt))
    snapshot_every = max(1, n_steps // 120)
    _, diag = sim.run(
        psi0=psi0,
        t_final=params.t_final,
        log_every=params.log_every,
        snapshot_every=snapshot_every,
    )

    plot_simulation_summary(
        grid=grid,
        potential=potential,
        diagnostics=diag,
        output_path=results_dir / "free_gaussian_overview.png",
    )
    create_density_gif(
        grid=grid,
        diagnostics=diag,
        output_path=results_dir / "free_gaussian_density.gif",
    )
    create_phase_gif(
        grid=grid,
        diagnostics=diag,
        output_path=results_dir / "free_gaussian_phase.gif",
    )

    print(f"Final norm: {diag.norms[-1]:.8f}")
    print(f"Final energy: {diag.energies[-1]:.8f}")
    print(
        "Saved figures: Results/free_gaussian_overview.png, "
        "Results/free_gaussian_overview_diagnostics.png, "
        "Results/free_gaussian_overview_potential.png, "
        "Results/free_gaussian_density.gif, "
        "Results/free_gaussian_phase.gif"
    )
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
