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
from src.boundary import DirichletBoundary2D
from src.grid import Grid2D
from src.initial_conditions import make_gaussian_packet
from src.integrators import build_integrator
from src.models import LinearSchrodingerModel
from src.potential import build_potential
from src.simulation import SimulationRunner2D
from src.visualization import create_density_gif, create_phase_gif, plot_simulation_summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    params = SimulationParameters()
    results_dir = PROJECT_ROOT / params.output.results_dir
    grid = Grid2D(params.grid.nx, params.grid.ny, params.grid.lx, params.grid.ly)
    boundary = DirichletBoundary2D()
    potential = build_potential(grid, params.model)
    psi0 = make_gaussian_packet(
        grid,
        boundary=boundary,
        sigma=params.initial_condition.sigma,
        x0=params.initial_condition.x0,
        y0=params.initial_condition.y0,
        kx=params.initial_condition.kx,
        ky=params.initial_condition.ky,
        target_particle_number=params.initial_condition.target_particle_number,
    )

    model = LinearSchrodingerModel(grid=grid, potential=potential, boundary=boundary)
    integrator = build_integrator(params.solver.integrator)
    sim = SimulationRunner2D(model=model, integrator=integrator, dt=params.solver.dt)
    n_steps = int(round(params.solver.t_final / params.solver.dt))
    snapshot_every = params.solver.snapshot_every or max(1, n_steps // 120)
    _, diag = sim.run(
        psi0=psi0,
        t_final=params.solver.t_final,
        log_every=params.solver.log_every,
        snapshot_every=snapshot_every,
    )

    if params.output.save_png:
        plot_simulation_summary(
            grid=grid,
            potential=potential,
            diagnostics=diag,
            output_path=results_dir / "free_gaussian_overview.png",
        )
    if params.output.save_gif:
        create_density_gif(
            grid=grid,
            diagnostics=diag,
            output_path=results_dir / "free_gaussian_density.gif",
            fps=params.output.gif_fps,
        )
        create_phase_gif(
            grid=grid,
            diagnostics=diag,
            output_path=results_dir / "free_gaussian_phase.gif",
            fps=params.output.gif_fps,
        )

    print(f"Final particle number: {diag.particle_numbers[-1]:.8f}")
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
